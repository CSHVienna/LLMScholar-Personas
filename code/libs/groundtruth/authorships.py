"""
Step A.0: works x authorships (the only scan of `works`), in resumable chunks of `works` rows.

Works kept: not paratext, not retracted, not xpac (unless include_xpac), publication year in [min_year, max_year].
xpac = OpenAlex "expansion pack" (2025 additions from DataCite / repositories; excluded by OpenAlex by default).

Chunk c = works rows [c * chunk_size, (c + 1) * chunk_size) (DuckDB rowid ranges are read without a full scan).
Each chunk is written to <output_dir>/_tmp_chunk=<c>/ and renamed to <output_dir>/chunk=<c>/ only when complete,
so a crashed run is resumed by re-running: finished chunks are skipped, partial ones redone. Layout:
    <output_dir>/chunk=<c>/bucket=<b>/*.parquet      (bucket = author_id % n_buckets)
    <output_dir>/_params.json                        (filters, chunk size, number of chunks: checked on every run)
Columns: author_id, work_id, year, cited_by_count, n_authors (authorships on the work),
field_id / subfield_id (of the work's PRIMARY topic; NULL if the work has none).

The same author can appear twice on a work in OpenAlex; such duplicates are removed downstream (bibliometrics.py).
"""

import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List

import duckdb

from groundtruth.db import sql_path

PARAMS_FILE = '_params.json'


def works_filter_sql(min_year: int, max_year: int, include_xpac: bool = False) -> str:
    xpac = '' if include_xpac else 'AND NOT coalesce(is_xpac, false) '
    return (f"NOT coalesce(is_paratext, false) AND NOT coalesce(is_retracted, false) {xpac}"
            f"AND publication_year BETWEEN {int(min_year)} AND {int(max_year)}")


def n_chunks(con: duckdb.DuckDBPyConnection, chunk_size: int) -> int:
    return math.ceil(con.execute("SELECT count(*) FROM works").fetchone()[0] / chunk_size)


def authorships_query(chunk: int, chunk_size: int, min_year: int, max_year: int, n_buckets: int,
                      include_xpac: bool = False) -> str:
    start = int(chunk) * int(chunk_size)
    return f"""
        WITH w AS (
            SELECT id, publication_year, cited_by_count, authorships,
                   primary_topic.field.id AS field_id, primary_topic.subfield.id AS subfield_id
            FROM works
            WHERE rowid >= {start} AND rowid < {start + int(chunk_size)}
              AND {works_filter_sql(min_year, max_year, include_xpac)}
        ), a AS (
            SELECT id AS work_id, publication_year AS year, coalesce(cited_by_count, 0) AS cited_by_count,
                   len(authorships) AS n_authors, field_id, subfield_id,
                   unnest(authorships).author.id AS author_id
            FROM w
        )
        SELECT author_id, work_id, year, cited_by_count, n_authors, field_id, subfield_id,
               (author_id % {int(n_buckets)})::INTEGER AS bucket
        FROM a
        WHERE author_id IS NOT NULL
    """


def chunk_dir(output_dir: Path | str, chunk: int) -> Path:
    return Path(output_dir) / f"chunk={chunk}"


def check_params(output_dir: Path | str, params: Dict, overwrite: bool = False) -> None:
    """Record the run parameters once (atomically); refuse to mix chunks written with different parameters.
    overwrite: replace the stored parameters (the caller recomputes every output)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PARAMS_FILE
    if overwrite:   # atomic replace; parallel jobs write the same content
        tmp = output_dir / f".{PARAMS_FILE}.{os.getpid()}"
        tmp.write_text(json.dumps(params, indent=1))
        os.replace(tmp, path)
        return
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # first process wins (GNU parallel safe)
        with os.fdopen(fd, 'w') as f:
            json.dump(params, f, indent=1)
        return
    except FileExistsError:
        pass
    stored = json.loads(path.read_text() or '{}')
    if stored and stored != params:
        raise ValueError(f"{path} was written with other parameters: {stored} != {params}. "
                         f"Use another --output_dir or --overwrite (recomputes every output).")


def write_chunk(con: duckdb.DuckDBPyConnection, output_dir: Path | str, chunk: int, chunk_size: int,
                min_year: int, max_year: int, n_buckets: int, include_xpac: bool = False) -> int:
    """Write one chunk (via a temporary folder, renamed when complete); returns the number of rows written."""
    final = chunk_dir(output_dir, chunk)
    tmp = Path(output_dir) / f"_tmp_chunk={chunk}"
    if tmp.exists():
        shutil.rmtree(tmp)                                          # leftover of a crashed run
    tmp.mkdir(parents=True)
    query = authorships_query(chunk, chunk_size, min_year, max_year, n_buckets, include_xpac)
    row = con.execute(f"COPY ({query}) TO {sql_path(tmp)} "
                      f"(FORMAT parquet, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE, COMPRESSION zstd)").fetchone()
    tmp.rename(final)
    return int(row[0]) if row else -1


def missing_chunks(output_dir: Path | str) -> List[int]:
    """Chunks recorded in _params.json that are not complete yet (raises if the step never ran)."""
    path = Path(output_dir) / PARAMS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No {PARAMS_FILE} in {output_dir}: run batch_gt_authorships.py first")
    expected = json.loads(path.read_text())['n_chunks']
    return [c for c in range(expected) if not chunk_dir(output_dir, c).exists()]
