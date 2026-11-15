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
from pathlib import Path
from typing import List

import duckdb

from groundtruth.db import copy_partitioned
from utils.batch import PARAMS_FILE


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


def write_chunk(con: duckdb.DuckDBPyConnection, output_dir: Path | str, chunk: int, chunk_size: int,
                min_year: int, max_year: int, n_buckets: int, include_xpac: bool = False) -> int:
    """Write one chunk (via a temporary folder, renamed when complete); returns the number of rows written."""
    query = authorships_query(chunk, chunk_size, min_year, max_year, n_buckets, include_xpac)
    n = copy_partitioned(con, query, chunk_dir(output_dir, chunk), 'bucket')
    return -1 if n is None else n


def missing_chunks(output_dir: Path | str) -> List[int]:
    """Chunks recorded in _params.json that are not complete yet (raises if the step never ran)."""
    path = Path(output_dir) / PARAMS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No {PARAMS_FILE} in {output_dir}: run batch_gt_authorships.py first")
    expected = json.loads(path.read_text())['n_chunks']
    return [c for c in range(expected) if not chunk_dir(output_dir, c).exists()]
