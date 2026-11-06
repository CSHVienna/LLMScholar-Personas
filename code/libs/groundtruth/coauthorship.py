"""
Coauthorship edge list (one table for the coauthor counts AND the coauthorship network). Only works with
<= max_team authors (default 100) create pairs: hyperauthorship works (up to 5000 authors in OpenAlex, i.e. 25M
pairs each) would dominate size and time. Each edge keeps enough to filter / weight later.

Edge columns (directed: every pair appears as (a, b) and (b, a), so an author's edges sit in its own bucket):
    author_id, coauthor_id
    n_shared_works    works they share
    newman_weight     sum over shared works of 1 / (team - 1)   (Newman 2001: a 3000-author paper weighs ~0)
    min_team          smallest team size among the shared works (<= max_team; e.g. keep min_team <= 20 for a
                      stricter collaboration network)
    first_year, last_year

Steps (all resumable, outputs appear only when complete):
    A.0b  pairs per A.0 chunk: every work lies inside one chunk, so the pairs of its authors come from a self-join
          within the chunk (~3M authorships; never a join against the full table)
              <pairs_dir>/chunk=<c>/bucket=<b>/*.parquet
    A.1a  edges per author bucket: the chunk pairs aggregated over all chunks (an author pair can share works in
          several chunks)
              <edges_dir>/bucket-<bb>.parquet
          n_coauthors of an author = its number of rows here (bibliometrics.py)
    A.5   network among given authors (e.g. the factual recommended authors): edges with both ends in the set
"""

import shutil
from pathlib import Path

import duckdb

from groundtruth.authorships import chunk_dir
from groundtruth.db import authorships_glob, sql_path


# --- A.0b: pairs per chunk ---------------------------------------------------------------------------------------
def pairs_query(authorships_dir: Path | str, chunk: int, n_buckets: int, max_team: int) -> str:
    files = sql_path(Path(chunk_dir(authorships_dir, chunk)) / 'bucket=*' / '*.parquet')
    return f"""
        WITH a AS (
            SELECT DISTINCT work_id, author_id, year FROM read_parquet({files}, hive_partitioning = false)
        ), t AS (   -- team = distinct authors of the work (duplicates / missing ids in OpenAlex removed)
            SELECT a.*, count(*) OVER (PARTITION BY work_id) AS team FROM a
        ), w AS (
            SELECT * FROM t WHERE team <= {int(max_team)}
        )
        SELECT x.author_id, y.author_id AS coauthor_id, count(*) AS n_shared_works,
               sum(1.0 / (x.team - 1)) AS newman_weight, min(x.team) AS min_team,
               min(x.year) AS first_year, max(x.year) AS last_year,
               (x.author_id % {int(n_buckets)})::INTEGER AS bucket
        FROM w x JOIN w y ON x.work_id = y.work_id AND x.author_id <> y.author_id
        GROUP BY x.author_id, y.author_id
    """


def write_pairs_chunk(con: duckdb.DuckDBPyConnection, authorships_dir: Path | str, pairs_dir: Path | str, chunk: int,
                      n_buckets: int, max_team: int) -> int:
    """Write the pairs of one chunk (via a temporary folder, renamed when complete); returns the number of rows."""
    final = chunk_dir(pairs_dir, chunk)
    tmp = Path(pairs_dir) / f"_tmp_chunk={chunk}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if not list(chunk_dir(authorships_dir, chunk).glob('bucket=*/*.parquet')):   # every work filtered (e.g. xpac)
        tmp.rename(final)
        return 0
    row = con.execute(f"COPY ({pairs_query(authorships_dir, chunk, n_buckets, max_team)}) TO {sql_path(tmp)} "
                      f"(FORMAT parquet, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE, COMPRESSION zstd)").fetchone()
    tmp.rename(final)
    return int(row[0]) if row else -1


# --- A.1a: edges per author bucket -------------------------------------------------------------------------------
def edges_query(pairs_dir: Path | str, bucket: int) -> str:
    return f"""
        SELECT author_id, coauthor_id, sum(n_shared_works)::BIGINT AS n_shared_works,
               sum(newman_weight) AS newman_weight, min(min_team) AS min_team,
               min(first_year) AS first_year, max(last_year) AS last_year
        FROM read_parquet({sql_path(authorships_glob(pairs_dir, bucket))}, hive_partitioning = false)
        GROUP BY author_id, coauthor_id
        ORDER BY author_id, coauthor_id
    """


def write_edges_bucket(con: duckdb.DuckDBPyConnection, pairs_dir: Path | str, output_file: Path | str,
                       bucket: int) -> int:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_name(output_file.name + '.tmp')
    query = edges_query(pairs_dir, bucket)
    if not list(Path(pairs_dir).glob(f"chunk=*/bucket={bucket}/*.parquet")):    # no pairs at all (tiny samples only)
        query = ("SELECT NULL::BIGINT AS author_id, NULL::BIGINT AS coauthor_id, NULL::BIGINT AS n_shared_works, "
                 "NULL::DOUBLE AS newman_weight, NULL::BIGINT AS min_team, NULL::INTEGER AS first_year, "
                 "NULL::INTEGER AS last_year WHERE false")
    con.execute(f"COPY ({query}) TO {sql_path(tmp)} (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(output_file)
    return con.execute(f"SELECT count(*) FROM read_parquet({sql_path(output_file)})").fetchone()[0]


# --- A.5: network among given authors ----------------------------------------------------------------------------
def network_query(edges_dir: Path | str, author_ids_file: Path | str) -> str:
    """Undirected edges (author_a < author_b) with both ends in the given set."""
    return f"""
        WITH ids AS (SELECT DISTINCT author_id FROM read_parquet({sql_path(author_ids_file)}))
        SELECT author_id AS author_a, coauthor_id AS author_b, n_shared_works, newman_weight, min_team,
               first_year, last_year
        FROM read_parquet({sql_path(Path(edges_dir) / 'bucket-*.parquet')})
        WHERE author_id < coauthor_id
          AND author_id IN (SELECT author_id FROM ids) AND coauthor_id IN (SELECT author_id FROM ids)
        ORDER BY author_a, author_b
    """


def write_network(con: duckdb.DuckDBPyConnection, output_file: Path | str, edges_dir: Path | str,
                  author_ids_file: Path | str) -> int:
    """Write the edge list (via a temporary file); returns the number of edges."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_name(output_file.name + '.tmp')
    con.execute(f"COPY ({network_query(edges_dir, author_ids_file)}) TO {sql_path(tmp)} (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(output_file)
    return con.execute(f"SELECT count(*) FROM read_parquet({sql_path(output_file)})").fetchone()[0]
