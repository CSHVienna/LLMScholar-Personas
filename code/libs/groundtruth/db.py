"""
DuckDB plumbing for the ground-truth steps: read-only OpenAlex connection, resource settings,
author buckets and shared CLI arguments.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import duckdb

from utils import constants as cons


def add_resource_arguments(parser: argparse.ArgumentParser) -> None:
    """DuckDB threads / memory / spill folder. Defaults are deliberately bounded: unlimited, DuckDB takes 80 % of the
    machine's RAM per process (this OOM-killed the shared server once)."""
    parser.add_argument('--threads', type=int, default=cons.DEFAULT_DUCKDB_THREADS, help='DuckDB threads')
    parser.add_argument('--memory_limit', default=cons.DEFAULT_DUCKDB_MEMORY, help="DuckDB memory limit, e.g. '40GB'")
    parser.add_argument('--temp_dir', default=cons.DEFAULT_DUCKDB_TEMP_DIR,
                        help='DuckDB spill folder (default <project>/tmp/duckdb)')


def add_duckdb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--openalex_db', default=cons.DEFAULT_OPENALEX_DB, help='OpenAlex DuckDB file (opened read-only)')
    add_resource_arguments(parser)


def add_bucket_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--n_buckets', type=int, default=cons.DEFAULT_GT_N_BUCKETS,
                        help='Authors are split by author_id %% n_buckets (must match the authorships step)')
    parser.add_argument('--buckets', nargs='+', type=int, default=None,
                        help='Buckets to process (default: all); GNU parallel: --buckets {}')
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing output files')


def connect(args: argparse.Namespace, database: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """Connection to the OpenAlex database (read-only), or an in-memory one if ``database`` is ':memory:'."""
    database = database or args.openalex_db
    con = duckdb.connect(database, read_only=database != ':memory:')
    if getattr(args, 'threads', None):
        con.execute(f"SET threads = {int(args.threads)}")
    if getattr(args, 'memory_limit', None):
        con.execute(f"SET memory_limit = '{args.memory_limit}'")
    if getattr(args, 'temp_dir', None):
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{args.temp_dir}'")
    con.execute("SET preserve_insertion_order = false")
    return con


def buckets_of(args: argparse.Namespace) -> List[int]:
    buckets = args.buckets if args.buckets is not None else list(range(args.n_buckets))
    bad = [b for b in buckets if not 0 <= b < args.n_buckets]
    if bad:
        raise ValueError(f"Buckets out of range [0, {args.n_buckets}): {bad}")
    return buckets


def bucket_file(folder: Path | str, bucket: int, extension: str = 'parquet') -> Path:
    """Per-bucket file: <folder>/bucket-<bb>.<extension>"""
    return Path(folder) / f"bucket-{bucket:02d}.{extension}"


def authorships_glob(folder: Path | str, bucket: Optional[int] = None) -> str:
    """Parquet glob of the authorships of one bucket (or of all buckets), over all complete chunks."""
    part = f"bucket={bucket}" if bucket is not None else "bucket=*"
    return str(Path(folder) / 'chunk=*' / part / '*.parquet')


def write_parquet_atomic(df, path: Path | str) -> None:
    """Write a DataFrame to <path>.tmp, then rename: a crash never leaves a partial file that looks complete."""
    from utils import ios
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    ios.to_parquet(df, tmp)
    tmp.replace(path)


def is_person_sql(name_column: str = 'display_name', orcid_column: str = 'orcid') -> str:
    """SQL boolean: the OpenAlex author is a person (has an ORCID, or the name does not look like an organisation)."""
    pattern = cons.ORGANIZATION_NAME_PATTERN.replace("'", "''")
    return f"({orcid_column} IS NOT NULL OR NOT regexp_matches(coalesce({name_column}, ''), '{pattern}'))"


def sql_path(path: Path | str) -> str:
    """Path as a single-quoted SQL literal."""
    return "'" + str(path).replace("'", "''") + "'"


def bucket_params(args: argparse.Namespace) -> str:
    """Log-file tag: bucket(s) processed."""
    buckets = buckets_of(args)
    if len(buckets) == args.n_buckets:
        return 'all'
    return f"bucket{buckets[0]:02d}" if len(buckets) == 1 else f"{len(buckets)}buckets"
