#!/usr/bin/env python3
"""
Step A.0: works x authorships (filtered works), in resumable chunks of `works` rows, partitioned by author bucket.

Works kept: not paratext, not retracted, not xpac (OpenAlex 2025 "expansion pack"; --include_xpac to keep them),
publication year in [--min_year, --max_year].
Writes <output_dir>/chunk=<c>/bucket=<b>/*.parquet (default results/groundtruth/authorships/) with
    author_id, work_id, year, cited_by_count, n_authors, field_id, subfield_id (primary topic)
A chunk appears only when complete: re-running skips finished chunks and redoes a crashed one.
This is the only scan of `works` (476M rows); the output is the lookup every later GT step reads.

Usage (from any folder):
    python batch_gt_authorships.py --count_chunks                                  # number of chunks (for parallel)
    parallel -j 12 python batch_gt_authorships.py --chunks {} --threads 3 ::: $(seq 0 237)
    python batch_gt_authorships.py --chunks 0 1 --chunk_size 20000 --output_dir /tmp/smoke/authorships   # smoke
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import shutil
from pathlib import Path

from groundtruth import db
from groundtruth.authorships import check_params, chunk_dir, n_chunks, write_chunk
from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs


def run_chunk(chunk: int, args: argparse.Namespace) -> str:
    if chunk_dir(args.output_dir, chunk).exists():
        return f"[chunk {chunk}] skipped (complete)"
    con = db.connect(args)
    n_rows = write_chunk(con, args.output_dir, chunk, args.chunk_size, args.min_year, args.max_year, args.n_buckets,
                         args.include_xpac)
    con.close()
    return f"[chunk {chunk}] {n_rows} authorships"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    db.add_duckdb_arguments(parser)
    parser.add_argument('--output_dir', default=cons.DEFAULT_GT_AUTHORSHIPS_DIR, help='Partitioned output folder')
    parser.add_argument('--n_buckets', type=int, default=cons.DEFAULT_GT_N_BUCKETS, help='Author buckets (author_id %% n)')
    parser.add_argument('--chunk_size', type=int, default=cons.DEFAULT_GT_WORKS_CHUNK_SIZE, help='works rows per chunk')
    parser.add_argument('--chunks', nargs='+', type=int, default=None,
                        help='Chunks to process (default: all); GNU parallel: --chunks {}')
    parser.add_argument('--count_chunks', action='store_true', help='Print the number of chunks and exit')
    parser.add_argument('--min_year', type=int, default=cons.GT_MIN_YEAR)
    parser.add_argument('--max_year', type=int, default=cons.GT_REFERENCE_YEAR)
    parser.add_argument('--include_xpac', action='store_true', default=cons.DEFAULT_GT_INCLUDE_XPAC,
                        help='Also count xpac works (OpenAlex expansion pack; excluded by default)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete the whole output folder first (do NOT use with parallel jobs)')
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.count_chunks:
        print(n_chunks(db.connect(args), args.chunk_size))
        return
    chunks = args.chunks
    tag = 'all' if chunks is None else f"chunk{chunks[0]:03d}" if len(chunks) == 1 else f"{len(chunks)}chunks"
    logs.setup_logging('gt_authorships', args, tag + ('_xpac' if args.include_xpac else ''))
    if args.overwrite and Path(args.output_dir).exists():
        shutil.rmtree(args.output_dir)

    total = n_chunks(db.connect(args), args.chunk_size)
    check_params(args.output_dir, {'min_year': args.min_year, 'max_year': args.max_year,
                                   'include_xpac': args.include_xpac, 'n_buckets': args.n_buckets,
                                   'chunk_size': args.chunk_size, 'n_chunks': total,
                                   'openalex_db': Path(args.openalex_db).name})
    chunks = list(range(total)) if chunks is None else chunks
    bad = [c for c in chunks if not 0 <= c < total]
    if bad:
        raise SystemExit(f"Chunks out of range [0, {total}): {bad}")
    ios.printf(f"Chunks: {len(chunks)} of {total} (chunk size {args.chunk_size})")
    batch.run_jobs(chunks, run_chunk, args)


if __name__ == '__main__':
    main()
