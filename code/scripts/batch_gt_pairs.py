#!/usr/bin/env python3
"""
Step A.0b: coauthor pairs per A.0 chunk (needs A.0 complete). Intermediate of the edge list (A.1a).

Writes <output_dir>/chunk=<c>/bucket=<b>/*.parquet (default results/groundtruth/coauthor_pairs/) with
author_id, coauthor_id, n_shared_works, newman_weight, min_team, first_year, last_year
(only works with <= --max_team authors, default 100).
Each chunk is a self-join inside the chunk; a chunk appears only when complete, so re-running resumes.
Can be deleted once A.1a (batch_gt_edges.py) is complete.

Usage (from any folder):
    python batch_gt_pairs.py --threads 12 --memory_limit 40GB --temp_dir <project>/tmp/duckdb   # all chunks, one by one
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import json
from pathlib import Path

from groundtruth import db
from groundtruth.authorships import PARAMS_FILE, check_params, chunk_dir, missing_chunks
from groundtruth.coauthorship import write_pairs_chunk
from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs


def run_chunk(chunk: int, args: argparse.Namespace) -> str:
    if chunk_dir(args.output_dir, chunk).exists():
        return f"[chunk {chunk}] skipped (complete)"
    con = db.connect(args, ':memory:')
    n_rows = write_pairs_chunk(con, args.authorships_dir, args.output_dir, chunk, args.n_buckets, args.max_team)
    con.close()
    return f"[chunk {chunk}] {n_rows} pairs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    db.add_resource_arguments(parser)
    parser.add_argument('--authorships_dir', default=cons.DEFAULT_GT_AUTHORSHIPS_DIR, help='Output of step A.0')
    parser.add_argument('--output_dir', default=cons.DEFAULT_GT_PAIRS_DIR)
    parser.add_argument('--max_team', type=int, default=cons.DEFAULT_GT_COAUTHOR_MAX_TEAM,
                        help='Only works with <= this many authors create coauthor pairs')
    parser.add_argument('--chunks', nargs='+', type=int, default=None, help='Chunks (default: all); GNU parallel: --chunks {}')
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chunks = args.chunks
    tag = 'all' if chunks is None else f"chunk{chunks[0]:03d}" if len(chunks) == 1 else f"{len(chunks)}chunks"
    logs.setup_logging('gt_pairs', args, tag)
    missing = missing_chunks(args.authorships_dir)
    if missing:
        raise SystemExit(f"Step A.0 is incomplete: {len(missing)} chunks missing (e.g. {missing[:5]})")
    a0 = json.loads((Path(args.authorships_dir) / PARAMS_FILE).read_text())
    args.n_buckets = a0['n_buckets']
    check_params(args.output_dir, {'n_buckets': a0['n_buckets'], 'n_chunks': a0['n_chunks'], 'max_team': args.max_team})
    chunks = list(range(a0['n_chunks'])) if chunks is None else chunks
    ios.printf(f"Chunks: {len(chunks)} of {a0['n_chunks']}")
    batch.run_jobs(chunks, run_chunk, args)


if __name__ == '__main__':
    main()
