#!/usr/bin/env python3
"""
Step A.1a: the coauthorship edge list, one author bucket at a time (needs A.0b complete).

Writes <output_dir>/bucket-<bb>.parquet (default results/groundtruth/coauthor_edges/), one row per directed pair:
    author_id, coauthor_id, n_shared_works, newman_weight, min_team, first_year, last_year
(see libs/groundtruth/coauthorship.py). Used by A.1 (n_coauthors = rows per author) and A.5 (network).

Usage (from any folder):
    python batch_gt_edges.py --threads 12 --memory_limit 40GB --temp_dir <project>/tmp/duckdb   # all buckets, one by one
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import json
from pathlib import Path

from groundtruth import db
from groundtruth.authorships import PARAMS_FILE, missing_chunks
from groundtruth.coauthorship import write_edges_bucket
from utils import batch
from utils import constants as cons
from utils import logs


def run_bucket(bucket: int, args: argparse.Namespace) -> str:
    output_file = db.bucket_file(args.output_dir, bucket)
    if output_file.exists() and not args.overwrite:
        return f"[bucket {bucket}] skipped (exists): {output_file}"
    con = db.connect(args, ':memory:')
    n_rows = write_edges_bucket(con, args.pairs_dir, output_file, bucket)
    con.close()
    return f"[bucket {bucket}] {n_rows} edges -> {output_file.name}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    db.add_resource_arguments(parser)
    db.add_bucket_arguments(parser)
    parser.add_argument('--pairs_dir', default=cons.DEFAULT_GT_PAIRS_DIR, help='Output of step A.0b')
    parser.add_argument('--output_dir', default=cons.DEFAULT_GT_EDGES_DIR, help='Where to write bucket-<bb>.parquet')
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logs.setup_logging('gt_edges', args, db.bucket_params(args))
    missing = missing_chunks(args.pairs_dir)
    if missing:
        raise SystemExit(f"Step A.0b is incomplete: {len(missing)} chunks missing (e.g. {missing[:5]}); re-run it first")
    n_buckets = json.loads((Path(args.pairs_dir) / PARAMS_FILE).read_text())['n_buckets']
    if n_buckets != args.n_buckets:
        raise SystemExit(f"--n_buckets {args.n_buckets} differs from the pairs' n_buckets {n_buckets}")
    batch.run_jobs(db.buckets_of(args), run_bucket, args)


if __name__ == '__main__':
    main()
