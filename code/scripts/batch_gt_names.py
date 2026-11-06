#!/usr/bin/env python3
"""
Step A.2: name index over ALL OpenAlex authors (display name + alternatives), one author bucket at a time.
Independent of steps A.0-A.1.

Writes <output_dir>/bucket-<bb>.parquet (default results/groundtruth/names/) with
    name_key (same normalisation as the recommendations' name_key), name_key_short (first + last token),
    author_id, is_display_name

Usage (from any folder):
    python batch_gt_names.py --buckets 0 --limit_authors 10000 --output_dir /tmp/smoke/names
    parallel -j 16 python batch_gt_names.py --buckets {} --threads 2 ::: $(seq 0 63)
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse

from groundtruth import db
from groundtruth.names import write_name_index
from utils import batch
from utils import constants as cons
from utils import logs


def run_bucket(bucket: int, args: argparse.Namespace) -> str:
    output_file = db.bucket_file(args.output_dir, bucket)
    if output_file.exists() and not args.overwrite:
        return f"[bucket {bucket}] skipped (exists): {output_file}"
    con = db.connect(args)
    n_rows = write_name_index(con, output_file, bucket, args.n_buckets, args.limit_authors)
    con.close()
    return f"[bucket {bucket}] {n_rows} (author, name) rows -> {output_file.name}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    db.add_duckdb_arguments(parser)
    db.add_bucket_arguments(parser)
    parser.add_argument('--limit_authors', type=int, default=None, help='Only the first N authors per bucket (smoke tests)')
    parser.add_argument('--output_dir', default=cons.DEFAULT_GT_NAMES_DIR, help='Where to write bucket-<bb>.parquet')
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = db.bucket_params(args) + (f"_limit{args.limit_authors}" if args.limit_authors else '')
    logs.setup_logging('gt_names', args, params)
    batch.run_jobs(db.buckets_of(args), run_bucket, args)


if __name__ == '__main__':
    main()
