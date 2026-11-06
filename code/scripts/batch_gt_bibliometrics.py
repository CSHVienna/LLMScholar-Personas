#!/usr/bin/env python3
"""
Step A.1: bibliometrics per author, one author bucket at a time (needs steps A.0 and A.1a).

Writes <output_dir>/bucket-<bb>.parquet (default results/groundtruth/authors/), one row per author with >= 1
counted work: first/last publication year, # works, # citations, h / i10 / e index, # unique coauthors
(works with <= 100 authors, from the A.1a edge list), works per prompt field (primary topic, field_mapping.json),
OpenAlex summary stats, latest affiliation country.

Usage (from any folder):
    python batch_gt_bibliometrics.py --buckets 0 --authorships_dir /tmp/smoke/authorships --output_dir /tmp/smoke/authors
    parallel -j 8 python batch_gt_bibliometrics.py --buckets {} --threads 4 ::: $(seq 0 63)
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from groundtruth import db
from groundtruth.authorships import missing_chunks
from groundtruth.bibliometrics import compute_bibliometrics
from groundtruth.fields import load_field_mapping
from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs


def run_bucket(bucket: int, args: argparse.Namespace) -> str:
    output_file = db.bucket_file(args.output_dir, bucket)
    if output_file.exists() and not args.overwrite:
        return f"[bucket {bucket}] skipped (exists): {output_file}"
    if not list(Path(args.authorships_dir).glob(f"chunk=*/bucket={bucket}/*.parquet")):
        return f"[bucket {bucket}] no authorships in {args.authorships_dir}"

    con = db.connect(args)
    df = compute_bibliometrics(con, args.authorships_dir, args.edges_dir, bucket, args.n_buckets, args.fields,
                               log=ios.printf)
    con.close()
    db.write_parquet_atomic(df, output_file)
    in_fields = {f.field: int((df[f.column] >= 1).sum()) for f in args.fields}
    return f"[bucket {bucket}] {len(df)} authors -> {output_file.name} | with >= 1 field work: {in_fields}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    db.add_duckdb_arguments(parser)
    db.add_bucket_arguments(parser)
    parser.add_argument('--authorships_dir', default=cons.DEFAULT_GT_AUTHORSHIPS_DIR, help='Output of step A.0')
    parser.add_argument('--field_mapping_file', default=cons.DEFAULT_FIELD_MAPPING_FILE)
    parser.add_argument('--edges_dir', default=cons.DEFAULT_GT_EDGES_DIR, help='Output of step A.1a (edge list)')
    parser.add_argument('--output_dir', default=cons.DEFAULT_GT_AUTHORS_DIR, help='Where to write bucket-<bb>.parquet')
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logs.setup_logging('gt_bibliometrics', args, db.bucket_params(args))
    missing = missing_chunks(args.authorships_dir)
    if missing:
        raise SystemExit(f"Step A.0 is incomplete: {len(missing)} chunks missing (e.g. {missing[:5]}); re-run it first")
    absent = [b for b in db.buckets_of(args) if not db.bucket_file(args.edges_dir, b).exists()]
    if absent:
        raise SystemExit(f"Step A.1a (batch_gt_edges.py) is incomplete: no edges for buckets {absent[:5]}...")
    args.fields = load_field_mapping(args.field_mapping_file)
    batch.run_jobs(db.buckets_of(args), run_bucket, args)


if __name__ == '__main__':
    main()
