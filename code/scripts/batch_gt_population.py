#!/usr/bin/env python3
"""
Step A.4: target population, seniority, prominence bins (L/M/H within field x seniority), top group and GT shares
(per field x seniority, field, seniority, all). Needs steps A.1 and A.3 (without A.3, gender/ethnicity = unknown).

Writes to <gt_dir>/ (default results/groundtruth/):
    gt_authors.parquet, gt_members.parquet, gt_shares.parquet, gt_cutoffs.parquet   (see libs/groundtruth/population.py)

Usage (from any folder):
    python batch_gt_population.py --authors_dir /tmp/smoke/authors --demographics_dir /tmp/smoke/demographics --gt_dir /tmp/smoke
    python batch_gt_population.py --threads 12 --memory_limit 40GB
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from groundtruth import db
from groundtruth.fields import load_field_mapping
from groundtruth.population import create_views, write_outputs
from utils import analysis_config as ac
from utils import constants as cons
from utils import ios
from utils import logs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    db.add_resource_arguments(parser)
    parser.add_argument('--authors_dir', default=cons.DEFAULT_GT_AUTHORS_DIR, help='Output of step A.1')
    parser.add_argument('--demographics_dir', default=cons.DEFAULT_GT_DEMOGRAPHICS_DIR, help='Output of step A.3')
    parser.add_argument('--gt_dir', default=cons.DEFAULT_GT_DIR, help='Where to write gt_*.parquet')
    parser.add_argument('--field_mapping_file', default=cons.DEFAULT_FIELD_MAPPING_FILE)
    parser.add_argument('--min_field_share', type=float, default=cons.DEFAULT_GT_MIN_FIELD_SHARE,
                        help='Minimum share of a scholar\'s works in a field to belong to it (default 0: any work)')
    parser.add_argument('--gender_edges', nargs='+', type=float, default=cons.DEFAULT_GENDER_EDGES,
                        help='Cut points on p_female (increasing, in (0, 1)), e.g. 0.1 0.9 or 0.1 0.4 0.6 0.9')
    parser.add_argument('--gender_labels', nargs='+', default=cons.DEFAULT_GENDER_LABELS,
                        help='One label per interval [0, e1), [e1, e2), ..., [ek, 1], e.g. male neutral female')
    parser.add_argument('--ethnicity_min_prob', type=float, default=cons.DEFAULT_ETHNICITY_MIN_PROB,
                        help='raceBERT: label = top class if its probability >= this (else unknown)')
    parser.add_argument('--reference_year', type=int, default=cons.GT_REFERENCE_YEAR, help='academic_age = year - first_pub_year')
    parser.add_argument('--junior_max', type=int, default=cons.DEFAULT_SENIORITY_JUNIOR_MAX)
    parser.add_argument('--senior_min', type=int, default=cons.DEFAULT_SENIORITY_SENIOR_MIN)
    parser.add_argument('--single_cutoff', type=int, default=cons.DEFAULT_SENIORITY_SINGLE_CUTOFF)
    parser.add_argument('--bin_edges', nargs='+', type=float, default=cons.DEFAULT_GT_BIN_EDGES,
                        help='Prominence: percent-rank cut points (increasing, in (0, 1)), e.g. 0.333 0.667')
    parser.add_argument('--bin_labels', nargs='+', default=cons.DEFAULT_GT_BIN_LABELS,
                        help='Prominence: one label per interval, e.g. L M H')
    parser.add_argument('--top_percentile', type=float, default=cons.DEFAULT_GT_TOP_PERCENTILE,
                        help='Top group: top x %% by works and by citations within field x seniority')
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing outputs')
    logs.add_log_arguments(parser)
    ac.add_config_argument(parser)
    return parser


def main() -> None:
    args = ac.parse_with_config(build_parser())
    logs.setup_logging('gt_population', args, f"minshare{args.min_field_share:g}")
    if (Path(args.gt_dir) / 'gt_shares.parquet').exists() and not args.overwrite:
        ios.printf(f"Skipped (exists, use --overwrite): {args.gt_dir}/gt_shares.parquet")
        return
    authors_files = sorted(Path(args.authors_dir).glob('bucket-*.parquet'))
    if not authors_files:
        raise SystemExit(f"No bucket-*.parquet in {args.authors_dir} (run batch_gt_bibliometrics.py first)")
    demo_files = sorted(Path(args.demographics_dir).glob('bucket-*.parquet'))
    if len(demo_files) < len(authors_files):
        ios.printf(f"WARNING: demographics for {len(demo_files)} of {len(authors_files)} buckets "
                   f"(missing authors get gender/ethnicity = unknown)")

    start = ios.datetime.now()
    con = db.connect(args, ':memory:')
    create_views(con, str(Path(args.authors_dir) / 'bucket-*.parquet'),
                 str(Path(args.demographics_dir) / 'bucket-*.parquet') if demo_files else None,
                 load_field_mapping(args.field_mapping_file), args.min_field_share, args.reference_year,
                 args.junior_max, args.senior_min, args.single_cutoff, args.bin_edges, args.top_percentile, args.bin_labels,
                 args.gender_edges, args.gender_labels, args.ethnicity_min_prob)
    counts = write_outputs(con, args.gt_dir)
    ios.printf(f"Done in {ios.datetime.now() - start} | rows: {counts} -> {args.gt_dir}")


if __name__ == '__main__':
    main()
