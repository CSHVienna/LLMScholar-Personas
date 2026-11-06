#!/usr/bin/env python3
"""
Step A.5: coauthorship network among a given set of authors (connectedness). Needs step A.1a (the edge list).

--author_ids_file: a parquet file with an `author_id` column, e.g. the matched recommended authors (B.6 author_id) or a GT
sample. Writes <output_dir>/coauthorship_<name of the ids file>.parquet (default results/groundtruth/coauthorship/)
with undirected edges author_a < author_b, n_shared_works, newman_weight, min_team, first_year, last_year.
Filter / weight later (e.g. min_team <= 100 to drop pairs that only met on hyperauthorship papers).

Usage (from any folder):
    python batch_gt_coauthorship.py --author_ids_file results/matching/authors/main/recommended_authors.parquet
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from groundtruth import db
from groundtruth.coauthorship import write_network
from utils import constants as cons
from utils import ios
from utils import logs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    db.add_resource_arguments(parser)
    parser.add_argument('--author_ids_file', required=True, help='Parquet file with an author_id column')
    parser.add_argument('--edges_dir', default=cons.DEFAULT_GT_EDGES_DIR, help='Output of step A.1a')
    parser.add_argument('--output_dir', default=cons.DEFAULT_GT_COAUTHORSHIP_DIR)
    parser.add_argument('--overwrite', action='store_true', help='Recompute an existing output file')
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    name = Path(args.author_ids_file).stem
    logs.setup_logging('gt_coauthorship', args, name)
    output_file = Path(args.output_dir) / f"coauthorship_{name}.parquet"
    if output_file.exists() and not args.overwrite:
        ios.printf(f"Skipped (exists, use --overwrite): {output_file}")
        return
    start = ios.datetime.now()
    con = db.connect(args, ':memory:')
    n_edges = write_network(con, output_file, args.edges_dir, args.author_ids_file)
    ios.printf(f"Done in {ios.datetime.now() - start} | {n_edges} edges -> {output_file}")


if __name__ == '__main__':
    main()
