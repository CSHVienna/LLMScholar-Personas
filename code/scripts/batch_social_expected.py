#!/usr/bin/env python3
"""
Step 9a: expected social metrics under a random recommender (libs/metrics/expected.py), once per variant, before
the social metrics (step 9 reads them to write <metric>_adj = observed - expected).

For every field x seniority of the GT and every list size n = 1..--n_max: expected parity / diversity of each
attribute (random labels with the GT shares), connectedness (random members and their coauthorship edges) and
similarity (mean pairwise cosine of random members). Uses the parity target and the unknown handling of the config.

Writes <output_dir>/social_expected/expected.parquet (field, seniority, metric, n, expected) and _params.json
(default output results/metrics/main/).

Usage (from any folder; run_metrics_pipeline.sh -s expected):
    python batch_social_expected.py --config config/analysis_smoke.yaml --gt_dir <smoke>/groundtruth/population/main \\
        --output_dir <smoke>/metrics/main --threads 4 --memory_limit 8GB
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from groundtruth import db
from groundtruth.authorships import check_params
from metrics import expected as ex
from metrics import social as so
from utils import analysis_config as ac
from utils import constants as cons
from utils import ios
from utils import logs

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gt_dir', default=cons.DEFAULT_GT_POPULATION_DIR, help='A.4 outputs of the variant')
    parser.add_argument('--output_dir', default=cons.DEFAULT_METRICS_DIR, help='Metrics folder of the variant')
    parser.add_argument('--include_unknown', action=argparse.BooleanOptionalAction, default=False,
                        help="'unknown' as a class (as the social metrics)")
    parser.add_argument('--parity_target', choices=so.SCOPES, default='field_x_seniority', help='GT target of parity')
    parser.add_argument('--n_max', type=int, default=ex.N_MAX, help='Longest list simulated (longer: the value at n_max)')
    parser.add_argument('--seed', type=int, default=cons.DEFAULT_SEED)
    parser.add_argument('--overwrite', action='store_true', help='Recompute an existing table')
    for option in ('--valid_response_tags', '--valid_rec_tags', '--top_country_share'):   # config options, no effect here
        parser.add_argument(option, nargs='*', default=None, help=argparse.SUPPRESS)
    for flag in ('--keep_duplicates', '--order_sensitive_keys', '--consistency_all_runs'):
        parser.add_argument(flag, action='store_true', help=argparse.SUPPRESS)
    db.add_resource_arguments(parser)
    logs.add_log_arguments(parser)
    ac.add_config_argument(parser)
    return parser


def main() -> None:
    args = ac.parse_with_config(build_parser())
    logs.setup_logging('social_expected', args, params=Path(args.output_dir).name)
    out = ex.expected_file(args.output_dir)
    params = ex.depends_on(args.gt_dir, args.include_unknown, args.parity_target) | {'n_max': args.n_max, 'seed': args.seed, 'label_draws': ex.LABEL_DRAWS,
                                      'people_draws': ex.PEOPLE_DRAWS, 'similarity_sample': ex.SIMILARITY_SAMPLE}
    if out.exists() and not args.overwrite:
        check_params(out.parent, params)
        ios.printf(f"Skipped (exists, use --overwrite): {out}")
        return
    if args.overwrite:
        (out.parent / '_params.json').unlink(missing_ok=True)
    check_params(out.parent, params)
    start = ios.datetime.now()
    gt = so.GroundTruth(args.gt_dir, db.connect(args, ':memory:'))
    shares = gt.shares[gt.shares['scope'] == 'field_x_seniority']
    groups = sorted(set(zip(shares['field'], shares['seniority'])))
    ios.printf(f"{len(groups)} field x seniority groups, n = 1..{args.n_max}, GT {args.gt_dir}")
    table = ex.expected_table(gt, groups, args.parity_target, args.include_unknown, args.seed, args.n_max)
    ios.to_parquet(table, out.with_name(out.name + '.tmp'))
    out.with_name(out.name + '.tmp').replace(out)
    summary = table[table['n'].isin([1, 2, 5, 10])].pivot_table(index='metric', columns='n', values='expected').round(3)
    ios.printf(f"Done in {ios.datetime.now() - start} | {len(table)} rows -> {out}\n"
               f"mean expected over the groups (n = 1, 2, 5, 10):\n{summary.to_string()}")


if __name__ == '__main__':
    main()
