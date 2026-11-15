#!/usr/bin/env python3
"""
Steps 7-8: technical metrics per configuration and per response (libs/metrics/technical.py), one model at a time.
Needs B.1-B.3 (responses / recommendations metadata) and B.6 (author matching, for factuality).

Writes, under --output_dir (default results/metrics/main/; robustness checks: results/metrics/robustness/<check>/):
    config/<model>.parquet     validity, refusals, behaviour / format / recommendation consistency
    response/<model>.parquet   outcome, k balance, duplicates, factuality (name, name_field, name_seniority, author)
The parameters (valid tag sets, robustness flags, matching folder) are recorded in <output_dir>/_params.json;
outputs written with other parameters are refused (use another --output_dir).

Usage (from any folder):
    python batch_metrics.py --models deepseek-r1-8b-0528-qwen3-q4_K_M --recommendations_metadata_dir <smoke>/recommendations_metadata \\
        --matching_dir <smoke>/matching/authors/main --output_dir <smoke>/metrics/main
    python batch_metrics.py --keep_duplicates --output_dir results/metrics/robustness/with_duplicates
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from utils.batch import check_params
from metrics import technical as tm
from utils import batch
from utils import analysis_config as ac
from utils import constants as cons
from utils import ios

def run_model(model: str, args: argparse.Namespace) -> str:
    config_file = batch.model_file(Path(args.output_dir) / 'config', model)
    response_file = batch.model_file(Path(args.output_dir) / 'response', model)
    if config_file.exists() and response_file.exists() and not args.overwrite:
        return f"[{model}] skipped (exists)"
    responses, recs, matches = batch.load_model_inputs(model, args, tm.INPUT_COLUMNS, filter_recs=True)

    config = tm.config_metrics(responses, recs, args.valid_response_tags, args.order_sensitive_keys,
                               args.consistency_all_runs)
    response = tm.response_metrics(responses, recs, matches, args.valid_response_tags, args.valid_rec_tags,
                                   args.keep_duplicates, args.order_sensitive_keys)
    ios.write_parquet_atomic(config, config_file)
    ios.write_parquet_atomic(response, response_file)
    return (f"[{model}] {len(config)} configurations (validity {config['validity'].mean():.3f}, "
            f"recommendation consistency {config['recommendation_consistency'].mean():.3f}) | {len(response)} responses "
            f"(duplicates {response['duplicates'].mean():.3f}, factuality_author {response['factuality_author'].mean():.3f})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_model_arguments(parser)
    batch.add_input_arguments(parser)
    parser.add_argument('--output_dir', default=cons.DEFAULT_METRICS_DIR,
                        help='results/metrics/main, or results/metrics/robustness/<check> for a robustness check')
    parser.add_argument('--valid_response_tags', nargs='+', default=cons.DEFAULT_VALID_RESPONSE_TAGS,
                        choices=cons.RESPONSE_TAGS, help='A response is valid if all its tags are in this set')
    parser.add_argument('--valid_rec_tags', nargs='+', default=cons.DEFAULT_VALID_REC_TAGS,
                        choices=cons.REC_TAGS, help='A recommendation is valid if all its tags are in this set')
    parser.add_argument('--keep_duplicates', action='store_true', help='Robustness: factuality with duplicates kept')
    parser.add_argument('--order_sensitive_keys', action='store_true',
                        help='Robustness: names compared in written order (duplicates, recommendation consistency)')
    parser.add_argument('--consistency_all_runs', action='store_true',
                        help='Robustness: recommendation consistency over all runs (invalid response = empty set)')
    batch.accept_runner_options(parser)        # e.g. --include_unknown (a social robustness flag): no effect here
    ac.add_config_argument(parser)
    return parser


def main() -> None:
    args = ac.parse_with_config(build_parser())
    check_params(args.output_dir, {key: (sorted(v) if isinstance(v, list) else v) for key, v in vars(args).items()
                                   if key in ('valid_response_tags', 'valid_rec_tags', 'keep_duplicates',
                                              'order_sensitive_keys', 'consistency_all_runs', 'languages')}
                 | {'matching_dir': str(Path(args.matching_dir).resolve())})
    batch.run_per_model('metrics', run_model, args)


if __name__ == '__main__':
    main()
