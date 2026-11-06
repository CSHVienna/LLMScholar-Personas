#!/usr/bin/env python3
"""
Steps 7-8: technical metrics per configuration and per response (libs/metrics/technical.py), one model at a time.
Needs B.1-B.3 (responses / recommendations metadata) and B.6 (author matching, for factuality).

Writes, under --output_dir (default results/metrics/main/; robustness checks: results/metrics/robustness/<check>/):
    config/<model>.parquet     validity, refusals, behaviour / format / recommendation consistency
    response/<model>.parquet   outcome, k ratio, duplicates, factuality (name, name_field, name_seniority, author)
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

from groundtruth.authorships import check_params
from metrics import technical as tm
from utils import batch
from utils import analysis_config as ac
from utils import constants as cons
from utils import ios

RESPONSE_COLUMNS = tm.RESPONSE_KEY + [cons.COL_RESPONSE_TAGS, 'n_persons']
REC_COLUMNS = tm.REC_KEY + ['name_raw', 'lastname_raw', 'first_name', 'last_name', 'name_key', cons.COL_REC_TAGS]
MATCH_COLUMNS = tm.REC_KEY + [f"matched_{level}" for level in tm.LEVELS]


def run_model(model: str, args: argparse.Namespace) -> str:
    config_file = batch.model_file(Path(args.output_dir) / 'config', model)
    response_file = batch.model_file(Path(args.output_dir) / 'response', model)
    if config_file.exists() and response_file.exists() and not args.overwrite:
        return f"[{model}] skipped (exists)"
    inputs = {'responses': batch.model_file(args.responses_metadata_dir, model),
              'recommendations': batch.model_file(args.recommendations_metadata_dir, model),
              'matching': batch.model_file(args.matching_dir, model)}
    missing = [f"{name}: {path}" for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"[{model}] missing inputs: {missing}")
    responses = ios.load_parquet(inputs['responses'], columns=RESPONSE_COLUMNS)
    recs = ios.load_parquet(inputs['recommendations'], columns=REC_COLUMNS)
    matches = ios.load_parquet(inputs['matching'], columns=MATCH_COLUMNS)
    if args.languages != cons.LANGUAGES:
        responses = responses[responses[cons.VAR_LANGUAGE].isin(args.languages)]
        recs = recs[recs[cons.VAR_LANGUAGE].isin(args.languages)]

    config = tm.config_metrics(responses, recs, args.valid_response_tags, args.order_sensitive_keys,
                               args.consistency_all_runs)
    response = tm.response_metrics(responses, recs, matches, args.valid_response_tags, args.valid_rec_tags,
                                   args.keep_duplicates, args.order_sensitive_keys)
    ios.to_parquet(config, config_file.with_name(config_file.name + '.tmp'))
    config_file.with_name(config_file.name + '.tmp').replace(config_file)
    ios.to_parquet(response, response_file.with_name(response_file.name + '.tmp'))
    response_file.with_name(response_file.name + '.tmp').replace(response_file)
    return (f"[{model}] {len(config)} configurations (validity {config['validity'].mean():.3f}, "
            f"recommendation consistency {config['recommendation_consistency'].mean():.3f}) | {len(response)} responses "
            f"(duplicates {response['duplicates'].mean():.3f}, factuality_author {response['factuality_author'].mean():.3f})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_model_arguments(parser)
    parser.add_argument('--responses_metadata_dir', default=cons.DEFAULT_RESPONSES_METADATA_DIR, help='Output of B.1-B.2')
    parser.add_argument('--recommendations_metadata_dir', default=cons.DEFAULT_RECOMMENDATIONS_METADATA_DIR, help='Output of B.3')
    parser.add_argument('--matching_dir', default=cons.DEFAULT_MATCH_AUTHORS_DIR, help='Output of B.6b (a variant folder)')
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
