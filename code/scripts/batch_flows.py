#!/usr/bin/env python3
"""
Flows (appendix figures): the people table, one model at a time (libs/metrics/flows.py): one row per (valid
response, factual person = matched author of a unique valid name) with the configuration (prompt variables, model)
and the person's GT attributes (perceived gender, race, ethnicity, country, productivity / influence /
collaboration, top groups). Any flow (prompt variable or model characteristic -> person attribute) and the location
drift are then computed in code/notebooks/5_flows.ipynb. Needs B.1-B.3, B.6 and the GT (A.4 gt_members).

Writes <output_dir>/flows/<model>.parquet (default results/metrics/main/); parameters in flows/_params.json.

Usage (from any folder; run_metrics_pipeline.sh -s flows runs one process per model):
    python batch_flows.py --models deepseek-r1-8b-0528-qwen3-q4_K_M --config config/analysis_smoke.yaml \\
        --responses_metadata_dir <smoke>/responses_metadata --recommendations_metadata_dir <smoke>/recommendations_metadata \\
        --matching_dir <smoke>/matching/authors/main --gt_dir <smoke>/groundtruth/population/main --output_dir <smoke>/metrics/main
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from utils.batch import check_params
from metrics import flows as fl
from metrics import social as so
from utils import analysis_config as ac
from utils import batch
from utils import constants as cons
from utils import ios

def run_model(model: str, args: argparse.Namespace) -> str:
    out = batch.model_file(Path(args.output_dir) / 'flows', model)
    if out.exists() and not args.overwrite:
        return f"[{model}] skipped (exists)"
    responses, recs, matches = batch.load_model_inputs(model, args, so.PEOPLE_INPUT_COLUMNS)
    people = fl.people_table(responses, recs, matches, args.gt, args.valid_response_tags, args.valid_rec_tags,
                             args.order_sensitive_keys)
    ios.write_parquet_atomic(people, out)
    return (f"[{model}] {len(people)} (response, person) rows, {people['author_id'].nunique()} distinct people | "
            f"unknown gender {fl.unknown_share(people, 'gender'):.1%}, country {fl.unknown_share(people, 'country'):.1%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_model_arguments(parser)
    batch.add_input_arguments(parser)
    parser.add_argument('--gt_dir', default=cons.DEFAULT_GT_POPULATION_DIR, help='A.4 outputs of the variant (gt_members)')
    parser.add_argument('--output_dir', default=cons.DEFAULT_METRICS_DIR)
    parser.add_argument('--valid_response_tags', nargs='+', default=cons.DEFAULT_VALID_RESPONSE_TAGS, choices=cons.RESPONSE_TAGS)
    parser.add_argument('--valid_rec_tags', nargs='+', default=cons.DEFAULT_VALID_REC_TAGS, choices=cons.REC_TAGS)
    parser.add_argument('--order_sensitive_keys', action='store_true', help='Robustness: names compared in written order')
    batch.accept_runner_options(parser)
    ac.add_config_argument(parser)
    return parser


def main() -> None:
    args = ac.parse_with_config(build_parser())
    check_params(Path(args.output_dir) / 'flows',
                 {'valid_response_tags': sorted(args.valid_response_tags), 'valid_rec_tags': sorted(args.valid_rec_tags),
                  'order_sensitive_keys': args.order_sensitive_keys, 'languages': args.languages,
                  'matching_dir': str(Path(args.matching_dir).resolve()), 'gt_dir': str(Path(args.gt_dir).resolve())},
                 overwrite=args.overwrite)
    args.gt = so.GroundTruth(args.gt_dir)
    batch.run_per_model('flows', run_model, args)


if __name__ == '__main__':
    main()
