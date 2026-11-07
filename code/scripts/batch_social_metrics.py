#!/usr/bin/env python3
"""
Step 9: social metrics per response (libs/metrics/social.py), one model at a time: parity, diversity and the unknown
share per attribute (gender, ethnicity, location, productivity, influence, collaboration), popularity (top group by
works / citations), connectedness and similarity, on the matched authors of the unique valid names of each valid
response. Needs B.1-B.3, B.6 and the GT (A.1, A.1a edges, A.4).

Writes, under --output_dir (default results/metrics/main/; robustness: results/metrics/robustness/<check>/):
    social/<model>.parquet       one row per response: n_people + the metrics
    social_gaps/<model>.parquet  one row per response x attribute x class: p, q (GT target), gap = p - q
Every social metric also as <metric>_adj = observed - expected under a random recommender (needs step 9a,
batch_social_expected.py, in the same --output_dir).
The parameters (valid tags, include_unknown, parity_target, ...) are recorded in <output_dir>/social/_params.json.

Usage (from any folder; run_metrics_pipeline.sh -s social runs one process per model):
    python batch_social_metrics.py --models deepseek-r1-8b-0528-qwen3-q4_K_M --config config/analysis_smoke.yaml \\
        --responses_metadata_dir <smoke>/responses_metadata --recommendations_metadata_dir <smoke>/recommendations_metadata \\
        --matching_dir <smoke>/matching/authors/main --gt_dir <smoke>/groundtruth/population/main --output_dir <smoke>/metrics/main
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from groundtruth.authorships import check_params
from metrics import expected as ex
from metrics import social as so
from metrics import technical as tm
from utils import analysis_config as ac
from utils import batch
from utils import constants as cons
from utils import ios

RESPONSE_COLUMNS = tm.RESPONSE_KEY + [cons.COL_RESPONSE_TAGS]
REC_COLUMNS = tm.REC_KEY + ['first_name', 'last_name', 'name_key', cons.COL_REC_TAGS]
MATCH_COLUMNS = tm.REC_KEY + ['matched_author', 'author_id']


def write_atomic(df, path: Path) -> None:
    tmp = path.with_name(path.name + '.tmp')
    ios.to_parquet(df, tmp)
    tmp.replace(path)


def run_model(model: str, args: argparse.Namespace) -> str:
    out = batch.model_file(Path(args.output_dir) / 'social', model)
    gaps_file = batch.model_file(Path(args.output_dir) / 'social_gaps', model)
    if out.exists() and gaps_file.exists() and not args.overwrite:
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
    people = so.response_people(responses, recs, matches, args.valid_response_tags, args.valid_rec_tags,
                                args.order_sensitive_keys)
    metrics, gaps = so.social_metrics(responses, people, args.gt, args.include_unknown, args.parity_target,
                                      args.top_country_share, expectations=args.expected)
    out.parent.mkdir(parents=True, exist_ok=True)
    gaps_file.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(metrics, out)
    write_atomic(gaps, gaps_file)
    with_people = metrics[metrics['n_people'] > 0]
    return (f"[{model}] {len(metrics)} responses, {len(with_people)} with matched people "
            f"(mean {with_people['n_people'].mean():.1f}) | parity_gender {metrics['parity_gender'].mean():.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_model_arguments(parser)
    parser.add_argument('--responses_metadata_dir', default=cons.DEFAULT_RESPONSES_METADATA_DIR, help='Output of B.1-B.2')
    parser.add_argument('--recommendations_metadata_dir', default=cons.DEFAULT_RECOMMENDATIONS_METADATA_DIR, help='Output of B.3')
    parser.add_argument('--matching_dir', default=cons.DEFAULT_MATCH_AUTHORS_DIR, help='Output of B.6b (a variant folder)')
    parser.add_argument('--gt_dir', default=cons.DEFAULT_GT_POPULATION_DIR, help='A.4 outputs of the variant (gt_members, gt_shares, gt_authors); coauthor_edges from the GT root')
    parser.add_argument('--output_dir', default=cons.DEFAULT_METRICS_DIR)
    parser.add_argument('--valid_response_tags', nargs='+', default=cons.DEFAULT_VALID_RESPONSE_TAGS, choices=cons.RESPONSE_TAGS)
    parser.add_argument('--valid_rec_tags', nargs='+', default=cons.DEFAULT_VALID_REC_TAGS, choices=cons.REC_TAGS)
    parser.add_argument('--include_unknown', action=argparse.BooleanOptionalAction, default=False,
                        help="Count 'unknown' as a class in parity / diversity (default: no, renormalise over the known)")
    parser.add_argument('--parity_target', choices=so.SCOPES, default='field_x_seniority',
                        help='Scope of the GT target of parity (requested field x seniority by default)')
    parser.add_argument('--top_country_share', type=float, default=so.DEFAULT_TOP_COUNTRY_SHARE,
                        help='Popularity by country: the fewest countries holding this share of the GT scholars')
    parser.add_argument('--order_sensitive_keys', action='store_true', help='Robustness: names compared in written order')
    parser.add_argument('--keep_duplicates', action='store_true',
                        help='Accepted for the runners (technical robustness flag); no effect: each person counts once')
    parser.add_argument('--consistency_all_runs', action='store_true',
                        help='Accepted for the runners (technical robustness flag); no effect here')
    ac.add_config_argument(parser)
    return parser


def main() -> None:
    args = ac.parse_with_config(build_parser())
    check_params(Path(args.output_dir) / 'social',
                 {'valid_response_tags': sorted(args.valid_response_tags), 'valid_rec_tags': sorted(args.valid_rec_tags),
                  'include_unknown': args.include_unknown, 'parity_target': args.parity_target,
                  'top_country_share': args.top_country_share,
                  'order_sensitive_keys': args.order_sensitive_keys, 'languages': args.languages,
                  'matching_dir': str(Path(args.matching_dir).resolve()), 'gt_dir': str(Path(args.gt_dir).resolve())})
    args.gt = so.GroundTruth(args.gt_dir)
    args.expected = ex.load(args.output_dir, args.gt_dir, args.include_unknown, args.parity_target)
    batch.run_per_model('social_metrics', run_model, args)


if __name__ == '__main__':
    main()
