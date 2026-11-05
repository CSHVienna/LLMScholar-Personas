#!/usr/bin/env python3
"""
Step B.3: tag every recommendation (multi-label; rules in libs/responses/recommendations.py):
correct, title_removed, fixed, split, initials_name, initials_lastname, initials,
placeholder, invalid.

Reads   <responses_metadata_dir>/<model>.parquet   (to select the responses)
and the raw response files (to read the recommendations), and writes
        <output_dir>/<model>.parquet               (default results/recommendations_metadata/)
with one row per recommended person: response identifiers
(model, date, language, role, location, k, seniority, field, subfield, run_id),
the response tags, the cleaned name, the recommendation tags, completeness and the full LLM record.

The recommendations of every response with named recommendations are stored (whatever its
tags), so that tags can be inspected (notebook responses_tags.ipynb) before deciding what goes downstream.

Usage (from any folder; paths default to the project folders):
    python batch_recommendations.py --models gpt-4_1-nano-2025-04-14 --limit_prompts 5 --responses_metadata_dir /tmp/smoke/responses --output_dir /tmp/smoke
    bash run_responses_pipeline.sh    # all steps, all models (GNU parallel, timestamped joblogs)
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse

from utils import batch
from utils import constants as cons
from utils import ios
from responses.reader import ResponseReader
from responses.recommendation_table import build_recommendations


def run_model(model: str, args: argparse.Namespace) -> str:
    """Tag the recommendations of one model and write its file. Returns a one-line report."""
    output_file = batch.model_file(args.output_dir, model)
    if output_file.exists() and not args.overwrite:
        return f"[{model}] skipped (exists): {output_file}"

    summary_file = batch.model_file(args.responses_metadata_dir, model)
    if not summary_file.exists():
        raise FileNotFoundError(f"missing responses metadata {summary_file} (run batch_responses_summary.py first)")

    summary = ios.load_parquet(summary_file)
    df = build_recommendations(ResponseReader.from_args(args), model, args.languages, summary)

    ios.to_parquet(df, output_file)
    counts = {tag: int(df[f"{cons.REC_TAG_PREFIX}{tag}"].sum()) for tag in cons.REC_TAGS}
    return f"[{model}] {len(df)} recommendations -> {output_file.name} | {counts}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_raw_response_arguments(parser)
    batch.add_model_arguments(parser)
    parser.add_argument('--responses_metadata_dir', default=cons.DEFAULT_RESPONSES_METADATA_DIR,
                        help='Folder with the responses metadata (<model>.parquet)')
    parser.add_argument('--output_dir', default=cons.DEFAULT_RECOMMENDATIONS_METADATA_DIR, help='Where to write <model>.parquet')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    batch.run_per_model('recommendations', run_model, args)


if __name__ == '__main__':
    main()
