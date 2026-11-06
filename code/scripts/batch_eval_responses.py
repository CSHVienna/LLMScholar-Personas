#!/usr/bin/env python3
"""
Step B.1-2: read the raw LLM responses and tag every response (multi-label; rules in libs/responses/taxonomy.py):
correct / python_literal / embedded / truncated, plus wrapped / toy, or one of
empty / empty_list / refused / invalid.

Writes one file per model:  <output_dir>/<model>.parquet   (default results/responses_metadata/)
Each row is one response, identified by
    model, date, language, role, location, k, seniority, field, subfield, run_id

Usage (from any folder; paths default to the project folders):
    python batch_eval_responses.py --models gpt-4_1-nano-2025-04-14 --limit_prompts 5 --output_dir /tmp/smoke
    bash run_responses_pipeline.sh    # all steps, all models (GNU parallel, timestamped joblogs)
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse

from utils import batch
from utils import constants as cons
from utils import ios
from responses.reader import ResponseReader
from responses.summary import summarize_model


def run_model(model: str, args: argparse.Namespace) -> str:
    """Summarise one model (all languages) and write its file. Returns a one-line report."""
    output_file = batch.model_file(args.output_dir, model)
    if output_file.exists() and not args.overwrite:
        return f"[{model}] skipped (exists): {output_file}"

    df = summarize_model(ResponseReader.from_args(args), model, args.languages)
    if df.empty:
        return f"[{model}] no responses found"

    ios.to_parquet(df, output_file)
    counts = {tag: int(df[f"{cons.RESPONSE_TAG_PREFIX}{tag}"].sum()) for tag in cons.RESPONSE_TAGS}
    return f"[{model}] {len(df)} responses -> {output_file.name} | {counts}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_raw_response_arguments(parser)
    batch.add_model_arguments(parser)
    parser.add_argument('--output_dir', default=cons.DEFAULT_RESPONSES_METADATA_DIR, help='Where to write <model>.parquet')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    batch.run_per_model('responses_summary', run_model, args)


if __name__ == '__main__':
    main()
