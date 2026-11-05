#!/usr/bin/env python3
"""
Draw a stratified sample for manual annotation (blind to the inferred labels).

For --task responses|recommendations, samples up to --per_stratum items per inferred tag
combination (rare combinations are all checked), at most --max_items, and writes
    <output_dir>/annotation_<task>_items.json   codebook + items, NO inferred labels -> load in the tool
    <output_dir>/annotation_<task>_key.parquet  item id -> identifiers + inferred tags (for evaluation only)
Open code/tools/annotation_tool.html in a browser, load the items file, annotate, export the CSV
(annotation_<task>_<annotator>.csv) into <output_dir>, then run batch_annotation_evaluate.py.

An existing sample is never overwritten (annotations refer to its item ids) unless --overwrite.
--refresh_codebook only replaces the codebook of an existing items file (items and ids unchanged).

Usage (from any folder):
    python code/scripts/batch_annotation_sample.py --task responses
    python code/scripts/batch_annotation_sample.py --task recommendations --per_stratum 30
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import json
from pathlib import Path

import pandas as pd

from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs
from annotation import tasks
from responses.context import PromptContext
from responses.reader import ResponseReader

REC_MATCH = [cons.COL_MODEL, cons.VAR_LANGUAGE, cons.COL_PROMPT_ID, cons.COL_RUN_ID, cons.COL_REC_INDEX]


# -----------------------------------------------------------------------------
# Sampling per task
# -----------------------------------------------------------------------------
def sample_responses(args: argparse.Namespace) -> pd.DataFrame:
    """Stratified sample of responses, with their prompt and raw text read from the raw files."""
    columns = cons.RESPONSE_ID_COLUMNS + [cons.COL_SOURCE, cons.COL_PROMPT_ID, cons.COL_RESPONSE_TAGS]
    summary = ios.load_parquet_folder(args.responses_metadata_dir, columns, args.models)
    sample = tasks.stratified_sample(summary, cons.COL_RESPONSE_TAGS, args.per_stratum, args.max_items, args.seed)

    # Raw texts, one raw file at a time
    reader = ResponseReader(args.responses_dir, PromptContext(args.context_dir))
    texts = {}
    for (model, language), group in sample.groupby([cons.COL_MODEL, cons.VAR_LANGUAGE], observed=True):
        keys = list(zip(group[cons.COL_PROMPT_ID].astype(int), group[cons.COL_RUN_ID].astype(int)))
        texts.update({(model, language, *key): value for key, value in reader.raw_texts(model, language, keys).items()})
    lookup = [texts[(r[cons.COL_MODEL], r[cons.VAR_LANGUAGE], int(r[cons.COL_PROMPT_ID]), int(r[cons.COL_RUN_ID]))]
              for _, r in sample.iterrows()]
    return sample.assign(prompt=[p for p, _ in lookup], raw_text=[t for _, t in lookup])


def sample_recommendations(args: argparse.Namespace) -> pd.DataFrame:
    """Stratified sample of recommendations; the (large) full records are read for the sample only."""
    columns = cons.RESPONSE_ID_COLUMNS + [cons.COL_PROMPT_ID, cons.COL_REC_INDEX, cons.COL_REC_TAGS,
                                          'name_raw', 'lastname_raw', 'first_name', 'last_name']
    recs = ios.load_parquet_folder(args.recommendations_metadata_dir, columns, args.models)
    sample = tasks.stratified_sample(recs, cons.COL_REC_TAGS, args.per_stratum, args.max_items, args.seed)

    records = []
    for model, group in sample.groupby(cons.COL_MODEL, observed=True):
        full = ios.load_parquet(batch.model_file(args.recommendations_metadata_dir, model), columns=REC_MATCH + ['record_json'])
        records.append(group.merge(full, on=REC_MATCH, how='left'))
    return pd.concat(records, ignore_index=True)


SAMPLERS = {cons.ANNOTATION_TASK_RESPONSES: sample_responses,
            cons.ANNOTATION_TASK_RECOMMENDATIONS: sample_recommendations}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, choices=cons.ANNOTATION_TASKS)
    parser.add_argument('--per_stratum', type=int, default=cons.DEFAULT_ANNOTATION_PER_STRATUM,
                        help='Items per inferred tag combination (all items if fewer)')
    parser.add_argument('--max_items', type=int, default=cons.DEFAULT_ANNOTATION_MAX_ITEMS, help='Cap on the sample size')
    parser.add_argument('--seed', type=int, default=cons.DEFAULT_SEED)
    parser.add_argument('--models', nargs='+', default=None, help='Subset of models (default: all)')
    parser.add_argument('--responses_metadata_dir', default=cons.DEFAULT_RESPONSES_METADATA_DIR)
    parser.add_argument('--recommendations_metadata_dir', default=cons.DEFAULT_RECOMMENDATIONS_METADATA_DIR)
    parser.add_argument('--responses_dir', default=cons.DEFAULT_RESPONSES_DIR)
    parser.add_argument('--context_dir', default=cons.DEFAULT_CONTEXT_DIR)
    parser.add_argument('--output_dir', default=cons.DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing sample (invalidates its annotations)')
    parser.add_argument('--refresh_codebook', action='store_true',
                        help='Only update the codebook of the existing items file (items and ids unchanged)')
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logs.setup_logging('annotation_sample', args, params=args.task)

    output_dir = Path(args.output_dir)
    items_file = output_dir / f"{cons.PREFIX_ANNOTATION}_{args.task}_items.json"
    key_file = output_dir / f"{cons.PREFIX_ANNOTATION}_{args.task}_key.parquet"
    if args.refresh_codebook:
        if not items_file.exists():
            raise SystemExit(f"{items_file} does not exist: draw a sample first (without --refresh_codebook).")
        payload = json.loads(items_file.read_text(encoding='utf-8'))
        payload['codebook'] = tasks.codebook(args.task)
        items_file.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
        ios.printf(f"codebook refreshed: {items_file} ({len(payload['items'])} items unchanged)")
        return
    if items_file.exists() and not args.overwrite:
        raise SystemExit(f"{items_file} exists (annotations refer to its item ids). Use --overwrite to replace it.")

    # 1. Sample (stratified by inferred tag combination) and opaque item ids
    sample = tasks.assign_item_ids(SAMPLERS[args.task](args), prefix=args.task[:3].upper(), seed=args.seed)
    counts = sample[cons.COL_RESPONSE_TAGS if args.task == cons.ANNOTATION_TASK_RESPONSES else cons.COL_REC_TAGS]
    ios.printf(f"{len(sample)} items from {counts.nunique()} tag combinations:\n{counts.value_counts().to_string()}")

    # 2. Blind items for the tool, and the key for the evaluation
    output_dir.mkdir(parents=True, exist_ok=True)
    items_file.write_text(json.dumps(tasks.items_payload(args.task, sample), ensure_ascii=False, indent=1), encoding='utf-8')
    ios.to_parquet(tasks.key_table(args.task, sample), key_file)
    ios.printf(f"items: {items_file}\nkey:   {key_file}\ntool:  {cons.ANNOTATION_TOOL_FILE}")


if __name__ == '__main__':
    main()
