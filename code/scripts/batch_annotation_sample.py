#!/usr/bin/env python3
"""
Draw a stratified sample for manual annotation (blind to the inferred labels).

For --task responses|recommendations, samples up to --per_stratum items per inferred tag
combination (rare combinations are all checked), at most --max_items, and writes
    <output_dir>/annotation_<task>_items.json   codebook + items, NO inferred labels -> load in the tool
    <output_dir>/annotation_<task>_key.parquet  item id -> identifiers + inferred tags (for evaluation only)
Open code/tools/annotation_tool.html in a browser, load the items file, annotate, export the CSV
(annotation_<task>_<annotator>.csv) into <output_dir>, then run batch_annotation_evaluate.py.

Verification tasks (annotation/verification.py; the pipeline's output IS shown, to be checked against OpenAlex):
    --task matching   distinct recommended names x requested field x seniority, stratified by the match outcome
                      (<rule>|linked, <rule>|name, none); needs B.6 (--matching_dir, --candidates_dir) and A.1
    --task authors    GT memberships (author x field), stratified by field x seniority; needs A.4 (--gt_dir) and A.1

Demographics (annotation/demographics.py; blind, name only):
    --task demographics  people (GT scholars + recommended people = B.6 matched authors), stratified by side x inferred
                         gender x inferred ethnicity; the annotator picks perceived gender and ethnicity; needs A.4
                         (--gt_dir) and, for the recommended side, B.6 (--matching_dir)

An existing sample is never overwritten (annotations refer to its item ids) unless --overwrite.
--refresh_codebook only replaces the codebook of an existing items file (items and ids unchanged), and
refreshes the inferred tags of its key from the current metadata (after re-running B.2-B.3).

Usage (from any folder):
    python code/scripts/batch_annotation_sample.py --task responses
    python code/scripts/batch_annotation_sample.py --task recommendations --per_stratum 30
    python code/scripts/batch_annotation_sample.py --task matching --per_stratum 15
    python code/scripts/batch_annotation_sample.py --task authors --per_stratum 10
    python code/scripts/batch_annotation_sample.py --task demographics --per_stratum 10
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import json
from pathlib import Path

import pandas as pd

from utils import analysis_config
from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs
from annotation import demographics
from annotation import tasks
from annotation import verification
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


def sample_matching(args: argparse.Namespace) -> pd.DataFrame:
    valid = analysis_config.load(args.config)['validity']['valid_rec_tags'] if args.config else cons.DEFAULT_VALID_REC_TAGS
    return verification.sample_matching(args.matching_dir, args.candidates_dir, args.authors_dir, args.per_stratum,
                                        args.max_items, args.seed, args.models, args.recommendations_metadata_dir, valid)


def sample_demographics(args: argparse.Namespace) -> pd.DataFrame:
    return demographics.sample_demographics(args.gt_dir, args.matching_dir, args.per_stratum, args.max_items, args.seed)


def sample_authors(args: argparse.Namespace) -> pd.DataFrame:
    return verification.sample_authors(args.gt_dir, args.authors_dir, args.per_stratum, args.max_items, args.seed)


SAMPLERS = {cons.ANNOTATION_TASK_RESPONSES: sample_responses,
            cons.ANNOTATION_TASK_RECOMMENDATIONS: sample_recommendations,
            cons.ANNOTATION_TASK_MATCHING: sample_matching,
            cons.ANNOTATION_TASK_AUTHORS: sample_authors,
            cons.ANNOTATION_TASK_DEMOGRAPHICS: sample_demographics}
STRATUM_COLUMN = {cons.ANNOTATION_TASK_RESPONSES: cons.COL_RESPONSE_TAGS,
                  cons.ANNOTATION_TASK_RECOMMENDATIONS: cons.COL_REC_TAGS,
                  cons.ANNOTATION_TASK_MATCHING: verification.STRATUM,
                  cons.ANNOTATION_TASK_AUTHORS: verification.STRATUM,
                  cons.ANNOTATION_TASK_DEMOGRAPHICS: demographics.STRATUM}


def builder_of(task: str):
    """Module with codebook / items_payload / key_table of a task."""
    if task == cons.ANNOTATION_TASK_DEMOGRAPHICS:
        return demographics
    return verification if task in verification.VERIFICATION_TASKS else tasks


def refresh_key(args: argparse.Namespace, key_file: Path) -> None:
    """Replace the key's inferred tags by those of the current metadata (same items)."""
    key = ios.load_parquet(key_file)
    if args.task == cons.ANNOTATION_TASK_RESPONSES:
        folder, tags_column, match = args.responses_metadata_dir, cons.COL_RESPONSE_TAGS, list(cons.RESPONSE_ID_COLUMNS)
    else:
        folder, tags_column, match = args.recommendations_metadata_dir, cons.COL_REC_TAGS, REC_MATCH
    models = sorted(key[cons.COL_MODEL].astype(str).unique())
    current = ios.load_parquet_folder(folder, match + [tags_column], models)
    as_text = lambda df: df[match].astype(str).agg('\x1f'.join, axis=1)   # match key, robust to categorical dtypes
    lookup = pd.Series(current[tags_column].to_numpy(), index=as_text(current))
    lookup = lookup[~lookup.index.duplicated()]
    new_tags = as_text(key).map(lookup)
    lost = int(new_tags.isna().sum())
    if lost:
        raise SystemExit(f"{lost} key items not found in {folder}: key not refreshed.")
    changed = int((new_tags.to_numpy() != key['inferred_tags'].to_numpy()).sum())
    key['inferred_tags'] = new_tags.to_numpy()
    ios.to_parquet(key, key_file)
    ios.printf(f"key refreshed: {key_file} ({changed} of {len(key)} items with new inferred tags)")


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
    parser.add_argument('--config', default=None,
                        help='matching: analysis config whose valid_rec_tags select the recommendations (default: '
                             'DEFAULT_VALID_REC_TAGS)')
    parser.add_argument('--matching_dir', default=cons.DEFAULT_MATCH_AUTHORS_DIR,
                        help='matching / demographics: B.6b output (a variant)')
    parser.add_argument('--candidates_dir', default=cons.DEFAULT_MATCH_CANDIDATES_DIR, help='matching: B.6a candidates')
    parser.add_argument('--gt_dir', default=cons.DEFAULT_GT_DIR,
                        help='authors / demographics: ground truth (gt_members / gt_authors.parquet)')
    parser.add_argument('--authors_dir', default=cons.DEFAULT_GT_AUTHORS_DIR, help='matching / authors: A.1 profiles')
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
        payload['codebook'] = builder_of(args.task).codebook(args.task)
        items_file.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
        ios.printf(f"codebook refreshed: {items_file} ({len(payload['items'])} items unchanged)")
        if key_file.exists() and args.task in cons.ANNOTATION_TAG_TASKS:   # other keys hold what was sampled: never refreshed
            refresh_key(args, key_file)
        return
    if items_file.exists() and not args.overwrite:
        raise SystemExit(f"{items_file} exists (annotations refer to its item ids). Use --overwrite to replace it.")

    # 1. Sample (stratified by inferred tag combination) and opaque item ids
    sample = tasks.assign_item_ids(SAMPLERS[args.task](args), prefix=args.task[:3].upper(), seed=args.seed)
    counts = sample[STRATUM_COLUMN[args.task]]
    ios.printf(f"{len(sample)} items from {counts.nunique()} strata:\n{counts.value_counts().to_string()}")

    # 2. Items for the tool (blind for the tag tasks), and the key for the evaluation
    builder = builder_of(args.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    items_file.write_text(json.dumps(builder.items_payload(args.task, sample), ensure_ascii=False, indent=1), encoding='utf-8')
    ios.to_parquet(builder.key_table(args.task, sample), key_file)
    ios.printf(f"items: {items_file}\nkey:   {key_file}\ntool:  {cons.ANNOTATION_TOOL_FILE}")


if __name__ == '__main__':
    main()
