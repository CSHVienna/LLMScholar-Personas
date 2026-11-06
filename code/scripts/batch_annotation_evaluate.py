#!/usr/bin/env python3
"""
Evaluate the inferred tags against the manual annotations of one task.

Reads   <annotations_dir>/annotation_<task>_key.parquet       (inferred tags, from batch_annotation_sample.py)
        <annotations_dir>/annotation_<task>_<annotator>.csv   (exports of code/tools/annotation_tool.html)
Writes  <annotations_dir>/annotation_<task>_eval_<table>.csv  for the tables
        summary         sample sizes, exact-match rate, valid-decision accuracy (+ name / count match)
        tags            per tag: support, predicted, TP/FP/FN, precision, recall, F1
        agreement       per tag: Krippendorff's alpha between annotators (if >= 2 annotators)
        disagreements   items whose inferred tags differ from the adjudicated ones (to inspect)

Verification tasks (matching, authors; annotation/verification.py) write instead
        rates           per label group, stratum and label: n, k, share, Wilson CI; 'overall (weighted)' rows
                        re-weight the strata to their population sizes
        agreement       per label group: Krippendorff's alpha (if >= 2 annotators)

Usage (from any folder):
    python code/scripts/batch_annotation_evaluate.py --task responses
    python code/scripts/batch_annotation_evaluate.py --task recommendations --valid_tags correct title_removed
    python code/scripts/batch_annotation_evaluate.py --task matching
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from utils import constants as cons
from utils import ios
from utils import logs
from annotation import evaluation
from annotation import verification

DEFAULT_VALID = {cons.ANNOTATION_TASK_RESPONSES: cons.DEFAULT_VALID_RESPONSE_TAGS,
                 cons.ANNOTATION_TASK_RECOMMENDATIONS: cons.DEFAULT_VALID_REC_TAGS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, choices=cons.ANNOTATION_TASKS)
    parser.add_argument('--valid_tags', nargs='+', default=None,
                        help='Tags that make a combination valid (default: DEFAULT_VALID_*_TAGS in constants)')
    parser.add_argument('--annotations_dir', default=cons.DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument('--responses_metadata_dir', default=cons.DEFAULT_RESPONSES_METADATA_DIR,
                        help='Responses metadata (for the pipeline count of person records)')
    logs.add_log_arguments(parser)
    return parser


def evaluate_verification(args: argparse.Namespace, folder: Path, prefix: str) -> None:
    key = ios.load_parquet(folder / f"{prefix}_key.parquet")
    annotations = evaluation.load_annotations(folder, args.task)
    ios.printf(f"{len(annotations)} annotations by {annotations['annotator'].nunique()} annotator(s) "
               f"on {annotations['item_id'].nunique()} of {len(key)} items")
    answers = verification.group_answers(annotations, args.task)
    gold = verification.adjudicate_groups(answers)
    tables = {'rates': verification.label_rates(key, gold, args.task),
              'agreement': verification.group_agreement(answers)}
    for name, table in tables.items():
        path = folder / f"{prefix}_eval_{name}.csv"
        ios.to_csv(table, path)
        ios.printf(f"{name}: {path}")
    overall = tables['rates'][tables['rates']['stratum'] == 'overall (weighted)']
    ios.printf(f"\n{overall.round(3).to_string(index=False)}")


def main() -> None:
    args = build_parser().parse_args()
    logs.setup_logging('annotation_evaluate', args, params=args.task)
    folder, prefix = Path(args.annotations_dir), f"{cons.PREFIX_ANNOTATION}_{args.task}"
    if args.task in verification.VERIFICATION_TASKS:
        evaluate_verification(args, folder, prefix)
        return
    valid_tags = args.valid_tags or DEFAULT_VALID[args.task]

    # 1. Inputs: key (inferred) and annotations (manual)
    key = ios.load_parquet(folder / f"{prefix}_key.parquet")
    if args.task == cons.ANNOTATION_TASK_RESPONSES:   # pipeline count of person records (k_returned), for the count check
        key = evaluation.add_person_counts(key, args.responses_metadata_dir)
    annotations = evaluation.load_annotations(folder, args.task)
    ios.printf(f"{len(annotations)} annotations by {annotations['annotator'].nunique()} annotator(s) "
               f"on {annotations['item_id'].nunique()} of {len(key)} items")

    # 2. Agreement, adjudication, comparison
    gold = evaluation.adjudicate(annotations, args.task)
    compared = evaluation.compare(key, gold)
    tables = {'summary': evaluation.summary(compared, valid_tags, annotations, gold),
              'tags': evaluation.tag_metrics(compared, args.task),
              'agreement': evaluation.agreement(annotations, args.task),
              'disagreements': evaluation.disagreements(compared)}

    # 3. Outputs
    for name, table in tables.items():
        path = folder / f"{prefix}_eval_{name}.csv"
        ios.to_csv(table, path)
        ios.printf(f"{name}: {path}")
    ios.printf(f"\n{tables['summary'].T.to_string(header=False)}\n\n{tables['tags'].round(3).to_string(index=False)}")


if __name__ == '__main__':
    main()
