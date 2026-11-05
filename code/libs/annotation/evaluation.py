"""
Evaluation of inferred tags against manual annotations (one or more annotators).

1. Agreement between annotators (if >= 2): per tag, Krippendorff's alpha (nominal, binary present/absent)
   and the share of items on which all annotators chose exactly the same tag set.
2. Adjudication: per item, a tag is in the gold set if more than half of the (non-unsure) annotators
   chose it. Items with a tie on any tag (e.g. two annotators who disagree) have no gold set and are
   excluded from the comparison (and counted), as are items where every annotator was unsure.
3. Inferred vs gold: per tag precision, recall, F1 (support = gold positives); exact-match rate of the tag
   sets; accuracy of the derived valid / not valid decision (valid = all tags in the valid set);
   for recommendations, agreement of the cleaned first / last name with the annotators' names; for
   responses, agreement of the number of person records (k_returned) with the annotators' count.
"""

from pathlib import Path
from typing import Iterable, List

import krippendorff
import numpy as np
import pandas as pd

from utils import constants as cons
from utils import ios
from annotation.tasks import ITEM_ID, vocabulary
from responses import tags as tag_utils
from responses.recommendations import normalize_tokens
from responses.tag_stats import is_valid

ANSWER_FIELDS = ['first_name', 'last_name', 'n_recommendations']   # free-text / number answers of the codebooks


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_annotations(folder: Path | str, task: str) -> pd.DataFrame:
    """All annotator exports ``annotation_<task>_<annotator>.csv`` in ``folder``."""
    files = [f for f in sorted(Path(folder).glob(f"{cons.PREFIX_ANNOTATION}_{task}_*.csv"))
             if not f.stem.startswith(f"{cons.PREFIX_ANNOTATION}_{task}_eval")]
    if not files:
        raise FileNotFoundError(f"No {cons.PREFIX_ANNOTATION}_{task}_<annotator>.csv files in {folder}")
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    df['unsure'] = df['unsure'].astype(int).astype(bool)
    return df


def tag_matrix(annotations: pd.DataFrame, task: str) -> pd.DataFrame:
    """Boolean matrix (item, annotator) x tag."""
    tags = vocabulary(task)
    chosen = annotations['tags'].map(lambda joined: set(tag_utils.split(joined)))
    matrix = pd.DataFrame({tag: chosen.map(lambda s, t=tag: t in s) for tag in tags})
    matrix.index = pd.MultiIndex.from_frame(annotations[[ITEM_ID, 'annotator']])
    return matrix


# -----------------------------------------------------------------------------
# 1. Agreement
# -----------------------------------------------------------------------------
def agreement(annotations: pd.DataFrame, task: str) -> pd.DataFrame:
    """Per tag: Krippendorff's alpha and number of items rated by >= 2 annotators (empty if one annotator)."""
    annotators = annotations['annotator'].unique()
    if len(annotators) < 2:
        return pd.DataFrame(columns=['tag', 'krippendorff_alpha', 'n_items_multiple_annotators'])
    matrix = tag_matrix(annotations[~annotations['unsure']], task)
    rows = []
    for tag in matrix.columns:
        reliability = matrix[tag].astype(float).unstack('annotator').reindex(columns=annotators).T.to_numpy()
        rated_twice = int(((~np.isnan(reliability)).sum(axis=0) >= 2).sum())   # items with >= 2 ratings
        alpha = (krippendorff.alpha(reliability_data=reliability, level_of_measurement='nominal')
                 if rated_twice and np.nanstd(reliability) > 0 else np.nan)
        rows.append({'tag': tag, 'krippendorff_alpha': alpha, 'n_items_multiple_annotators': rated_twice})
    return pd.DataFrame(rows)


def exact_agreement(annotations: pd.DataFrame) -> float:
    """Share of items (rated by >= 2 annotators) on which all annotators chose the same tag set."""
    sets = annotations[~annotations['unsure']].groupby(ITEM_ID)['tags'].agg(list)
    multiple = sets[sets.map(len) >= 2]
    return float(multiple.map(lambda tags: len(set(tags)) == 1).mean()) if len(multiple) else np.nan


# -----------------------------------------------------------------------------
# 2. Adjudication
# -----------------------------------------------------------------------------
def adjudicate(annotations: pd.DataFrame, task: str) -> pd.DataFrame:
    """Gold tag set per item (majority of non-unsure annotators), with ``is_tie`` for items without a majority.

    Items with only unsure answers are dropped.
    """
    sure = annotations[~annotations['unsure']]
    matrix = tag_matrix(sure, task)
    votes = matrix.groupby(level=ITEM_ID).mean()
    gold = votes.apply(lambda row: tag_utils.join([t for t in row.index if row[t] > 0.5], vocabulary(task)), axis=1)
    n_annotators = sure.groupby(ITEM_ID)['annotator'].nunique()
    result = pd.DataFrame({ITEM_ID: gold.index, 'gold_tags': gold.values,
                           'is_tie': (votes == 0.5).any(axis=1).values,
                           'n_annotators': n_annotators.reindex(gold.index).values})
    for field in ANSWER_FIELDS:
        if field in sure.columns:   # non-empty answer given by a strict majority (else no gold value)
            answers = sure[sure[field] != ''].groupby(ITEM_ID)[field].agg(_majority)
            result[f'gold_{field}'] = result[ITEM_ID].map(answers).fillna('')
    return result


# -----------------------------------------------------------------------------
# 3. Inferred vs gold
# -----------------------------------------------------------------------------
def compare(key: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Key (inferred) joined with gold, only items with a gold label (ties excluded)."""
    return key.merge(gold[~gold['is_tie']], on=ITEM_ID, how='inner')


def tag_metrics(compared: pd.DataFrame, task: str) -> pd.DataFrame:
    """Per tag: support, predicted, TP, FP, FN, precision, recall, F1."""
    inferred = compared['inferred_tags'].map(lambda j: set(tag_utils.split(j)))
    gold = compared['gold_tags'].map(lambda j: set(tag_utils.split(j)))
    rows = []
    for tag in vocabulary(task):
        pred, true = inferred.map(lambda s, t=tag: t in s), gold.map(lambda s, t=tag: t in s)
        tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if (tp + fp) and (tp + fn) else np.nan)
        rows.append({'tag': tag, 'support': int(true.sum()), 'predicted': int(pred.sum()), 'tp': tp, 'fp': fp,
                     'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1})
    return pd.DataFrame(rows)


def summary(compared: pd.DataFrame, valid_tags: Iterable[str], annotations: pd.DataFrame,
            gold: pd.DataFrame) -> pd.DataFrame:
    """One-row summary: sample sizes, exact-match rate, valid-decision accuracy (and name agreement)."""
    valid_tags = list(valid_tags)
    inferred_valid = compared['inferred_tags'].map(lambda j: is_valid(j, valid_tags))
    gold_valid = compared['gold_tags'].map(lambda j: is_valid(j, valid_tags))
    row = {'n_items': len(compared), 'n_annotators': annotations['annotator'].nunique(),
           'n_items_tied': int(gold['is_tie'].sum()), 'n_unsure_answers': int(annotations['unsure'].sum()),
           'exact_match_tags': float((compared['inferred_tags'] == compared['gold_tags']).mean()),
           'valid_accuracy': float((inferred_valid == gold_valid).mean()),
           'valid_false_positives': int((inferred_valid & ~gold_valid).sum()),    # kept but should be dropped
           'valid_false_negatives': int((~inferred_valid & gold_valid).sum()),    # dropped but should be kept
           'exact_agreement_annotators': exact_agreement(annotations)}
    if 'gold_first_name' in compared.columns:
        has_gold = compared['gold_last_name'] != ''
        row['name_match'] = float(_same_name(compared[has_gold]).mean()) if has_gold.any() else np.nan
    if 'gold_n_recommendations' in compared.columns and 'n_persons' in compared.columns:
        counted = compared[compared['gold_n_recommendations'] != '']
        gold_n = pd.to_numeric(counted['gold_n_recommendations'], errors='coerce')
        row['count_match'] = float((gold_n == counted['n_persons']).mean()) if len(counted) else np.nan
        row['count_mean_abs_diff'] = float((gold_n - counted['n_persons']).abs().mean()) if len(counted) else np.nan
    return pd.DataFrame([row])


def disagreements(compared: pd.DataFrame) -> pd.DataFrame:
    """Items whose inferred tag set differs from the gold one (for inspection)."""
    return compared[compared['inferred_tags'] != compared['gold_tags']].reset_index(drop=True)


def _majority(values: pd.Series) -> str:
    """The value given by more than half of the answers, else '' (no gold value)."""
    top = values.mode().iloc[0]
    return top if (values == top).mean() > 0.5 else ''


def _same_name(df: pd.DataFrame) -> pd.Series:
    """Cleaned name equals the annotators' name (case/accent-insensitive, token order kept)."""
    def norm(first, last) -> List[str]:
        return normalize_tokens(f"{first or ''} {last or ''}")
    return pd.Series([norm(r['first_name'], r['last_name']) == norm(r['gold_first_name'], r['gold_last_name'])
                      for _, r in df.iterrows()], index=df.index)


RESPONSE_MATCH = [cons.COL_MODEL, cons.VAR_LANGUAGE, cons.COL_PROMPT_ID, cons.COL_RUN_ID]


def add_person_counts(key: pd.DataFrame, responses_metadata_dir: Path | str) -> pd.DataFrame:
    """``key`` with the pipeline count of person records (``n_persons`` = k_returned) of each response."""
    models = list(key[cons.COL_MODEL].astype(str).unique())
    summary = ios.load_parquet_folder(responses_metadata_dir, RESPONSE_MATCH + ['n_persons'], models)
    as_text = lambda df: df.assign(**{c: df[c].astype(str) for c in RESPONSE_MATCH})   # same dtypes for the merge
    merged = as_text(key).merge(as_text(summary), on=RESPONSE_MATCH, how='left')
    return merged.assign(**{c: key[c].values for c in RESPONSE_MATCH})           # original identifier dtypes

