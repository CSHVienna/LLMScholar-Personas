"""
Technical metrics (plan §2.3; steps 7-8): per configuration and per response, from the B.1-B.3 tags and the
B.6 author matching. Pure pandas on one model at a time.

Configuration (CONFIG_ID_COLUMNS = model, date, language, location, role, k, field, subfield, seniority):
    n_runs, n_valid, n_refused, n_invalid
    validity                    n_valid / n_runs
    refusals                    n_refused / n_runs
    behaviour_consistency       share of run pairs with the same outcome (refused > valid > invalid), all runs;
                                sum n_c (n_c - 1) / 2  /  R (R - 1) / 2;  NaN if R < 2
    format_consistency          mean pairwise Jaccard of the full response-tag sets, all runs (robustness)
    recommendation_consistency  mean pairwise Jaccard of the SETS of recommended names over the pairs of runs with
                                a valid response; every recommendation counts whatever its tag (key: name_key, or
                                for a recommendation without one the normalised raw 'first|last' in written order);
                                NaN if < 2 valid runs.  n_valid_runs = number of runs used
Response (RESPONSE_ID_COLUMNS = configuration + run_id):
    outcome, is_valid, k_requested, k_returned (person records, all tags), k_ratio (valid responses only)
    n_valid_recs, n_unique_names (among the valid recommendations, by name_key)
    duplicates                  1 - n_unique_names / n_valid_recs (valid responses with >= 1 valid recommendation)
    factuality_name / _name_field / _name_seniority / _author
                                share of the UNIQUE valid names matched at that level (B.6 matched_*; a repeated
                                name is judged by its first occurrence); n_factual_author = their number

Robustness options (plan §2.4; each written to its own folder by the caller):
    keep_duplicates       factuality over all valid recommendations (duplicates kept)
    order_sensitive_keys  name keys in written order ('first|last') for duplicates and recommendation consistency
    consistency_all_runs  recommendation consistency over all runs (an invalid response = an empty set; two empty
                          sets = 1)
"""

from itertools import combinations
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from responses.recommendations import normalize_tokens
from utils import constants as cons

OUTCOME_VALID, OUTCOME_REFUSED, OUTCOME_INVALID = 'valid', 'refused', 'invalid'
OUTCOMES = [OUTCOME_VALID, OUTCOME_REFUSED, OUTCOME_INVALID]
LEVELS = ['name', 'name_field', 'name_seniority', 'author']
CONFIG_KEY = cons.CONFIG_ID_COLUMNS
RESPONSE_KEY = cons.RESPONSE_ID_COLUMNS
REC_KEY = RESPONSE_KEY + [cons.COL_REC_INDEX]


# --- ids ----------------------------------------------------------------------------------------------------------
def row_id(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """One string per row from the id columns (missing values -> ''): joins never depend on dtypes / NaN keys."""
    parts = [df[c].map(lambda v: '' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for c in columns]
    return pd.Series(['\x1f'.join(values) for values in zip(*parts)], index=df.index)


# --- tags --------------------------------------------------------------------------------------------------------
def tag_set(tags: Optional[str]) -> frozenset:
    return frozenset(tags.split(cons.TAG_SEPARATOR)) if isinstance(tags, str) and tags else frozenset()


def is_valid(tags: pd.Series, valid_tags: Iterable[str]) -> pd.Series:
    """A response / recommendation is valid if it has tags and ALL of them are in the valid set."""
    valid = frozenset(valid_tags)
    return tags.map(lambda t: bool(tag_set(t)) and tag_set(t) <= valid)


def outcome(tags: pd.Series, valid_tags: Iterable[str]) -> pd.Series:
    """refused (tagged refused) > valid (all tags valid) > invalid."""
    refused = tags.map(lambda t: cons.RESPONSE_REFUSED in tag_set(t))
    valid = is_valid(tags, valid_tags)
    return pd.Series(np.where(refused, OUTCOME_REFUSED, np.where(valid, OUTCOME_VALID, OUTCOME_INVALID)), index=tags.index)


# --- name keys ---------------------------------------------------------------------------------------------------
def written_order_key(first: Optional[str], last: Optional[str]) -> Optional[str]:
    """'first|last' of normalised tokens in written order (placeholders, order-sensitive robustness)."""
    first_norm, last_norm = ' '.join(normalize_tokens(first or '')), ' '.join(normalize_tokens(last or ''))
    return f"{first_norm}|{last_norm}" if first_norm or last_norm else None


def consistency_keys(recs: pd.DataFrame, order_sensitive: bool = False) -> pd.Series:
    """Key of every recommendation for recommendation consistency (every recommendation, whatever its tag)."""
    raw = pd.Series([written_order_key(f, l) for f, l in zip(recs['name_raw'], recs['lastname_raw'])], index=recs.index)
    if order_sensitive:
        cleaned = pd.Series([written_order_key(f, l) if (isinstance(f, str) or isinstance(l, str)) else None
                             for f, l in zip(recs['first_name'], recs['last_name'])], index=recs.index)
        return cleaned.where(cleaned.notna(), raw)
    return recs['name_key'].where(recs['name_key'].notna(), raw)


def duplicate_keys(recs: pd.DataFrame, order_sensitive: bool = False) -> pd.Series:
    """Key of a (valid) recommendation for duplicates and unique-name factuality."""
    if order_sensitive:
        return pd.Series([written_order_key(f, l) for f, l in zip(recs['first_name'], recs['last_name'])], index=recs.index)
    return recs['name_key']


# --- pairwise ----------------------------------------------------------------------------------------------------
def jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def mean_pairwise_jaccard(sets: Sequence[frozenset]) -> float:
    pairs = list(combinations(sets, 2))
    return float(np.mean([jaccard(a, b) for a, b in pairs])) if pairs else np.nan


def pairwise_agreement(counts: Sequence[int]) -> float:
    """Share of pairs with the same category: sum n_c (n_c - 1) / 2 over R (R - 1) / 2; NaN if R < 2."""
    total = sum(counts)
    if total < 2:
        return np.nan
    return sum(n * (n - 1) / 2 for n in counts) / (total * (total - 1) / 2)


# --- configuration metrics -------------------------------------------------------------------------------------
def config_metrics(responses: pd.DataFrame, recs: pd.DataFrame, valid_response_tags: Iterable[str],
                   order_sensitive_keys: bool = False, consistency_all_runs: bool = False) -> pd.DataFrame:
    """One row per configuration (CONFIG_KEY)."""
    responses = responses.copy()
    responses['outcome'] = outcome(responses[cons.COL_RESPONSE_TAGS], valid_response_tags)
    recs = recs.assign(ckey=consistency_keys(recs, order_sensitive_keys), rid=row_id(recs, RESPONSE_KEY))
    names = recs.dropna(subset=['ckey']).groupby('rid')['ckey'].agg(frozenset)
    responses['names'] = row_id(responses, RESPONSE_KEY).map(names)
    responses['names'] = responses['names'].map(lambda s: s if isinstance(s, frozenset) else frozenset())
    responses['cid'] = row_id(responses, CONFIG_KEY)

    rows = []
    for _, group in responses.groupby('cid', sort=True):
        counts = group['outcome'].value_counts().reindex(OUTCOMES, fill_value=0)
        runs = group if consistency_all_runs else group[group['outcome'] == OUTCOME_VALID]
        rows.append({**group.iloc[0][CONFIG_KEY].to_dict(),
                     'n_runs': len(group), 'n_valid': int(counts[OUTCOME_VALID]),
                     'n_refused': int(counts[OUTCOME_REFUSED]), 'n_invalid': int(counts[OUTCOME_INVALID]),
                     'validity': counts[OUTCOME_VALID] / len(group), 'refusals': counts[OUTCOME_REFUSED] / len(group),
                     'behaviour_consistency': pairwise_agreement(counts.tolist()),
                     'format_consistency': mean_pairwise_jaccard([tag_set(t) for t in group[cons.COL_RESPONSE_TAGS]]),
                     'n_valid_runs': len(runs),
                     'recommendation_consistency': mean_pairwise_jaccard(runs['names'].tolist())})
    return pd.DataFrame(rows)


# --- response metrics --------------------------------------------------------------------------------------------
def response_metrics(responses: pd.DataFrame, recs: pd.DataFrame, matches: pd.DataFrame,
                     valid_response_tags: Iterable[str], valid_rec_tags: Iterable[str],
                     keep_duplicates: bool = False, order_sensitive_keys: bool = False) -> pd.DataFrame:
    """One row per response (RESPONSE_KEY)."""
    out = responses[RESPONSE_KEY].copy()
    out['outcome'] = outcome(responses[cons.COL_RESPONSE_TAGS], valid_response_tags).to_numpy()
    out['is_valid'] = out['outcome'] == OUTCOME_VALID
    out['k_requested'] = responses[cons.VAR_K].to_numpy()
    out['k_returned'] = responses['n_persons'].to_numpy()
    out['k_ratio'] = np.where(out['is_valid'], out['k_returned'] / out['k_requested'], np.nan)

    valid = recs[is_valid(recs[cons.COL_REC_TAGS], valid_rec_tags)].copy()
    valid['dkey'] = duplicate_keys(valid, order_sensitive_keys)
    valid['rid'], valid['xid'] = row_id(valid, RESPONSE_KEY), row_id(valid, REC_KEY)
    match_columns = [f"matched_{level}" for level in LEVELS]
    matched = matches.assign(xid=row_id(matches, REC_KEY)).set_index('xid')[match_columns]
    if not matched.index.is_unique:
        raise ValueError("Author matching has duplicated recommendations")
    valid = valid.join(matched, on='xid')
    missing = valid[match_columns[0]].isna() & valid['dkey'].notna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} valid recommendations have no author matching (re-run B.6)")
    valid[match_columns] = valid[match_columns].astype(bool)
    valid = valid.sort_values(REC_KEY)
    judged = valid if keep_duplicates else valid.drop_duplicates(['rid', 'dkey'])
    per = valid.groupby('rid').agg(n_valid_recs=('dkey', 'size'), n_unique_names=('dkey', 'nunique'))
    fact = judged.groupby('rid')[match_columns].mean().rename(columns=lambda c: f"factuality_{c[len('matched_'):]}")
    fact['n_factual_author'] = judged.groupby('rid')['matched_author'].sum()
    rid = row_id(out, RESPONSE_KEY)
    for column in ['n_valid_recs', 'n_unique_names']:
        out[column] = rid.map(per[column]).fillna(0).astype(int).to_numpy()
    for column in fact.columns:
        out[column] = rid.map(fact[column]).to_numpy()
    has_recs = out['is_valid'] & (out['n_valid_recs'] > 0)
    out['duplicates'] = np.where(has_recs, 1 - out['n_unique_names'] / out['n_valid_recs'].where(has_recs, 1), np.nan)
    for column in [f"factuality_{l}" for l in LEVELS] + ['n_factual_author']:
        out[column] = out[column].where(has_recs)
    return out

