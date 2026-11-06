"""Summaries of the author matching (B.6b outputs) for the notebook matching_stats.ipynb."""

from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd

from utils import constants as cons

NO_NAME, NO_MATCH = 'no name', 'no match'
RULE_ORDER = cons.MATCH_RULES + [NO_MATCH, NO_NAME]

LEVELS = ['matched_name', 'matched_name_field', 'matched_name_seniority', 'matched_author']
OUTCOME_AUTHOR, OUTCOME_FIELD, OUTCOME_SENIORITY = 'author', 'field only', 'seniority only'
OUTCOME_AMBIGUOUS, OUTCOME_NAME = 'ambiguous', 'name only'
OUTCOME_ORDER = [OUTCOME_AUTHOR, OUTCOME_FIELD, OUTCOME_SENIORITY, OUTCOME_AMBIGUOUS, OUTCOME_NAME, NO_MATCH, NO_NAME]


REQUIRED_COLUMNS = ['match_rule', 'rec_name', 'jw_full_max', 'jw_parts_max', *LEVELS, 'matched_author_field',
                    'matched_author_seniority', 'n_name', 'n_name_field', 'n_name_seniority', 'n_author', 'author_id']


def load_matches(folder: Path | str, models: Optional[List[str]] = None) -> pd.DataFrame:
    """All <model>.parquet files of a B.6b output folder (one row per recommendation); fails clearly if they were
    written by an older version of the matching (missing columns)."""
    files = sorted(Path(folder).glob('*.parquet'))
    if models:
        files = [f for f in files if f.stem in models]
    if not files:
        raise FileNotFoundError(f"No <model>.parquet in {folder}")
    df = duckdb.sql(f"SELECT * FROM read_parquet({[str(f) for f in files]}, union_by_name = true)").df()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{folder} was written by an older version (missing {missing}): re-run the matching "
                         f"(rm -r <results>/matching; run_matching_pipeline.sh)")
    return df


def rule(df: pd.DataFrame) -> pd.Series:
    """match_rule, or 'no match' (a name without candidates) / 'no name' (no first + last name)."""
    return df['match_rule'].where(df['match_rule'].notna(),
                                  np.where(df['rec_name'].isna(), NO_NAME, NO_MATCH))


def outcome(df: pd.DataFrame) -> pd.Series:
    """Author-level outcome of every recommendation (see matching/decide.py)."""
    conditions = [df['matched_author'], df['matched_author_field'], df['matched_author_seniority'],
                  (df['n_name_field'] > 0) & (df['n_name_seniority'] > 0),
                  df['matched_name'], df['rec_name'].notna()]
    choices = [OUTCOME_AUTHOR, OUTCOME_FIELD, OUTCOME_SENIORITY, OUTCOME_AMBIGUOUS, OUTCOME_NAME, NO_MATCH]
    return pd.Series(np.select(conditions, choices, NO_NAME), index=df.index)


def shares(df: pd.DataFrame, column: pd.Series, order: List[str], by: str = cons.COL_MODEL) -> pd.DataFrame:
    """Share of each category per group (rows: groups + 'all models' = mean of the group shares)."""
    counts = pd.crosstab(df[by], column).reindex(columns=order, fill_value=0)
    table = counts.div(counts.sum(axis=1), axis=0)
    table.loc['all models'] = table.mean()
    return table


def level_rates(df: pd.DataFrame, by: str = cons.COL_MODEL) -> pd.DataFrame:
    """Share of recommendations (with a name) matched at each level, per group + 'all models' (mean of groups)."""
    named = df[df['rec_name'].notna()]
    table = named.groupby(by)[LEVELS].mean()
    table.loc['all models'] = table.mean()
    return table


def tier_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Per decided rule: number of recommendations and the median candidates of every rule (overlapping)."""
    matched = df[df['match_rule'].notna()]
    columns = [f"n_rule_{r}" for r in cons.MATCH_RULES]
    table = matched.groupby('match_rule')[columns].median().reindex(cons.MATCH_RULES)
    table.insert(0, 'n_recommendations', matched['match_rule'].value_counts().reindex(cons.MATCH_RULES).fillna(0).astype(int))
    return table
