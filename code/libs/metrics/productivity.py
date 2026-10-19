"""
productivity.py — Low / med / high tiers of publications and citations,
relative to the author's field.
"""

import numpy as np
import pandas as pd

from libs.metrics import constants


def classify_source_bucket(
    df: pd.DataFrame,
    *,
    ss_col: str = "author_status",
    oa_col: str = "oa_status",
) -> pd.Series:
    """Classify each row into a coverage bucket.

    Returns a Series with values ``'SS + OA'``, ``'OA only'``,
    ``'SS only (no OA)'``, or ``'not found'`` depending on whether the row was
    matched in Semantic Scholar (``ss_col == 'found'``) and/or OpenAlex
    (``oa_col == 'found'``).
    """
    ss = df[ss_col].eq("found")
    oa = df[oa_col].eq("found")
    conds = [ss & oa, ~ss & oa, ss & ~oa]
    choices = ["SS + OA", "OA only", "SS only (no OA)"]
    return pd.Series(np.select(conds, choices, default="not found"), index=df.index)


def productivity_thresholds(
    df: pd.DataFrame,
    *,
    field_col: str = "field_en",
    metric_cols=None,
) -> dict:
    """Per-field p33/p67 of every productivity metric.

    Returns ``{metric_col: DataFrame(p33, p67) indexed by field}``.
    Used by every step that bins authors into low / med / high tiers.
    """
    cols = list(metric_cols or constants.PRODUCTIVITY_OA_FIELDS_MAP)
    out = {}
    for col in cols:
        th = (
            df.groupby(field_col)[col]
            .quantile([0.33, 0.67])
            .unstack()
            .rename(columns={0.33: "p33", 0.67: "p67"})
        )
        out[col] = th
    return out


def assign_productivity_tier(
    values: pd.Series, fields: pd.Series, th: pd.DataFrame
) -> pd.Series:
    """Bin ``values`` into ``{low, med, high}`` per-field tiers.

    ``th`` is the DataFrame returned by :func:`productivity_thresholds` for
    the same metric. Rows whose field is unknown (or whose value is NaN) get
    ``NaN``.
    """
    p33 = fields.map(th["p33"])
    p67 = fields.map(th["p67"])
    out = pd.Series(np.nan, index=values.index, dtype=object)
    mask = values.notna() & p33.notna() & p67.notna()
    out.loc[mask & (values <= p33)] = "low"
    out.loc[mask & (values > p33) & (values <= p67)] = "med"
    out.loc[mask & (values > p67)] = "high"
    return out


def tier_fractions(
    df: pd.DataFrame,
    tier_col: str,
    label: str,
    *,
    group_keys=None,
    tier_labels=None,
) -> pd.DataFrame:
    """Per-call fractions of authors in each tier (low / med / high).

    Returns a DataFrame with one row per group, columns
    ``pct_low_{label}``, ``pct_med_{label}``, ``pct_high_{label}``. NaN where
    the group has no tier-tagged authors.
    """
    group_keys = list(group_keys or constants.CALL_KEYS)
    tier_labels = list(tier_labels or constants.PRODUCTIVITY_TIER_LABELS)
    cts = (
        df.dropna(subset=[tier_col])
        .groupby(group_keys + [tier_col], dropna=False)
        .size()
        .unstack(tier_col, fill_value=0)
        .reindex(columns=tier_labels, fill_value=0)
    )
    total = cts.sum(axis=1).replace(0, np.nan)
    frac = cts.div(total, axis=0)
    return frac.rename(columns=lambda t: f"pct_{t}_{label}").reset_index()
