"""Loaders for the distribution CSVs produced by
``scripts/metrics/build_ethnicity_distributions.py``.

Consolidates the I/O + reindex-by-order pattern used by
``notebooks/analysis/ethnicity_metrics.ipynb``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _ordered_series(df: pd.DataFrame, key_col: str, order: list[str]) -> pd.Series:
    """Index ``df`` by ``key_col`` and reindex by ``order`` (intersection only)."""
    s = df.set_index(key_col)["count"]
    return s.reindex([e for e in order if e in s.index])


def _ordered_pivot(
    df: pd.DataFrame, index_col: str, columns_col: str, order: list[str]
) -> pd.DataFrame:
    """Pivot a long CSV (index, columns, count) and reindex columns by ``order``."""
    p = df.pivot(index=index_col, columns=columns_col, values="count").fillna(0).astype(int)
    return p[[c for c in order if c in p.columns]]


def load_ethnicity_distributions(
    dist_dir: Path | str, order: list[str]
) -> dict[str, pd.Series | pd.DataFrame]:
    """Load every distribution CSV written by ``build_ethnicity_distributions.py``.

    Returns a dict with ``gt_overall``, ``rec_overall`` (Series indexed by ethnicity
    in ``order``) and ``gt_per_field``, ``rec_per_field``, ``rec_per_model``
    (DataFrames pivoted and reordered by ``order``).
    """
    d = Path(dist_dir)
    return {
        "gt_overall": _ordered_series(pd.read_csv(d / "gt_overall.csv"), "ethnicity", order),
        "rec_overall": _ordered_series(pd.read_csv(d / "rec_overall.csv"), "ethnicity", order),
        "gt_per_field": _ordered_pivot(
            pd.read_csv(d / "gt_per_field.csv"), "field", "ethnicity", order
        ),
        "rec_per_field": _ordered_pivot(
            pd.read_csv(d / "rec_per_field.csv"), "field_en", "perceived_ethnicity", order
        ),
        "rec_per_model": _ordered_pivot(
            pd.read_csv(d / "rec_per_model.csv"), "model", "perceived_ethnicity", order
        ),
    }
