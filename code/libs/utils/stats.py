"""
Percentile bootstrap summaries, one rule for every analysis: the (1 - confidence) / 2 and 1 - (1 - confidence) / 2
quantiles of the bootstrap draws, their standard deviation (ddof 1) and the number of draws.
"""

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def tails(confidence: float) -> Tuple[float, float]:
    """Lower and upper quantile of a two-sided percentile interval, e.g. 0.95 -> (0.025, 0.975)."""
    alpha = (1 - confidence) / 2
    return alpha, 1 - alpha


def percentile_ci(draws: np.ndarray, confidence: float, axis=None, skipna: bool = False) -> Tuple:
    """(low, high) percentile interval of the draws along axis (skipna: NaN draws left out)."""
    quantile = np.nanquantile if skipna else np.quantile
    low, high = quantile(draws, list(tails(confidence)), axis=axis)
    return low, high


def percentile_summary(draws: Sequence[float], confidence: float) -> dict:
    """ci_low, ci_high, std and n_boot of one quantity's draws (non-finite draws left out; NaN when none)."""
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    low, high = percentile_ci(values, confidence) if len(values) else (np.nan, np.nan)
    return {'ci_low': low, 'ci_high': high, 'std': values.std(ddof=1) if len(values) > 1 else np.nan,
            'n_boot': len(values)}


def grouped_percentile_summary(draws: pd.DataFrame, by, column: str, confidence: float,
                               skipna: bool = False) -> pd.DataFrame:
    """Per group of the draws: ci_low, ci_high, std (ddof 1) and n_boot (non-NaN draws) of `column`. skipna=False: a
    NaN draw makes the interval NaN (pandas otherwise skips it)."""
    low_q, high_q = tails(confidence)
    if skipna:
        grouped = draws.groupby(by)[column]
        return pd.DataFrame({'ci_low': grouped.quantile(low_q), 'ci_high': grouped.quantile(high_q),
                             'std': grouped.std(ddof=1), 'n_boot': grouped.count()})
    return draws.groupby(by)[column].agg(
        ci_low=lambda v: np.quantile(v, low_q), ci_high=lambda v: np.quantile(v, high_q),
        std=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, n_boot='count')
