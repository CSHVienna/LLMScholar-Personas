"""
p-values from the cluster bootstrap and multiple-testing corrections, shared by RQ1, RQ2 and the interactions.

    wald_p      p from an estimate and its bootstrap SE (the SD of the replicates): one-sided (H1: estimate > 0; for
                omega^2 / R^2 shares) or two-sided (deviations). A percentile p (share of replicates beyond 0) cannot go
                below 1 / (B + 1), too coarse for corrections over hundreds of tests; the Wald p is continuous.
                Assumes a roughly normal bootstrap distribution (fine for means / deviations; skewed for omega^2 near
                0, which only matters for borderline values).
    holm        Holm (1979) step-down, family-wise error rate
    benjamini_hochberg   BH (1995) false discovery rate; valid for independent / positively dependent tests (PRDS)
    benjamini_yekutieli  BY (2001) false discovery rate under ANY dependence (metrics correlated in both directions):
                BH x sum_{i<=m} 1/i, stricter
    adjust      one of them by name ('holm' | 'bh' | 'by'); NaN p-values are kept and not counted
"""

from typing import Sequence

import numpy as np
from scipy.stats import norm

from utils import constants as cons

CORRECTIONS = ('holm', 'bh', 'by')


def wald_p(estimate, se, sided: str = 'two') -> np.ndarray:
    """p-values of estimate / se (NaN where se is missing or 0). sided = 'two' | 'greater'."""
    estimate, se = np.asarray(estimate, dtype=float), np.asarray(se, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(np.isfinite(se) & (se > 0), estimate / se, np.nan)
    if sided == 'greater':
        return norm.sf(z)
    if sided == 'two':
        return 2 * norm.sf(np.abs(z))
    raise ValueError(f"sided must be 'two' or 'greater', not {sided!r}")


def _sorted_finite(p: Sequence[float]):
    p = np.asarray(p, dtype=float)
    ok = np.flatnonzero(np.isfinite(p))
    return p, ok[np.argsort(p[ok], kind='stable')]


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values."""
    p, order = _sorted_finite(p)
    out = np.full(len(p), np.nan)
    m = len(order)
    out[order] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    return out


def benjamini_hochberg(p: Sequence[float], dependence_free: bool = False) -> np.ndarray:
    """BH adjusted p-values; dependence_free: the BY version (x sum 1/i)."""
    p, order = _sorted_finite(p)
    out = np.full(len(p), np.nan)
    m = len(order)
    if not m:
        return out
    c = np.sum(1 / np.arange(1, m + 1)) if dependence_free else 1.0
    scaled = p[order] * m * c / np.arange(1, m + 1)
    out[order] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    return out


def benjamini_yekutieli(p: Sequence[float]) -> np.ndarray:
    return benjamini_hochberg(p, dependence_free=True)


def adjust(p: Sequence[float], method: str = 'by') -> np.ndarray:
    if method == 'holm':
        return holm(p)
    if method == 'bh':
        return benjamini_hochberg(p)
    if method == 'by':
        return benjamini_yekutieli(p)
    raise ValueError(f"method must be one of {CORRECTIONS}, not {method!r}")


def add_p_values(table, value: str, sided: str = 'two'):
    """Copy of a results table with p_<value> (Wald, <value> / <value>_std) and p_<value>_<method> for every
    correction. The family = ALL rows of the table (e.g. every variable x metric loaded), not one figure's columns."""
    out = table.copy()
    out[f"p_{value}"] = wald_p(out[value], out[f"{value}_std"], sided)
    for method in CORRECTIONS:
        out[f"p_{value}_{method}"] = adjust(out[f"p_{value}"], method)
    return out


def significant(table, value: str, correction=None, alpha: float = cons.ALPHA, sided: str = 'two') -> np.ndarray:
    """Boolean per row: adjusted p < alpha (correction 'holm' | 'bh' | 'by'; p-values added if missing), or, with
    correction=None, the uncorrected percentile CI excludes 0 (one-sided 'greater': its lower limit > 0)."""
    if correction is None:
        low, high = table[f"{value}_ci_low"].to_numpy(float), table[f"{value}_ci_high"].to_numpy(float)
        with np.errstate(invalid='ignore'):
            return (low > 0) if sided == 'greater' else ((low > 0) | (high < 0))
    column = f"p_{value}_{correction}"
    if column not in table:
        table = add_p_values(table, value, sided)
    with np.errstate(invalid='ignore'):
        return (table[column].to_numpy(float) < alpha)
