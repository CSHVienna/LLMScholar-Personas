"""
RQ1: global influence of each configuration variable (and of the model) on each metric.

Data: one row per model x configuration (metrics/summary.configuration_table); per metric, rows with a missing value
are left out. Model: metric ~ model + language + role + location + k + seniority + field + subfield(field)
(categorical main effects, ordinary least squares). Subfield is NESTED in field: it is coded by within-field contrasts
(effect coding inside each field), so it captures only differences between the subfields of the same field and
field keeps its own effect.

    partial omega^2 (Olejnik & Algina 2003), from type-II sums of squares (each factor after all the others, except
        the factors nested in it, following the marginality principle; degrees of freedom = added rank):
        omega2_p = (SS_f - df_f MS_e) / (SS_f + (N - df_f) MS_e)
    Shapley R^2 (dominance analysis; Budescu 1993, Groemping 2007): the R^2 of every subset of factors; a factor's
        share = its mean increment over all orders of entry. The shares sum to the R^2 of the full model.
    Confidence intervals: cluster bootstrap over MODELS (a model is resampled with all its configurations; a model
        drawn twice counts as two clusters), percentile intervals. Point estimates are on the original data.

Effect sizes, not p-values: with this many configurations every effect is "significant" (Lin, Lucas & Shmueli 2013).
"""

from itertools import combinations
from math import factorial
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from utils import constants as cons

MODEL_FACTOR = cons.COL_MODEL
DEFAULT_FACTORS = [MODEL_FACTOR] + cons.CONFIG_VARIABLES
NESTED = {cons.VAR_SUBFIELD: cons.VAR_FIELD}      # factor -> the factor it is nested in (used when both are present)


# --- design ------------------------------------------------------------------------------------------------------
def dummies(values: pd.Series) -> np.ndarray:
    """Treatment-coded indicator columns of one categorical factor (first level dropped; none if 1 level)."""
    codes, levels = pd.factorize(values, sort=True)
    if len(levels) < 2:
        return np.empty((len(values), 0))
    return (codes[:, None] == np.arange(1, len(levels))[None, :]).astype(float)


def nested_contrasts(values: pd.Series, parents: pd.Series) -> np.ndarray:
    """Within-parent effect coding of a nested factor: inside each parent level with m >= 2 child levels, m - 1
    columns (+1 for child j, -1 for the parent's first child, 0 elsewhere). The columns are zero outside their parent,
    so the factor only explains differences between the children of the same parent."""
    columns = []
    for parent in sorted(parents.unique()):
        inside = (parents == parent).to_numpy()
        children = sorted(values[inside].unique())
        for child in children[1:]:
            columns.append(np.where(inside & (values == child).to_numpy(), 1.0,
                                    np.where(inside & (values == children[0]).to_numpy(), -1.0, 0.0)))
    return np.column_stack(columns) if columns else np.empty((len(values), 0))


def residual_ss(y: np.ndarray, blocks: Sequence[np.ndarray]) -> float:
    """Residual sum of squares of y on an intercept + the given dummy blocks (direct least squares)."""
    X = np.column_stack([np.ones(len(y)), *blocks]) if blocks else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    return float(residual @ residual)


class Gram:
    """Cross products of [1, all dummy blocks] and y, computed once: the residual SS of any subset of factors is then
    a small solve, RSS(S) = y'y - c_S' G_SS^+ c_S (same result as fitting the subset; 128 subsets cost ~1 fit)."""

    def __init__(self, y: np.ndarray, blocks: Dict[str, np.ndarray]):
        X = np.column_stack([np.ones(len(y)), *blocks.values()])
        self.G, self.c, self.yy = X.T @ X, X.T @ y, float(y @ y)
        self.columns, start = {}, 1
        for factor, block in blocks.items():
            self.columns[factor] = list(range(start, start + block.shape[1]))
            start += block.shape[1]

    def _sub(self, factors: Sequence[str]):
        idx = [0] + [c for f in factors for c in self.columns[f]]
        return self.G[np.ix_(idx, idx)], self.c[idx]

    def rss(self, factors: Sequence[str]) -> float:
        G, c = self._sub(factors)
        beta, *_ = np.linalg.lstsq(G, c, rcond=None)
        return max(self.yy - float(c @ beta), 0.0)

    def rank(self, factors: Sequence[str]) -> int:
        """Rank of the design (intercept + factors): dependent dummy columns (empty cells, unbalanced data) don't count."""
        return int(np.linalg.matrix_rank(self._sub(factors)[0]))


def _prepare(df: pd.DataFrame, metric: str, factors: Sequence[str]):
    data = df[[metric, *factors]].dropna(subset=[metric])
    y = data[metric].to_numpy(dtype=float)
    blocks = {f: nested_contrasts(data[f].astype(str), data[NESTED[f]].astype(str))
              if NESTED.get(f) in factors else dummies(data[f].astype(str)) for f in factors}
    return y, blocks


def nested_in(factor: str, factors: Sequence[str]) -> List[str]:
    """The factors nested in `factor` (left out of its type-II comparison)."""
    return [f for f in factors if NESTED.get(f) == factor]


# --- effect sizes ------------------------------------------------------------------------------------------------
def omega_squared(y: np.ndarray, blocks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Partial omega^2 of every factor from type-II sums of squares. Degrees of freedom are ranks (a factor's df = the
    rank it adds after all the others). NaN when not estimable: a single level / no added rank, or no variation."""
    n = len(y)
    gram = Gram(y, blocks)
    factors = list(blocks)
    full, rank_full = gram.rss(factors), gram.rank(factors)
    df_error = n - rank_full
    if df_error <= 0 or float(((y - y.mean()) ** 2).sum()) == 0:
        return {f: np.nan for f in factors}
    ms_error = full / df_error
    out = {}
    for factor in factors:
        base = [f for f in factors if f != factor and f not in nested_in(factor, factors)]
        df_f = gram.rank(base + [factor]) - gram.rank(base)
        if df_f == 0:
            out[factor] = np.nan
            continue
        ss_f = gram.rss(base) - gram.rss(base + [factor])
        denominator = ss_f + (n - df_f) * ms_error
        out[factor] = (ss_f - df_f * ms_error) / denominator if denominator > 0 else np.nan
    return out


def shapley_r2(y: np.ndarray, blocks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Shapley decomposition of R^2 over the factors (exact: every subset is fitted once)."""
    factors = list(blocks)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return {f: np.nan for f in factors}
    gram = Gram(y, blocks)
    r2 = {}
    for size in range(len(factors) + 1):
        for subset in combinations(factors, size):
            r2[frozenset(subset)] = 1 - gram.rss(subset) / total
    k = len(factors)
    out = {}
    for factor in factors:
        others = [f for f in factors if f != factor]
        value = 0.0
        for size in range(k):
            weight = factorial(size) * factorial(k - size - 1) / factorial(k)
            for subset in combinations(others, size):
                value += weight * (r2[frozenset(subset) | {factor}] - r2[frozenset(subset)])
        out[factor] = value
    return out


# --- bootstrap ---------------------------------------------------------------------------------------------------
def resample_models(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Models drawn with replacement, each with all its configurations; a repeated model gets its own label."""
    models = df[MODEL_FACTOR].unique()
    groups = {m: g for m, g in df.groupby(MODEL_FACTOR, sort=False)}
    picks = rng.choice(models, size=len(models), replace=True)
    return pd.concat([groups[m].assign(**{MODEL_FACTOR: f"{m}#{i}"}) for i, m in enumerate(picks)], ignore_index=True)


def influence(df: pd.DataFrame, metric: str, factors: Sequence[str] = DEFAULT_FACTORS,
              n_boot: int = cons.DEFAULT_BOOTSTRAP_SAMPLES, n_boot_shapley: int | None = None,
              confidence: float = cons.DEFAULT_CONFIDENCE, seed: int = cons.DEFAULT_SEED) -> pd.DataFrame:
    """One row per factor: partial omega^2 and Shapley R^2 with cluster-bootstrap CIs (+ std of the replicates)."""
    n_boot_shapley = n_boot if n_boot_shapley is None else n_boot_shapley
    y, blocks = _prepare(df, metric, factors)
    point_omega, point_shapley = omega_squared(y, blocks), shapley_r2(y, blocks)
    full_r2 = sum(v for v in point_shapley.values() if np.isfinite(v))
    rng = np.random.default_rng(seed)
    omega_boot, shapley_boot = [], []
    for b in range(max(n_boot, n_boot_shapley)):
        y_b, blocks_b = _prepare(resample_models(df, rng), metric, factors)
        if b < n_boot:
            omega_boot.append(omega_squared(y_b, blocks_b))
        if b < n_boot_shapley:
            shapley_boot.append(shapley_r2(y_b, blocks_b))
    alpha = (1 - confidence) / 2
    rows = []
    for factor in factors:
        row = {'metric': metric, 'variable': factor, 'family': variable_family(factor),
               'n_configs': len(y), 'n_models': df.loc[df[metric].notna(), MODEL_FACTOR].nunique(), 'r2_full': full_r2}
        for name, point, boot in (('omega2', point_omega, omega_boot), ('shapley_r2', point_shapley, shapley_boot)):
            values = np.array([r[factor] for r in boot], dtype=float)
            values = values[np.isfinite(values)]
            row[name] = point[factor]
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = (np.quantile(values, [alpha, 1 - alpha]) if len(values)
                                                             else (np.nan, np.nan))
            row[f"{name}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"{name}_n_boot"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def variable_family(variable: str) -> str:
    if variable == MODEL_FACTOR:
        return 'model'
    return 'persona' if variable in cons.PERSONA_VARIABLES else 'context'
