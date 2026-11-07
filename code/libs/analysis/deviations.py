"""
RQ2: local influence of each VALUE of the configuration variables on each metric.

Same data and model as RQ1 (analysis/influence.py): one row per model x configuration; per metric, rows with a missing
value are left out; metric ~ model + language + location + role + k + field + subfield(field) + seniority, additive
OLS (subfield nested in field).

Per level of a variable, weighted effect coding (te Grotenhuis et al. 2017): every level is compared with the GRAND
MEAN, no reference level is needed.
    EMM(level)       estimated marginal mean: the mean prediction over all configurations with the variable set to
                     the level, the other variables kept as observed (weights = observed shares)
    deviation(level) EMM(level) - grand mean (= the mean of the metric; the deviations, weighted by the level shares,
                     sum to 0)
    Nested subfield: setting a subfield also sets its field, so its EMM includes the field's effect; a field's EMM
    averages its subfields with their observed shares.
    contrast(level)  EMM(level) - EMM(reference level) (treatment coding; references: cons.RQ2_REFERENCE_LEVELS,
                     else the first value in the design order; a subfield vs the reference subfield of its field)
Fit: r2, r2_adj (rank-based degrees of freedom), delta_r2_model = r2 - r2 without the model.
Confidence intervals: cluster bootstrap over MODELS (as RQ1), percentile intervals; point estimates on the data.
The model is a control in the regression; its own levels are not reported (model comparisons: the benchmark figures
of 3_metrics.ipynb; a bootstrap over models gives no CI for one model).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.influence import DEFAULT_FACTORS, MODEL_FACTOR, NESTED, Gram, _prepare, resample_models, variable_family
from utils import constants as cons


def fit(y: np.ndarray, blocks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """OLS on [1, blocks]; the contribution of every factor to every fitted value (block @ its coefficients)."""
    X = np.column_stack([np.ones(len(y)), *blocks.values()])
    G, c = X.T @ X, X.T @ y
    beta, *_ = np.linalg.lstsq(G, c, rcond=None)
    out, start = {}, 1
    for factor, block in blocks.items():
        out[factor] = block @ beta[start:start + block.shape[1]]
        start += block.shape[1]
    return out


def level_deviations(data: pd.DataFrame, contributions: Dict[str, np.ndarray], factors: Sequence[str]) -> pd.DataFrame:
    """One row per (variable, parent, level): n, deviation from the grand mean (see module docstring).
    The counterfactual 'set the variable to the level' changes the variable's contribution, plus its parent's (a
    nested variable) or the contributions of the variables nested in it (averaged over the level's rows)."""
    rows = []
    for factor in factors:
        if factor == MODEL_FACTOR:
            continue
        parent = NESTED.get(factor) if NESTED.get(factor) in factors else None
        total = contributions[factor] + sum(contributions[f] for f in factors if NESTED.get(f) == factor) \
            + (contributions[parent] if parent else 0)
        frame = pd.DataFrame({'level': data[factor].astype(str).to_numpy(), 'value': total,
                              'parent': data[parent].astype(str).to_numpy() if parent else ''})
        grand = float(total.mean())
        for (parent_level, level), g in frame.groupby(['parent', 'level'], sort=False):
            rows.append({'variable': factor, 'parent': parent_level, 'level': level, 'n': len(g),
                         'deviation': float(g['value'].mean()) - grand})
    return pd.DataFrame(rows)


def design_order(variable: str, levels) -> list:
    """Levels in the design order (cons.VARIABLE_VALUES), unknown levels last (alphabetical)."""
    known = [str(x) for x in cons.VARIABLE_VALUES.get(variable, [])]
    levels = set(levels)
    return [x for x in known if x in levels] + sorted(levels - set(known))


def add_contrasts(table: pd.DataFrame, references: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """reference and contrast (deviation - deviation of the reference) of every row; per (variable, parent)."""
    references = cons.RQ2_REFERENCE_LEVELS if references is None else references
    out = []
    for (variable, _), g in table.groupby(['variable', 'parent'], sort=False):
        wanted = references.get(variable)
        reference = wanted if wanted in set(g['level']) else design_order(variable, g['level'])[0]
        base = float(g.loc[g['level'] == reference, 'deviation'].iloc[0])
        out.append(g.assign(reference=reference, contrast=g['deviation'] - base))
    return pd.concat(out).sort_index()


def fit_statistics(y: np.ndarray, blocks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """R^2, adjusted R^2 and the R^2 the model (LLM) adds over the configuration variables."""
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return {'r2': np.nan, 'r2_adj': np.nan, 'delta_r2_model': np.nan}
    gram, factors = Gram(y, blocks), list(blocks)
    r2 = 1 - gram.rss(factors) / total
    rank = gram.rank(factors)
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - rank) if len(y) > rank else np.nan
    without = [f for f in factors if f != MODEL_FACTOR]
    delta = r2 - (1 - gram.rss(without) / total) if MODEL_FACTOR in factors else np.nan
    return {'r2': r2, 'r2_adj': r2_adj, 'delta_r2_model': delta}


def deviations(df: pd.DataFrame, metric: str, factors: Sequence[str] = DEFAULT_FACTORS,
               n_boot: int = cons.DEFAULT_BOOTSTRAP_SAMPLES, confidence: float = cons.DEFAULT_CONFIDENCE,
               seed: int = cons.DEFAULT_SEED, references: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """One row per level of every configuration variable: share, grand_mean, emm, deviation, reference, contrast
    (+ _ci_low, _ci_high, _std, _n_boot for the three), r2, r2_adj, delta_r2_model, n_configs, n_models."""
    data = df[[metric, *factors]].dropna(subset=[metric])
    y, blocks = _prepare(data, metric, factors)
    point = add_contrasts(level_deviations(data, fit(y, blocks), factors), references)
    key = ['variable', 'parent', 'level']
    rng = np.random.default_rng(seed)
    boot: List[pd.DataFrame] = []
    for b in range(n_boot):
        sample = resample_models(data, rng)
        y_b, blocks_b = _prepare(sample, metric, factors)
        replicate = add_contrasts(level_deviations(sample, fit(y_b, blocks_b), factors), references)
        boot.append(replicate.assign(b=b, grand_mean=float(y_b.mean())))
    alpha = (1 - confidence) / 2
    grand_mean = float(y.mean())
    point['share'] = point['n'] / point.groupby('variable')['n'].transform('sum')
    point['grand_mean'], point['emm'] = grand_mean, grand_mean + point['deviation']
    if boot:
        draws = pd.concat(boot, ignore_index=True)
        draws['emm'] = draws['grand_mean'] + draws['deviation']
        for name in ('deviation', 'emm', 'contrast'):
            stats = draws.groupby(key)[name].agg(
                ci_low=lambda v: np.quantile(v, alpha), ci_high=lambda v: np.quantile(v, 1 - alpha),
                std=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, n_boot='count')
            point = point.merge(stats.add_prefix(f"{name}_").reset_index(), on=key, how='left')
    else:
        for name in ('deviation', 'emm', 'contrast'):
            for s in ('ci_low', 'ci_high', 'std'):
                point[f"{name}_{s}"] = np.nan
            point[f"{name}_n_boot"] = 0
    point.insert(0, 'metric', metric)
    point.insert(2, 'family', point['variable'].map(variable_family))
    for name, value in fit_statistics(y, blocks).items():
        point[name] = value
    point['n_configs'], point['n_models'] = len(y), data[MODEL_FACTOR].nunique()
    return point
