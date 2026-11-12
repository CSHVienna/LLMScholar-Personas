"""
RQ3: socio-technical trade-off. One point per model: technical score = sum of the technical metrics, social score =
sum of the social metrics (defaults: cons.RQ3_TECHNICAL_METRICS / cons.RQ3_SOCIAL_METRICS). A metric where lower is
better (direction LOWER in metrics/summary.METRICS: refusals, duplicates) enters as its complement 1 - x, so larger is
better on both axes and a raw metric adds [0, 1] to its axis. A size-adjusted social metric (<key>_adj = observed - E,
E = <key>_expected, the value of a random recommender of the same n; notebooks: ADJUSTED) enters as is (-x if lower
is better) and can only take [-E, 1 - E] ([E - 1, E] if lower is better): with E ~ 0.8 (parity at n ~ 10) little room
is left above random.

A model's metric = the mean over its configurations (the configuration table of metrics/summary); its attainable
range per axis = the sum of the per-metric ranges with E = its mean over the model's configurations (<axis>_min,
<axis>_max: they differ between models through n). The axis range covers every model: [min of the minima, max of
the maxima]. Without the _expected columns, an adjusted metric falls back to [-1, 1] (with a warning).
Quadrants:
    relative   split at the per-axis MEDIANS across the models (models compared with each other)
    absolute   split at 1/2 per raw metric and 0 per adjusted one: the middle of the axis (raw), the random
               recommender (adjusted; not the middle of the lopsided range)
    Q1 high technical / high social, Q2 low / high, Q3 low / low, Q4 high / low (>= the split counts as high).
Pareto front: models not dominated on both axes by another model.
"""

import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd

from metrics import summary
from utils import constants as cons

QUADRANTS = {'Q1': 'high tech / high social', 'Q2': 'low tech / high social', 'Q3': 'low tech / low social',
             'Q4': 'high tech / low social'}
MODES = cons.RQ3_MODES


def is_adjusted(key: str) -> bool:
    """A size-adjusted metric (<key>_adj: observed - expected, in [-1, 1])."""
    return key.endswith(summary.ADJ_SUFFIX)


def oriented(values: pd.Series, metric: summary.Metric) -> pd.Series:
    """The metric so that larger is better: the complement 1 - x if lower is better (-x if size-adjusted)."""
    if metric.direction != summary.LOWER:
        return values
    return -values if is_adjusted(metric.key) else 1 - values


def expected_column(key: str) -> str:
    """<base>_expected of a size-adjusted key <base>_adj (the value of a random recommender)."""
    return key[:-len(summary.ADJ_SUFFIX)] + summary.EXPECTED_SUFFIX


def metric_range(configs: pd.DataFrame, metric: summary.Metric) -> pd.DataFrame:
    """Per model: the attainable (low, high) of the oriented model mean. Raw: (0, 1); adjusted: (-E, 1 - E), or
    (E - 1, E) if lower is better, E = mean of <base>_expected over the model's configurations with a value."""
    models = configs[cons.COL_MODEL].drop_duplicates()
    if not is_adjusted(metric.key):
        return pd.DataFrame({'low': 0.0, 'high': 1.0}, index=pd.Index(models, name=cons.COL_MODEL))
    column = expected_column(metric.key)
    if column not in configs:
        warnings.warn(f"RQ3: no {column} in the configuration table; {metric.key} assumed in [-1, 1]", UserWarning,
                      stacklevel=3)
        return pd.DataFrame({'low': -1.0, 'high': 1.0}, index=pd.Index(models, name=cons.COL_MODEL))
    e = configs[configs[metric.key].notna()].groupby(cons.COL_MODEL)[column].mean().reindex(models)
    e.index.name = cons.COL_MODEL
    if metric.direction == summary.LOWER:
        return pd.DataFrame({'low': e - 1, 'high': e})
    return pd.DataFrame({'low': -e, 'high': 1 - e})


def model_scores(configs: pd.DataFrame, technical: Sequence[str] = tuple(cons.RQ3_TECHNICAL_METRICS),
                 social: Sequence[str] = tuple(cons.RQ3_SOCIAL_METRICS)) -> pd.DataFrame:
    """One row per model: each metric (mean over configurations, oriented), technical / social score, the model's
    attainable range per axis (<axis>_min, <axis>_max; module docstring), the absolute split per axis (<axis>_split:
    1/2 per raw metric, 0 per adjusted one) and n_missing (metrics without a value, left out of the sum)."""
    check_adjusted_axes(technical, social)
    keys = list(dict.fromkeys([*technical, *social]))
    missing = [k for k in keys if k not in configs]
    if missing:
        raise ValueError(f"Metrics not in the configuration table: {missing} (run the metrics / social step)")
    means = configs.groupby(cons.COL_MODEL)[keys].mean()
    for key in keys:
        means[key] = oriented(means[key], summary.METRICS[key])
    scores = means.copy()
    scores['technical'] = means[list(technical)].sum(axis=1, min_count=1)
    scores['social'] = means[list(social)].sum(axis=1, min_count=1)
    for axis, axis_keys in (('technical', technical), ('social', social)):
        ranges = [metric_range(configs, summary.METRICS[k]) for k in axis_keys]
        scores[f'{axis}_min'] = sum(r['low'] for r in ranges).reindex(scores.index)
        scores[f'{axis}_max'] = sum(r['high'] for r in ranges).reindex(scores.index)
        scores[f'{axis}_split'] = 0.5 * sum(not is_adjusted(k) for k in axis_keys)
    scores['n_missing'] = means[keys].isna().sum(axis=1)
    return scores.reset_index()


def quadrants(scores: pd.DataFrame, mode: str = 'relative') -> pd.Series:
    """Q1-Q4 of every model (see module docstring)."""
    split_x, split_y = splits(scores, mode)
    high_t, high_s = scores['technical'] >= split_x, scores['social'] >= split_y
    return pd.Series(np.select([high_t & high_s, ~high_t & high_s, ~high_t & ~high_s], ['Q1', 'Q2', 'Q3'], 'Q4'),
                     index=scores.index)


def splits(scores: pd.DataFrame, mode: str = 'relative') -> tuple:
    """(technical, social) value where the quadrants are split."""
    if mode == 'relative':
        return float(scores['technical'].median()), float(scores['social'].median())
    if mode == 'absolute':
        return tuple(float(scores[f'{axis}_split'].iloc[0]) for axis in ('technical', 'social'))
    raise ValueError(f"mode must be one of {MODES}")


def axis_range(scores: pd.DataFrame, axis: str) -> tuple:
    """(minimum, maximum) attainable score of an axis ('technical' / 'social') over all models: the lowest model
    minimum and the highest model maximum."""
    return float(np.nanmin(scores[f'{axis}_min'])), float(np.nanmax(scores[f'{axis}_max']))


def pareto_front(scores: pd.DataFrame) -> pd.Series:
    """True for the models no other model beats on both axes (>= on both, > on one)."""
    t, s = scores['technical'].to_numpy(), scores['social'].to_numpy()
    dominated = [bool(np.any((t >= t[i]) & (s >= s[i]) & ((t > t[i]) | (s > s[i])))) for i in range(len(t))]
    return pd.Series(~np.array(dominated), index=scores.index)


def check_adjusted_axes(technical: Sequence[str], social: Sequence[str]) -> None:
    """Warn when an axis sum is not interpretable: (1) raw and size-adjusted metrics mixed (different zero, range and
    split); (2) a metric without a better direction (summary.NEUTRAL: popularity, cohesion, locality), whose sign
    would cancel the others (more elite / local than random is not 'more social'). Several higher-is-better
    metrics, raw ([0, 1]) or adjusted ([-1, 1]: excess over a random recommender), e.g. parity + diversity, are fine."""
    for name, keys in (('technical', technical), ('social', social)):
        adjusted = [k for k in keys if is_adjusted(k)]
        if adjusted and len(adjusted) < len(keys):
            warnings.warn(f"RQ3 {name} axis mixes raw {[k for k in keys if not is_adjusted(k)]} and size-adjusted "
                          f"{adjusted} metrics: different scales, axis range and absolute split.",
                          UserWarning, stacklevel=3)
        neutral = [k for k in keys if summary.METRICS[k].direction == summary.NEUTRAL]
        if neutral:
            warnings.warn(f"RQ3 {name} axis sums {neutral}, which have no better direction (higher is not better): "
                          f"their sign cancels the other metrics, and the axis range assumes [0, 1] / [-1, 1] per "
                          f"metric (similarity is a cosine: [-1, 1], adj [-2, 2]).", UserWarning, stacklevel=3)


def axis_label(keys: Sequence[str]) -> str:
    """e.g. 'validity + refusals$^c$ + duplicates$^c$ + $\\sum$ factuality' (a subgroup of > 1 metric is summed);
    size-adjusted social metrics: '$\\sum$ parity$^{adj}$' (as the metric labels of RQ1 / RQ2)."""
    parts: List[str] = []
    groups = [summary.METRICS[k].group for k in keys]
    i = 0
    while i < len(keys):
        j = i
        while j + 1 < len(keys) and groups[j + 1] == groups[i]:
            j += 1
        metric = summary.METRICS[keys[i]]
        adj = is_adjusted(metric.key)
        if j > i:
            parts.append(f"$\\sum$ {groups[i].removesuffix(summary.ADJ_SUFFIX)}" + ('$^{adj}$' if adj else ''))
        else:
            name = metric.key.split('_')[0] if metric.family == 'social' else metric.key
            marks = ['adj'] * adj + ['c'] * (metric.direction == summary.LOWER)   # c: complement (adj: -x)
            sup = ','.join(marks)
            parts.append(name + (f"$^{sup}$" if len(sup) == 1 else f"$^{{{sup}}}$" if marks else ''))
        i = j + 1
    return ' + '.join(parts)
