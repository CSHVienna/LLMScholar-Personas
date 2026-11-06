"""
RQ3: socio-technical trade-off. One point per model: technical score = sum of the technical metrics, social score =
sum of the social metrics (defaults: cons.RQ3_TECHNICAL_METRICS / cons.RQ3_SOCIAL_METRICS). A metric where lower is
better (direction LOWER in metrics/summary.METRICS: refusals, duplicates) enters as its complement 1 - x, so larger is
better on both axes and the maximum of an axis is its number of metrics.

A model's metric = the mean over its configurations (the configuration table of metrics/summary).
Quadrants:
    relative   split at the per-axis MEDIANS across the models (models compared with each other)
    absolute   split at HALF the maximum possible score of each axis
    Q1 high technical / high social, Q2 low / high, Q3 low / low, Q4 high / low (>= the split counts as high).
Pareto front: models not dominated on both axes by another model.
"""

from typing import List, Sequence

import numpy as np
import pandas as pd

from metrics import summary
from utils import constants as cons

QUADRANTS = {'Q1': 'high tech / high social', 'Q2': 'low tech / high social', 'Q3': 'low tech / low social',
             'Q4': 'high tech / low social'}
MODES = cons.RQ3_MODES


def oriented(values: pd.Series, metric: summary.Metric) -> pd.Series:
    """The metric so that larger is better: the complement 1 - x if lower is better."""
    return 1 - values if metric.direction == summary.LOWER else values


def model_scores(configs: pd.DataFrame, technical: Sequence[str] = tuple(cons.RQ3_TECHNICAL_METRICS),
                 social: Sequence[str] = tuple(cons.RQ3_SOCIAL_METRICS)) -> pd.DataFrame:
    """One row per model: each metric (mean over configurations, oriented), technical / social score, their maxima,
    n_missing (metrics without a value, left out of the sum)."""
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
    scores['technical_max'], scores['social_max'] = len(technical), len(social)
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
        return float(scores['technical_max'].iloc[0]) / 2, float(scores['social_max'].iloc[0]) / 2
    raise ValueError(f"mode must be one of {MODES}")


def pareto_front(scores: pd.DataFrame) -> pd.Series:
    """True for the models no other model beats on both axes (>= on both, > on one)."""
    t, s = scores['technical'].to_numpy(), scores['social'].to_numpy()
    dominated = [bool(np.any((t >= t[i]) & (s >= s[i]) & ((t > t[i]) | (s > s[i])))) for i in range(len(t))]
    return pd.Series(~np.array(dominated), index=scores.index)


def axis_label(keys: Sequence[str]) -> str:
    """e.g. 'validity + refusals$^c$ + duplicates$^c$ + $\\sum$ factuality' (a subgroup of > 1 metric is summed)."""
    parts: List[str] = []
    groups = [summary.METRICS[k].group for k in keys]
    i = 0
    while i < len(keys):
        j = i
        while j + 1 < len(keys) and groups[j + 1] == groups[i]:
            j += 1
        metric = summary.METRICS[keys[i]]
        if j > i:
            parts.append(f"$\\sum$ {groups[i]}")
        else:
            name = metric.key.split('_')[0] if metric.family == 'social' else metric.key
            parts.append(name + ('$^c$' if metric.direction == summary.LOWER else ''))
        i = j + 1
    return ' + '.join(parts)
