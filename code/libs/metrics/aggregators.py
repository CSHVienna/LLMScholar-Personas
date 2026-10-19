"""
aggregators.py — Averages and confidence intervals for the plots.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from tqdm.auto import tqdm

from libs.metrics import constants

tqdm.pandas()


def ci95(series: pd.Series) -> float:
    """95% CI half-width using Student t-distribution (non-binary metrics)."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, df=len(s) - 1) * s.sem())


def ci_wilson(series: pd.Series) -> tuple:
    """Return (mean, half-width) for Wilson 95% CI of a 0/1 series."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return (np.nan, np.nan)
    k = float(s.sum())
    low, high = proportion_confint(count=k, nobs=n, alpha=0.05, method="wilson")
    return (k / n, (high - low) / 2.0)


def aggregate_scores(
    all_calls_df: pd.DataFrame,
    group_by: list,
    metrics: list = constants.ALL_METRICS,
    binary_metrics: list = constants.BINARY_METRICS,
) -> pd.DataFrame:
    """Mean ± 95% CI aggregation.

    Wilson score CI for binary metrics (validity, refusals);
    Student-t CI for everything else.
    """
    rows = []
    for keys, g in all_calls_df.groupby(group_by, dropna=False):
        row = dict(zip(group_by, keys if isinstance(keys, tuple) else (keys,)))
        row["n"] = len(g)
        for m in metrics:
            s = g[m].dropna() if m in g.columns else pd.Series(dtype=float)
            if m in binary_metrics:
                mean_v, ci_v = ci_wilson(s)
                row[f"{m}_mean"] = mean_v
                row[f"{m}_ci"] = ci_v
            else:
                row[f"{m}_mean"] = s.mean() if len(s) else np.nan
                row[f"{m}_ci"] = ci95(s)
            row[f"{m}_std"] = s.std() if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_attempt(df, group_cols, metric_agg):

    def pct(x):
        return x["n_counts"] / x["n_total"]

    def pct_inverse(x):
        return 1 - x["n_counts"] / x["n_total"]

    if type(metric_agg) == tuple:
        per_attempt = (
            df.groupby(group_cols, as_index=False)
            .agg(metric=metric_agg)
            .reset_index(drop=True)
        )
    else:
        k = [k for k in metric_agg.keys() if k != "ntotal"]

        if len(k) == 1:
            k = k[0]
            per_attempt = (
                df.groupby(group_cols, as_index=False)
                .agg(n_total=metric_agg["ntotal"], n_counts=metric_agg[k])
                .assign(
                    metric=lambda x: pct(x) if k in ["npositive"] else pct_inverse(x)
                )
                .reset_index(drop=True)
            )
        else:
            raise ValueError(f"Metric {metric_agg} not supported")

    return per_attempt


def aggregate_factuality_task(df_factuality_task, metric):
    """Per-attempt match rate for a task-level metric (field / seniority / location).

    ``metric`` is a key of ``constants.FACTUALITY_TASK_STATUS_COLS``; the status
    column it maps to holds ``<prefix>_match`` / ``<prefix>_mismatch`` /
    ``<prefix>_unknown`` / ``not_applicable``. Only match and mismatch are
    evaluable — the other two mean there was nothing to check against (e.g. an
    author with no topics on the field side), so they are dropped rather than
    counted as failures, exactly as step 6 of build_valid_calls.py does.
    """
    status_col = constants.FACTUALITY_TASK_STATUS_COLS[metric]
    prefix = status_col.split("_")[0]
    df_factuality_task_clean = df_factuality_task.drop_duplicates(
        subset=[
            "model",
            "grounded",
            "temperature",
            "date",
            "time",
            "task_name",
            "task_param",
            "task_attempt",
            "clean_name",
        ]
    ).copy()
    eligible = df_factuality_task_clean[
        df_factuality_task_clean[status_col].isin(
            [f"{prefix}_match", f"{prefix}_mismatch"]
        )
    ].copy()
    eligible[status_col] = (eligible[status_col] == f"{prefix}_match").astype(int)
    metric_agg = (
        status_col,
        "mean",
    )  # how many factual field-author / location-author / seniority-author checks are true
    per_attempt = aggregate_per_attempt(
        eligible, constants.BENCHMARK_PER_ATTEMPT_COLS, metric_agg
    )
    return per_attempt
