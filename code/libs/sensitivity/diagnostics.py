"""Diagnostics: upper bound on variance explainable by the measured design."""

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VarianceCeiling:
    metric: str
    n: int
    n_cells: int
    share_replicated: float            # share of cells with >= 2 rows
    r2_ceiling: float                  # saturated-model R^2
    r2_ceiling_adj: float              # penalized for one parameter per cell
    observed_within_var: float
    expected_sampling_var: float | None = None   # p(1-p)/k, if k given

    def to_dict(self) -> dict:
        return asdict(self)


def design_cells(df: pd.DataFrame, cell_vars: list[str]) -> pd.Series:
    """One label per unique configuration of the design variables."""
    return df[cell_vars].astype(str).agg('|'.join, axis=1)


def variance_ceiling(df: pd.DataFrame,
                     metric: str,
                     cell_vars: list[str],
                     k_col: str | None = None) -> VarianceCeiling:
    """Decompose variance of `metric` into between- and within-cell parts."""
    y = df[metric].astype(float)
    g = y.groupby(design_cells(df, cell_vars))
    n, n_cells = len(y), g.ngroups

    cell_mean = g.transform('mean')
    ss_total = ((y - y.mean()) ** 2).sum()
    ss_within = ((y - cell_mean) ** 2).sum()
    dof_within = n - n_cells

    r2 = 1 - ss_within / ss_total
    r2_adj = (1 - (ss_within / dof_within) / (ss_total / (n - 1))
              if dof_within > 0 else np.nan)
    within_var = ss_within / dof_within if dof_within > 0 else np.nan

    expected = None
    if k_col is not None:
        k = df[k_col].astype(float)
        expected = float((cell_mean * (1 - cell_mean) / k).mean())

    return VarianceCeiling(
        metric=metric, n=n, n_cells=n_cells,
        share_replicated=float((g.size() >= 2).mean()),
        r2_ceiling=float(r2), r2_ceiling_adj=float(r2_adj),
        observed_within_var=float(within_var),
        expected_sampling_var=expected,
    )