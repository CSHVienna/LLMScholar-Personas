"""
Interactions (plan D.7.3): do two-way interactions add to the additive RQ1 / RQ2 model? Stored and plotted apart from
the main results; RQ1 / RQ2 stay additive.

Data and base model as RQ1 (analysis/influence.py): one row per model x configuration, metric ~ model + language +
role + location + k + seniority + field + subfield(field), additive OLS. A block = the treatment-coded products of two
factors ((a - 1)(b - 1) columns), added to the base model:
    screening       every pair of cons.INTERACTION_VARIABLES (15) + LLM x each of them (6); the pre-specified ones
                    (cons.INTERACTION_PRESPECIFIED) are marked
    decomposition   model characteristic x variable (cons.INTERACTION_DECOMPOSITIONS, e.g. size x k): nested in the
                    LLM x variable block (a characteristic is constant within an LLM; its main effect is in the LLM
                    term); share = its Delta R^2 / the LLM block's Delta R^2 (how much of the model differences in the
                    variable's effect it explains); descriptive, not causal
    global          base + all screening blocks (one number per metric: is the additive model adequate?)
Per block (each vs the base model, not vs the other blocks):
    delta_r2        R^2(base + block) - R^2(base);  delta_r2_adj: the same with adjusted R^2
    f2              local Cohen's f^2 = delta_r2 / (1 - R^2(base + block))  (Selya et al. 2012)
    omega2          partial omega^2 = (SS - df MS_e) / (SS + (N - df) MS_e), SS = RSS(base) - RSS(base + block),
                    MS_e of base + block, df = added rank (Olejnik & Algina 2003; corrects for the block size)
    max_abs_deviation  largest |cell deviation| over the block's cells (the SESOI statistic, metric units)
    p_value         one-sided, omega2 > 0: 1 - Phi(omega2 / bootstrap SE) (Wald with the cluster-bootstrap SE; a
                    percentile p cannot go below 1 / (B + 1), too coarse for Holm / BY over hundreds of tests)
Cells (one row per level combination): observed = mean of the metric, additive = mean of the base prediction,
deviation = observed - additive (= the interaction the additive model misses; exact for a balanced design).
Confidence intervals: cluster bootstrap over MODELS (as RQ1), percentile; none for the cells of LLM blocks (a
bootstrap over models gives no CI for one model). Multiple-testing correction and flags: flag(), on the combined
table of all metrics (load()).
"""

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.influence import DEFAULT_FACTORS, MODEL_FACTOR, Gram, prepare, resample_models
from analysis.multiple_testing import add_p_values, benjamini_hochberg, benjamini_yekutieli, holm, wald_p
from utils import constants as cons
from utils.stats import grouped_percentile_summary, percentile_summary

Block = Tuple[str, str]
GLOBAL = 'all'
PRESPECIFIED, SCREENING, DECOMPOSITION, GLOBAL_ROLE = 'pre-specified', 'screening', 'decomposition', 'global'


def block_name(block: Block) -> str:
    return f"{block[0]}:{block[1]}"


def screening_blocks(variables: Sequence[str] = cons.INTERACTION_VARIABLES, model: bool = True) -> List[Block]:
    """Every pair of the variables (in their order) + (model, variable) for each."""
    return list(combinations(variables, 2)) + ([(MODEL_FACTOR, v) for v in variables] if model else [])


def is_prespecified(block: Block, prespecified: Sequence[Block] = cons.INTERACTION_PRESPECIFIED) -> bool:
    return frozenset(block) in {frozenset(b) for b in prespecified}


# --- design ------------------------------------------------------------------------------------------------------
def _codes(values: pd.Series, levels: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Integer codes of a factor (sorted levels, or the given ones: absent values -> -1)."""
    values = values.astype(str)
    levels = sorted(values.unique()) if levels is None else list(levels)
    return pd.Categorical(values, categories=levels).codes.astype(int), levels


def interaction_block(a: np.ndarray, na: int, b: np.ndarray, nb: int) -> sp.csr_matrix:
    """Treatment-coded products of two factors (codes; level 0 dropped): (na - 1)(nb - 1) sparse columns."""
    rows = np.flatnonzero((a > 0) & (b > 0))
    columns = (a[rows] - 1) * (nb - 1) + (b[rows] - 1)
    return sp.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(a), max((na - 1) * (nb - 1), 0)))


class Design:
    """y, the base blocks (RQ1 coding) and the interaction blocks of one data set, with their Gram cross products.
    levels: fixed levels per non-model factor (so the cells of a bootstrap replicate line up with the data's)."""

    def __init__(self, data: pd.DataFrame, metric: str, base: Sequence[str], blocks: Sequence[Block],
                 levels: Optional[Dict[str, List[str]]] = None):
        self.y, base_blocks = prepare(data, metric, base)
        self.base = list(base)
        self.levels, self.codes = {}, {}          # factor -> level names; factor -> (codes, number of levels)
        for factor in dict.fromkeys(f for block in blocks for f in block):
            fixed = None if factor == MODEL_FACTOR else (levels or {}).get(factor)   # models: relabelled per replicate
            codes, self.levels[factor] = _codes(data[factor], fixed)
            self.codes[factor] = (codes, len(self.levels[factor]))
        names = [block_name(b) for b in blocks]
        inter = {name: interaction_block(*self.codes[a], *self.codes[b]) for name, (a, b) in zip(names, blocks)}
        self.X_base = np.column_stack([np.ones(len(self.y)), *base_blocks.values()])
        self.gram = Gram(self.y, {**base_blocks, **inter})
        self.n = len(self.y)
        self.total = float(((self.y - self.y.mean()) ** 2).sum())

    def additive_prediction(self) -> np.ndarray:
        return self.X_base @ self.gram.coefficients(self.base)

    def cells(self, block: Block, prediction: np.ndarray) -> pd.DataFrame:
        """Observed and additive cell means of a block (one row per level combination present)."""
        (a, na), (b, nb) = self.codes[block[0]], self.codes[block[1]]
        ok = (a >= 0) & (b >= 0)
        key = a[ok] * nb + b[ok]
        n = np.bincount(key, minlength=na * nb)
        observed = np.bincount(key, self.y[ok], minlength=na * nb)
        additive = np.bincount(key, prediction[ok], minlength=na * nb)
        present = np.flatnonzero(n)
        return pd.DataFrame({'ia': present // nb, 'ib': present % nb, 'n': n[present],
                             'observed': observed[present] / n[present], 'additive': additive[present] / n[present]})


def block_statistics(design: Design, factors: Sequence[str]) -> Dict[str, float]:
    """delta_r2, delta_r2_adj, f2, omega2, df of `factors` (interaction block names) added to the base model."""
    gram, base, n = design.gram, design.base, design.n
    nan = {k: np.nan for k in ('delta_r2', 'delta_r2_adj', 'f2', 'omega2', 'df')}
    if design.total == 0:
        return nan
    rss_base, rank_base = gram.rss(base), gram.rank(base)
    full = base + list(factors)
    rss_full, rank_full = gram.rss(full), gram.rank(full)
    dof, df_error = rank_full - rank_base, n - rank_full
    if dof <= 0 or df_error <= 0 or n - rank_base <= 0:
        return {**nan, 'df': float(dof)}
    ss, ms_error = max(rss_base - rss_full, 0.0), rss_full / df_error
    denominator = ss + (n - dof) * ms_error
    adj = lambda rss, rank: 1 - (rss / (n - rank)) / (design.total / (n - 1))
    return {'delta_r2': ss / design.total, 'delta_r2_adj': adj(rss_full, rank_full) - adj(rss_base, rank_base),
            'f2': ss / rss_full if rss_full > 0 else np.nan,
            'omega2': (ss - dof * ms_error) / denominator if denominator > 0 else np.nan, 'df': float(dof)}


def _estimates(design: Design, screening: Sequence[Block], decompositions: Dict[Block, Block]):
    """Block statistics (dict name -> stats) and cell tables (dict name -> DataFrame) of one data set."""
    prediction = design.additive_prediction()
    stats, cells = {}, {}
    for block in [*screening, *decompositions]:
        name = block_name(block)
        stats[name] = block_statistics(design, [name])
        cells[name] = design.cells(block, prediction)
        dev = cells[name]['observed'] - cells[name]['additive']
        stats[name]['max_abs_deviation'] = float(dev.abs().max()) if len(dev) else np.nan
    for block, parent in decompositions.items():
        name, parent_name = block_name(block), block_name(parent)
        parent_r2 = stats[parent_name]['delta_r2'] if parent_name in stats else np.nan
        stats[name]['share'] = stats[name]['delta_r2'] / parent_r2 if parent_r2 and parent_r2 > 0 else np.nan
    stats[GLOBAL] = block_statistics(design, [block_name(b) for b in screening])
    stats[GLOBAL]['max_abs_deviation'] = np.nan
    return stats, cells


def interactions(df: pd.DataFrame, metric: str, base: Sequence[str] = DEFAULT_FACTORS,
                 screening: Optional[Sequence[Block]] = None,
                 decompositions: Optional[Dict[Block, Block]] = None,
                 prespecified: Sequence[Block] = cons.INTERACTION_PRESPECIFIED,
                 n_boot: int = cons.DEFAULT_BOOTSTRAP_SAMPLES, confidence: float = cons.DEFAULT_CONFIDENCE,
                 seed: int = cons.DEFAULT_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(blocks, cells) of one metric. blocks: one row per screening block, decomposition and the global check (see
    module docstring); cells: one row per level combination of every block. Decompositions whose characteristic
    column is missing from df are left out."""
    screening = screening_blocks() if screening is None else list(screening)
    decompositions = dict(cons.INTERACTION_DECOMPOSITIONS if decompositions is None else decompositions)
    decompositions = {b: p for b, p in decompositions.items() if b[0] in df and p in screening}
    blocks = [*screening, *decompositions]
    columns = list(dict.fromkeys([metric, *base, *(f for b in blocks for f in b)]))
    data = df[columns].dropna(subset=[metric]).reset_index(drop=True)
    design = Design(data, metric, base, blocks)
    point, point_cells = _estimates(design, screening, decompositions)
    rng = np.random.default_rng(seed)
    boot_stats, boot_cells = [], []
    for b in range(n_boot):
        replicate = Design(resample_models(data, rng), metric, base, blocks, design.levels)
        stats, cells = _estimates(replicate, screening, decompositions)
        boot_stats.append(stats)
        boot_cells.append({name: c.assign(b=b) for name, c in cells.items() if MODEL_FACTOR not in name.split(':')})
    measures = ['delta_r2', 'delta_r2_adj', 'f2', 'omega2', 'max_abs_deviation', 'share']
    rows = []
    for name in [*map(block_name, blocks), GLOBAL]:
        if name == GLOBAL:
            row = {'block': GLOBAL, 'factor_a': '', 'factor_b': '', 'role': GLOBAL_ROLE, 'parent': ''}
        else:
            a, b = name.split(':')
            decomposition = (a, b) in decompositions
            role = DECOMPOSITION if decomposition else (PRESPECIFIED if is_prespecified((a, b), prespecified)
                                                        else SCREENING)
            row = {'block': name, 'factor_a': a, 'factor_b': b, 'role': role,
                   'parent': block_name(decompositions[(a, b)]) if decomposition else ''}
        row['metric'] = metric
        row['df'] = point[name]['df']
        for measure in measures:
            if measure not in point[name]:
                continue
            row[measure] = point[name][measure]
            draws = [s[name].get(measure, np.nan) for s in boot_stats]
            row.update({f"{measure}_{k}": v for k, v in percentile_summary(draws, confidence).items()})
        se = row.get('omega2_std', np.nan)
        row['p_value'] = float(wald_p(row['omega2'], se, 'greater')) if np.isfinite(se) and se > 0 and np.isfinite(
            row['omega2']) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table['r2_base'] = 1 - design.gram.rss(design.base) / design.total if design.total else np.nan
    table['n_configs'], table['n_models'] = design.n, data[MODEL_FACTOR].nunique()
    cells = _cell_table(metric, blocks, design, point_cells, boot_cells, confidence)
    return table, cells


def _cell_table(metric: str, blocks: Sequence[Block], design: Design, point: Dict[str, pd.DataFrame],
                boot: List[Dict[str, pd.DataFrame]], confidence: float) -> pd.DataFrame:
    """One row per cell of every block: observed, additive, deviation (+ _ci_low, _ci_high, _std, _n_boot)."""
    out = []
    for a, b in blocks:
        name = block_name((a, b))
        frame = point[name].copy()
        frame['deviation'] = frame['observed'] - frame['additive']
        draws = [r[name] for r in boot if name in r]
        if draws:
            d = pd.concat(draws, ignore_index=True)
            d['deviation'] = d['observed'] - d['additive']
            ci = grouped_percentile_summary(d, ['ia', 'ib'], 'deviation', confidence).add_prefix('deviation_')
            frame = frame.merge(ci.reset_index(), on=['ia', 'ib'], how='left')
        else:
            for s in ('ci_low', 'ci_high', 'std'):
                frame[f"deviation_{s}"] = np.nan
            frame['deviation_n_boot'] = 0
        frame.insert(0, 'level_a', [design.levels[a][i] for i in frame['ia']])
        frame.insert(1, 'level_b', [design.levels[b][i] for i in frame['ib']])
        out.append(frame.drop(columns=['ia', 'ib']).assign(metric=metric, block=name, factor_a=a, factor_b=b))
    columns = ['metric', 'block', 'factor_a', 'level_a', 'factor_b', 'level_b']
    table = pd.concat(out, ignore_index=True)
    return table[columns + [c for c in table.columns if c not in columns]]


# --- multiple testing and flags ----------------------------------------------------------------------------------
def sesoi(metric: str, default: float = cons.INTERACTION_SESOI,
          by_metric: Optional[Dict[str, float]] = None) -> float:
    by_metric = cons.INTERACTION_SESOI_BY_METRIC if by_metric is None else by_metric
    return by_metric.get(metric, by_metric.get(metric.removesuffix(cons.ADJ_SUFFIX), default))


def flag(table: pd.DataFrame, alpha: float = cons.INTERACTION_ALPHA, omega2_min: float = cons.INTERACTION_OMEGA2_MIN,
         sesoi_default: float = cons.INTERACTION_SESOI, sesoi_by_metric: Optional[Dict[str, float]] = None
         ) -> pd.DataFrame:
    """Adjusted p-values and flags on the blocks of ALL metrics together (the correction runs over the whole table):
        p_by        BY over every screening block (pre-specified included) x metric  -> flag_screening (+ omega2);
                    used by the figures (metrics are correlated, in both directions)
        p_bh        BH over the same tests -> flag_screening_bh (+ omega2): for comparison only
        p_holm      Holm over the pre-specified blocks x metrics -> flag_prespecified (+ max |deviation| >= SESOI);
                    Holm over the global rows (one per metric) -> flag_global (+ omega2)
        reported    decompositions: their LLM block is flag_prespecified for the metric (else not reported)."""
    out = table.copy()
    blocks = out['role'].isin([PRESPECIFIED, SCREENING])
    pre, glob = out['role'] == PRESPECIFIED, out['role'] == GLOBAL_ROLE
    out['p_by'], out['p_bh'], out['p_holm'] = np.nan, np.nan, np.nan
    out.loc[blocks, 'p_by'] = benjamini_yekutieli(out.loc[blocks, 'p_value'])
    out.loc[blocks, 'p_bh'] = benjamini_hochberg(out.loc[blocks, 'p_value'])
    out.loc[pre, 'p_holm'] = holm(out.loc[pre, 'p_value'])
    out.loc[glob, 'p_holm'] = holm(out.loc[glob, 'p_value'])
    out['sesoi'] = [sesoi(m, sesoi_default, sesoi_by_metric) for m in out['metric']]
    big_omega = out['omega2'] >= omega2_min
    out['flag_screening'] = blocks & (out['p_by'] < alpha) & big_omega
    out['flag_screening_bh'] = blocks & (out['p_bh'] < alpha) & big_omega
    out['flag_prespecified'] = pre & (out['p_holm'] < alpha) & (out['max_abs_deviation'] >= out['sesoi'])
    out['flag_global'] = glob & (out['p_holm'] < alpha) & big_omega
    parent_flag = out.loc[pre].set_index(['metric', 'block'])['flag_prespecified']
    out['reported'] = [bool(parent_flag.get((m, p), False)) if r == DECOMPOSITION else True
                       for m, p, r in zip(out['metric'], out['parent'], out['role'])]
    return out


# --- outputs -----------------------------------------------------------------------------------------------------
def load(folder: Path | str, **flag_args) -> pd.DataFrame:
    """The block tables of every metric (<folder>/<metric>.parquet), with the adjusted p-values and flags (flag())."""
    files = sorted(Path(folder).glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"No interaction results in {folder} (run_metrics_pipeline.sh -s interactions)")
    return flag(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), **flag_args)


def load_cells(folder: Path | str) -> pd.DataFrame:
    """The cell tables of every metric (<folder>/cells/<metric>.parquet), with p_deviation (two-sided Wald, deviation /
    its bootstrap SE) and p_deviation_<holm|bh|by> over all cells with a CI (LLM cells have none: NaN)."""
    files = sorted((Path(folder) / 'cells').glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"No interaction cells in {folder}/cells")
    return add_p_values(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), 'deviation', 'two')
