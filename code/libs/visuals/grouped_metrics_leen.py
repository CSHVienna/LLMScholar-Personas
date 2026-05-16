"""
grouped_metrics.py

Reusable grouped-bar metric grid plot.

Produces the figure style used in the paper (e.g. Figure 2, "Infrastructure-level
performance"): a left label column listing section/row labels, then one panel per
metric with horizontal bars (mean ± 95% CI), best-in-group values bolded, and
arrows indicating directional preference (↑/↓) on metric titles.

Typical use:

    from libs.visuals.grouped_metrics import plot_grouped_metrics

    plot_grouped_metrics(
        calls_df,
        group_configs=[
            {'label': 'Access', 'column': 'model_access', 'color': '#4A90D9',
             'order': ['Open', 'Proprietary']},
            {'label': 'Size',   'column': 'model_size',   'color': '#5DB85D',
             'order': ['Small', 'Medium', 'Large', 'XL']},
        ],
        metrics=['factuality', 'duplicates', 'consistency', ...],
        metric_directions={'factuality': '↑', 'duplicates': '↓'},
        save_path='out.pdf',
    )
"""

from __future__ import annotations

import textwrap as _textwrap
from typing import Optional, Sequence, Mapping, Any

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy import stats as _stats
from statsmodels.stats.proportion import proportion_confint


# ── Default style constants (mirror gridcons.py) ─────────────────────────────
FIG_DPI         = 600
TICK_FONT_SIZE  = 8
TICK_FONT_COLOR = '#828282'
LABEL_FONT_SIZE = 11
SPINE_LW        = 0.3

DEFAULT_DIRECTIONS = {
    'validity':                    '↑',
    'refusals':                    '↓',
    'factuality_author':                  '↑',
    'factuality_field':            '↑',
    'factuality_seniority':        '↑',
    'factuality_location':         '↑',
    'consistency':                 None,
    'duplicates':                  '↓',
    'div_gender':                  None,
    'div_ethnicity':               None,
    'div_location':                None,
    'div_productivity_works':      None,
    'div_productivity_citations':  None,
    'parity_gender':               '↑',
    'parity_ethnicity':            '↑',
    'parity_works':                '↑',
    'parity_citations':            '↑',
    'popularity_works':            None,
    'popularity_citations':        None,
}

# Bernoulli (binary 0/1) metrics — Wilson score CI was used when aggregating
# raw per-call 0/1 indicators. With two-stage aggregation (per-cell mean first,
# then group mean), cell-level values are continuous in [0,1] and Wilson no
# longer applies — Student-t is used for everything. Re-populate this set if
# you ever pass raw per-call 0/1 data again.
BINARY_METRICS: set[str] = set()


def _ci95(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    return float(_stats.t.ppf(0.975, df=len(s) - 1) * s.sem())


def _ci_wilson(s: pd.Series) -> tuple[float, float]:
    """Return (mean, half-width) of Wilson 95% CI for a 0/1 series."""
    s = s.dropna()
    n = len(s)
    if n == 0:
        return (np.nan, np.nan)
    k = float(s.sum())
    low, high = proportion_confint(count=k, nobs=n, alpha=0.05, method='wilson')
    return (k / n, (high - low) / 2.0)


def _resolve_order(src_df: pd.DataFrame, col: str, order: Optional[Sequence] = None) -> list:
    available = set(src_df[col].dropna().unique())
    if order is not None:
        out = [v for v in order if v in available]
        out += sorted([v for v in available if v not in order], key=str)
        return out
    return sorted(available, key=str)


def _row_metric_stats(grp: pd.DataFrame, metric: str) -> tuple[float, float, int, str]:
    """Return (mean, ci_half, n, method) for one (group, metric) cell."""
    s = grp[metric].dropna() if metric in grp.columns else pd.Series(dtype=float)
    if metric in BINARY_METRICS:
        mean_v, ci_v = _ci_wilson(s)
        method = 'wilson'
    else:
        mean_v = float(s.mean()) if len(s) else np.nan
        ci_v   = _ci95(s)
        method = 'student-t'
    return mean_v, ci_v, len(s), method


def compute_grouped_metrics_table(
    all_calls_df: pd.DataFrame,
    group_configs: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    long: bool = False,
) -> pd.DataFrame:
    """
    Return the per-(section, row, metric) numeric table that
    `plot_grouped_metrics` would render.

    Parameters
    ----------
    long : bool
        If True, returns long-format (one row per metric × section × label) with
        columns [section, label, n_calls, metric, mean, ci_half, ci_low, ci_high,
        method]. If False (default), returns wide-format keyed by (section, label)
        with one column block (`{metric}_mean`, `{metric}_ci`, `{metric}_n`) per
        metric — same shape that `plot_grouped_metrics` uses internally.
    """
    long_rows: list[dict] = []
    wide_rows: list[dict] = []
    for gc in group_configs:
        col = gc['column']
        src_df = all_calls_df
        for fk, fv in gc.get('filter', {}).items():
            if fk in src_df.columns:
                src_df = src_df[src_df[fk] == fv]
        if col not in src_df.columns:
            continue
        order = _resolve_order(src_df, col, gc.get('order'))
        for val in order:
            grp = src_df[src_df[col] == val]
            wide = {'section': gc['label'], 'label': str(val), 'n_calls': len(grp)}
            for m in metrics:
                mean_v, ci_v, n_m, method = _row_metric_stats(grp, m)
                wide[f'{m}_mean'] = mean_v
                wide[f'{m}_ci']   = ci_v
                wide[f'{m}_n']    = n_m
                long_rows.append({
                    'section': gc['label'],
                    'label':   str(val),
                    'metric':  m,
                    'n':       n_m,
                    'mean':    mean_v,
                    'ci_half': ci_v,
                    'ci_low':  mean_v - ci_v if pd.notna(mean_v) and pd.notna(ci_v) else np.nan,
                    'ci_high': mean_v + ci_v if pd.notna(mean_v) and pd.notna(ci_v) else np.nan,
                    'method':  method,
                })
            wide_rows.append(wide)
    return pd.DataFrame(long_rows) if long else pd.DataFrame(wide_rows)


def _shades(hex_color: str, n: int) -> list[tuple]:
    """Return n shades of `hex_color`, light → dark."""
    base  = np.array(mcolors.to_rgb(hex_color))
    white = np.ones(3)
    if n == 1:
        return [tuple(white * 0.25 + base * 0.75)]
    return [tuple(white * (1 - t) + base * t)
            for t in np.linspace(0.35, 1.0, n)]


def _nice_metric_name(m: str) -> str:
    return (m.replace('_', ' ')
             .replace('div ', 'Div ')
             .replace('parity ', 'Par ')
             .replace('factuality field', 'Field Match')
             .replace('factuality seniority', 'Sen. Match')
             .title())


def plot_grouped_metrics(
    all_calls_df: pd.DataFrame,
    group_configs: Sequence[Mapping[str, Any]],
    *,
    metrics: Optional[Sequence[str]] = None,
    metric_directions: Optional[Mapping[str, Optional[str]]] = None,
    metric_labels: Optional[Mapping[str, str]] = None,
    force_sections = False,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    tick_font_size: int = TICK_FONT_SIZE,
    tick_font_color: str = TICK_FONT_COLOR,
    label_font_size: int = LABEL_FONT_SIZE,
    spine_lw: float = SPINE_LW,
    fig_dpi: int = FIG_DPI,
    panel_width: float = 1.6,
    show: bool = True,
    textwrap_width: int = 11,
    
) -> plt.Figure:
    """
    Plot grouped horizontal-bar metric panels with section labels and 95% CIs.

    Parameters
    ----------
    all_calls_df : DataFrame
        Per-call metric values. Each metric in `metrics` must be a column.
    group_configs : list of dict
        Each dict defines one section (block of rows). Keys:
          - 'label' (str): section label shown in the left column.
          - 'column' (str): DataFrame column whose values become rows.
          - 'color' (str hex): base color; rows get a gradient of shades.
          - 'order' (list, optional): preferred row order; unknown values appended.
          - 'filter' (dict, optional): equality filters applied to the DataFrame
            before grouping (e.g. {'field_en': 'Biology'}).
    metrics : list of str, optional
        Metric column names to plot. Defaults to columns of `all_calls_df`
        present in DEFAULT_DIRECTIONS.
    metric_directions : dict, optional
        Per-metric direction: '↑' (higher better), '↓' (lower better), or None.
        Best-in-group cells are bolded for directional metrics.
    figsize : tuple, optional
        (width, height) in inches; auto-sized if omitted.
    save_path : str, optional
        If given, savefig to this path (bbox_inches='tight').
    show : bool
        If True, plt.show() and plt.close() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    if metrics is None:
        metrics = [f"{m}_mean" for m in all_calls_df.columns if m in DEFAULT_DIRECTIONS]
    dirs = {**DEFAULT_DIRECTIONS, **(metric_directions or {})}

    # ── Build sections ────────────────────────────────────────────────────────
    sections = []
    
    for gc in group_configs:
        col    = gc['column']

        src_df = all_calls_df # if not force_sections else all_calls_df.query("column==@col").rename(columns={'label':col}).copy()

        for fk, fv in gc.get('filter', {}).items():
            if fk in src_df.columns:
                src_df = src_df[src_df[fk] == fv]
        if col not in src_df.columns:
            print(f"Warning: '{col}' not found — skipping '{gc['label']}'.")
            continue
        available = set(src_df[col].dropna().unique())
        if 'order' in gc:
            order  = [v for v in gc['order'] if v in available]
            order += sorted([v for v in available if v not in gc['order']], key=str)
        else:
            order = sorted(available, key=str)
        rows = []
        for val in order:
            grp = src_df[src_df[col] == val]
            row = {'label': str(val)}
            for m in metrics:
                m_mean = f"{m}_mean"
                tmp_mean = grp[m_mean].dropna()
                val_mean = tmp_mean.iloc[0] if m_mean in grp.columns and not tmp_mean.empty else np.nan
                row[m_mean] = val_mean

                m_ci = f"{m}_ci"
                tmp_ci = grp[m_ci].dropna()
                val_ci = tmp_ci.iloc[0] if m_ci in grp.columns and not tmp_ci.empty else np.nan
                row[m_ci] = val_ci
                
                m_n = f"{m}_n"
                cn = 'n'
                val_n = grp[cn].dropna().iloc[0] if cn in grp.columns else np.nan
                row[m_n] = val_n

            rows.append(row)
        if rows:
            sections.append({'label': gc['label'], 'color': gc['color'], 'rows': rows})

    if not sections or not metrics:
        raise ValueError("No valid sections or metrics.")

    # ── Y layout ─────────────────────────────────────────────────────────────
    SECTION_GAP = 0.8
    y = 0.0
    y_lookup, sec_ranges = {}, []
    for si, sec in enumerate(sections):
        if si > 0:
            y += SECTION_GAP
        y_top = y
        for ri in range(len(sec['rows'])):
            y_lookup[(si, ri)] = y
            y += 1.0
        sec_ranges.append((y_top, y - 1.0))
    y_max  = y - 1.0
    sep_ys = [(sec_ranges[i][1] + sec_ranges[i + 1][0]) / 2
              for i in range(len(sections) - 1)]

    # ── Label column width + bracket position ────────────────────────────────
    # Layout: [section label]──gap──[bracket]──gap──[row labels (right-aligned)]
    # Estimate text widths in inches (serif font, conservative ~0.70 in/char ratio).
    max_label_chars = max(
        (len(row['label']) for sec in sections for row in sec['rows']), default=10
    )
    max_sec_chars = max(
        max((len(line) for line in _textwrap.fill(sec['label'], width=textwrap_width,
                                                  break_long_words=True).split('\n')),
            default=8)
        for sec in sections
    )
    _row_label_in = max_label_chars * tick_font_size       * 0.70 / 72
    _sec_label_in = max_sec_chars   * label_font_size * 0.72 * 0.70 / 72

    SEC_LEFT_PAD  = 0.10   # inches: left margin for section labels
    SEC_BX_GAP    = 0.18   # inches: gap between section label and bracket
    BX_ROW_GAP    = 0.12   # inches: gap between bracket end and row labels
    BRACKET_W_IN  = 0.10   # inches: horizontal bracket extent
    RIGHT_PAD     = 0.10   # inches: margin right of row labels

    # When there's only one section, the section label and bracket are hidden
    # (see below), so reclaim that horizontal space.
    _single_section = len(sections) == 1
    
    if _single_section:
        label_col_w = max(1.0, SEC_LEFT_PAD + _row_label_in + RIGHT_PAD) #10e-100
    else:
        label_col_w = max(
            1.8,
            SEC_LEFT_PAD + _sec_label_in + SEC_BX_GAP + BRACKET_W_IN
            + BX_ROW_GAP + _row_label_in + RIGHT_PAD,
        )
    
    # Bracket positioned so that its right edge sits BX_ROW_GAP to the left of
    # the longest row label. Right-edge of row labels is at x≈0.98 normalized.
    _row_label_left = 0.98 - _row_label_in / label_col_w
    BX = _row_label_left - BX_ROW_GAP / label_col_w - BRACKET_W_IN / label_col_w


    # ── Figure ────────────────────────────────────────────────────────────────
    n_m  = len(metrics)
    pad  = 0.4
    if figsize is None:
        w = label_col_w + panel_width * n_m + 0.5
        h = max((y_max + 2 * pad) * 0.65, 1.2)
        figsize = (w, h)

    fig, axes = plt.subplots(
        1, n_m + 1, figsize=figsize,
        gridspec_kw={'width_ratios': [label_col_w] + [panel_width] * n_m, 'wspace': 0.12},
    )
    axes = np.atleast_1d(axes)
    y_lo, y_hi = -pad, y_max + pad

    # ── Label column ──────────────────────────────────────────────────────────
    lax = axes[0]
    lax.set_xlim(0, 1); lax.set_ylim(y_lo, y_hi)
    lax.invert_yaxis(); lax.axis('off')

    bracket_w_norm = BRACKET_W_IN / label_col_w
    sec_x_norm     = SEC_LEFT_PAD / label_col_w

    # Skip section label + bracket entirely when there is only one section —
    # the bracket adds visual noise without conveying any grouping information.
    draw_section_chrome = len(sections) > 1
    for si, (sec, (y_top, y_bot)) in enumerate(zip(sections, sec_ranges)):
        y_c = (y_top + y_bot) / 2
        if draw_section_chrome:
            # break_long_words=False prevents mid-word breaks (e.g. "Mathematics" → "Mathematic\ns")
            wrapped = _textwrap.fill(sec['label'], width=textwrap_width, break_long_words=True)
            lax.text(sec_x_norm, y_c, wrapped, ha='right', va='center', multialignment='right', fontsize=label_font_size * 0.72, fontweight='bold')
            lax.plot([BX, BX],                  [y_top - 0.3, y_bot + 0.3], color='#444', lw=spine_lw * 3)
            lax.plot([BX, BX + bracket_w_norm], [y_top - 0.3, y_top - 0.3], color='#444', lw=spine_lw * 3)
            lax.plot([BX, BX + bracket_w_norm], [y_bot + 0.3, y_bot + 0.3], color='#444', lw=spine_lw * 3)
        for ri, row in enumerate(sec['rows']):
            wrapped = _textwrap.fill(row['label'], width=textwrap_width, break_long_words=True)
            lax.text(0.98, y_lookup[(si, ri)], wrapped, ha='right', va='center', fontsize=tick_font_size, color=tick_font_color)

    # ── Metric panels ─────────────────────────────────────────────────────────
    # Estimate label text width in data-coord units (panel spans [0, 1] in data
    # coords mapped onto `panel_width` inches).
    CHAR_W_DATA = (tick_font_size - 1) / 72 * 0.65 / panel_width  # ~data units per char
    MARGIN      = 0.97   # don't let labels go past here

    labels = metric_labels or {}
    for m, ax in zip(metrics, axes[1:]):
        direction = dirs.get(m)
        arrow = f' {direction}' if direction else ''
        nice  = labels.get(m, _nice_metric_name(m))
        ax.set_title(f'{nice}{arrow}', fontsize=tick_font_size, fontweight='bold', pad=4)
        ax.set_xlim(0, 1); ax.set_ylim(y_lo, y_hi)
        ax.invert_yaxis(); ax.set_yticks([])
        # Use short labels ('0', '', '1') to prevent "1.00.0" overlap between adjacent panels
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '', '1'])
        ax.tick_params(axis='x', labelsize=tick_font_size, labelcolor=tick_font_color,
                       width=spine_lw)
        ax.spines['bottom'].set_linewidth(spine_lw)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.grid(axis='x', linewidth=0.4, alpha=0.3, zorder=0)
        for sy in sep_ys:
            ax.axhline(sy, color='#bbb', lw=0.8, ls='--', zorder=1)

        for si, sec in enumerate(sections):
            n_rows = len(sec['rows'])
            colors = _shades(sec['color'], n_rows)
            vals   = [row[f'{m}_mean'] for row in sec['rows']]
            cis    = [row[f'{m}_ci']   for row in sec['rows']]
            ns     = [row[f'{m}_n']    for row in sec['rows']]

            # print(vals, cis, ns)

            good = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
            if direction == '↑' and good:
                best = max(v for _, v in good)
            elif direction == '↓' and good:
                best = min(v for _, v in good)
            else:
                best = None

            for ri, (val, ci, n, color) in enumerate(zip(vals, cis, ns, colors)):
                yv = y_lookup[(si, ri)]
                if not np.isfinite(val):
                    continue
                is_best = best is not None and abs(val - best) < 1e-9
                low_n   = n < 3
                ax.barh(yv, val, height=0.55, color=color, alpha=0.92, zorder=2)
                ci_val = ci if (np.isfinite(ci) and not low_n) else 0.0
                if ci_val:
                    ax.errorbar(val, yv, xerr=ci_val, fmt='none',
                                color='#333', lw=0.9, capsize=2, zorder=3)

                txt   = f'{val:.2f}{"*" if low_n else ""}'
                fw    = 'bold' if is_best else 'normal'
                gap   = ci_val + 0.015
                x_out = val + gap           # outside (right of bar)
                x_in  = val - gap           # inside  (left of bar end)
                txt_w = len(txt) * CHAR_W_DATA

                # Place outside unless it would overflow the axis margin
                if x_out + txt_w <= MARGIN:
                    ax.text(x_out, yv, txt, ha='left', va='center',
                            fontsize=tick_font_size - 1, fontweight=fw,
                            zorder=4, clip_on=True)
                else:
                    ax.text(x_in, yv, txt, ha='right', va='center',
                            fontsize=tick_font_size - 1, fontweight=fw,
                            zorder=4, clip_on=True)

    fig.tight_layout(pad=0.3, w_pad=0.0, h_pad=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=fig_dpi)
        print(f'Saved → {save_path}')
    if show:
        plt.show()
        plt.close()
    return fig



def plot_grouped_metrics_labeled(
    all_calls_df: pd.DataFrame,
    group_configs: Sequence[Mapping[str, Any]],
    *,
    section_labels: Optional[Mapping[str, str]] = None,
    row_labels: Optional[Mapping[str, str]] = None,
    **plot_kwargs,
):
    """
    Same as ``plot_grouped_metrics`` plus optional label remapping for the
    section header and the row labels.
 
    Parameters
    ----------
    all_calls_df, group_configs, **plot_kwargs
        Forwarded to ``plot_grouped_metrics`` after preprocessing. See its
        docstring for ``metrics``, ``metric_directions``, ``metric_labels``,
        ``figsize``, ``save_path``, ``show``, etc.
    section_labels : dict, optional
        Global mapping ``{gc['label']: display}`` for section headers.
        Equivalent in spirit to ``metric_labels`` but applied to ``label``.
    row_labels : dict, optional
        Global mapping ``{df_value: display}`` for unique values of every
        ``gc['column']``, which appear as the row labels.
 
    Per-config overrides (optional keys inside each ``group_config`` dict):
        - ``label_labels``   — overrides ``section_labels`` for that section
        - ``column_labels``  — overrides ``row_labels`` for that section
 
    Resolution order: per-config > top-level > original value.
    Unknown values fall through unchanged.
 
    Implementation note
    -------------------
    Row-label remapping is done by writing a derived column to a local
    copy of the DataFrame and pointing ``gc['column']`` at it; this keeps
    the downstream function (incl. aggregation, ordering, best-in-group
    highlighting) entirely unchanged.
    """
    df = all_calls_df.copy()
    new_configs: list[dict] = []
 
    # force_sections = plot_kwargs.get('force_sections', False)
    
    for i, gc in enumerate(group_configs):
        # Strip our extension keys before forwarding.
        new_gc = {k: v for k, v in gc.items()
                  if k not in ('label_labels', 'column_labels')}
        
        # 1) Row label remap — rewrite values into a derived column so the
        #    existing renderer naturally picks them up.
        col_map = _merge_label_map(row_labels, gc.get('column_labels'))
        src_col = gc.get('column')

        if col_map and src_col and src_col in df.columns:
            derived = f'__lbl__{src_col}__{i}'
            df[derived] = _remap_series(df[src_col], col_map)
            new_gc['column'] = derived
            if 'order' in gc:
                new_gc['order'] = _remap_order(gc['order'], col_map)
            # NB: ``filter`` references its own column(s), unrelated to
            # gc['column'], so nothing to do for filters.
 
        new_configs.append(new_gc)
 
    return plot_grouped_metrics(df, new_configs, **plot_kwargs)

# ── Internal helpers ─────────────────────────────────────────────────────────
def _merge_label_map(
    global_map: Optional[Mapping[str, str]],
    local_map: Optional[Mapping[str, str]],
) -> dict:
    """Merge a global mapping with a per-config one; local wins."""
    return {**(global_map or {}), **(local_map or {})}
 
 
def _remap_series(s: pd.Series, mapping: Mapping[str, str]) -> pd.Series:
    """Apply a {orig: display} mapping to a Series, preserving NaN and
    leaving unmapped values unchanged."""
    return s.map(lambda v: mapping.get(v, v) if pd.notna(v) else v)
 
 
def _remap_order(
    order: Optional[Sequence], mapping: Mapping[str, str]
) -> Optional[list]:
    """Apply mapping to an order list. Accepts either original or already-
    mapped values, so users can specify ``order`` in whichever form they
    have on hand."""
    if order is None:
        return None
    return [mapping.get(v, v) for v in order]