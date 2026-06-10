"""Plotting for sensitivity analysis. Persona vs context is the visual anchor."""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from libs.sensitivity.spec import ModelSpec
from libs.visuals.constants import PROMPT_VAR_COLORS as PALETTE
from libs.visuals.vis import FIG_DPI

def create_grid(fe_models, metrics=None, figsize_per_panel=(5, 6), nrows=1, sharey=True):
    if metrics is None:
        metrics = list(fe_models.keys())

    ncols = int(np.ceil(len(metrics) / nrows))
    fig, axes = plt.subplots(nrows, int(ncols), figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows), sharey=sharey)

    grid = {}
    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            if idx < len(metrics):
                ax = axes[row, col] if nrows > 1 else axes[col]
                grid[metrics[idx]] = {'ax': ax, 'irow': row, 'icol': col, 'index': idx, 'ncols':ncols, 'nrows':nrows}

    return fig, grid

# ---------- helpers ----------

def _kind(term: str, spec) -> str:
    base = term.split(':')[0]
    # Strip C(...)[T...] decoration
    base = re.sub(r'^C\(([^)]+)\).*', r'\1', base)
    base = re.sub(r'\[T\..+\]$', '', base)
    if base in spec.persona_vars: return 'persona'
    if base in spec.context_vars: return 'context'
    return 'other'


def _coef_terms_for(prompt_var: str, fe_params_index) -> list[str]:
    """Return all fe_param terms that belong to `prompt_var`.

    Handles three cases:
      - numeric:        'prompt_var'
      - categorical:    'prompt_var[T.level]'
      - patsy-wrapped:  'C(prompt_var)[T.level]'
    """
    pattern = re.compile(
        rf'^(?:C\()?{re.escape(prompt_var)}(?:\))?(\[T\..+\])?$'
    )
    return [t for t in fe_params_index if pattern.match(t)]


def _interaction_term(main_term: str, moderator: str, mod_level: str,
                      fe_params_index) -> str | None:
    """Find the interaction term that combines `main_term` with mod level.
    Patsy may order interactions either way, so check both."""
    mod_part = f'C({moderator})[T.{mod_level}]'
    for candidate in (f'{main_term}:{mod_part}', f'{mod_part}:{main_term}'):
        if candidate in fe_params_index:
            return candidate
    return None


def _contrast_sum(model, terms: list[str]) -> tuple[float, float, float]:
    """Compute estimate, SE, p-value for the sum of `terms`.
    Works for both OLS and MixedLM via params_table()."""
    pt = model.params_table()
    params = pt['params']
    idx = params.index.tolist()
    L = np.zeros((1, len(idx)))
    for term in terms:
        if term not in idx:
            return (np.nan, np.nan, np.nan)
        L[0, idx.index(term)] = 1.0
    tt = model.result.t_test(L)   # t_test exists on both OLSResults and MixedLMResults
    return float(tt.effect[0]), float(tt.sd[0]), float(tt.pvalue)


def marginal_effects(model, spec, prompt_var: str, moderator: str) -> pd.DataFrame:
    pt = model.params_table()
    params, ses, ci = pt['params'], pt['se'], pt['conf_int']
    pvals = pt['pvalues']
    idx = params.index

    main_terms = _coef_terms_for(prompt_var, idx)
    if not main_terms:
        return pd.DataFrame()

    mod_levels = [t.split(f'C({moderator})[T.')[1].rstrip(']')
                  for t in idx
                  if t.startswith(f'C({moderator})[T.') and ':' not in t]

    rows = []
    for main_term in main_terms:
        prompt_label = re.sub(r'^C\(|\)(?=\[)', '', main_term)

        rows.append({
            'prompt_var': prompt_var, 'prompt_level': prompt_label,
            'mod_level': 'reference',
            'estimate': params[main_term], 'se': ses[main_term],
            'ci_low':   ci.loc[main_term, 0], 'ci_high': ci.loc[main_term, 1],
            'pval':     pvals[main_term],
        })

        for lvl in mod_levels:
            inter = _interaction_term(main_term, moderator, lvl, idx)
            if inter is None:
                continue
            est, se, pval = _contrast_sum(model, [main_term, inter])
            if np.isnan(est):
                continue
            rows.append({
                'prompt_var': prompt_var, 'prompt_level': prompt_label,
                'mod_level': lvl,
                'estimate': est, 'se': se,
                'ci_low':   est - 1.96 * se, 'ci_high': est + 1.96 * se,
                'pval':     pval,
            })
    return pd.DataFrame(rows)

def _reference_level(prompt_var: str, df: pd.DataFrame) -> str | None:
    """Look up the reference level Patsy used for a categorical variable.

    Handles pd.Categorical (uses category order), object/string columns
    (Patsy uses alphabetical order), and boolean columns (False is ref).
    Returns None for numeric variables.
    """
    if df is None or prompt_var not in df.columns:
        return None
    col = df[prompt_var]
    if hasattr(col, 'cat'):
        return str(col.cat.categories[0])
    if col.dtype == bool:
        return 'False'
    if col.dtype == object or pd.api.types.is_string_dtype(col):
        vals = sorted(col.dropna().unique())
        return str(vals[0]) if vals else None
    return None


def _strip_patsy(term: str, prompt_var: str) -> str:
    """Turn 'role_en[T.phd student]' into 'phd student'.
       Turn 'C(role_en)[T.phd student]' into 'phd student'.
       Turn 'popularity_citations' (numeric) into 'popularity_citations'."""
    m = re.match(rf'^(?:C\()?{re.escape(prompt_var)}(?:\))?\[T\.(.+)\]$', term)
    return m.group(1) if m else term


def _level_sort_key(term: str, prompt_var: str, df: pd.DataFrame | None):
    """Stable, metric-INDEPENDENT ordering key for a level term.

    The y-axis order must be identical across every panel of a shared-y grid,
    so it must NOT depend on the fitted coefficient (which differs per metric).
    Order preference:
      1. categorical -> follow the dataframe's declared category order
      2. numeric-looking levels (e.g. '5.0', '10.0') -> by numeric value
      3. everything else -> alphabetical
    """
    level = _strip_patsy(term, prompt_var)
    if df is not None and prompt_var in df.columns and hasattr(df[prompt_var], 'cat'):
        cats = list(map(str, df[prompt_var].cat.categories))
        if level in cats:
            return (0, cats.index(level), '')
    try:
        return (1, float(level), '')
    except (TypeError, ValueError):
        return (2, 0.0, level)


def _pretty(var_name: str) -> str:
    """role_en -> Role, popularity_citations -> Popularity citations.
    Customize this dict for paper-ready labels."""
    overrides = {
        'role_en':              'Role',
        'language_en':          'Language',
        'location_en':          'Location',
        'popularity_citations': 'Popularity (citations)',
        'parity_gender':        'Gender parity',
        'div_gender':           'Gender diversity',
        'validity':             'Validity',
        'consistency':          'Consistency',
        'factuality_author':    'Factuality (author)',
        'factuality_field':     'Factuality (field)',
    }
    if var_name in overrides:
        return overrides[var_name]
    return var_name.replace('_', ' ').capitalize()


def _add_row_banding(ax, pdf, alpha: float = 0.07,
                     restart_at_header: bool = True):
    """Draw alternating colored bands behind level rows for readability.

    Bands are tinted by the row's kind (persona = orange, context = blue),
    very transparent so they don't compete with the dots.

    Args:
        ax: Axes to draw on.
        pdf: DataFrame with 'is_header' and 'kind' columns, indexed 0..N-1.
        alpha: opacity of the bands. 0.05–0.12 works well.
        restart_at_header: if True, restart the band/no-band alternation
                           at each group header so the first level under
                           each header is always banded.
    """
    band_idx = 0
    for i, row in pdf.iterrows():
        if row['is_header']:
            if restart_at_header:
                band_idx = 0
            continue
        if band_idx % 2 == 0:
            ax.axhspan(
                i - 0.5, i + 0.5,
                color=PALETTE[row['kind']],
                alpha=alpha,
                zorder=0,
                linewidth=0,
            )
        band_idx += 1

# ---------- plot 1: Model 1 ----------

def plot_main_effects(model, spec, df=None, ax=None,
                       title: str = 'Average effect across LLMs',
                       group_order: list[str] | None = None,
                       name_map: dict[str, str] | None = None,
                       figsize: tuple[float, float] | None = None,
                       banding: bool = True,       
                       band_alpha: float = 0.07,
                       r2_kwargs: dict = {},
                       ):
    """Forest plot of main effects with grouped y-axis labels.

    Args:
        model: a fitted SensitivityModel (FixedEffectsModel or MixedEffectsModel).
        spec:  the ModelSpec.
        df:    original dataframe — needed to look up reference levels.
        ax:    matplotlib Axes. If None, a new figure is created.
        title: plot title.
        group_order: explicit order of prompt variables. Defaults to
                     persona first, context second, in spec-declared order.
        name_map: optional dict for relabelling variables AND levels in the
                  display. Applies to both group headers and level rows.
                  Example: {'role_en': 'Role', 'phd student': 'PhD student',
                            'language_en': 'Language', 'k': 'Number of items'}
        figsize: (width, height) in inches. Defaults to auto-scaled by row count.
    """
    name_map = name_map or {}

    pt = model.params
    params, ci = pt['params'], pt['conf_int']
    idx = params.index

    if group_order is None:
        group_order = list(spec.persona_vars) + list(spec.context_vars)

    def relabel(s: str) -> str:
        """Apply name_map override; fall back to prettified original."""
        return name_map.get(s, _pretty(s))

    # ---- Build plot rows ----
    plot_rows = []
    for pv in group_order:
        terms = _coef_terms_for(pv, idx)
        if not terms:
            continue
        kind = _kind(pv, spec)
        ref_level = _reference_level(pv, df) if df is not None else None

        # Header
        header = relabel(pv)
        if ref_level is not None:
            header = f'{header} [ref: {relabel(ref_level)}]'
        plot_rows.append({
            'is_header': True, 'label': header, 'kind': kind,
            'estimate': None, 'ci_low': None, 'ci_high': None,
        })

        # Level rows.
        # NOTE: order by the variable's natural level order (metric-INDEPENDENT),
        # never by params[t] (metric-dependent) -- otherwise a shared-y grid
        # shows mismatched labels vs dots across panels.
        for term in sorted(terms, key=lambda t: _level_sort_key(t, pv, df)):
            raw_level = _strip_patsy(term, pv)
            level_label = '    ' + relabel(raw_level)
            plot_rows.append({
                'is_header': False, 'label': level_label, 'kind': kind,
                'estimate': params[term],
                'ci_low':   ci.loc[term, 0],
                'ci_high':  ci.loc[term, 1],
            })

    pdf = pd.DataFrame(plot_rows)
    if pdf.empty:
        raise ValueError("No prompt-variable terms found in params.")

    # ---- Figure ----
    if ax is None:
        if figsize is None:
            figsize = (7.5, max(3, 0.32 * len(pdf)))
        _, ax = plt.subplots(figsize=figsize)

    # ---- Row banding for readability ----
    if banding:
        _add_row_banding(ax, pdf, alpha=band_alpha)
    
    # ---- Plot points and CIs ---
    for i, row in pdf.iterrows():
        if row['is_header']:
            continue
        c = PALETTE[row['kind']]
        ax.errorbar(row['estimate'], i,
                    xerr=[[row['estimate'] - row['ci_low']],
                          [row['ci_high'] - row['estimate']]],
                    fmt='o', color=c, ecolor=c, capsize=3, markersize=6)

    ax.axvline(0, color='gray', ls='--', lw=0.5)

    # Annotate
    r2 = model.r2() if hasattr(model, 'r2') else None
    llm_share = model.llm_share if hasattr(model, 'llm_share') else None
    s = ''
    if r2 is not None:
        s1 = "R$^2$ = {:.2f}".format(r2['r2']) 
        s2 = "R$^2 adj.$ = {:.2f}".format(r2['r2_adj'])
        s = s1 + '\n' + s2
    if llm_share is not None:
        s3 = "LLM id.: {:.0%} of R$^2$".format(llm_share)
        s = s + ('\n' + s3 if s3 else '')
    if s and s!='':
        ax.text(s=s, 
                x=r2_kwargs.get('x', 0.98), 
                y=r2_kwargs.get('y', 0.02), 
                transform=ax.transAxes, 
                ha=r2_kwargs.get('ha', 'right'), 
                va=r2_kwargs.get('va', 'bottom'),
                fontsize=r2_kwargs.get('fontsize', 11)
                )

    # ---- Y-axis ----
    ax.set_yticks(range(len(pdf)))
    ax.set_yticklabels(pdf['label'], fontsize=10)
    ax.set_ylim(len(pdf) - 0.5, -0.5)   # top-to-bottom: row 0 at top, last row at bottom

    # Color every label (header AND levels) by its persona/context kind
    for tick_label, (_, row) in zip(ax.get_yticklabels(), pdf.iterrows()):
        tick_label.set_color(PALETTE[row['kind']])
        if row['is_header']:
            tick_label.set_fontweight('bold')

    ax.margins(x=0.05)
    ax.set_xlabel('Effect on metric')
    ax.set_title(title)
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color=PALETTE['persona'], lw=0, label='Persona'),
        Line2D([0], [0], marker='o', color=PALETTE['context'], lw=0, label='Context'),
    ], frameon=True, loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


# ---------- plot 2: Model 2 heterogeneity grid ----------

def plot_heterogeneity_grid(model, spec, moderators=None,
                            figsize_per_panel=(5.5, None)):
    """One panel per moderator. Y-axis: each prompt_level (categorical dummy
    or numeric var). X-axis: marginal effect at each moderator level."""
    moderators = moderators or spec.llm_attrs
    n = len(moderators)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
                              sharey=False)
    if n == 1:
        axes = [axes]

    for ax, mod in zip(axes, moderators):
        # Collect marginal effects across all prompt vars
        me = pd.concat([marginal_effects(model, spec, pv, mod)
                        for pv in spec.prompt_vars], ignore_index=True)
        if me.empty:
            continue

        me['kind'] = me['prompt_var'].apply(
            lambda v: 'persona' if v in spec.persona_vars else 'context'
        )

        # Order y-axis: persona first, then context, sorted within each
        me_sorted = me.sort_values(['kind', 'prompt_var', 'prompt_level'])
        y_labels = me_sorted['prompt_level'].unique().tolist()
        y_map = {pl: i for i, pl in enumerate(y_labels)}

        mod_levels = ['reference'] + sorted(
            [l for l in me['mod_level'].unique() if l != 'reference']
        )
        markers = ['o', 's', '^', 'D', 'v', 'P'][:len(mod_levels)]
        jitter = np.linspace(-0.25, 0.25, len(mod_levels))

        for marker, lvl, dy in zip(markers, mod_levels, jitter):
            sub = me[me['mod_level'] == lvl]
            for _, row in sub.iterrows():
                c = PALETTE[row['kind']]
                y = y_map[row['prompt_level']] + dy
                ax.errorbar(row['estimate'], y,
                            xerr=[[row['estimate'] - row['ci_low']],
                                  [row['ci_high'] - row['estimate']]],
                            fmt=marker, color=c, ecolor=c,
                            capsize=2, markersize=5, alpha=0.85)

        ax.axvline(0, color='gray', ls='--', lw=0.5)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel('Effect on metric')
        ax.set_title(f'Moderated by {mod}')

        handles = [Line2D([0], [0], marker=m, color='gray', lw=0, label=l)
                   for m, l in zip(markers, mod_levels)]
        ax.legend(handles=handles, frameon=False, fontsize=7, loc='best')

    fig.suptitle('Heterogeneity of prompt effects across LLM attributes', y=1.01)
    fig.tight_layout()
    return fig


# ---------- plot 3: robustness — m2 vs m2_by_attr ----------

def plot_robustness_comparison(m2, m2_by_attr: dict, spec: ModelSpec):
    """Scatter m2 interaction coefficients against m2_by_attr equivalents.
    Points near the y=x line → models agree → robust to multicollinearity."""
    c2 = m2.coefs()
    rows = []
    for attr, model in m2_by_attr.items():
        cs = model.coefs()
        # Match interaction terms across the two
        for term in cs['term']:
            if attr in term and ':' in term:
                if term in c2['term'].values:
                    full_est = c2.loc[c2['term'] == term, 'estimate'].iloc[0]
                    marg_est = cs.loc[cs['term'] == term, 'estimate'].iloc[0]
                    pv = term.split(':')[0]
                    rows.append({
                        'term': term, 'attr': attr, 'kind': _kind(pv, spec),
                        'full_model': full_est, 'marginal_model': marg_est,
                    })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 6))
    for kind, color in PALETTE.items():
        sub = df[df['kind'] == kind]
        ax.scatter(sub['marginal_model'], sub['full_model'],
                   color=color, label=kind.capitalize(), alpha=0.7, s=40)

    lim = max(abs(df[['full_model', 'marginal_model']].values).max() * 1.1, 0.01)
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.5)
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_xlabel('Interaction coefficient (single-moderator model)')
    ax.set_ylabel('Interaction coefficient (full m2)')
    ax.set_title('Robustness: do single- and full-moderator models agree?')
    ax.legend(frameon=False)
    return fig

def get_star_suffix(p: float) -> str:
        """Stars by descending strictness: ***  **  *."""
        t = sorted((0.05, 0.01, 0.001))  # ascending [strict, ..., lax]
        if p < t[0]:
            return '***'
        if p < t[1]:
            return '**'
        if p < t[2]:
            return '*'
        return ''

def plot_residuals_diagnostic(analysis, metrics_name_map, fn_summary_table, fn_plot):
    import statsmodels.stats.api as sms
    import statsmodels.api as sm
    import scipy.stats as stats

    nmetrics = len(analysis.results.items())
    ncols = 6
    nrows = (nmetrics + ncols - 1) // ncols
    width = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*width, nrows*width), sharex=True, sharey=True)

    df_summary_table = pd.DataFrame(columns=['metric', 'shapiro_stat', 'shapiro_p', 'bp_stat', 'bp_p', 'outside_0_1_pct'])

    for index, (metric, obj) in enumerate(analysis.results.items()):
        nrow = index // ncols
        ncol = index % ncols
        ax = axes[nrow][ncol]

        model = analysis.results[metric]['m1'].result

        resid = model.resid
        sm.qqplot(resid, line='45', fit=True, ax=ax)
        stat, p = stats.shapiro(resid)   # use the plot as the primary evidence at large N
        ax.text(0.02, 0.98, f'Shapiro-Wilk: {stat:.3f} {get_star_suffix(p)}', transform=ax.transAxes, ha='left', va='top')
        
        bp = sms.het_breuschpagan(model.resid, model.model.exog)
        ax.text(0.02, 0.90, f'Breusch-Pagan: {bp[0]:.3f} {get_star_suffix(bp[1])}', transform=ax.transAxes, ha='left', va='top')
        ax.text(0.98, 0.02, metrics_name_map[metric], transform=ax.transAxes, ha='right', va='bottom', fontsize='large',)
        ax.set_xlabel('' if nrow < nrows - 1 else 'Theoretical quantiles')
        ax.set_ylabel('Sample quantiles' if ncol == 0 else '')

        obj = {
            'metric': metric,
            'shapiro_stat': stat,
            'shapiro_p': p,
            'bp_stat': bp[0],
            'bp_p': bp[1],
            'outside_0_1_pct': obj['m1'].model_info['outside_0_1_pct']}
        df_summary_table = pd.concat([df_summary_table, pd.DataFrame([obj])], ignore_index=True)
        df_summary_table.to_csv(fn_summary_table, index=False)

    if nmetrics < ncols * nrows:
        for index in range(nmetrics, ncols * nrows):
            nrow = index // ncols
            ncol = index % ncols
            fig.delaxes(axes[nrow][ncol])
            axes[nrow-1][ncol].set_xlabel('Theoretical quantiles')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(fn_plot, dpi=FIG_DPI)
    plt.show()
    plt.close()

    return df_summary_table

