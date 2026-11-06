"""
Flow figures (appendix): who is recommended, by prompt variable or model characteristic (data: metrics/flows.py).

    plot_flow(ax, table, ...)     one alluvial panel: source values (left) -> classes of a person attribute (right);
                                  every source value has the same height (its people's composition), so the bands are
                                  within-source shares and the right nodes the mean composition over the sources
    plot_flow_grid(people, source=... | target=...)
                                  one prompt variable vs several person attributes (source fixed: the source values
                                  are written once per row), or one person attribute vs several prompt variables /
                                  model characteristics (target fixed: the classes written once per row); ncols sets
                                  the layout (e.g. 3 -> 2 x 3 for six panels, 6 -> 1 x 6)
    plot_flows(people, pairs)     a figure of any (source, target) panels
    plot_drift_flow(people)       the drift as a flow: location -> country, own-country bands in colour, others faded
    plot_drift(people, drift)     prompt location -> the people's country: (a) heatmap of the country shares per
                                  prompted location, the prompt's own country outlined; (b) share from the prompt's
                                  country vs the baseline (other locations prompted), with CIs over models
Colours: gender diverging (blue - grey - red, in the label order of the config), ethnicity / country categorical in a
fixed, colour-vision-checked order ('other' grey), prominence classes sequential; source nodes in the prompt family
colour (persona orange, context blue) or grey (model characteristics). 'unknown' is left out unless included.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import Rectangle

from metrics import flows as fl
from metrics import summary
from utils import constants as cons
from viz import constants as viscons
from viz.metric_plots import text_size

TARGET_TITLES = {'gender': 'perceived gender', 'ethnicity': 'perceived ethnicity', 'country': 'country (latest affiliation)',
                 fl.AUTHOR_SENIORITY: 'seniority', 'productivity': 'productivity', 'influence': 'influence',
                 'collaboration': 'collaboration'}
SENIORITY_ORDER = [cons.SENIORITY_JUNIOR, 'mid', cons.SENIORITY_SENIOR]
MODEL_SOURCES = list(summary.MODEL_ATTRIBUTES) + ['family']


# --- orders, labels, colours -------------------------------------------------------------------------------------
def source_title(source: str) -> str:
    if source in summary.MODEL_ATTRIBUTES:
        return summary.MODEL_ATTRIBUTES[source][0]
    return {'family': 'Model family', cons.COL_MODEL: 'Model'}.get(source, summary.PROMPT_TITLES.get(source, source))


def source_family(source: str) -> str:
    if source in MODEL_SOURCES or source == cons.COL_MODEL:
        return 'model'
    return summary.PERSONA if source in cons.PERSONA_VARIABLES else summary.CONTEXT


def source_order(source: str, values: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(values, labels) in the design order (prompt variables) or the order of the model characteristics."""
    present = set(map(str, values))
    if source in summary.MODEL_ATTRIBUTES:
        _, levels, labels = summary.MODEL_ATTRIBUTES[source]
        pairs = [(str(l), lab) for l, lab in zip(levels, labels) if str(l) in present]
        rest = sorted(present - {p for p, _ in pairs})
        return [p for p, _ in pairs] + rest, [lab for _, lab in pairs] + rest
    from analysis.deviations import design_order
    order = design_order(source, present)
    return order, [summary.PROMPT_LABELS.get(v, v.capitalize() if v.islower() else v) for v in order]


def _ramp(colors: Sequence[str], n: int) -> List[str]:
    cmap = LinearSegmentedColormap.from_list('ramp', list(colors))
    return [to_hex(cmap(x)) for x in (np.linspace(0, 1, n) if n > 1 else [0.5])]


def target_order_colors(target: str, classes: Sequence[str], gender_labels: Optional[Sequence[str]] = None,
                        bin_labels: Optional[Sequence[str]] = None) -> Tuple[List[str], Dict[str, str]]:
    """Order and colour of the classes of a person attribute ('other' and 'unknown' last, in grey)."""
    present = [c for c in dict.fromkeys(map(str, classes))]
    special = [c for c in (fl.OTHER, fl.UNKNOWN) if c in present]
    rest = [c for c in present if c not in special]
    if target == 'gender':
        scheme = list(gender_labels or cons.DEFAULT_GENDER_LABELS)
        order = [c for c in scheme if c in rest] + sorted(set(rest) - set(scheme))
        full = _ramp(viscons.FLOW_DIVERGING, len(scheme))
        colors = {c: full[scheme.index(c)] if c in scheme else viscons.FLOW_OTHER_COLOR for c in order}
    elif target in ('productivity', 'influence', 'collaboration', fl.AUTHOR_SENIORITY):
        scheme = list(SENIORITY_ORDER if target == fl.AUTHOR_SENIORITY else bin_labels or cons.DEFAULT_GT_BIN_LABELS)
        order = [c for c in scheme if c in rest] + sorted(set(rest) - set(scheme))
        full = _ramp(viscons.FLOW_SEQUENTIAL, len(scheme))
        colors = {c: full[scheme.index(c)] if c in scheme else viscons.FLOW_OTHER_COLOR for c in order}
    else:
        if target == 'ethnicity':
            order = [c for c in fl.ETHNICITY_ORDER if c in rest] + sorted(set(rest) - set(fl.ETHNICITY_ORDER))
        else:   # countries: the prompt countries first (design order: the same colour in every figure), then by share
            prompt = prompt_countries()
            order = [c for c in prompt if c in rest] + [c for c in rest if c not in prompt]
        palette = viscons.FLOW_CATEGORICAL
        # consecutive slots of the validated order (neighbours stay distinguishable); the prompt countries are kept in
        # every country table (prompt_countries()), so they take the first slots: the same colour in every figure
        colors = {c: palette[i] if i < len(palette) else viscons.FLOW_OTHER_COLOR for i, c in enumerate(order)}
    colors.update({fl.OTHER: viscons.FLOW_OTHER_COLOR, fl.UNKNOWN: viscons.FLOW_UNKNOWN_COLOR})
    return order + special, colors


def prompt_countries() -> List[str]:
    """Country codes of the prompt locations, in the design order (kept out of 'Other' in every country table)."""
    return [fl.LOCATION_COUNTRY[l] for l in map(str, cons.VARIABLE_VALUES[cons.VAR_LOCATION]) if l in fl.LOCATION_COUNTRY]


def country_table(people: pd.DataFrame, source: str, target: str, include_unknown: bool) -> pd.DataFrame:
    """flow_table with the top countries pooled (prompt countries always kept); other targets unchanged."""
    if target != 'country':
        return fl.flow_table(people, source, target, include_unknown)
    return fl.flow_table(people, source, target, include_unknown, viscons.FLOW_TOP_COUNTRIES, keep=prompt_countries())


def class_label(target: str, value: str) -> str:
    if value in (fl.OTHER, fl.UNKNOWN):
        return value.capitalize()
    return value if target in ('country', 'productivity', 'influence', 'collaboration') else value.capitalize()


# --- one panel ---------------------------------------------------------------------------------------------------
def _smooth(t: np.ndarray) -> np.ndarray:
    return 3 * t ** 2 - 2 * t ** 3


def plot_flow(ax: plt.Axes, table: pd.DataFrame, source: str, target: str, gender_labels=None, bin_labels=None,
              label_width: Tuple[float, float] = (0.0, 0.0), left_labels: bool = True, right_labels: bool = True,
              n_source: Optional[pd.Series] = None, title: Optional[str] = None, note: Optional[str] = None,
              target_order: Optional[Sequence[str]] = None, highlight: Optional[Dict[str, str]] = None) -> None:
    """One alluvial panel from flow_table() output (see module docstring). label_width: (left, right) label room as a
    share of the flow width (x data units); left_labels / right_labels: write the source values / target classes
    (off for panels sharing them with a neighbour); n_source: the n written under each source value (default: the
    mentions with a known class of this target); note: a line under the title; target_order: the classes' order
    (default by share); highlight: source value -> class whose band is emphasised (the others faded)."""
    sources, source_labels = source_order(source, table['source'].unique())
    by_share = list(target_order) if target_order else table.groupby('target')['n'].sum().sort_values(ascending=False).index
    targets, colors = target_order_colors(target, by_share, gender_labels, bin_labels)
    if highlight is not None:   # only the highlighted classes in colour
        colors = {c: (v if c in set(highlight.values()) else viscons.FLOW_OTHER_COLOR) for c, v in colors.items()}
    share = table.pivot_table(index='source', columns='target', values='share', aggfunc='sum').reindex(
        index=sources, columns=targets).fillna(0.0)
    n_source = (table.groupby('source')['n_source'].first() if n_source is None else n_source).reindex(sources)
    gap = 0.08
    w = viscons.FLOW_NODE_WIDTH
    node_color = viscons.METRIC_BAR_COLORS.get(source_family(source), viscons.METRIC_BAR_COLORS['model'])

    # left nodes: equal heights
    left_top = {s: i * (1 + gap) for i, s in enumerate(sources)}
    total_left = len(sources) * (1 + gap) - gap
    # right nodes: mean composition, scaled to the same total height
    right_size = share.sum(axis=0)
    scale = (total_left - gap * (len(targets) - 1)) / right_size.sum() if right_size.sum() else 1.0
    right_top, y = {}, 0.0
    for t in targets:
        right_top[t] = y
        y += right_size[t] * scale + gap
    x = np.linspace(w, 1 - w, 60)
    s_curve = _smooth((x - w) / (1 - 2 * w))
    right_fill = {t: right_top[t] for t in targets}
    for s in sources:
        y_left = left_top[s]
        for t in targets:
            h = share.at[s, t]
            if h <= 0:
                continue
            h_right = h * scale
            top = y_left + (right_fill[t] - y_left) * s_curve
            bottom = top + h + (h_right - h) * s_curve
            emphasis = highlight is None or highlight.get(s) == t
            ax.fill_between(x, top, bottom, color=colors[t] if emphasis else viscons.FLOW_UNKNOWN_COLOR,
                            alpha=viscons.FLOW_BAND_ALPHA if highlight is None else (0.85 if emphasis else 0.35),
                            linewidth=0)
            if (h >= viscons.FLOW_BAND_LABEL_MIN and highlight is None) or (highlight is not None and emphasis and h > 0):
                ax.text(w + 0.012, y_left + h / 2, f"{h:.0%}", ha='left', va='center', fontsize='xx-small',
                        color=viscons.METRIC_ROW_LABEL_COLOR)
            y_left += h
            right_fill[t] += h_right
    for s, label in zip(sources, source_labels):
        ax.add_patch(Rectangle((0, left_top[s]), w, 1, color=node_color, linewidth=0))
        if not left_labels:
            continue
        ax.text(-0.015, left_top[s] + 0.4, label, ha='right', va='center', fontsize='x-small',
                color=viscons.METRIC_ROW_LABEL_COLOR)
        ax.text(-0.015, left_top[s] + 0.75, f"n = {int(n_source[s]):,}", ha='right', va='center', fontsize='xx-small',
                color=viscons.COLOR_TEXT_SECONDARY)
    mean_share = right_size / right_size.sum() if right_size.sum() else right_size
    for t in targets:
        h = right_size[t] * scale
        ax.add_patch(Rectangle((1 - w, right_top[t]), w, h, color=colors[t], linewidth=0))
        if not right_labels:
            continue
        ax.text(1.015, right_top[t] + h / 2, f"{class_label(target, t)}  {mean_share[t]:.0%}", ha='left', va='center',
                fontsize='x-small', color=viscons.METRIC_ROW_LABEL_COLOR)
    ax.set_xlim(-label_width[0], 1 + label_width[1])
    ax.set_ylim(max(total_left, y - gap), -0.05)
    ax.axis('off')
    ax.set_title(title or f"{source_title(source)} → {TARGET_TITLES.get(target, target)}", fontsize='small',
                 fontweight='bold')
    if note:
        ax.text(0.5, -0.04, note, ha='center', va='bottom', fontsize='xx-small', color=viscons.COLOR_TEXT_SECONDARY,
                transform=ax.get_xaxis_transform())


# --- figures -----------------------------------------------------------------------------------------------------
def unknown_note(people: pd.DataFrame, target: str, include_unknown: bool) -> Optional[str]:
    """'unknown x% left out' (n under the source values counts every mention, the bands only the known classes)."""
    share = fl.unknown_share(people, target)
    return None if include_unknown or not share else f"unknown {share:.0%} left out"


def plot_drift_flow(people: pd.DataFrame, top: int = viscons.FLOW_TOP_COUNTRIES, include_unknown: bool = False,
                    width: float = viscons.PAPER_TEXT_WIDTH_IN * 0.6, title: Optional[str] = None) -> plt.Figure:
    """Drift as a flow: prompted location -> the people's country; each location's band to its own country in
    colour, the others faded; the prompt countries always shown (first), then the top countries, then Other."""
    locations, labels = source_order(cons.VAR_LOCATION, people[cons.VAR_LOCATION].astype(str).unique())
    codes = fl.location_codes(locations)
    table = fl.flow_table(people, cons.VAR_LOCATION, 'country', include_unknown, top, keep=list(codes.values()))
    others = [c for c in table.groupby('target')['n'].sum().sort_values(ascending=False).index
              if c not in codes.values() and c not in (fl.OTHER, fl.UNKNOWN)]
    present = set(table['target'])
    order = [codes[l] for l in locations if codes[l] in present] + others   # a country without mentions: not shown
    n_rows = max(len(locations), table['target'].nunique())
    fig = plt.figure(figsize=(width, n_rows * viscons.FLOW_ROW_IN * 1.6 + 0.6))
    left_in = max(text_size(l, 'x-small')[0] for l in labels) + 0.1
    right_in = max(text_size(f"{c}  100%", 'x-small')[0] for c in table['target'].unique()) + 0.1
    flow_in = max(width - left_in - right_in - 0.2, 0.5)
    ax = fig.add_axes([0.1 / width, 0.06, (width - 0.2) / width, 0.84])
    plot_flow(ax, table, cons.VAR_LOCATION, 'country', label_width=(left_in / flow_in, right_in / flow_in),
              n_source=people[cons.VAR_LOCATION].astype(str).value_counts(), target_order=order,
              highlight={l: codes[l] for l in locations}, note=unknown_note(people, 'country', include_unknown),
              title=title or 'Prompted location → country of the recommended people')
    return fig



def with_model_columns(people: pd.DataFrame, metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The people table with the model characteristics (access, size, reasoning, origin, family) as columns."""
    metadata = summary.load_model_metadata() if metadata is None else metadata
    columns = [c for c in metadata.columns if c in MODEL_SOURCES and c not in people]
    return people.merge(metadata[[cons.COL_MODEL] + columns], on=cons.COL_MODEL, how='left')


def plot_flows(people: pd.DataFrame, pairs: Sequence[Tuple[str, str]], include_unknown: bool = False,
               gender_labels=None, bin_labels=None, ncols: Optional[int] = None,
               width: float = viscons.PAPER_TEXT_WIDTH_IN, title: Optional[str] = None) -> plt.Figure:
    """A row (or grid) of flow panels, one per (source, target)."""
    ncols = ncols or min(len(pairs), 3)
    nrows = int(np.ceil(len(pairs) / ncols))
    tables = [country_table(people, s, t, include_unknown)
              for s, t in pairs]
    n_nodes = [max(t['source'].nunique(), t['target'].nunique()) for t in tables]
    rows_in = [max(n_nodes[r * ncols:(r + 1) * ncols]) * viscons.FLOW_ROW_IN * 1.6 + 0.45 for r in range(nrows)]
    fig = plt.figure(figsize=(width, sum(rows_in)))
    panel_w = width / ncols
    top = 1.0
    for r in range(nrows):
        h = rows_in[r] / sum(rows_in)
        for c in range(ncols):
            i = r * ncols + c
            if i >= len(pairs):
                break
            (source, target), table = pairs[i], tables[i]
            _, labels = source_order(source, table['source'].unique())
            left_in = max(text_size(l, 'x-small')[0] for l in labels) + 0.08
            right_in = max(text_size(f"{class_label(target, t)}  100%", 'x-small')[0] for t in table['target'].unique()) + 0.08
            flow_in = max(panel_w - left_in - right_in - 0.1, 0.4)
            ax = fig.add_axes([(c * panel_w + 0.05) / width, top - h + 0.06 * h, (panel_w - 0.1) / width, 0.8 * h])
            plot_flow(ax, table, source, target, gender_labels, bin_labels, (left_in / flow_in, right_in / flow_in),
                      n_source=people[source].astype(str).value_counts(),
                      note=unknown_note(people, target, include_unknown))
        top -= h
    if title:
        fig.suptitle(title, fontsize='small', y=1.0)
    return fig


def plot_drift(people: pd.DataFrame, drift: pd.DataFrame, top: int = viscons.FLOW_TOP_COUNTRIES,
               width: float = viscons.PAPER_TEXT_WIDTH_IN, title: Optional[str] = None) -> plt.Figure:
    """Location drift (see module docstring)."""
    known = people[people['country'].astype(str) != fl.UNKNOWN].copy()
    known[cons.VAR_LOCATION] = known[cons.VAR_LOCATION].astype(str)
    locations, _ = source_order(cons.VAR_LOCATION, known[cons.VAR_LOCATION].unique())
    codes = fl.location_codes(locations)
    others = [c for c in fl.top_classes(known['country'], top + len(codes)) if c not in codes.values()]
    columns = [codes[l] for l in locations] + others[:max(top - len(codes), 0)]
    known['col'] = known['country'].where(known['country'].isin(columns), fl.OTHER)
    columns += [fl.OTHER]
    shares = pd.crosstab(known[cons.VAR_LOCATION], known['col'], normalize='index').reindex(
        index=locations, columns=columns).fillna(0.0)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(width, 0.42 * len(locations) + 1.2), layout='constrained',
                                 gridspec_kw={'width_ratios': [len(columns), 7]})
    vmax = max(float(shares.to_numpy().max()), 0.05)
    cmap = LinearSegmentedColormap.from_list('share', ['#ffffff', viscons.METRIC_BAR_COLORS['persona'], '#8a3f0a'])
    ax.imshow(shares.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    for i, loc in enumerate(locations):
        for j, col in enumerate(columns):
            v = shares.iat[i, j]
            ax.text(j, i, f"{v:.0%}" if v >= 0.005 else '', ha='center', va='center', fontsize='xx-small',
                    color='white' if v > 0.6 * vmax else viscons.METRIC_ROW_LABEL_COLOR)
        j = columns.index(codes[loc])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=viscons.METRIC_ROW_LABEL_COLOR,
                               linewidth=1.2))
    ax.set_xticks(range(len(columns)), [c if c != fl.OTHER else 'Other' for c in columns], fontsize='x-small')
    ax.set_yticks(range(len(locations)), locations, fontsize='x-small')
    ax.set_xlabel("Country of the recommended people", fontsize='x-small')
    ax.set_ylabel("Prompted location", fontsize='x-small')
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    d = drift.set_index('location').reindex(locations)
    ys = np.arange(len(locations))
    bx.hlines(ys, d['baseline'], d['share'], color=viscons.COLOR_GRID, linewidth=2, zorder=1)
    bx.scatter(d['baseline'], ys, color=viscons.METRIC_BAR_COLORS['model'], s=22, zorder=2, label='other locations prompted')
    if 'share_ci_low' in d:
        bx.errorbar(d['share'], ys, xerr=[(d['share'] - d['share_ci_low']).clip(lower=0),
                                         (d['share_ci_high'] - d['share']).clip(lower=0)],
                    fmt='none', ecolor=viscons.METRIC_CI_COLOR, elinewidth=viscons.METRIC_CI_LINEWIDTH,
                    capsize=viscons.METRIC_CI_CAPSIZE, zorder=2)
    bx.scatter(d['share'], ys, color=viscons.METRIC_BAR_COLORS['persona'], s=26, zorder=3, label='this location prompted')
    for y, (share, lift) in zip(ys, d[['share', 'lift']].to_numpy()):
        if np.isfinite(lift):
            bx.text(share, y - 0.32, f"×{lift:.1f}", ha='center', va='bottom', fontsize='xx-small',
                    color=viscons.COLOR_TEXT_SECONDARY)
    bx.set_yticks(ys, [f"{l} ({codes[l]})" for l in locations], fontsize='x-small')
    bx.set_ylim(len(locations) - 0.5, -0.8)
    bx.set_xlim(0, max(0.05, float(np.nanmax(d[['share', 'baseline'] + (['share_ci_high'] if 'share_ci_high' in d else [])]
                                               .to_numpy())) * 1.15))
    bx.set_xlabel("Share from the prompt's country", fontsize='x-small')
    bx.tick_params(labelsize='x-small')
    bx.grid(axis='x', color=viscons.COLOR_GRID, linewidth=0.5)
    for side in ('top', 'right'):
        bx.spines[side].set_visible(False)
    bx.legend(fontsize='xx-small', frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
    if title:
        fig.suptitle(title, fontsize='small')
    return fig


def plot_flow_grid(people: pd.DataFrame, source: Optional[str] = None, target: Optional[str] = None,
                   targets: Sequence[str] = tuple(fl.TARGETS), sources: Sequence[str] = tuple(cons.CONFIG_VARIABLES),
                   ncols: int = viscons.FLOW_GRID_NCOLS, include_unknown: bool = False, gender_labels=None,
                   bin_labels=None, width: Optional[float] = None, title: Optional[str] = None) -> plt.Figure:
    """Source fixed (source=..., panels = targets) or target fixed (target=..., panels = sources); see the module
    docstring. width: default the paper width for <= 3 columns, wider for more."""
    if (source is None) == (target is None):
        raise ValueError("give exactly one of source= (vary the person attribute) or target= (vary the prompt variable)")
    pairs = [(source, t) for t in targets] if source else [(s, target) for s in sources]
    ncols = min(ncols, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    width = width or viscons.PAPER_TEXT_WIDTH_IN * max(1.0, ncols / 3)
    tables = [country_table(people, s, t, include_unknown)
              for s, t in pairs]
    mentions = {s: people[s].astype(str).value_counts() for s, _ in pairs}       # n = all mentions (also unknown)
    n_nodes = [max(t['source'].nunique(), t['target'].nunique()) for t in tables]
    rows_in = [max(n_nodes[r * ncols:(r + 1) * ncols]) * viscons.FLOW_ROW_IN * 1.6 + 0.45 for r in range(nrows)]
    fig = plt.figure(figsize=(width, sum(rows_in) + (0.25 if title else 0)))

    def left_room(i):
        s, t = pairs[i]
        _, labels = source_order(s, tables[i]['source'].unique())
        return max(text_size(l, 'x-small')[0] for l in labels) + 0.1

    def right_room(i):
        s, t = pairs[i]
        return max(text_size(f"{class_label(t, c)}  100%", 'x-small')[0] for c in tables[i]['target'].unique()) + 0.08

    top = 1.0 - (0.25 / (sum(rows_in) + 0.25) if title else 0)
    total_h = sum(rows_in) + (0.25 if title else 0)
    for r in range(nrows):
        idx = list(range(r * ncols, min((r + 1) * ncols, len(pairs))))
        # label room: source fixed -> left labels only in the first panel; target fixed -> right labels only in the last
        lefts = [left_room(i) if (target or i == idx[0]) else 0.05 for i in idx]
        rights = [right_room(i) if (source or i == idx[-1]) else 0.05 for i in idx]
        flow_in = max((width - sum(lefts) - sum(rights) - 0.1 * len(idx)) / len(idx), 0.35)
        x = 0.05
        h = rows_in[r] / total_h
        for k, i in enumerate(idx):
            s, t = pairs[i]
            panel_w = lefts[k] + flow_in + rights[k]
            ax = fig.add_axes([x / width, top - h + 0.06 * h, panel_w / width, 0.8 * h])
            panel_title = (TARGET_TITLES.get(t, t) if source else source_title(s)).capitalize()
            plot_flow(ax, tables[i], s, t, gender_labels, bin_labels, (lefts[k] / flow_in, rights[k] / flow_in),
                      left_labels=bool(target) or k == 0, right_labels=bool(source) or k == len(idx) - 1,
                      n_source=mentions[s], title=panel_title, note=unknown_note(people, t, include_unknown))
            x += panel_w + 0.1
        top -= h
    fig.suptitle(title or (f"{source_title(source)} (prompt) → recommended people" if source
                           else f"Prompt → {TARGET_TITLES.get(target, target)} of the recommended people"),
                 fontsize='small', y=0.995)
    return fig
