"""
File names of every figure (without extension): one convention for all families.

    <family>_<what>[_<selection>][_<kind>][_adj]

Parts are joined by '_'; words inside a part by '-' (an '_' inside a part becomes '-'), so the parts can always be
split back: rq2_emm_paper-technical_bar. The size-adjusted twin of a figure ends in '_adj' (it sorts next to it).
Empty parts are left out. kind is given where one 'what' has several layouts (bar, heatmap, forest, scatter, inset,
cells).

    figure_name(family, *parts, adjusted=False)        the generic builder
    metrics_figure_name(rows, metrics, exclude)        metrics_models_paper_bar[_adj]
    rq_figure_name(rq, kind, measure, metrics, ...)    rq1_omega2_all_heatmap_adj, rq2_emm_paper-technical_bar
    rq3_figure_name(mode, technical, social, kind)     rq3_absolute_inset-access_adj
    interaction_figure_name(kind, block, metrics)      interactions_omega2_all_heatmap, interactions_model-k_flagged_cells
    gt_figure_name / flows_figure_name / tags_figure_name / matching_figure_name: figure_name of their family
"""

from typing import Optional, Sequence

from analysis import tradeoff
from metrics import summary
from utils import constants as cons

GT, MATCHING, TAGS, METRICS, FLOWS, INTERACTIONS = 'gt', 'matching', cons.PREFIX_PLOT_TAGS, 'metrics', 'flows', 'interactions'
RQ_KINDS = ('bar', 'heatmap', 'forest')
RQ3_KINDS = ('scatter', 'inset')


def _part(value) -> str:
    return str(value).replace('_', '-').replace(':', '-')


def figure_name(family: str, *parts, adjusted: bool = False) -> str:
    """<family>_<part>_<part>...[_adj]; empty parts (None, '') left out, '_' / ':' inside a part -> '-'."""
    return '_'.join([family, *(_part(p) for p in parts if p not in (None, ''))]) + (cons.ADJ_SUFFIX if adjusted else '')


def gt_figure_name(*parts) -> str:
    return figure_name(GT, *parts)


def flows_figure_name(*parts) -> str:
    return figure_name(FLOWS, *parts)


def tags_figure_name(*parts) -> str:
    return figure_name(TAGS, *parts)


def matching_figure_name(*parts) -> str:
    return figure_name(MATCHING, *parts)


def metrics_figure_name(rows: str, metrics, exclude=None) -> str:
    """metrics_<rows>_<selection>_bar[_adj] (rows: 'models', 'model-families', 'model-characteristics', 'prompt')."""
    selection, adjusted = summary.split_adjusted(metrics, exclude)
    return figure_name(METRICS, rows, selection, 'bar', adjusted=adjusted)


def rq_figure_name(rq: int, kind: str, measure: str, metrics: Optional[str | Sequence[str]] = None,
                   exclude: Optional[str | Sequence[str]] = None) -> str:
    """rq<rq>_<measure>_<selection>_<kind>[_adj]; selection = 'all' / the selection items joined by '-' (+ '-no-<excluded>'),
    kind = 'bar' | 'heatmap' | 'forest' (e.g. rq1_omega2_paper_bar_adj, rq2_emm_technical-factuality_bar)."""
    if kind not in RQ_KINDS:
        raise ValueError(f"kind must be one of {RQ_KINDS}, not {kind!r}")
    selection, adjusted = summary.split_adjusted(metrics, exclude)
    return figure_name(f'rq{rq}', measure, selection, kind, adjusted=adjusted)


def rq1_figure_name(kind: str, measure: str, metrics=None, exclude=None) -> str:
    return rq_figure_name(1, kind, measure, metrics, exclude)


def rq2_figure_name(kind: str, value: str = 'deviation', metrics=None, exclude=None) -> str:
    return rq_figure_name(2, kind, value, metrics, exclude)


def rq3_figure_name(mode: str, technical: Optional[Sequence[str]] = None, social: Optional[Sequence[str]] = None,
                    kind: str = 'scatter', color_by: Optional[str] = None) -> str:
    """rq3_<mode>[_<technical keys>-vs-<social keys>]_<kind>[-<color_by>][_adj]: the selection only for non-default
    metric sets; kind 'scatter' | 'inset' (color_by: the inset's model characteristic); '_adj' if any metric is
    size-adjusted."""
    if kind not in RQ3_KINDS:
        raise ValueError(f"kind must be one of {RQ3_KINDS}, not {kind!r}")
    technical, social = tradeoff.axis_metrics(technical, social)
    technical, adj_t = summary.split_adjusted(technical)
    social, adj_s = summary.split_adjusted(social)
    default = technical == '-'.join(cons.RQ3_TECHNICAL_METRICS) and social == '-'.join(cons.RQ3_SOCIAL_METRICS)
    selection = None if default else f"{technical}-vs-{social}"
    return figure_name('rq3', mode, selection, kind + (f'-{color_by}' if color_by else ''), adjusted=adj_t or adj_s)


def interaction_figure_name(kind: str, block: Optional[str] = None, metrics: Optional[str | Sequence[str]] = None,
                            measure: str = 'omega2') -> str:
    """kind 'heatmap': interactions_<measure>_<selection>_heatmap[_adj]; kind 'cells': interactions_<block a>-<block b>
    _<selection>_cells[_adj]. metrics: a selection, or the name of one (e.g. 'flagged', 'exploratory')."""
    if kind not in ('heatmap', 'cells'):
        raise ValueError(f"kind must be 'heatmap' or 'cells', not {kind!r}")
    selection, adjusted = summary.split_adjusted(metrics)
    what = measure if kind == 'heatmap' else block
    return figure_name(INTERACTIONS, what, selection, kind, adjusted=adjusted)
