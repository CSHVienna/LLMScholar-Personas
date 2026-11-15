"""
Order and colour of the classes of a person attribute, the same in every figure (flows, GT composition / population /
countries): gender diverging, race / ethnicity / country categorical in a fixed slot per class, prominence and
seniority sequential, 'other' / 'unknown' grey ('unknown' also hatched).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from metrics import flows as fl
from utils import constants as cons
from viz import constants as viscons
from viz.figures import ramp


def target_order_colors(target: str, classes: Sequence[str], gender_labels: Optional[Sequence[str]] = None,
                        bin_labels: Optional[Sequence[str]] = None) -> Tuple[List[str], Dict[str, str]]:
    """Order and colour of the classes of a person attribute ('other' and 'unknown' last, in grey)."""
    present = [c for c in dict.fromkeys(map(str, classes))]
    special = [c for c in (cons.OTHER, cons.UNKNOWN) if c in present]
    rest = [c for c in present if c not in special]
    if target == 'gender':
        scheme = list(gender_labels or cons.DEFAULT_GENDER_LABELS)
        order = [c for c in scheme if c in rest] + sorted(set(rest) - set(scheme))
        full = ramp(viscons.FLOW_DIVERGING, len(scheme), single=0.5)
        colors = {c: full[scheme.index(c)] if c in scheme else viscons.FLOW_OTHER_COLOR for c in order}
    elif target in ('productivity', 'influence', 'collaboration', fl.AUTHOR_SENIORITY):
        scheme = list(cons.SENIORITY_CLASSES if target == fl.AUTHOR_SENIORITY else bin_labels or cons.DEFAULT_GT_BIN_LABELS)
        order = [c for c in scheme if c in rest] + sorted(set(rest) - set(scheme))
        full = ramp(viscons.FLOW_SEQUENTIAL, len(scheme), single=0.5)
        colors = {c: full[scheme.index(c)] if c in scheme else viscons.FLOW_OTHER_COLOR for c in order}
    elif target in ('race', 'ethnicity'):   # fixed slot per class (the same colour in every figure)
        scheme = fl.RACE_ORDER if target == 'race' else viscons.FLOW_SHOWN_CLASSES['ethnicity']
        order = [c for c in scheme if c in rest] + sorted(set(rest) - set(scheme))
        palette = viscons.FLOW_CATEGORICAL
        colors = {c: palette[scheme.index(c)] if c in scheme and scheme.index(c) < len(palette)
                  else viscons.FLOW_OTHER_COLOR for c in order}
    else:   # countries: the prompt countries first (design order: the same colour in every figure), then by share
        prompt = prompt_countries()
        order = [c for c in prompt if c in rest] + [c for c in rest if c not in prompt]
        palette = viscons.FLOW_CATEGORICAL
        # consecutive slots of the validated order (neighbours stay distinguishable); the prompt countries are kept in
        # every country table (prompt_countries()), so they take the first slots: the same colour in every figure
        colors = {c: palette[i] if i < len(palette) else viscons.FLOW_OTHER_COLOR for i, c in enumerate(order)}
    colors.update({cons.OTHER: viscons.FLOW_OTHER_COLOR, cons.UNKNOWN: viscons.FLOW_UNKNOWN_COLOR})
    return order + special, colors


def class_hatch(value: str) -> dict:
    """Extra patch style of a class: 'unknown' is hatched (its pale colour alone is too close to the surface)."""
    return {'hatch': viscons.UNKNOWN_HATCH, 'edgecolor': viscons.UNKNOWN_HATCH_COLOR} if value == cons.UNKNOWN else {}


def prompt_countries() -> List[str]:
    """Country codes of the prompt locations, in the design order (kept out of 'Other' in every country table)."""
    return [fl.LOCATION_COUNTRY[l] for l in map(str, cons.VARIABLE_VALUES[cons.VAR_LOCATION]) if l in fl.LOCATION_COUNTRY]


def pool_shown(values: pd.Series, target: str) -> pd.Series:
    """Classes outside viscons.FLOW_SHOWN_CLASSES[target] (e.g. the smaller ethnicities) pooled as 'other'."""
    shown = viscons.FLOW_SHOWN_CLASSES.get(target)
    if not shown:
        return values
    return values.where(values.isin(set(shown) | {cons.OTHER, cons.UNKNOWN}), cons.OTHER)
