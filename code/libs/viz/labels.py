"""Short display labels (models, tag combinations) used by all plots."""

import re
from typing import Dict, Iterable

from utils import constants as cons
from viz import constants as viscons

_RULES = [(re.compile(pattern), replacement) for pattern, replacement in viscons.MODEL_LABEL_RULES]


def model_label(model: str) -> str:
    """Short label of one model slug (rules, then overrides), e.g. 'gpt-4_1-mini-2025-04-14' -> 'gpt-4.1-mini'."""
    label = model
    for pattern, replacement in _RULES:
        label = pattern.sub(replacement, label)
    return viscons.MODEL_LABEL_OVERRIDES.get(model, label)


def model_labels(models: Iterable[str]) -> Dict[str, str]:
    """Model slug -> short label; raises if two models would share a label."""
    labels = {model: model_label(model) for model in models}
    seen: Dict[str, str] = {}
    for model, label in labels.items():
        if label in seen and seen[label] != model:
            raise ValueError(f"Models {seen[label]!r} and {model!r} share the label {label!r}: "
                             f"add an entry to MODEL_LABEL_OVERRIDES in viz/constants.py")
        seen[label] = model
    return labels


def sample_size_label(n_models: int = None, n_responses: int = None, n_recommendations: int = None,
                      separator: str = ', ') -> str:
    """e.g. '$N_m$ = 43, $N_r$ = 928,800' (matplotlib mathtext: subscripts; only the sizes given)."""
    sizes = ((viscons.SYMBOL_N_MODELS, n_models), (viscons.SYMBOL_N_RESPONSES, n_responses),
             (viscons.SYMBOL_N_RECOMMENDATIONS, n_recommendations))
    return separator.join(f"{symbol} = {value:,}" for symbol, value in sizes if value is not None)


def tag_label(category: str, short: bool = False) -> str:
    """'embedded|wrapped' -> 'embedded + wrapped', or with ``short`` 'embed+wrap'."""
    if not short:
        return category.replace(cons.TAG_SEPARATOR, viscons.TAGS_SEPARATOR_DISPLAY)
    tags = [viscons.TAGS_SHORT_LABELS.get(tag, tag) for tag in category.split(cons.TAG_SEPARATOR)]
    return viscons.TAGS_SEPARATOR_DISPLAY_SHORT.join(tags)
