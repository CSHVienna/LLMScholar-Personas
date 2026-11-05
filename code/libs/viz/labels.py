"""Short display labels (models, tag combinations) used by all plots."""

import re
from typing import Dict, Iterable

from utils import constants as cons

_RULES = [(re.compile(pattern), replacement) for pattern, replacement in cons.MODEL_LABEL_RULES]


def model_label(model: str) -> str:
    """Short label of one model slug (rules, then overrides), e.g. 'gpt-4_1-mini-2025-04-14' -> 'gpt-4.1-mini'."""
    label = model
    for pattern, replacement in _RULES:
        label = pattern.sub(replacement, label)
    return cons.MODEL_LABEL_OVERRIDES.get(model, label)


def model_labels(models: Iterable[str]) -> Dict[str, str]:
    """Model slug -> short label; raises if two models would share a label."""
    labels = {model: model_label(model) for model in models}
    seen: Dict[str, str] = {}
    for model, label in labels.items():
        if label in seen and seen[label] != model:
            raise ValueError(f"Models {seen[label]!r} and {model!r} share the label {label!r}: "
                             f"add an entry to MODEL_LABEL_OVERRIDES in constants.py")
        seen[label] = model
    return labels


def sample_size_label(n_models: int = None, n_responses: int = None, n_recommendations: int = None,
                      separator: str = ', ') -> str:
    """e.g. '$N_m$ = 43, $N_r$ = 928,800' (matplotlib mathtext: subscripts; only the sizes given)."""
    sizes = ((cons.SYMBOL_N_MODELS, n_models), (cons.SYMBOL_N_RESPONSES, n_responses),
             (cons.SYMBOL_N_RECOMMENDATIONS, n_recommendations))
    return separator.join(f"{symbol} = {value:,}" for symbol, value in sizes if value is not None)


def tag_label(category: str, short: bool = False) -> str:
    """'wrapped|extra_incomplete' -> 'wrapped + extra_incomplete', or with ``short`` 'wrap+extra_inc'."""
    if not short:
        return category.replace(cons.TAG_SEPARATOR, cons.TAG_SEPARATOR_DISPLAY)
    tags = [cons.TAG_SHORT_LABELS.get(tag, tag) for tag in category.split(cons.TAG_SEPARATOR)]
    return cons.TAG_SEPARATOR_DISPLAY_SHORT.join(tags)
