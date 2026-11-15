"""
Display names of every plot: one name per concept (same name in every figure). Plots ask here and nothing else.

    variable_title(v)        prompt variable / LLM / model characteristic: 'Language', 'k', 'LLM', 'Access', ...
    level_label(v, value)    value of a prompt variable or model characteristic: 'Recruiter', 'Open', 'Yes', ...
    attribute_title(a)       person attribute (flows, GT): 'Gender', 'Country', 'Seniority', 'Productivity', ...
    class_label(a, value)    class of a person attribute: 'East Asian', 'Female', 'Junior', 'Other', countries as is
    MEASURE_LABELS           effect sizes (RQ1, interactions); fade_rule(): the 'faded' note of every heatmap
Metric headers (Par.$_{gender}$, short 'Gender' / '# works') live with the metric registry (metrics/summary.py:
METRICS, ATTRIBUTE_NAMES); prompt titles in summary.PROMPT_TITLES, model characteristics in summary.MODEL_ATTRIBUTES.
Rule: sentence case for titles, axis labels and classes ('k' and symbols as they are).
"""

import re
from typing import Dict, Iterable, Optional

from metrics import flows as fl
from metrics import summary
from utils import constants as cons
from viz import constants as viscons

LLM = 'LLM'                                         # the model factor, in every figure (never 'Model')
FAMILY_TITLES = {cons.FAMILY_MODEL: LLM, cons.FAMILY_PERSONA: 'Persona', cons.FAMILY_CONTEXT: 'Context',
                 cons.FAMILY_INTERACTION: 'Interaction'}
_SOURCE_TITLES = {cons.COL_MODEL: LLM, cons.MODEL_FAMILY: f'{LLM} family'}
ATTRIBUTE_TITLES = {'gender': 'Gender', 'race': 'Race', 'ethnicity': 'Ethnicity', 'country': 'Country',
                    fl.AUTHOR_SENIORITY: 'Seniority', 'productivity': 'Productivity', 'influence': 'Influence',
                    'collaboration': 'Collaboration'}     # no 'perceived' in figures (the captions say it once)
_AS_IS_CLASSES = ('country', 'productivity', 'influence', 'collaboration')   # country codes, L / M / H bins
MEASURE_LABELS = {'omega2': r'Partial $\omega^2$', 'shapley_r2': r'Shapley $R^2$', 'delta_r2': r'$\Delta R^2$',
                  'f2': r'Local $f^2$', 'max_abs_deviation': 'Max |deviation|'}
CORRECTION_LABELS = {'holm': 'Holm', 'bh': 'BH', 'by': 'BY'}


def variable_title(variable: str) -> str:
    """'language' -> 'Language', 'k' -> 'k', the model factor -> 'LLM', 'access' -> 'Access'."""
    if variable in _SOURCE_TITLES:
        return _SOURCE_TITLES[variable]
    if variable in summary.MODEL_ATTRIBUTES:
        return summary.MODEL_ATTRIBUTES[variable][0]
    return summary.PROMPT_TITLES.get(variable, variable.replace('_', ' ').capitalize())


def level_label(variable: str, value) -> str:
    """Value of a prompt variable (summary.level_label) or of a model characteristic (its label, e.g. True -> 'Yes')."""
    if variable in summary.MODEL_ATTRIBUTES:
        _, levels, labels = summary.MODEL_ATTRIBUTES[variable]
        return dict(zip(map(str, levels), labels)).get(str(value), str(value))
    return summary.level_label(value)


def attribute_title(attribute: str) -> str:
    """Person attribute (flows, GT): 'gender' -> 'Gender', 'author_seniority' / 'seniority' -> 'Seniority'."""
    return ATTRIBUTE_TITLES.get(attribute, attribute.replace('_', ' ').capitalize())


def class_label(attribute: str, value: str) -> str:
    """Class of a person attribute: 'east_asian' -> 'East Asian', 'female' -> 'Female', 'junior' -> 'Junior',
    'other' -> 'Other'; country codes and L / M / H bins as they are."""
    if value in (cons.OTHER, cons.UNKNOWN):
        return value.capitalize()
    if attribute in _AS_IS_CLASSES:
        return value
    return value.replace('_', ' ').title()


def fade_rule(correction: Optional[str], alpha: float) -> str:
    """What a faded heatmap value means: 'BY-adjusted p ≥ 0.05', or '95% CI includes 0' without a correction."""
    if correction is None:
        return '95% CI includes 0'
    return f"{CORRECTION_LABELS[correction]}-adjusted p ≥ {alpha:g}"

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
