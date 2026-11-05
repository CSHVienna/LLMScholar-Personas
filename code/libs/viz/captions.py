"""
Figure captions for the paper (Nature style: a one-sentence title, then what is shown, the sample
sizes, how the summary and error bars are computed, and every abbreviation used in the figure).

Captions are built from the same inputs as the figures, so numbers and tag lists never drift.
They are printed in the notebook, not drawn on the figure.
"""

import textwrap
from typing import Dict, Iterable, List

from utils import constants as cons
from responses.tag_stats import CombinationCategories
from viz.labels import tag_label


def tags_figure_caption(unit: str, categories: CombinationCategories, valid_tags: Iterable[str],
                        n_models: int, n_responses: int, n_units: int) -> str:
    """Caption of a paper figure of tag combinations (per-model panel + all-models panel).

    ``unit`` is 'responses' or 'recommendations'; ``n_units`` is the number of those units
    (equal to ``n_responses`` for responses).
    """
    is_responses = unit == 'responses'
    title = ('Output format of the LLM responses.' if is_responses
             else 'Name quality of the recommended scholars.')
    # Sample sizes with the symbols annotated in the figure (matplotlib mathtext = LaTeX, ready to paste)
    n_m, n_r, n_rec = cons.SYMBOL_N_MODELS, cons.SYMBOL_N_RESPONSES, cons.SYMBOL_N_RECOMMENDATIONS
    sample = (f'{n_r} = {n_responses:,} responses from {n_m} = {n_models} models' if is_responses
              else f'{n_rec} = {n_units:,} recommendations in the {n_r} = {n_responses:,} responses that contain '
                   f'recommendations, from {n_m} = {n_models} models')
    valid = ', '.join(tag_label(tag, short=True) for tag in valid_tags)
    definitions = cons.RESPONSE_TAG_DEFINITIONS if is_responses else cons.REC_TAG_DEFINITIONS
    top_n = _number_word(cons.TOP_COMBINATIONS_PER_FAMILY)

    sentences = [
        title,
        f'Left, share of each model\'s {unit} per tag combination ({sample}); models are sorted by '
        f'their share of valid {unit}, written above each bar when below 100%.',
        f'Right, mean of the model-level shares (models weighted equally); error bars, '
        f'{cons.DEFAULT_CONFIDENCE:.0%} confidence intervals from a hierarchical bootstrap '
        f'({cons.DEFAULT_BOOTSTRAP_SAMPLES:,} replicates; models resampled, then prompt configurations within models).',
        f'A {unit[:-1]} is valid if all its tags are among {valid} (green); all other combinations are red '
        f'and hatched. Darker shades denote more frequent combinations; the {top_n} most '
        f'frequent combinations of each group are shown and the rest are pooled as '
        f'"{cons.LABEL_OTHER_VALID}" and "{cons.LABEL_OTHER_INVALID}".',
        f'Tags (+, co-occurring tags): {_definitions(categories, definitions)}.',
    ]
    return ' '.join(sentences)


def print_caption(caption: str, width: int = cons.CAPTION_PRINT_WIDTH) -> None:
    """Print a caption wrapped at ``width`` characters."""
    print(textwrap.fill(caption, width=width))


def _definitions(categories: CombinationCategories, definitions: Dict[str, str]) -> str:
    """'correct, ...; wrap, ...' for every tag that appears in the figure, in tag order."""
    shown = {tag for category in categories.categories for tag in category.split(cons.TAG_SEPARATOR)}
    order: List[str] = [tag for tag in definitions if tag in shown]
    return '; '.join(f'{tag_label(tag, short=True)}, {definitions[tag]}' for tag in order)


def _number_word(n: int) -> str:
    """Numbers below ten as words (journal style), others as digits."""
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return words[n] if 0 <= n < 10 else str(n)
