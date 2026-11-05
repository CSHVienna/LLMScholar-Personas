"""
Annotation tasks: what is sampled, what the annotator sees, and the codebook (labels + definitions).

Each task produces two files (written by ``batch_annotation_sample.py``):
* ``annotation_<task>_items.json`` — blind: codebook + items (content only; no inferred labels),
  loaded by the local tool ``code/tools/annotation_tool.html``;
* ``annotation_<task>_key.parquet`` — item id -> identifiers and inferred tags, used only by
  ``batch_annotation_evaluate.py``.
"""

import json
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from utils import constants as cons

ITEM_ID = 'item_id'


# -----------------------------------------------------------------------------
# Codebooks
# -----------------------------------------------------------------------------
def codebook(task: str) -> dict:
    """Labels (with definitions and keyboard order), rules and free-text fields of a task."""
    if task == cons.ANNOTATION_TASK_RESPONSES:
        return {'task': task,
                'unit': 'response',
                'instructions': 'Read the prompt and the raw response of the LLM. Tick every tag that applies '
                                'to the response as a whole. Judge only the format, not whether the people exist.',
                'labels': [_label(tag, cons.RESPONSE_TAG_DEFINITIONS) for tag in cons.RESPONSE_TAGS],
                'rules': ['"correct" applies only if no other tag applies.',
                          'empty, empty_list, refused and invalid are used alone (no named recommendation).',
                          'Every other response gets all its issue tags (e.g. wrapped + extra_incomplete).',
                          'Count every person the response lists, also those without a usable name.'],
                'fields': [{'id': 'n_recommendations', 'type': 'number',
                            'label': 'Number of recommended people (all person entries; 0 if none)'}]}
    if task == cons.ANNOTATION_TASK_RECOMMENDATIONS:
        return {'task': task,
                'unit': 'recommendation',
                'instructions': 'Look at the raw name and lastname fields of one recommended person. Tick every tag '
                                'that applies, and write the first and last name as you would record them '
                                '(leave empty if there is no usable name).',
                'labels': [_label(tag, cons.REC_TAG_DEFINITIONS) for tag in cons.REC_TAGS],
                'rules': ['"correct" applies only if no other tag applies.',
                          '"placeholder" is used alone.'],
                'fields': [{'id': 'first_name', 'label': 'First name (as you would record it)'},
                           {'id': 'last_name', 'label': 'Last name (as you would record it)'}]}
    raise ValueError(f"Unknown annotation task {task!r} (expected one of {cons.ANNOTATION_TASKS})")


def _label(tag: str, definitions: Dict[str, str]) -> dict:
    return {'id': tag, 'definition': definitions[tag]}


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, stratum_column: str, per_stratum: int, max_items: int,
                      seed: int) -> pd.DataFrame:
    """Up to ``per_stratum`` random rows per stratum (all rows of small strata), at most ``max_items`` overall.

    Rare strata are thereby over-represented on purpose (every inferred combination gets checked).
    If the cap binds, the per-stratum quota is lowered until the sample fits.
    """
    sizes = df[stratum_column].value_counts()
    quota = per_stratum
    while quota > 1 and sizes.clip(upper=quota).sum() > max_items:
        quota -= 1
    parts = [group.sample(n=min(len(group), quota), random_state=seed)
             for _, group in df.groupby(stratum_column, observed=True)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)   # shuffled


def assign_item_ids(df: pd.DataFrame, prefix: str, seed: int) -> pd.DataFrame:
    """Opaque item ids (random hex), so that ids reveal nothing about strata or models."""
    rng = np.random.default_rng(seed)
    ids = {f"{prefix}-{rng.integers(0, 16 ** 6):06x}" for _ in range(3 * len(df))}   # oversample for uniqueness
    return df.assign(**{ITEM_ID: sorted(ids)[:len(df)]}).sample(frac=1, random_state=seed + 1)


# -----------------------------------------------------------------------------
# Items (what the annotator sees)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Section:
    """One block of an item, e.g. the prompt or the raw response."""
    title: str
    text: str
    preformatted: bool = False

    def to_dict(self) -> dict:
        return {'title': self.title, 'text': self.text, 'preformatted': self.preformatted}


def response_item(row: pd.Series) -> List[Section]:
    """Prompt and raw response text (as returned by the API)."""
    return [Section('Prompt', row['prompt']),
            Section('Raw response', row['raw_text'] if row['raw_text'] else '(no content)', preformatted=True)]


def recommendation_item(row: pd.Series) -> List[Section]:
    """Raw name / lastname values (exactly as returned, quoted) and the full record."""
    def quoted(value) -> str:
        return '(missing)' if value is None or (isinstance(value, float) and np.isnan(value)) else json.dumps(value, ensure_ascii=False)
    record = json.dumps(json.loads(row['record_json']), indent=2, ensure_ascii=False)
    return [Section('name', quoted(row['name_raw']), preformatted=True),
            Section('lastname', quoted(row['lastname_raw']), preformatted=True),
            Section('Full record', record, preformatted=True)]


ITEM_BUILDERS: Dict[str, Callable[[pd.Series], List[Section]]] = {
    cons.ANNOTATION_TASK_RESPONSES: response_item,
    cons.ANNOTATION_TASK_RECOMMENDATIONS: recommendation_item,
}


def items_payload(task: str, sample: pd.DataFrame) -> dict:
    """The blind JSON for the annotation tool: codebook + items (no inferred labels)."""
    build = ITEM_BUILDERS[task]
    items = [{ITEM_ID: row[ITEM_ID], 'sections': [s.to_dict() for s in build(row)]} for _, row in sample.iterrows()]
    return {'codebook': codebook(task), 'items': items, 'tag_separator': cons.TAG_SEPARATOR}


def key_table(task: str, sample: pd.DataFrame) -> pd.DataFrame:
    """Item id -> identifiers and inferred tags (kept apart from the items)."""
    tags_column = cons.COL_RESPONSE_TAGS if task == cons.ANNOTATION_TASK_RESPONSES else cons.COL_REC_TAGS
    columns = [ITEM_ID] + cons.RESPONSE_ID_COLUMNS + [cons.COL_PROMPT_ID, tags_column]
    if task == cons.ANNOTATION_TASK_RECOMMENDATIONS:
        columns += [cons.COL_REC_INDEX, 'first_name', 'last_name']
    key = sample[[c for c in columns if c in sample.columns]].copy()
    return key.rename(columns={tags_column: 'inferred_tags'})


def vocabulary(task: str) -> List[str]:
    """All tags of a task, in tag order."""
    return cons.RESPONSE_TAGS if task == cons.ANNOTATION_TASK_RESPONSES else cons.REC_TAGS

