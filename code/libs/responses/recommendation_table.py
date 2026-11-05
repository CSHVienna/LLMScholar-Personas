"""
Recommendations table: one row per recommended person, with response identifiers and tags.

Every response with named recommendations (correct, or any issue tag) is tagged,
so that all tags can be counted before deciding what goes downstream.
Responses are re-read from the raw files with the same :class:`ResponseReader`; if the
recomputed response tags disagree with the summary, the summary is outdated and the step stops.
"""

from typing import Dict, List, Tuple

import pandas as pd

from utils import constants as cons
from utils.frames import apply_categories
from responses import tags as tag_utils
from responses.reader import ResponseReader
from responses.recommendations import NameTagger, tag_recommendations

RESPONSE_COLUMNS = (cons.RESPONSE_ID_COLUMNS + [cons.COL_SOURCE, cons.COL_PROMPT_ID]
                    + tag_utils.column_names(cons.RESPONSE_TAGS, cons.RESPONSE_TAG_PREFIX) + [cons.COL_PARSE_METHOD])
RECOMMENDATION_COLUMNS = ([cons.COL_REC_INDEX, 'name_raw', 'lastname_raw', 'first_name', 'last_name',
                           'full_name', 'name_key']
                          + tag_utils.column_names(cons.REC_TAGS, cons.REC_TAG_PREFIX)
                          + ['name_fix', 'missing_fields', 'is_complete', 'affiliations', 'areas', 'reason',
                             'source_url', 'extra_fields', 'record_json'])

ResponseKey = Tuple[str, int, int]  # (language, prompt_id, run_id)


class OutdatedSummaryError(RuntimeError):
    """Raised when the response summary was produced by a different parser version."""


def build_recommendations(reader: ResponseReader, model: str, languages: List[str],
                          summary: pd.DataFrame) -> pd.DataFrame:
    """Tag the recommendations of every response of ``model`` that has named recommendations."""
    selected = _selected_responses(summary)
    tagger, rows = NameTagger(), []

    for response in reader.iter_model(model, languages):
        key = (response.ids[cons.VAR_LANGUAGE], response.ids[cons.COL_PROMPT_ID], response.ids[cons.COL_RUN_ID])
        if key not in selected:
            continue
        recomputed = tag_utils.join(response.tags, cons.RESPONSE_TAGS)
        if recomputed != selected[key]:
            raise OutdatedSummaryError(f"[{model}] {key}: summary tags={selected[key]!r} but recomputed "
                                       f"tags={recomputed!r}. Re-run batch_responses_summary.py --overwrite (responses metadata).")

        # Response identifiers + one row per recommended person
        response_ids = {**response.ids,
                        **tag_utils.tag_columns(response.tags, cons.RESPONSE_TAGS, cons.RESPONSE_TAG_PREFIX),
                        cons.COL_PARSE_METHOD: response.parsed.method}
        rows.extend({**response_ids, **rec} for rec in tag_recommendations(response.persons, tagger))

    return apply_categories(pd.DataFrame(rows, columns=RESPONSE_COLUMNS + RECOMMENDATION_COLUMNS))


def _selected_responses(summary: pd.DataFrame) -> Dict[ResponseKey, str]:
    """Map (language, prompt_id, run_id) -> joined tags, for the responses with named recommendations."""
    columns = [f"{cons.RESPONSE_TAG_PREFIX}{tag}" for tag in cons.RESPONSE_TAGS_WITH_RECOMMENDATIONS]
    mask = summary[columns].any(axis=1)
    rows = summary[mask]
    keys = zip(rows[cons.VAR_LANGUAGE].astype(str), rows[cons.COL_PROMPT_ID].astype(int), rows[cons.COL_RUN_ID].astype(int))
    return dict(zip(keys, rows[cons.COL_RESPONSE_TAGS].astype(str)))
