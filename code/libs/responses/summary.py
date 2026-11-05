"""Response summary table: one row per response with its tags (no recommendation content)."""

from typing import List

import pandas as pd

from utils import constants as cons
from utils.frames import apply_categories
from responses import tags as tag_utils
from responses.persons import has_name, missing_fields
from responses.reader import ParsedResponse, ResponseReader
from responses.text_signals import find_toy_disclaimer

SUMMARY_COLUMNS = (cons.RESPONSE_ID_COLUMNS
                   + [cons.COL_SOURCE, cons.COL_PROMPT_ID]
                   + tag_utils.column_names(cons.RESPONSE_TAGS, cons.RESPONSE_TAG_PREFIX)
                   + [cons.COL_PARSE_METHOD, 'json_root', 'n_persons', 'n_named_persons', 'n_complete_persons',
                      'toy_keyword', 'created_at', 'finish_reason',
                      'prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'duration_s',
                      'api_error', 'api_refusal', 'parse_error'])


def summary_row(response: ParsedResponse) -> dict:
    """Flatten a :class:`ParsedResponse` into one summary record."""
    raw, parsed = response.raw, response.parsed
    return {**response.ids,
            **tag_utils.tag_columns(response.tags, cons.RESPONSE_TAGS, cons.RESPONSE_TAG_PREFIX),
            cons.COL_PARSE_METHOD: parsed.method if parsed else cons.PARSE_NONE,
            'json_root': parsed.root if parsed else None,
            'n_persons': len(response.persons),
            'n_named_persons': sum(has_name(p) for p in response.persons),
            'n_complete_persons': sum(not missing_fields(p) for p in response.persons),
            'toy_keyword': find_toy_disclaimer(raw.text) if cons.RESPONSE_TOY in response.tags else None,
            'created_at': raw.created_at,
            'finish_reason': raw.finish_reason,
            'prompt_tokens': raw.prompt_tokens,
            'completion_tokens': raw.completion_tokens,
            'reasoning_tokens': raw.reasoning_tokens,
            'duration_s': raw.duration_s,
            'api_error': raw.api_error,
            'api_refusal': raw.api_refusal,
            'parse_error': parsed.error if parsed else None}


def summarize_model(reader: ResponseReader, model: str, languages: List[str]) -> pd.DataFrame:
    """Build the response summary of one model across ``languages``."""
    rows = [summary_row(response) for response in reader.iter_model(model, languages)]
    return apply_categories(pd.DataFrame(rows, columns=SUMMARY_COLUMNS))
