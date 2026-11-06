"""
Response taxonomy (multi-label). "Valid" is defined later as a combination of these tags.

A response with >= 1 named recommendation (a name that is neither missing nor a template) gets
every issue tag that applies:
        python_literal  python syntax instead of JSON (single quotes, True/None, trailing commas)
        embedded        JSON surrounded by prose
        truncated       cut off: parsed up to the last complete record, trailing partial entry discarded
        wrapped         the JSON root is an object (e.g. {"candidates": [...]}), not the requested array
        toy             the names are disclaimed as illustrative / fictional ("plausible example")
and is tagged ``correct`` (alone) iff none of them applies, i.e. a strict JSON array (after cosmetic
cleaning: <think> traces, markdown code fences) without a disclaimer. Completeness (requested keys
absent from a person's object) is a per-recommendation attribute, not a response tag.

A response without any named recommendation gets exactly one failure tag, checked in order:
    refused     the API signalled a refusal (GPT `refusal` field, Gemini block/safety)
    invalid     API / transport error
    empty       no content (empty string or whitespace only, also after removing <think> traces)
    empty_list  an object wrapping only empty lists, e.g. {"candidates": []}
    refused     the text reads like a refusal / abstention and holds no person fields
                (a refusal carries no candidate names)
    invalid     anything else: unparsable or broken JSON, [], no person-like objects, only templates

An API-level refusal always wins (even if names were returned). Refusal keywords in the
text are only checked when no named recommendation was found, so words such as "cannot"
inside a ``reason`` never turn a usable answer into a refusal.
"""

import re
from typing import List, Optional

from utils import constants as cons
from responses import tags as tag_utils
from responses.content_parser import ParsedContent
from responses.persons import has_name
from responses.sources import RawResponse
from responses.text_signals import contains_refusal, find_toy_disclaimer

# A JSON key of a person field, e.g. "lastname": (used to tell broken recommendations from refusals)
_PERSON_FIELD = re.compile(r'["\']\s*(' + '|'.join(map(re.escape, cons.PERSON_FIRSTNAME_KEYS + cons.PERSON_LASTNAME_KEYS))
                           + r')\s*["\']\s*:', flags=re.IGNORECASE)


def classify(raw: RawResponse, parsed: Optional[ParsedContent]) -> List[str]:
    """Return the tags of one response, in tag order (see module docstring for the rules)."""
    # API-level outcomes
    if raw.api_refusal:
        return [cons.RESPONSE_REFUSED]
    if raw.api_error:
        return [cons.RESPONSE_INVALID]

    # No content
    if parsed is None or (not parsed.is_decoded and not parsed.has_content):
        return [cons.RESPONSE_EMPTY]

    # At least one named recommendation: its issues, or `correct` if there are none
    if any(has_name(person) for person in parsed.persons):
        issues = _issue_tags(raw, parsed)
        return tag_utils.ordered(issues, cons.RESPONSE_TAGS) if issues else [cons.RESPONSE_CORRECT]

    # No named recommendation: one failure tag
    if parsed.is_empty_wrapper:
        return [cons.RESPONSE_EMPTY_LIST]
    if contains_refusal(raw.text) and not _PERSON_FIELD.search(raw.text):
        return [cons.RESPONSE_REFUSED]
    return [cons.RESPONSE_INVALID]


def has_recommendations(tags: List[str]) -> bool:
    """True if the response has named recommendations (correct, or any issue tag)."""
    return any(tag in cons.RESPONSE_TAGS_WITH_RECOMMENDATIONS for tag in tags)


def _issue_tags(raw: RawResponse, parsed: ParsedContent) -> List[str]:
    tags = [cons.RESPONSE_TAG_BY_PARSE_METHOD[parsed.method]] if parsed.method != cons.PARSE_STRICT else []
    if parsed.root == cons.JSON_ROOT_OBJECT:
        tags.append(cons.RESPONSE_WRAPPED)
    if find_toy_disclaimer(raw.text):
        tags.append(cons.RESPONSE_TOY)
    return tags
