"""
Decoding of the generated text into person-like objects.

Decoding is attempted in steps of increasing invasiveness; the first step that
succeeds is recorded as ``method`` (and becomes a response tag, see ``taxonomy``):

0. cosmetic cleaning (not a repair): strip whitespace, ``<think>`` blocks and markdown code fences
1. ``strict``          json.loads
2. ``python_literal``  ast.literal_eval (single quotes, True/None, trailing commas ...)
3. ``embedded``        outermost JSON block found inside surrounding prose
4. ``truncated``       the JSON is syntactically valid up to the end of the text, where it was cut
                       (typically the token budget): the complete records are kept and the
                       trailing partial entry is discarded

JSON that is broken anywhere else (e.g. a missing comma mid-text) is not repaired.

Person-like objects are then searched recursively, so that any wrapper is
accepted (``{"candidates": [...]}``, ``{"1": {...}}``, ``{"profesor1": {...}}``, a bare dict, ...).
Ollama's JSON mode forces a top-level object, so wrappers are not a model failure.
"""

import ast
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from utils import constants as cons

# Cosmetic patterns
_THINK_BLOCK = re.compile(r'<think>.*?</think>', flags=re.DOTALL | re.IGNORECASE)
_THINK_DANGLING = re.compile(r'^.*?</think>', flags=re.DOTALL | re.IGNORECASE)   # opening tag cut off
_CODE_FENCE = re.compile(r'^```[a-zA-Z]*\s*(.*?)\s*```$', flags=re.DOTALL)

_PERSON_KEYS = frozenset(cons.PERSON_FIRSTNAME_KEYS + cons.PERSON_LASTNAME_KEYS)
_JSON_DECODER = json.JSONDecoder()


@dataclass(frozen=True)
class ParsedContent:
    """Result of decoding one generated text."""
    method: str                                       # one of cons.PARSE_METHODS
    persons: List[dict] = field(default_factory=list)  # person-like objects, in output order
    has_content: bool = False                         # decoded structure holds any non-blank string
    root: Optional[str] = None                        # 'array' / 'object' (None if not decoded as a whole)
    is_empty_wrapper: bool = False                    # an object holding only empty lists, e.g. {"candidates": []}
    error: Optional[str] = None                       # last decoding error (if nothing decoded)

    @property
    def is_decoded(self) -> bool:
        return self.method != cons.PARSE_NONE


class ContentParser:
    """Decode generated text into person-like objects (see module docstring)."""

    def parse(self, text: Optional[str]) -> ParsedContent:
        """Decode ``text``; never raises."""
        cleaned = clean_cosmetics(text or '')
        if not cleaned:
            return ParsedContent(method=cons.PARSE_NONE)

        # Steps 1-3: decode a whole JSON value
        value, method, error = self._decode_value(cleaned)
        if method != cons.PARSE_NONE:
            return ParsedContent(method=method, persons=find_persons(value), has_content=_has_text(value),
                                 root=cons.JSON_ROOT_ARRAY if isinstance(value, list) else cons.JSON_ROOT_OBJECT,
                                 is_empty_wrapper=is_empty_wrapper(value))

        # Step 4: truncated JSON -> keep the complete records
        segment = truncated_json(cleaned)
        if segment is not None:
            return ParsedContent(method=cons.PARSE_TRUNCATED, persons=complete_persons(segment), has_content=True,
                                 root=cons.JSON_ROOT_ARRAY if segment[0] == '[' else cons.JSON_ROOT_OBJECT)

        return ParsedContent(method=cons.PARSE_NONE, has_content=True, error=error)

    # -------------------------------------------------------------------------
    # Decoding steps
    # -------------------------------------------------------------------------
    def _decode_value(self, text: str) -> Tuple[Any, str, Optional[str]]:
        """Try strict JSON, python literal, then embedded JSON. Returns (value, method, error)."""
        error = None
        steps = ((cons.PARSE_STRICT, text, json.loads),
                 (cons.PARSE_LITERAL, text, _literal_eval),
                 (cons.PARSE_EMBEDDED, extract_embedded_block(text), _loads_json_or_literal))

        for method, candidate, decode in steps:
            if candidate is None or (method == cons.PARSE_EMBEDDED and candidate == text):
                continue  # nothing new to try
            try:
                value = decode(candidate)
                if isinstance(value, (dict, list)):
                    return value, method, None
            except Exception as exc:  # json/ast/recursion errors: try the next step
                error = f"{type(exc).__name__}: {exc}"[:500]
        return None, cons.PARSE_NONE, error


# -----------------------------------------------------------------------------
# Helpers (module-level so they can be tested and reused)
# -----------------------------------------------------------------------------
def clean_cosmetics(text: str) -> str:
    """Remove whitespace, reasoning traces and markdown fences (no content repair)."""
    text = _THINK_BLOCK.sub('', text)
    if '</think>' in text.lower():
        text = _THINK_DANGLING.sub('', text)
    text = text.strip()
    fence = _CODE_FENCE.match(text)
    return fence.group(1).strip() if fence else text


def extract_embedded_block(text: str) -> Optional[str]:
    """Return the span from the first '[' or '{' to the last matching closing bracket."""
    starts = [i for i in (text.find('['), text.find('{')) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    end = text.rfind(']' if text[start] == '[' else '}')
    return text[start:end + 1] if end > start else None


def truncated_json(text: str) -> Optional[str]:
    """Return the JSON segment (from the first '[' or '{') if it is a valid JSON prefix cut at the end, else None.

    A prefix is valid when decoding fails only because the input ended: the error is
    at the very end of the text, or a string is left unterminated (it runs to the end).
    """
    starts = [i for i in (text.find('['), text.find('{')) if i >= 0]
    if not starts:
        return None
    segment = text[min(starts):]
    try:
        json.loads(segment)
        return None  # complete JSON: not truncated
    except json.JSONDecodeError as error:
        ends_early = error.pos >= len(segment) or error.msg.startswith('Unterminated string')
        return segment if ends_early else None
    except RecursionError:
        return None


def complete_persons(text: str) -> List[dict]:
    """Every *complete* person-like JSON object in ``text``, in order (a trailing partial one is dropped).

    Scans each '{' and tries to decode one JSON object from there; after a
    successful decode the scan jumps past it, so nested objects are not duplicated.
    """
    persons, position = [], text.find('{')
    while position >= 0:
        try:
            value, end = _JSON_DECODER.raw_decode(text, position)
            if isinstance(value, dict) and is_person(value):
                persons.append(value)
                position = text.find('{', end)
                continue
        except (json.JSONDecodeError, RecursionError):
            pass
        position = text.find('{', position + 1)
    return persons


def find_persons(value: Any) -> List[dict]:
    """Recursively collect person-like dicts (a person's own nested objects are not searched)."""
    try:
        if isinstance(value, dict):
            if is_person(value):
                return [value]
            return [p for child in value.values() for p in find_persons(child)]
        if isinstance(value, list):
            return [p for child in value for p in find_persons(child)]
    except RecursionError:
        pass
    return []


def is_empty_wrapper(value: Any) -> bool:
    """True for a non-empty object whose values are all empty lists, e.g. {"candidates": []}."""
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(child, list) and not child for child in value.values()))


def is_person(obj: dict) -> bool:
    """A dict is person-like if it has at least one first-name or last-name key."""
    return any(str(key).strip().lower() in _PERSON_KEYS for key in obj)


def _loads_json_or_literal(text: str) -> Any:
    """json.loads, falling back to a python literal."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _literal_eval(text)


def _literal_eval(text: str) -> Any:
    """ast.literal_eval without its warnings (e.g. 'invalid escape sequence \\(' for LLM-made escapes).

    Note: Python reads some escapes differently from what the model may have meant (e.g. '\\032' as an
    octal character), so recovered *string values* can differ slightly; names are rarely affected.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ast.literal_eval(text)


def _has_text(value: Any) -> bool:
    """True if a decoded structure contains at least one non-blank string (keys excluded)."""
    try:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            return any(_has_text(v) for v in value.values())
        if isinstance(value, list):
            return any(_has_text(v) for v in value)
    except RecursionError:
        return True
    return value is not None and not isinstance(value, (dict, list))
