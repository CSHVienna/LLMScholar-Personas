"""Accessors for person-like objects returned by the LLMs: names, templates and required fields."""

import re
from typing import List, Optional, Tuple

from utils import constants as cons

_FIRSTNAME_KEYS = frozenset(cons.PERSON_FIRSTNAME_KEYS)
_LASTNAME_KEYS = frozenset(cons.PERSON_LASTNAME_KEYS)
_MISSING_VALUES = frozenset(cons.NAME_MISSING_VALUES)
_TEMPLATE_VALUES = frozenset(cons.NAME_TEMPLATE_VALUES)
_TEMPLATE_PATTERNS = [re.compile(pattern) for pattern in cons.NAME_TEMPLATE_PATTERNS]


# -----------------------------------------------------------------------------
# Names
# -----------------------------------------------------------------------------
def raw_name_parts(person: dict) -> Tuple[Optional[str], Optional[str]]:
    """Return the raw (first name, last name) values as given (non-strings become None).

    The first matching key (in the order of the object) is used for each part.
    """
    first = last = None
    found_first = found_last = False
    for key, value in person.items():
        key = str(key).strip().lower()
        if not found_first and key in _FIRSTNAME_KEYS:
            first, found_first = (value if isinstance(value, str) else None), True
        elif not found_last and key in _LASTNAME_KEYS:
            last, found_last = (value if isinstance(value, str) else None), True
    return first, last


def name_parts(person: dict) -> Tuple[Optional[str], Optional[str]]:
    """Return (first name, last name) with whitespace collapsed; missing values and templates -> None."""
    first, last = raw_name_parts(person)
    return clean_name_value(first), clean_name_value(last)


def has_name(person: dict) -> bool:
    """True if the person has a first or last name that is neither missing nor a template."""
    first, last = name_parts(person)
    return first is not None or last is not None


def clean_name_value(value) -> Optional[str]:
    """Collapse whitespace; map non-strings, missing values and templates to None."""
    if not isinstance(value, str):
        return None
    value = ' '.join(value.split())
    return None if is_missing(value) or is_template(value) else value


def is_missing(value: Optional[str]) -> bool:
    """True for None and 'no value' strings ('', 'N/A', 'none', ...)."""
    return value is None or ' '.join(value.split()).lower() in _MISSING_VALUES


def is_template(value: Optional[str]) -> bool:
    """True for schema/instruction echoes ('Last Name', 'Your Name') and template names ('Candidate 1')."""
    if not isinstance(value, str):
        return False
    lowered = ' '.join(value.split()).lower()
    return lowered in _TEMPLATE_VALUES or any(pattern.match(lowered) for pattern in _TEMPLATE_PATTERNS)


# -----------------------------------------------------------------------------
# Required fields
# -----------------------------------------------------------------------------
def missing_fields(person: dict) -> List[str]:
    """Requested non-name fields (in prompt order) that are absent or not populated in ``person``.

    Names are assessed by the recommendation tags, not here. Keys are compared lower-cased;
    values must be non-empty, and 'N/A' is accepted only where the prompt allows it (``source``).
    """
    lowered = {str(key).strip().lower(): value for key, value in person.items()}
    return [field for field in cons.COMPLETENESS_FIELDS if not _is_populated(field, lowered.get(field))]


def _is_populated(field: str, value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and (field in cons.FIELDS_ALLOWING_NA or not is_missing(value))
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True
