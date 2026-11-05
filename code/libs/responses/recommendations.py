"""
Recommendation tagging (multi-label): correct, title_removed, fixed, split,
initials_name, initials_lastname, initials, placeholder, invalid.

Each recommended person has a first name (``name``) and a last name (``lastname``).
Every tag that applies is kept (tag order as in ``cons.REC_TAGS``); "valid" is defined
later as a combination of tags.

1. a part is a template instead of a name ("Candidate 1", "Last Name")    -> placeholder (alone)
2. cosmetic cleaning: missing values (N/A, none, ...) -> empty; parentheses and edge
   punctuation removed; honorifics/degrees removed (Dr., Prof., PhD)      -> title_removed
3. only one part (or both identical):
     nothing left, or a single word only                                -> invalid
     >1 word: split into first + last name (lower-case particles such as
     'van', 'de la' go to the last name; "Smith, John" is last, first)   -> split
4. both parts, one repeated inside the other, e.g. name="John Smith"/lastname="Smith",
   name="Yamamoto Hiroshi"/lastname="Yamamoto", name="John"/lastname="John Smith":
   the repetition is removed                                              -> fixed
5. initials only (after 2-4): first name -> initials_name, last name -> initials_lastname,
   both -> initials
6. no title removal, split or fix was needed, and no part is initials only -> correct

Duplicates are not tagged here: they are an audit metric computed on the valid recommendations
of a response, using ``name_key`` (name tokens without case/accents, in any order).

Comparisons ignore case and accents; outputs keep the original spelling.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from utils import constants as cons
from responses import tags as tag_utils
from responses.persons import clean_name_value, is_template, missing_fields, raw_name_parts

_TITLES = frozenset(cons.NAME_TITLES)
_PARTICLES = frozenset(cons.NAME_PARTICLES)
_PARENTHESES = re.compile(r'\([^)]*\)|\[[^\]]*\]')
_EDGE_PUNCTUATION = '.;:"\'“”‘’`*'   # commas are kept: they mark "Last, First"
# Tags that prevent `correct`: any repair, or initials instead of a full name part
_NOT_CORRECT_TAGS = frozenset([cons.REC_TITLE_REMOVED, cons.REC_FIXED, cons.REC_SPLIT,
                               cons.REC_INITIALS_NAME, cons.REC_INITIALS_LASTNAME, cons.REC_INITIALS])

# Recommendation metadata kept from the LLM output (output column -> accepted keys, lower-case)
METADATA_KEYS = {'affiliations': ['current_affiliations', 'affiliations', 'affiliation'],
                 'areas': ['areas_of_research_or_work', 'areas_of_research', 'research_areas'],
                 'reason': ['reason'],
                 'source_url': ['source', 'url']}

# Every key we interpret; anything else is listed in `extra_fields` (the full object is kept in `record_json`)
_KNOWN_KEYS = frozenset(cons.REQUIRED_FIELDS + cons.PERSON_FIRSTNAME_KEYS + cons.PERSON_LASTNAME_KEYS
                        + [key for keys in METADATA_KEYS.values() for key in keys])

# Name fixes (recorded in the `name_fix` column: *how* a name was split or fixed)
FIX_LASTNAME_IN_NAME = 'lastname_in_name'
FIX_NAME_IN_LASTNAME = 'name_in_lastname'
FIX_SPLIT_NAME = 'split_name'
FIX_SPLIT_LASTNAME = 'split_lastname'
FIX_SPLIT_COMMA = 'split_comma'
FIX_MISSING = 'missing'
FIX_SINGLE_WORD = 'single_word'


@dataclass(frozen=True)
class TaggedName:
    """Cleaned name parts of one recommendation and its tags."""
    first_name: Optional[str]
    last_name: Optional[str]
    tags: List[str] = field(default_factory=list)
    fix: Optional[str] = None

    @property
    def full_name(self) -> Optional[str]:
        parts = [part for part in (self.first_name, self.last_name) if part]
        return ' '.join(parts) or None

    @property
    def key(self) -> Optional[str]:
        """Order-insensitive comparison key: name tokens without case/accents (used by the duplicates metric)."""
        return ' '.join(sorted(normalize_tokens(self.full_name or ''))) or None


class NameTagger:
    """Tag a (first name, last name) pair; see module docstring (rules 1-6)."""

    def tag(self, raw_first: Optional[str], raw_last: Optional[str]) -> TaggedName:
        # Rule 1: templates
        if is_template(raw_first) or is_template(raw_last):
            return TaggedName(None, None, [cons.REC_PLACEHOLDER])

        # Rule 2: cosmetic cleaning
        first, first_had_title = clean_name(raw_first)
        last, last_had_title = clean_name(raw_last)
        tags = [cons.REC_TITLE_REMOVED] if first_had_title or last_had_title else []

        # Rules 3-4: split a single part, or remove a repetition between both parts
        if first is None or last is None or normalize_tokens(first) == normalize_tokens(last):
            first, last, fix = split_single(first or last, FIX_SPLIT_NAME if first else FIX_SPLIT_LASTNAME)
            if first is None:
                return TaggedName(None, last, tag_utils.ordered(tags + [cons.REC_INVALID], cons.REC_TAGS), fix)
            tags.append(cons.REC_SPLIT)
        else:
            first, last, fix = remove_repetition(first, last)
            if fix:
                tags.append(cons.REC_FIXED)

        # Rules 5-6: initials; correct if nothing had to be repaired and both parts are full names
        tags += initials_tags(first, last)
        if not _NOT_CORRECT_TAGS.intersection(tags):
            tags.append(cons.REC_CORRECT)
        return TaggedName(first, last, tag_utils.ordered(tags, cons.REC_TAGS), fix)


# -----------------------------------------------------------------------------
# Tagging all recommendations of a response
# -----------------------------------------------------------------------------
def tag_recommendations(persons: List[dict], tagger: Optional[NameTagger] = None) -> List[dict]:
    """Tag every person of one response (in output order).

    Returns one record per person with the cleaned name, tags, completeness and the full LLM record.
    """
    tagger = tagger or NameTagger()
    rows = []

    for index, person in enumerate(persons, start=1):
        raw_first, raw_last = raw_name_parts(person)
        tagged = tagger.tag(raw_first, raw_last)
        missing = missing_fields(person)

        rows.append({cons.COL_REC_INDEX: index,
                     'name_raw': raw_first, 'lastname_raw': raw_last,
                     'first_name': tagged.first_name, 'last_name': tagged.last_name,
                     'full_name': tagged.full_name, 'name_key': tagged.key,
                     **tag_utils.tag_columns(tagged.tags, cons.REC_TAGS, cons.REC_TAG_PREFIX),
                     'name_fix': tagged.fix,
                     'missing_fields': ','.join(missing) or None, 'is_complete': not missing,
                     **_metadata(person),
                     'extra_fields': ','.join(extra_fields(person)) or None,
                     'record_json': json.dumps(person, ensure_ascii=False, default=str)})
    return rows


# -----------------------------------------------------------------------------
# Name helpers (module-level so they can be tested and reused, e.g. for GT names)
# -----------------------------------------------------------------------------
def clean_name(value) -> Tuple[Optional[str], bool]:
    """Remove placeholders, parentheses, edge punctuation and honorifics. Returns (name, title_removed)."""
    value = clean_name_value(value)
    if value is None:
        return None, False
    tokens = [t.strip(_EDGE_PUNCTUATION) for t in _PARENTHESES.sub(' ', value).split()]
    tokens = [t for t in tokens if t]

    kept = list(tokens)
    while kept and _is_title(kept[0]):
        kept.pop(0)
    while kept and _is_title(kept[-1]):
        kept.pop()
    return clean_name_value(' '.join(kept).strip(' ,')), len(kept) < len(tokens)


def remove_repetition(first: str, last: str) -> Tuple[str, str, Optional[str]]:
    """If one part repeats the other (at its start or end), remove it. Returns (first, last, fix)."""
    first_tokens, last_tokens = first.split(), last.split()
    first_norm, last_norm = normalize_tokens(first), normalize_tokens(last)
    n_first, n_last = len(first_norm), len(last_norm)

    # Last name repeated inside the first name ("John Smith" / "Smith", "Yamamoto Hiroshi" / "Yamamoto")
    if n_first > n_last:
        if first_norm[-n_last:] == last_norm:
            return ' '.join(first_tokens[:-n_last]), last, FIX_LASTNAME_IN_NAME
        if first_norm[:n_last] == last_norm:
            return ' '.join(first_tokens[n_last:]), last, FIX_LASTNAME_IN_NAME

    # First name repeated inside the last name ("John" / "John Smith")
    if n_last > n_first:
        if last_norm[:n_first] == first_norm:
            return first, ' '.join(last_tokens[n_first:]), FIX_NAME_IN_LASTNAME
        if last_norm[-n_first:] == first_norm:
            return first, ' '.join(last_tokens[:-n_first]), FIX_NAME_IN_LASTNAME

    return first, last, None


def split_single(text: Optional[str], fix: str) -> Tuple[Optional[str], Optional[str], str]:
    """Split one full-name string into (first, last, fix); first is None if it cannot be split (invalid)."""
    if text is None:
        return None, None, FIX_MISSING
    if ',' in text:  # "Smith, John"
        last, first = (part.strip() for part in text.split(',', 1))
        if first and last:
            return first, last, FIX_SPLIT_COMMA
        text = first or last
    tokens = text.split()
    if len(tokens) < 2:
        return None, text, FIX_SINGLE_WORD
    first, last = split_full_name(tokens)
    return first, last, fix


def initials_tags(first: Optional[str], last: Optional[str]) -> List[str]:
    """initials (both parts), initials_name or initials_lastname (one part), or nothing."""
    first_initials, last_initials = is_initials(first), is_initials(last)
    if first_initials and last_initials:
        return [cons.REC_INITIALS]
    if first_initials:
        return [cons.REC_INITIALS_NAME]
    if last_initials:
        return [cons.REC_INITIALS_LASTNAME]
    return []


def split_full_name(tokens: List[str]) -> Tuple[str, str]:
    """Split tokens into (first, last): last token plus preceding lower-case particles is the last name."""
    start = len(tokens) - 1
    while start > 1 and tokens[start - 1] in _PARTICLES:  # keep >= 1 token for the first name
        start -= 1
    return ' '.join(tokens[:start]), ' '.join(tokens[start:])


def normalize_tokens(text: str) -> List[str]:
    """Lower-case, accent-free tokens without punctuation (hyphens split tokens)."""
    decomposed = unicodedata.normalize('NFKD', text)
    stripped = ''.join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
    return re.sub(r"[^\w\s]", ' ', stripped.replace('-', ' ')).split()


def is_initials(text: Optional[str]) -> bool:
    """True if every token is a single letter (e.g. 'M. T.', 'K')."""
    tokens = normalize_tokens(text or '')
    return bool(tokens) and all(len(token) == 1 for token in tokens)


def _is_title(token: str) -> bool:
    return token.lower().strip(',').rstrip('.') in _TITLES


def extra_fields(person: dict) -> List[str]:
    """Keys returned by the LLM beyond the requested fields and the known name/metadata aliases."""
    return [str(key) for key in person if str(key).strip().lower() not in _KNOWN_KEYS]


def _metadata(person: dict) -> dict:
    """Recommendation metadata as strings (nested values serialised as JSON)."""
    lowered = {str(key).strip().lower(): value for key, value in person.items()}
    record = {}
    for column, keys in METADATA_KEYS.items():
        value = next((lowered[key] for key in keys if key in lowered), None)
        record[column] = value if value is None or isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return record
