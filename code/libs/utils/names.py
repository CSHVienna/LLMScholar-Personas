"""
Person names, the same rules on every side (recommended names, the OpenAlex name index, the demographic models):

    normalize_tokens(text)      lower-case, accent-free, transliterated tokens ('Bjørn Ó-Hara' -> ['bjorn', 'o', 'hara'])
    is_initials(text)           every token a single letter ('M. T.')
    split_full_name(tokens)     (first, last): the last token plus preceding lower-case particles is the last name
    first_last_order(name)      'Last, First' -> 'First Last' (other names unchanged)
    hangul_family_last(name)    Hangul names (family name first, no space) -> given name(s) + family name
"""

import re
import unicodedata
from typing import List, Optional, Tuple

from utils import constants as cons
from utils.text import clean_text

_PARTICLES = frozenset(cons.NAME_PARTICLES)
# Letters without a Unicode decomposition, transliterated for comparisons ('ß' -> 'ss' already by casefold)
_TRANSLITERATION = str.maketrans({'ø': 'o', 'ł': 'l', 'æ': 'ae', 'œ': 'oe', 'đ': 'd', 'ð': 'd', 'þ': 'th',
                                  'ı': 'i', 'ħ': 'h', 'ŧ': 't', 'ŀ': 'l'})
HANGUL_NAME = re.compile(r'[가-힣]{2,4}')                                # family name first, no space


def normalize_tokens(text: str) -> List[str]:
    """Lower-case, accent-free tokens without punctuation (hyphens split tokens), after clean_text (both sides:
    recommendation keys and the OpenAlex name index, A.2). Letters without a decomposition are transliterated
    ('Bjørn' -> 'bjorn', 'Łukasz' -> 'lukasz', 'Straße' -> 'strasse')."""
    decomposed = unicodedata.normalize('NFKD', clean_text(text) or '')
    stripped = ''.join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold().translate(_TRANSLITERATION)
    return re.sub(r"[^\w\s]", ' ', stripped.replace('-', ' ')).split()


def is_initials(text: Optional[str]) -> bool:
    """True if every token is a single letter (e.g. 'M. T.', 'K')."""
    tokens = normalize_tokens(text or '')
    return bool(tokens) and all(len(token) == 1 for token in tokens)


def split_full_name(tokens: List[str]) -> Tuple[str, str]:
    """Split tokens into (first, last): last token plus preceding lower-case particles is the last name."""
    start = len(tokens) - 1
    while start > 1 and tokens[start - 1] in _PARTICLES:  # keep >= 1 token for the first name
        start -= 1
    return ' '.join(tokens[:start]), ' '.join(tokens[start:])


def first_last_order(name: str) -> str:
    """'Last, First' -> 'First Last' (split at the first comma; parts stripped); a name without a comma unchanged."""
    if ',' not in name:
        return name
    last, _, first = name.partition(',')
    return f"{first.strip()} {last.strip()}"


def hangul_family_last(name: str) -> str:
    """Hangul names are written family name first without a space: move it last ('김민수' -> '민수 김'); other
    tokens unchanged (the name is re-joined with single spaces)."""
    return ' '.join(f"{t[1:]} {t[0]}" if HANGUL_NAME.fullmatch(t) else t for t in name.split())
