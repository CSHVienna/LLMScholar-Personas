"""
Text clean-up of person names, applied to BOTH sides (OpenAlex names and LLM-recommended names) before any
normalisation, matching or demographic inference. It only repairs the text; corrections of the name itself
(titles, repetitions, splitting) are for recommendations only (responses/recommendations.py).

clean_text, in this order:
    1. escapes written as text are decoded: 'Garc\\u00EDa' -> 'García' (a surrogate pair -> one character;
       lone surrogates are left as written)
    2. ftfy (mojibake, i.e. text decoded with the wrong encoding: 'GarcÃ\\xada' -> 'García', also when doubly
       encoded; HTML entities '&amp;'; full-width letters; curly apostrophes; NFC). Must run BEFORE step 3:
       the soft hyphen in 'Ã\\xad' is part of the mojibake of 'í'.
    3. invisible formatting characters removed (zero-width spaces / joiners, soft hyphens, direction marks, BOM):
       'Tarcí\\xadsio' -> 'Tarcísio'; control characters (tabs, newlines) become spaces.
Plain ASCII text without '&' or '\\u' is returned unchanged (fast path).
"""

import re
import unicodedata
from typing import Optional

# Escapes written as text by the LLM ('\\u00ED'); a surrogate pair is decoded as one character
_TEXT_ESCAPE = re.compile(r'\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})|\\u([0-9a-fA-F]{4})')
_ASCII_CONTROL = re.compile(r'[\x00-\x1f\x7f]')


def _decode(match: re.Match) -> str:
    if match.group(1):
        high, low = int(match.group(1), 16), int(match.group(2), 16)
        return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    code = int(match.group(3), 16)
    return match.group(0) if 0xD800 <= code <= 0xDFFF else chr(code)


def _invisible_removed(text: str) -> str:
    return ''.join('' if unicodedata.category(ch) == 'Cf' else ' ' if unicodedata.category(ch) == 'Cc' else ch
                   for ch in text)


def clean_text(text: Optional[str]) -> Optional[str]:
    """Repaired text (see module docstring); None / non-strings are returned as they are."""
    if not isinstance(text, str) or not text:
        return text
    if '\\u' in text:
        text = _TEXT_ESCAPE.sub(_decode, text)
    if text.isascii() and '&' not in text:
        return _ASCII_CONTROL.sub(' ', text) if _ASCII_CONTROL.search(text) else text
    import ftfy   # imported lazily: only needed for non-ASCII text
    return _invisible_removed(ftfy.fix_text(text))
