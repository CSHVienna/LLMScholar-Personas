"""
names.py — How a recommended name is compared with a researcher's name.

  normalize                lowercase, no accents, no titles (Dr, Prof, ...), no punctuation
  parse_name               (full name, first, second, last)
  block_key                first 2 letters of the last name: names are only
                           compared within the same block
  _JW                      Jaro-Winkler similarity
  _passes_per_token_check  every word of the query (initials aside) must have
                           a near-identical word in the researcher's name

A researcher matches a recommended name when JW(full names) >= DN_THRESHOLD
and the per-token check passes with PER_TOKEN_THRESHOLD.
"""

import re
import unicodedata

from rapidfuzz.distance import JaroWinkler

DN_THRESHOLD = 0.85  # JW threshold for (display_name, display_name)

PER_TOKEN_THRESHOLD = (
    0.95  # JW threshold per token (>1 char). Initials (1 char) are ignored.
)

_TITLE_RE = re.compile(
    r"\b(dr|mr|mrs|ms|prof|professor|phd|ph\.d|md|m\.d|dsc|ing|lic|msc)\b\.?",
    re.IGNORECASE,
)

_PAREN_RE = re.compile(r"\(.*?\)")

_JW = JaroWinkler.similarity


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _TITLE_RE.sub(" ", text)
    text = _PAREN_RE.sub(" ", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_name(full: str) -> tuple[str, str, str, str]:
    """Returns (display_name, first_name, second_name, last_name)."""
    tokens = full.split()
    if not tokens:
        return "", "", "", ""
    first = tokens[0]
    last = tokens[-1] if len(tokens) > 1 else ""
    second = tokens[1] if len(tokens) > 2 else ""
    return full, first, second, last


def block_key(last: str) -> str:
    return last[:2] if len(last) >= 2 else last[:1]


def _passes_per_token_check(q_dn: str, r_dn: str, threshold: float) -> bool:
    """
    Post-JW filter: for each token >1 char in the query, require that a
    token >1 char exists in the ref with JW >= threshold. 1-char tokens
    (initials) are ignored on both the query and ref sides.

    Catches cases like:
      "john doe smith" → "john e smith" : doe vs {john,smith} → max ≈ 0.5 → REJECT
      "mariette jacobs" → "maretha jacobs" : mariette vs {maretha,jacobs} → max ≈ 0.85 → REJECT
    Accepts:
      "maria gonzalez" → "maria gonzalez" : all tokens match → ACCEPT
    """
    q_toks = [t for t in q_dn.split() if len(t) > 1]
    r_toks = [t for t in r_dn.split() if len(t) > 1]
    if not q_toks:
        return True  # trivial query, let it pass
    if not r_toks:
        return False  # ref with no useful tokens
    for qt in q_toks:
        best = max(_JW(qt, rt) for rt in r_toks)
        if best < threshold:
            return False
    return True
