"""
oa_common.py — What the OpenAlex steps share, kept apart so other folders can
import it (a file whose name starts with a number cannot be imported).

  normalize_name   how names are compared with OpenAlex (also used by
                   scripts/1_ground_truth/ and the affiliation match)
  WORKS_TMP_DIR    DuckDB temp folder ([data].oa_works_tmp_dir)
  WORKS_AGG_DIR    the per-year chunks of country and institution per author,
                   written by 1_openalex.py and read by 3_affiliation.py
"""

import os
import re
import unicodedata

from libs.utils.config import config_default

WORKS_TMP_DIR = config_default("oa_works_tmp_dir") or "."
WORKS_AGG_DIR = os.path.join(WORKS_TMP_DIR, "oa_works_agg_chunks")

_NON_LETTER_RE = re.compile(r"[^a-z\s]")


def normalize_name(name: str) -> str:
    """Lowercase, strip accents (NFD), keep only letters and single spaces."""
    if not isinstance(name, str) or not name.strip():
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = _NON_LETTER_RE.sub(" ", name.lower())
    return " ".join(name.split())
