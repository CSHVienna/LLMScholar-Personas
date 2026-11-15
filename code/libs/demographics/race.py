"""
Perceived race with the raceBERT race model (Parasurama 2022; Hugging Face 'pparasurama/raceBERT'; US census
categories, trained on Florida voter names), on the full name.

The name is normalised as the raceBERT package does ('first LAST': lower-case given names, upper-case last name;
'Last, First' is reordered first). The model was trained on Latin-script names and labels e.g. CJK-script names as
white, so names with non-Latin letters are transliterated first (unidecode: '明彦 田中' -> 'Ming Yan Tian Zhong');
Latin names keep their accents. Classes (cons.RACEBERT_RACE_LABELS): api -> asian, nh_black -> black, hispanic,
nh_white -> white, aian -> other (merged, too few). All five probabilities are kept (p_race_asian ... p_race_other).
Label (A.4): argmax class if its probability >= min_prob (default 0.5), unknown otherwise (or if the name is empty).

GPU if there is one, else CPU (~800 names/s with 18 threads); unique names are inferred once (demographics/stages.py).
"""

import re
from typing import Optional

from demographics.racebert import RaceBertInferrer
from utils import constants as cons
from utils.names import first_last_order, hangul_family_last
from utils.text import clean_text

MODEL = 'pparasurama/raceBERT'
PREFIX = 'p_race_'
CLASSES = list(dict.fromkeys(cons.RACEBERT_RACE_LABELS.values()))
PROB_COLUMNS = [f"{PREFIX}{c}" for c in CLASSES]
_NON_LATIN = re.compile(r'[^\W\d_](?<![\u0000-ɏḀ-ỿ])')   # a letter outside the Latin blocks


def race_name(full_name: Optional[str]) -> Optional[str]:
    """'Maria Garcia' -> 'maria GARCIA' (raceBERT 'first LAST' strategy), after clean_text."""
    full_name = clean_text(full_name)
    if not full_name or not full_name.strip():
        return None
    name = full_name.strip()
    if _NON_LATIN.search(name):
        from unidecode import unidecode   # imported lazily (only needed by this step)
        # Hangul names are written without a space, family name first: move it last ('김민수' -> '민수 김' -> 'minsu GIM')
        name = unidecode(hangul_family_last(name)).strip()
        if not name:
            return None
    tokens = first_last_order(name).split()
    if not tokens:                                  # e.g. ',' (no name left after reordering)
        return None
    return ' '.join([t.lower() for t in tokens[:-1]] + [tokens[-1].upper()])


class RaceInferrer(RaceBertInferrer):
    """The five class probabilities (PROB_COLUMNS) and the race label per normalised name (race_name)."""

    def __init__(self, min_prob: float = cons.DEFAULT_RACE_MIN_PROB,
                 batch_size: int = cons.DEFAULT_RACEBERT_BATCH_SIZE, torch_threads: Optional[int] = None):
        super().__init__(MODEL, PREFIX, cons.RACEBERT_RACE_LABELS, race_name, 'race', min_prob, batch_size,
                         torch_threads)
