"""
Perceived ethnicity with the raceBERT ethnicity model ('pparasurama/raceBERT-ethnicity'; 13 Wikipedia name classes,
Ambekar et al. 2009 taxonomy, cons.RACEBERT_ETHNICITY_LABELS), on the full name: what
racebert.RaceBERT().predict_ethnicity runs, but with ALL 13 scores (p_eth_<class>), batches and a GPU when there is one.

Name as the package normalises it ('first_last': lower case, words joined by '_'), in the ORIGINAL script: no
transliteration (its vocabulary covers CJK, Cyrillic, Arabic; unidecode gives kanji Chinese readings and drops Arabic
vowels, which loses the Japanese / Muslim signal). 'Last, First' and Hangul names (family name first, no space) are
put in first-last order. Label (A.4): argmax class if its probability >= min_prob, unknown otherwise.
"""

from typing import Optional

from demographics.racebert import RaceBertInferrer
from utils import constants as cons
from utils.names import first_last_order, hangul_family_last
from utils.text import clean_text

MODEL = 'pparasurama/raceBERT-ethnicity'
PREFIX = 'p_eth_'
CLASSES = list(cons.RACEBERT_ETHNICITY_LABELS.values())
PROB_COLUMNS = [f"{PREFIX}{c}" for c in CLASSES]


def ethnicity_name(full_name: Optional[str]) -> Optional[str]:
    """'Maria Garcia' -> 'maria_garcia' (the package's 'first_last' strategy) after clean_text, in the original script."""
    full_name = clean_text(full_name)
    if not full_name or not full_name.strip():
        return None
    tokens = first_last_order(hangul_family_last(full_name)).split()
    return '_'.join(tokens).lower() if tokens else None


class EthnicityInferrer(RaceBertInferrer):
    """All 13 class probabilities (PROB_COLUMNS) and the ethnicity label per normalised name (ethnicity_name)."""

    def __init__(self, min_prob: float = cons.DEFAULT_ETHNICITY_MIN_PROB,
                 batch_size: int = cons.DEFAULT_RACEBERT_BATCH_SIZE, torch_threads: Optional[int] = None):
        super().__init__(MODEL, PREFIX, cons.RACEBERT_ETHNICITY_LABELS, ethnicity_name, 'ethnicity', min_prob,
                         batch_size, torch_threads)
