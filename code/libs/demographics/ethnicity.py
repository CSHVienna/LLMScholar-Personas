"""
Perceived ethnicity with the raceBERT ethnicity model ('pparasurama/raceBERT-ethnicity'; 13 Wikipedia name classes,
Ambekar et al. 2009 taxonomy, cons.RACEBERT_ETHNICITY_LABELS), on the full name: what
racebert.RaceBERT().predict_ethnicity runs, but with ALL 13 scores (p_eth_<class>), batches and a GPU when there is one.

Name as the package normalises it ('first_last': lower case, words joined by '_'), in the ORIGINAL script: no
transliteration (its vocabulary covers CJK, Cyrillic, Arabic; unidecode gives kanji Chinese readings and drops Arabic
vowels, which loses the Japanese / Muslim signal). 'Last, First' and Hangul names (family name first, no space) are
put in first-last order. Label (A.4): argmax class if its probability >= min_prob, unknown otherwise.
"""

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from demographics.race import HANGUL_NAME, labels
from utils import constants as cons
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
    name = ' '.join(f"{t[1:]} {t[0]}" if HANGUL_NAME.fullmatch(t) else t for t in full_name.split())
    if ',' in name:
        last, _, first = name.partition(',')
        name = f"{first.strip()} {last.strip()}"
    tokens = name.split()
    return '_'.join(tokens).lower() if tokens else None


def gpu_device():
    """The GPU for the pipeline's device= (CUDA 0, or Apple 'mps'), or None when there is none (CPU)."""
    import torch
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, 'mps', None)
    return 'mps' if mps is not None and mps.is_available() else None


class EthnicityInferrer:
    """All 13 class probabilities (PROB_COLUMNS) per normalised name (ethnicity_name); GPU if there is one."""

    def __init__(self, min_prob: float = cons.DEFAULT_ETHNICITY_MIN_PROB,
                 batch_size: int = cons.DEFAULT_RACEBERT_BATCH_SIZE, torch_threads: Optional[int] = None):
        import torch   # imported lazily (heavy)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
        if torch_threads:
            torch.set_num_threads(torch_threads)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        unknown = set(model.config.id2label.values()) ^ set(cons.RACEBERT_ETHNICITY_LABELS)
        if unknown:
            raise ValueError(f"raceBERT ethnicity labels differ from cons.RACEBERT_ETHNICITY_LABELS: {sorted(unknown)}")
        device = gpu_device()
        kwargs = {} if device is None else {'device': device}       # no GPU: device not passed (CPU)
        self.device = device
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, **kwargs)
        self.min_prob = min_prob
        self.batch_size = batch_size

    def probabilities(self, names: List[str]) -> List[Dict[str, float]]:
        """All 13 class probabilities (p_eth_<class>) per normalised name; one pipeline call for the whole list."""
        outputs = self.pipeline(names, top_k=None, batch_size=self.batch_size, truncation=True, max_length=64)
        return [{f"{PREFIX}{cons.RACEBERT_ETHNICITY_LABELS[o['label']]}": float(o['score']) for o in output}
                for output in outputs]

    def infer(self, full_names: Iterable[Optional[str]]) -> pd.DataFrame:
        """Columns: PROB_COLUMNS + ethnicity (label)."""
        normalised = [ethnicity_name(n) for n in full_names]
        unique = sorted({n for n in normalised if n})
        lookup = dict(zip(unique, self.probabilities(unique))) if unique else {}
        empty = {column: np.nan for column in PROB_COLUMNS}
        df = pd.DataFrame([lookup.get(n, empty) if n else empty for n in normalised], columns=PROB_COLUMNS)
        df['ethnicity'] = labels(df, PROB_COLUMNS, self.min_prob)
        return df
