"""
Perceived race with the raceBERT race model (Parasurama 2022; Hugging Face 'pparasurama/raceBERT'; US census
categories, trained on Florida voter names), on the full name.

The name is normalised as the raceBERT package does ('first LAST': lower-case given names, upper-case last name;
'Last, First' is reordered first). The model was trained on Latin-script names and labels e.g. CJK-script names as
white, so names with non-Latin letters are transliterated first (unidecode: '明彦 田中' -> 'Ming Yan Tian Zhong');
Latin names keep their accents. Classes (cons.RACEBERT_RACE_LABELS): api -> asian, nh_black -> black, hispanic,
nh_white -> white, aian -> other (merged, too few). All five probabilities are kept (p_race_asian ... p_race_other).
Label (A.4): argmax class if its probability >= min_prob (default 0.5), unknown otherwise (or if the name is empty).

CPU only (~800 names/s with 18 threads); unique names are inferred once (demographics/stages.py).
"""

import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils import constants as cons
from utils.text import clean_text

MODEL = 'pparasurama/raceBERT'
PREFIX = 'p_race_'
CLASSES = list(dict.fromkeys(cons.RACEBERT_RACE_LABELS.values()))
PROB_COLUMNS = [f"{PREFIX}{c}" for c in CLASSES]
_NON_LATIN = re.compile(r'[^\W\d_](?<![\u0000-ɏḀ-ỿ])')   # a letter outside the Latin blocks
HANGUL_NAME = re.compile(r'[가-힣]{2,4}')                                # family name first, no space


def race_name(full_name: Optional[str]) -> Optional[str]:
    """'Maria Garcia' -> 'maria GARCIA' (raceBERT 'first LAST' strategy), after clean_text."""
    full_name = clean_text(full_name)
    if not full_name or not full_name.strip():
        return None
    name = full_name.strip()
    if _NON_LATIN.search(name):
        from unidecode import unidecode
        # Hangul names are written without a space, family name first: move it last ('김민수' -> '민수 김' -> 'minsu GIM')
        name = ' '.join(f"{t[1:]} {t[0]}" if HANGUL_NAME.fullmatch(t) else t for t in name.split())
        name = unidecode(name).strip()
        if not name:
            return None
    if ',' in name:
        last, _, first = name.partition(',')
        name = f"{first.strip()} {last.strip()}"
    tokens = name.split()
    if not tokens:                                  # e.g. ',' (no name left after reordering)
        return None
    return ' '.join([t.lower() for t in tokens[:-1]] + [tokens[-1].upper()])


def labels(probs: pd.DataFrame, columns: List[str], min_prob: float) -> List[str]:
    """Argmax class (column name without its prefix 'p_<model>_') if its probability >= min_prob, else unknown."""
    out = []
    for row in probs[columns].to_numpy():
        if np.isnan(row).all():
            out.append(cons.DEMOGRAPHICS_UNKNOWN)
            continue
        best = int(np.nanargmax(row))
        out.append(columns[best].split('_', 2)[2] if row[best] >= min_prob else cons.DEMOGRAPHICS_UNKNOWN)
    return out


class RaceInferrer:
    """The five class probabilities (PROB_COLUMNS) per normalised name (race_name); CPU."""

    def __init__(self, min_prob: float = cons.DEFAULT_RACE_MIN_PROB,
                 batch_size: int = cons.DEFAULT_RACEBERT_BATCH_SIZE, torch_threads: Optional[int] = None):
        import torch   # imported lazily (heavy)
        from transformers import AutoTokenizer, RobertaForSequenceClassification, TextClassificationPipeline
        if torch_threads:
            torch.set_num_threads(torch_threads)
        model = RobertaForSequenceClassification.from_pretrained(MODEL)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1)
        self.min_prob = min_prob
        self.batch_size = batch_size

    def probabilities(self, names: List[str]) -> List[Dict[str, float]]:
        """Class probabilities (p_race_<class>) per normalised name."""
        outputs = self.pipeline(names, top_k=None, batch_size=self.batch_size, truncation=True, max_length=64)
        return [{f"{PREFIX}{cons.RACEBERT_RACE_LABELS[o['label']]}": float(o['score']) for o in output}
                for output in outputs]

    def infer(self, full_names: Iterable[Optional[str]]) -> pd.DataFrame:
        """Columns: PROB_COLUMNS + race (label)."""
        normalised = [race_name(n) for n in full_names]
        unique = sorted({n for n in normalised if n})
        lookup = dict(zip(unique, self.probabilities(unique))) if unique else {}
        empty = {column: np.nan for column in PROB_COLUMNS}
        df = pd.DataFrame([lookup.get(n, empty) if n else empty for n in normalised], columns=PROB_COLUMNS)
        df['race'] = labels(df, PROB_COLUMNS, self.min_prob)
        return df
