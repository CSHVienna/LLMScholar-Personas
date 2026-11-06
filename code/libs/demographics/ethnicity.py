"""
Perceived ethnicity with raceBERT (Parasurama 2022; Hugging Face model 'pparasurama/raceBERT'), on the full name.

The name is normalised as the raceBERT package does ('first LAST': lower-case given names, upper-case last name;
'Last, First' is reordered first). raceBERT was trained on Latin-script names and labels e.g. CJK-script names as
white, so names with non-Latin letters are transliterated first (unidecode: '明彦 田中' -> 'Ming Yan Tian Zhong');
Latin names keep their accents. Classes: api -> asian, nh_black -> black, hispanic, nh_white -> white,
aian -> other (merged, too few). All five probabilities are kept (p_asian ... p_other).
Label: argmax class if its probability >= min_prob (default 0.5), unknown otherwise (or if the name is empty).

CPU only here (~800 names/s with 18 threads); unique names are inferred once.
"""

import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils import constants as cons
from utils.text import clean_text

RACEBERT_MODEL = 'pparasurama/raceBERT'
_NON_LATIN = re.compile(r'[^\W\d_](?<![\u0000-\u024F\u1E00-\u1EFF])')   # a letter outside the Latin blocks
_HANGUL_NAME = re.compile(r'[가-힣]{2,4}')
PROB_COLUMNS = [f"p_{label}" for label in dict.fromkeys(cons.RACEBERT_LABELS.values())]


def racebert_name(full_name: Optional[str]) -> Optional[str]:
    """'Maria Garcia' -> 'maria GARCIA' (raceBERT 'first LAST' strategy), after clean_text."""
    full_name = clean_text(full_name)
    if not full_name or not full_name.strip():
        return None
    name = full_name.strip()
    if _NON_LATIN.search(name):
        from unidecode import unidecode
        # Hangul names are written without a space, family name first: move it last ('김민수' -> '민수 김' -> 'minsu GIM')
        name = ' '.join(f"{t[1:]} {t[0]}" if _HANGUL_NAME.fullmatch(t) else t for t in name.split())
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


class EthnicityInferrer:
    def __init__(self, min_prob: float = cons.DEFAULT_ETHNICITY_MIN_PROB,
                 batch_size: int = cons.DEFAULT_RACEBERT_BATCH_SIZE, torch_threads: Optional[int] = None):
        import torch   # imported lazily (heavy)
        from transformers import AutoTokenizer, RobertaForSequenceClassification, TextClassificationPipeline
        if torch_threads:
            torch.set_num_threads(torch_threads)
        model = RobertaForSequenceClassification.from_pretrained(RACEBERT_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(RACEBERT_MODEL, use_fast=True)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1)
        self.min_prob = min_prob
        self.batch_size = batch_size

    def probabilities(self, names: List[str]) -> List[Dict[str, float]]:
        """Class probabilities (our class names) per normalised name."""
        outputs = self.pipeline(names, top_k=None, batch_size=self.batch_size, truncation=True, max_length=64)
        return [{f"p_{cons.RACEBERT_LABELS[o['label']]}": float(o['score']) for o in output} for output in outputs]

    def infer(self, full_names: Iterable[Optional[str]]) -> pd.DataFrame:
        """Columns: p_asian, p_black, p_hispanic, p_white, p_other, ethnicity."""
        normalised = [racebert_name(n) for n in full_names]
        unique = sorted({n for n in normalised if n})
        lookup = dict(zip(unique, self.probabilities(unique))) if unique else {}
        empty = {column: np.nan for column in PROB_COLUMNS}
        df = pd.DataFrame([lookup.get(n, empty) if n else empty for n in normalised], columns=PROB_COLUMNS)
        df['ethnicity'] = self.labels(df)
        return df

    def labels(self, probs: pd.DataFrame) -> List[str]:
        values = probs[PROB_COLUMNS].to_numpy()
        labels = []
        for row in values:
            if np.isnan(row).all():
                labels.append(cons.ETHNICITY_UNKNOWN)
                continue
            best = int(np.nanargmax(row))
            labels.append(PROB_COLUMNS[best][2:] if row[best] >= self.min_prob else cons.ETHNICITY_UNKNOWN)
        return labels
