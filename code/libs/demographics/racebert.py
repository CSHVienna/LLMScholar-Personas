"""
The raceBERT models (Parasurama 2022) shared by race.py and ethnicity.py: one inferrer class (model + tokenizer +
classification pipeline, on the GPU when there is one, else the CPU), the class probabilities per distinct name, and
the label rule (argmax class if its probability >= min_prob, else unknown).
"""

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils import constants as cons


def gpu_device():
    """The GPU for the pipeline's device= (CUDA 0, or Apple 'mps'), or None when there is none (CPU)."""
    import torch   # imported lazily (heavy)
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, 'mps', None)
    return 'mps' if mps is not None and mps.is_available() else None


def argmax_labels(probs: pd.DataFrame, columns: List[str], prefix: str, min_prob: float) -> List[str]:
    """Per row: the class of the most probable column (column name without ``prefix``) if its probability >= min_prob,
    else unknown (also when every probability is missing)."""
    out = []
    for row in probs[columns].to_numpy():
        if np.isnan(row).all():
            out.append(cons.UNKNOWN)
            continue
        best = int(np.nanargmax(row))
        out.append(columns[best][len(prefix):] if row[best] >= min_prob else cons.UNKNOWN)
    return out


class RaceBertInferrer:
    """Class probabilities (<prefix><class>) and labels of one raceBERT model, per normalised name.

    model_id: Hugging Face id; labels: model label -> class (every model label must be listed); normalise: full
    name -> the name the model reads; label_column: name of the label column of infer()."""

    def __init__(self, model_id: str, prefix: str, labels: Dict[str, str],
                 normalise: Callable[[Optional[str]], Optional[str]], label_column: str, min_prob: float,
                 batch_size: int = cons.DEFAULT_RACEBERT_BATCH_SIZE, torch_threads: Optional[int] = None):
        import torch   # imported lazily (heavy)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
        if torch_threads:
            torch.set_num_threads(torch_threads)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        unknown = set(model.config.id2label.values()) ^ set(labels)
        if unknown:
            raise ValueError(f"{model_id} labels differ from the configured ones: {sorted(unknown)}")
        device = gpu_device()
        kwargs = {} if device is None else {'device': device}       # no GPU: device not passed (CPU)
        self.device = device
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, **kwargs)
        self.prefix, self.labels, self.normalise, self.label_column = prefix, labels, normalise, label_column
        self.prob_columns = [f"{prefix}{c}" for c in dict.fromkeys(labels.values())]
        self.min_prob = min_prob
        self.batch_size = batch_size

    def probabilities(self, names: List[str]) -> List[Dict[str, float]]:
        """Class probabilities (<prefix><class>) per normalised name; one pipeline call for the whole list."""
        outputs = self.pipeline(names, top_k=None, batch_size=self.batch_size, truncation=True, max_length=64)
        return [{f"{self.prefix}{self.labels[o['label']]}": float(o['score']) for o in output} for output in outputs]

    def infer(self, full_names: Iterable[Optional[str]]) -> pd.DataFrame:
        """Columns: the class probabilities + the label (unique names inferred once)."""
        normalised = [self.normalise(n) for n in full_names]
        unique = sorted({n for n in normalised if n})
        lookup = dict(zip(unique, self.probabilities(unique))) if unique else {}
        empty = {column: np.nan for column in self.prob_columns}
        df = pd.DataFrame([lookup.get(n, empty) if n else empty for n in normalised], columns=self.prob_columns)
        df[self.label_column] = argmax_labels(df, self.prob_columns, self.prefix, self.min_prob)
        return df
