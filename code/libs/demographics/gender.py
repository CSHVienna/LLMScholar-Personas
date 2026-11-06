"""
Perceived gender with nomquamgender (Van Buskirk et al. 2023), on the FIRST NAME only.

nomquamgender also looks up whole strings: given a full name it can hit a multi-token reference entry with
zero counts (e.g. 'John Smith' -> p(gf) = 0.0, classified male), so only the first name is passed. The first name is
the first non-initial token before the last name ('J. Michael Smith' -> 'Michael'; 'Smith, John' -> 'John').
Names without counts in the reference get p_female = NaN -> unknown.

Label: female if p_female >= .5, male otherwise, but only if the uncertainty 0.5 - |0.5 - p| <= threshold
(nomquamgender's rule; default 0.1); unknown otherwise.
"""

import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from utils import constants as cons
from utils.text import clean_text

_TOKEN = re.compile(r"[^\W\d_]", re.UNICODE)   # a letter


def first_name_of(full_name: Optional[str]) -> Optional[str]:
    """First non-initial given-name token of a full name (after clean_text), or None."""
    full_name = clean_text(full_name)
    if not full_name or not full_name.strip():
        return None
    name = full_name.strip()
    if ',' in name:                                 # 'Last, First'
        last, _, first = name.partition(',')
        name = f"{first.strip()} {last.strip()}"
    tokens = name.split()
    given = tokens[:-1] if len(tokens) > 1 else tokens
    for token in given:
        letters = ''.join(_TOKEN.findall(token))
        if len(letters) > 1 and not re.fullmatch(r"(?:[^\W\d_]\.)+[^\W\d_]?\.?", token):   # skip 'J.', 'J.M.'
            return token.strip('.,;:()"\'')
    return None


class GenderInferrer:
    def __init__(self, threshold: float = cons.DEFAULT_GENDER_UNCERTAINTY):
        import nomquamgender as nqg   # imported lazily (only needed by this step)
        self.model = nqg.NBGC()
        self.threshold = threshold

    def p_female(self, first_names: Iterable[Optional[str]]) -> np.ndarray:
        """p(female) per first name (NaN if missing or not in the reference); unique names looked up once."""
        names = [n.lower() if isinstance(n, str) and n else None for n in first_names]
        unique = sorted({n for n in names if n})
        lookup = {}
        if unique:
            for given, _used, _sources, counts, pgf in self.model.annotate(unique):
                lookup[given] = float(pgf) if counts and pgf == pgf else np.nan
        return np.array([lookup.get(n, np.nan) if n else np.nan for n in names], dtype=float)

    def labels(self, p_female: np.ndarray) -> List[str]:
        uncertainty = 0.5 - np.abs(0.5 - p_female)
        return [cons.GENDER_UNKNOWN if np.isnan(p) or u > self.threshold
                else cons.GENDER_FEMALE if p >= 0.5 else cons.GENDER_MALE
                for p, u in zip(p_female, uncertainty)]

    def infer(self, first_names: Iterable[Optional[str]]) -> pd.DataFrame:
        """Columns: gender_first_name, p_female, gender."""
        first_names = list(first_names)
        p = self.p_female(first_names)
        return pd.DataFrame({'gender_first_name': first_names, 'p_female': p, 'gender': self.labels(p)})

    def infer_full_names(self, full_names: Iterable[Optional[str]]) -> pd.DataFrame:
        return self.infer([first_name_of(n) for n in full_names])
