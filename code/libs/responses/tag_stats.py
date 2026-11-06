"""
Statistics of tag combinations (inputs of the tag plots).

Tags are multi-label, so each response / recommendation is summarised by its *combination*
(the joined tags, e.g. 'embedded|wrapped'). A combination is **valid** if all its tags are in
the valid set (alone or combined). For readability, each family (valid / not valid) keeps its
``top_n`` most frequent combinations and folds the rest into "other valid" / "other invalid".
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils import constants as cons
from responses import tags as tag_utils

RESPONSE_KEY = [cons.COL_MODEL, cons.VAR_LANGUAGE, cons.COL_PROMPT_ID, cons.COL_RUN_ID]
CONFIGURATION_KEY = [cons.COL_MODEL, cons.VAR_LANGUAGE, cons.COL_PROMPT_ID]   # a prompt with its runs


def is_valid(joined_tags: str, valid_tags: Iterable[str]) -> bool:
    """True if every tag of the combination is in ``valid_tags`` (an empty combination is not valid)."""
    tags = tag_utils.split(joined_tags)
    return bool(tags) and set(tags) <= set(valid_tags)


@dataclass(frozen=True)
class CombinationCategories:
    """Display categories of tag combinations: valid first, then invalid, each by decreasing frequency."""
    categories: List[str]              # display order
    validity: Dict[str, bool]          # category -> valid?
    category_of: Dict[str, str]        # combination -> category (itself, or an "other" label)

    @classmethod
    def from_combinations(cls, combinations: pd.Series, valid_tags: Iterable[str],
                          top_n: int = cons.TOP_COMBINATIONS_PER_FAMILY) -> 'CombinationCategories':
        counts = combinations.value_counts()
        categories, validity, category_of = [], {}, {}
        for family_valid, other_label in ((True, cons.LABEL_OTHER_VALID), (False, cons.LABEL_OTHER_INVALID)):
            family = [c for c in counts.index if is_valid(c, valid_tags) == family_valid]  # already by frequency
            kept, folded = family[:top_n], family[top_n:]
            categories += kept + ([other_label] if folded else [])
            validity.update({category: family_valid for category in kept + [other_label]})
            category_of.update({c: c for c in kept} | {c: other_label for c in folded})
        return cls(categories, validity, category_of)

    @property
    def valid_categories(self) -> List[str]:
        return [c for c in self.categories if self.validity[c]]

    def assign(self, combinations: pd.Series) -> pd.Categorical:
        """Map combinations to their display category."""
        return pd.Categorical(combinations.map(self.category_of), categories=self.categories, ordered=True)


# -----------------------------------------------------------------------------
# Shares
# -----------------------------------------------------------------------------
def shares_by_group(df: pd.DataFrame, tags_column: str, categories: CombinationCategories,
                    by: str = cons.COL_MODEL, order: Optional[List[str]] = None) -> pd.DataFrame:
    """% of rows per category within each group (rows: groups, columns: categories).

    Groups are sorted by decreasing valid share, unless ``order`` (a list of groups) is given,
    e.g. to show the models of two figures in the same order.
    """
    table = pd.crosstab(df[by], categories.assign(df[tags_column]), normalize='index', dropna=False) * 100
    table = table.reindex(columns=categories.categories, fill_value=0.0)
    if order is not None:
        return table.reindex([group for group in order if group in table.index])
    valid_share = table[categories.valid_categories].sum(axis=1)
    return table.loc[valid_share.sort_values(ascending=False).index]


def overall_shares(df: pd.DataFrame, tags_column: str, categories: CombinationCategories) -> pd.DataFrame:
    """Number and % of rows per category over all rows."""
    counts = pd.Series(categories.assign(df[tags_column])).value_counts().reindex(categories.categories, fill_value=0)
    return pd.DataFrame({'category': categories.categories, 'n': counts.values,
                         'pct': 100 * counts.values / counts.sum(),
                         'is_valid': [categories.validity[c] for c in categories.categories]})


def model_mean_shares(df: pd.DataFrame, tags_column: str, categories: CombinationCategories,
                      n_boot: int = cons.DEFAULT_BOOTSTRAP_SAMPLES, confidence: float = cons.DEFAULT_CONFIDENCE,
                      seed: int = cons.DEFAULT_SEED) -> pd.DataFrame:
    """Mean over models of each model's share (%) per category, with a hierarchical bootstrap CI.

    Every model weighs equally, so the estimate is the average of the per-model bars. The CI
    resamples (1) models with replacement and (2) within each sampled model its configurations
    (a prompt with its runs, kept together), because responses of one model / configuration are
    not independent. Percentile interval over ``n_boot`` replicates (reproducible with ``seed``).
    Also returned: the std of the model shares (between-model heterogeneity, descriptive).
    """
    counts = _counts_per_configuration(df, tags_column, categories)   # {model: (n_configs x K)}
    rng = np.random.default_rng(seed)

    # 1. Point estimate: per-model shares, then their mean
    model_shares = np.vstack([_shares(c.sum(axis=0, keepdims=True))[0] for c in counts.values()])  # (M x K)

    # 2. Bootstrap, level 2: resample configurations within each model -> (M x B x K) shares
    boot_shares = np.stack([_shares(rng.multinomial(len(c), np.full(len(c), 1 / len(c)), size=n_boot) @ c)
                            for c in counts.values()])

    # 3. Bootstrap, level 1: resample models -> (B x K) means
    n_models = len(counts)
    picks = rng.integers(0, n_models, size=(n_boot, n_models))
    boot_means = boot_shares[picks, np.arange(n_boot)[:, None], :].mean(axis=1)

    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot_means, [alpha, 1 - alpha], axis=0)
    return pd.DataFrame({'category': categories.categories, 'mean': model_shares.mean(axis=0),
                         'ci_low': low, 'ci_high': high, 'std_models': model_shares.std(axis=0, ddof=1),
                         'n_models': n_models, 'n_rows': len(df),
                         'is_valid': [categories.validity[c] for c in categories.categories]})


def _counts_per_configuration(df: pd.DataFrame, tags_column: str,
                              categories: CombinationCategories) -> Dict[str, np.ndarray]:
    """Per model: counts per configuration (rows) and category (columns)."""
    table = pd.crosstab([df[c] for c in CONFIGURATION_KEY], categories.assign(df[tags_column]), dropna=False)
    table = table.reindex(columns=categories.categories, fill_value=0)
    table = table[table.sum(axis=1) > 0]
    return {model: group.to_numpy(dtype=float) for model, group in table.groupby(level=cons.COL_MODEL, observed=True)}


def _shares(counts: np.ndarray) -> np.ndarray:
    """Row-normalised counts in % (last axis = categories)."""
    totals = counts.sum(axis=-1, keepdims=True)
    return 100 * np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
