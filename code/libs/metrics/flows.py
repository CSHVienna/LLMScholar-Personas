"""
Flows: who the LLMs recommend, by prompt variable or model characteristic (descriptive; appendix figures).

People table (batch_flows.py, one file per model): one row per (valid response, factual person), where the factual
people are the matched authors (B.6 author_id) of the unique valid names of a valid response (metrics/social.py
response_people; a person named twice in a response counts once, a person recommended in several responses counts
each time). Each row carries the configuration (prompt variables, model) and the person's attributes from the GT:
perceived gender, perceived ethnicity, country (last known affiliation), the person's own seniority class
(author_seniority), productivity / influence / collaboration classes and the top-group flags, taken from the gt_members row of the REQUESTED field (another field's row if the person is not a member of it).

    flow_table(people, source, target)   counts and shares of target classes per source value (source: a prompt
                                         variable or a model characteristic; target: a person attribute)
    drift(people)                        prompt location -> the people's country: share of people from the prompt's
                                         country, the same share when other locations are prompted (baseline), and
                                         lift = share / baseline; cluster-bootstrap CIs over models
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from metrics import social as so
from metrics import technical as tm
from utils import constants as cons

UNKNOWN = so.UNKNOWN
AUTHOR_SENIORITY = 'author_seniority'   # the person's own seniority class (A.4), not the prompt's 'seniority'
PERSON_ATTRIBUTES = ['gender', 'ethnicity', 'country', AUTHOR_SENIORITY, 'productivity', 'influence', 'collaboration',
                     'is_top_productivity', 'is_top_influence']
CLASS_ATTRIBUTES = ['gender', 'ethnicity', 'country', AUTHOR_SENIORITY, 'productivity', 'influence', 'collaboration']
TARGETS = ['gender', 'ethnicity', 'country', AUTHOR_SENIORITY, 'productivity', 'influence']   # default panels
OTHER = 'other'
# prompt location -> ISO 3166-1 alpha-2 (OpenAlex country codes)
LOCATION_COUNTRY = {'South Africa': 'ZA', 'Germany': 'DE', 'Canada': 'CA', 'Ecuador': 'EC', 'Japan': 'JP',
                    'United States': 'US', 'USA': 'US', 'United Kingdom': 'GB', 'China': 'CN', 'India': 'IN',
                    'Brazil': 'BR', 'Mexico': 'MX', 'France': 'FR', 'Spain': 'ES', 'Nigeria': 'NG', 'Kenya': 'KE'}
ETHNICITY_ORDER = ['asian', 'black', 'hispanic', 'white', 'other']


# --- people ------------------------------------------------------------------------------------------------------
def people_table(responses: pd.DataFrame, recs: pd.DataFrame, matches: pd.DataFrame, gt: so.GroundTruth,
                 valid_response_tags: Sequence[str], valid_rec_tags: Sequence[str],
                 order_sensitive_keys: bool = False) -> pd.DataFrame:
    """RESPONSE_KEY + author_id + PERSON_ATTRIBUTES, one row per (valid response, factual person)."""
    people = so.response_people(responses, recs, matches, valid_response_tags, valid_rec_tags, order_sensitive_keys)
    if people.empty:
        return people.assign(**{a: pd.Series(dtype=object) for a in PERSON_ATTRIBUTES})
    members = gt.members(people['author_id'].unique()).rename(columns={'seniority': AUTHOR_SENIORITY})
    members = members[['author_id', 'field'] + PERSON_ATTRIBUTES]
    own = people.merge(members, on=['author_id', cons.VAR_FIELD], how='left')
    fallback = members.drop_duplicates('author_id').drop(columns='field').set_index('author_id')
    missing = own['gender'].isna() & own['author_id'].isin(fallback.index)
    for column in PERSON_ATTRIBUTES:
        own.loc[missing, column] = own.loc[missing, 'author_id'].map(fallback[column])
    for column in CLASS_ATTRIBUTES:
        own[column] = own[column].fillna(UNKNOWN).astype(str)
    return own[tm.RESPONSE_KEY + ['author_id'] + PERSON_ATTRIBUTES]


# --- flows -------------------------------------------------------------------------------------------------------
def top_classes(values: pd.Series, n: int) -> List[str]:
    """The n most frequent known classes (the rest becomes 'other')."""
    known = values[values != UNKNOWN]
    return list(known.value_counts().index[:n])


def flow_table(people: pd.DataFrame, source: str, target: str, include_unknown: bool = False,
               top: Optional[int] = None, keep: Sequence[str] = ()) -> pd.DataFrame:
    """Long table source value x target class: n, share (within the source value), n_source. n counts MENTIONS of
    factual people (rows of the people table: a person recommended in several responses counts each time). top: keep
    the top classes of the target (plus the classes in `keep`), the others pooled as 'other' (e.g. countries)."""
    df = people[[source, target]].copy()
    df[source] = df[source].astype(str)
    df[target] = df[target].astype(str)
    if not include_unknown:
        df = df[df[target] != UNKNOWN]
    if top:
        kept = set(top_classes(df[target], top)) | set(keep)
        df.loc[~df[target].isin(kept) & (df[target] != UNKNOWN), target] = OTHER
    counts = df.groupby([source, target]).size().rename('n').reset_index()
    counts['n_source'] = counts.groupby(source)['n'].transform('sum')
    counts['share'] = counts['n'] / counts['n_source']
    return counts.rename(columns={source: 'source', target: 'target'})


def unknown_share(people: pd.DataFrame, target: str) -> float:
    return float((people[target].astype(str) == UNKNOWN).mean()) if len(people) else np.nan


# --- drift -------------------------------------------------------------------------------------------------------
def location_codes(locations: Sequence[str]) -> Dict[str, str]:
    missing = [loc for loc in locations if loc not in LOCATION_COUNTRY]
    if missing:
        raise ValueError(f"No country code for the prompt locations {missing}: add them to flows.LOCATION_COUNTRY")
    return {loc: LOCATION_COUNTRY[loc] for loc in locations}


def _drift_point(people: pd.DataFrame, codes: Dict[str, str]) -> pd.DataFrame:
    known = people[people['country'] != UNKNOWN]
    rows = []
    for location, code in codes.items():
        prompted = known[known[cons.VAR_LOCATION] == location]
        others = known[known[cons.VAR_LOCATION] != location]
        share = float((prompted['country'] == code).mean()) if len(prompted) else np.nan
        baseline = float((others['country'] == code).mean()) if len(others) else np.nan
        rows.append({'location': location, 'country': code, 'n': len(prompted), 'share': share, 'baseline': baseline,
                     'lift': share / baseline if baseline and baseline > 0 else np.nan})
    return pd.DataFrame(rows)


def drift(people: pd.DataFrame, n_boot: int = cons.DEFAULT_BOOTSTRAP_SAMPLES, seed: int = cons.DEFAULT_SEED,
          confidence: float = cons.DEFAULT_CONFIDENCE) -> pd.DataFrame:
    """Per prompt location: share of the people from that country (known countries), baseline (same country when
    the other locations are prompted), lift; CIs: cluster bootstrap over models (percentile)."""
    codes = location_codes(sorted(people[cons.VAR_LOCATION].astype(str).unique()))
    point = _drift_point(people.assign(**{cons.VAR_LOCATION: people[cons.VAR_LOCATION].astype(str)}), codes)
    groups = {m: g for m, g in people.groupby(cons.COL_MODEL, observed=True)}
    models = list(groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = pd.concat([groups[m] for m in rng.choice(models, size=len(models), replace=True)], ignore_index=True)
        draws.append(_drift_point(sample.assign(**{cons.VAR_LOCATION: sample[cons.VAR_LOCATION].astype(str)}), codes))
    alpha = (1 - confidence) / 2
    if draws:
        boot = pd.concat(draws, ignore_index=True)
        for name in ('share', 'baseline', 'lift'):
            q = boot.groupby('location')[name].quantile([alpha, 1 - alpha]).unstack()
            point[f"{name}_ci_low"] = point['location'].map(q[alpha])
            point[f"{name}_ci_high"] = point['location'].map(q[1 - alpha])
    point['n_models'] = len(models)
    return point


def load_people(metrics_dir, models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """All <metrics_dir>/flows/<model>.parquet files (batch_flows.py)."""
    from pathlib import Path
    files = sorted((Path(metrics_dir) / 'flows').glob('*.parquet'))
    files = [f for f in files if not models or f.stem in models]
    if not files:
        raise FileNotFoundError(f"No people tables in {Path(metrics_dir) / 'flows'} (run run_metrics_pipeline.sh -s flows)")
    people = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    missing = [c for c in PERSON_ATTRIBUTES if c not in people]
    if missing:
        raise ValueError(f"The people tables in {Path(metrics_dir) / 'flows'} lack {missing} (written by an older "
                         f"version): re-run  run_metrics_pipeline.sh -s flows -x \"--overwrite\"")
    return people
