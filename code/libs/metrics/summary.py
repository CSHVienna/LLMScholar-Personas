"""
Benchmark summaries for the figures: a metric registry, one table per configuration, and one estimation engine
(mean + confidence interval) for any rows (models, model characteristics, prompt variables) and any metrics.

Unit of analysis = the CONFIGURATION (a prompt with its runs), never the response: responses of one configuration
are not independent, so response metrics are first averaged within their configuration. Estimates:
    one model                 mean over its configurations; CI: bootstrap of its configurations
    a group of models / a     mean over the models of each model's mean (models weigh equally); CI: hierarchical
    prompt-variable level     bootstrap (1) models, (2) configurations within each sampled model
Percentile intervals over n_boot replicates, reproducible with the seed. Missing values (e.g. consistency with < 2
valid runs, factuality of an invalid response) are left out of their mean.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from metrics import technical as tm
from utils import constants as cons

HIGHER, LOWER, NEUTRAL = 1, -1, 0


@dataclass(frozen=True)
class Metric:
    key: str
    label: str                  # column header (matplotlib mathtext allowed)
    direction: int              # HIGHER / LOWER is better, or NEUTRAL (no arrow, no "best")
    family: str                 # technical | factuality | social
    group: str                  # subgroup within the family (e.g. parity, diversity) — used to select columns
    decimals: int = 2
    limits: Optional[tuple] = (0.0, 1.0)   # None: scaled to the data (e.g. effect sizes)
    short: str = ''                        # compact label under a subgroup bracket (heatmaps), e.g. 'Name'

    @property
    def band(self) -> str:
        """Top-level block of columns in the heatmaps: technical (incl. factuality) or social."""
        return 'social' if self.family == 'social' else 'technical'


METRICS: Dict[str, Metric] = {m.key: m for m in [
    Metric('refusals', 'Refusals', LOWER, 'technical', 'refusals', short='Refusals'),
    Metric('validity', 'Validity', HIGHER, 'technical', 'validity', short='Validity'),
    Metric('behaviour_consistency', 'Consist.$_{behav}$', NEUTRAL, 'technical', 'consistency', short='Behaviour'),
    Metric('recommendation_consistency', 'Consist.$_{recs}$', NEUTRAL, 'technical', 'consistency', short='Recs'),
    Metric('duplicates', 'Duplicates', LOWER, 'technical', 'duplicates', short='Duplicates'),
    Metric('k_ratio', 'k ratio', NEUTRAL, 'technical', 'k_ratio', limits=(0.0, 2.0), short='k ratio'),
    Metric('factuality_name', 'Fact.$_{name}$', HIGHER, 'factuality', 'factuality', short='Name'),
    Metric('factuality_name_field', 'Fact.$_{field}$', HIGHER, 'factuality', 'factuality', short='Field'),
    Metric('factuality_name_seniority', 'Fact.$_{seniority}$', HIGHER, 'factuality', 'factuality', short='Seniority'),
    Metric('factuality_author', 'Fact.$_{author}$', HIGHER, 'factuality', 'factuality', short='Author'),
]}
_SOCIAL_SHORT = {'gender': 'Gender', 'ethnicity': 'Ethnicity', 'location': 'Location', 'productivity': 'Works',
                 'influence': 'Citations', 'collaboration': 'Coauthors'}       # heatmap column labels
_SOCIAL_SUB = {'gender': 'gender', 'ethnicity': 'ethn', 'location': 'loc', 'productivity': 'works',
               'influence': 'cit', 'collaboration': 'coauth'}                 # subscripts of the bar headers
for _attribute in _SOCIAL_SHORT:
    METRICS[f"parity_{_attribute}"] = Metric(f"parity_{_attribute}", f"Par.$_{{{_SOCIAL_SUB[_attribute]}}}$", HIGHER,
                                             'social', 'parity', short=_SOCIAL_SHORT[_attribute])
for _attribute in _SOCIAL_SHORT:
    METRICS[f"diversity_{_attribute}"] = Metric(f"diversity_{_attribute}", f"Div.$_{{{_SOCIAL_SUB[_attribute]}}}$",
                                                HIGHER, 'social', 'diversity', short=_SOCIAL_SHORT[_attribute])
for _attribute in ('productivity', 'influence', 'location'):
    METRICS[f"popularity_{_attribute}"] = Metric(f"popularity_{_attribute}", f"Pop.$_{{{_SOCIAL_SUB[_attribute]}}}$",
                                                 NEUTRAL, 'social', 'popularity', short=_SOCIAL_SHORT[_attribute])
METRICS['connectedness'] = Metric('connectedness', 'Coh.$_{conn}$', NEUTRAL, 'social', 'cohesion', short='Connect.')
METRICS['similarity'] = Metric('similarity', 'Coh.$_{sim}$', NEUTRAL, 'social', 'cohesion', short='Sim.')
# Default columns of each family (the figures of the paper); social metrics are added with step 9
COLUMN_SETS = cons.METRIC_COLUMN_SETS          # named column sets: all, technical, factuality, technical_factuality, ...
FAMILY_DEFAULTS = COLUMN_SETS                    # (old name)
_unknown = sorted({k for keys in COLUMN_SETS.values() for k in keys} - set(METRICS))
if _unknown:
    raise ValueError(f"METRIC_COLUMN_SETS (utils/constants.py) names unknown metrics: {_unknown}")
CONFIG_METRICS = ['validity', 'refusals', 'behaviour_consistency', 'format_consistency', 'recommendation_consistency']
RESPONSE_METRICS = ['duplicates', 'k_ratio'] + [f"factuality_{level}" for level in tm.LEVELS]
SOCIAL_METRICS = [k for k, m in METRICS.items() if m.family == 'social']


def select_metrics(selection: str | Sequence[str]) -> List[Metric]:
    """A named column set (cons.METRIC_COLUMN_SETS: 'all', 'technical', 'technical_factuality', 'social', 'parity',
    ...), a subgroup, one metric key, or a list of metric keys."""
    if isinstance(selection, str):
        if selection in COLUMN_SETS:
            keys = COLUMN_SETS[selection]
        else:
            keys = [m.key for m in METRICS.values() if m.group == selection] or [selection]
    else:
        keys = list(selection)
    unknown = [k for k in keys if k not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; known: {list(METRICS)}")
    return [METRICS[k] for k in keys]


# --- rows --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RowGroup:
    """A block of rows: one row per value of `column` (in `levels` order), titled `title`, drawn in `color_key`."""
    title: str
    column: str
    levels: Sequence
    labels: Sequence[str]
    color_key: str


MODEL_ATTRIBUTES = {   # column of the model metadata -> (title, levels, labels)
    'access': ('Access', ['open-weight', 'proprietary'], ['Open', 'Proprietary']),
    'size': ('Size', ['T', 'S', 'M', 'L', 'XL'], ['Tiny', 'Small', 'Medium', 'Large', 'XL']),
    'reasoning': ('Reasoning', [True, False], ['True', 'False']),
    'origin': ('Origin', ['USA', 'FRA', 'CHN', 'ARE'], ['USA', 'FRA', 'CHN', 'ARE']),
}
PROMPT_TITLES = {cons.VAR_LANGUAGE: 'Language', cons.VAR_LOCATION: 'Location', cons.VAR_ROLE: 'Role', cons.VAR_SUBFIELD: 'Subfield',
                 cons.VAR_K: 'K', cons.VAR_FIELD: 'Field', cons.VAR_SENIORITY: 'Target'}
PROMPT_LABELS = {'Director/Recruiter': 'Recruiter', 'PhD student': 'Student', 'Computer Science': 'Computer Sc.'}
PERSONA, CONTEXT = 'persona', 'context'


def model_rows(models: Sequence[str], labels: Optional[Dict[str, str]] = None) -> List[RowGroup]:
    """One block with one row per model (in the given order)."""
    labels = labels or {}
    return [RowGroup('', cons.COL_MODEL, list(models), [labels.get(m, m) for m in models], 'model')]


def model_family_rows(models: Sequence[str], metadata: Optional[pd.DataFrame] = None,
                      labels: Optional[Dict[str, str]] = None) -> List[RowGroup]:
    """One block per model family (metadata 'family', in the order the families first appear in `models`), one row
    per model, each family in its own colour."""
    metadata = load_model_metadata() if metadata is None else metadata
    family_of = dict(zip(metadata[cons.COL_MODEL], metadata['family']))
    labels = labels or {}
    families: Dict[str, List[str]] = {}
    for model in models:
        families.setdefault(family_of.get(model, 'other'), []).append(model)
    return [RowGroup(family, cons.COL_MODEL, members, [labels.get(m, m) for m in members], f"family:{i}")
            for i, (family, members) in enumerate(families.items())]


def attribute_rows(attributes: Sequence[str] = tuple(MODEL_ATTRIBUTES)) -> List[RowGroup]:
    return [RowGroup(MODEL_ATTRIBUTES[a][0], a, MODEL_ATTRIBUTES[a][1], MODEL_ATTRIBUTES[a][2], a) for a in attributes]


def prompt_rows(variables: Sequence[str] = (cons.VAR_LANGUAGE, cons.VAR_LOCATION, cons.VAR_ROLE,
                                            cons.VAR_K, cons.VAR_FIELD, cons.VAR_SENIORITY)) -> List[RowGroup]:
    groups = []
    for v in variables:
        levels = list(cons.VARIABLE_VALUES[v])
        labels = [PROMPT_LABELS.get(str(x), x.capitalize() if isinstance(x, str) and x.islower() else str(x))
                  for x in levels]
        groups.append(RowGroup(PROMPT_TITLES.get(v, v), v, levels, labels,
                               PERSONA if v in cons.PERSONA_VARIABLES else CONTEXT))
    return groups


def load_model_metadata(path: Path | str = cons.DEFAULT_MODELS_METADATA_FILE) -> pd.DataFrame:
    """One row per model with the model characteristics (access, size, reasoning, origin, ...)."""
    raw = json.loads(Path(path).read_text())
    return pd.DataFrame([{cons.COL_MODEL: model, **{k: v for k, v in meta.items() if not isinstance(v, dict)}}
                         for model, meta in raw.items()])


def with_model_attributes(configs: pd.DataFrame, metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The configuration table with the model characteristics as columns (for attribute_rows)."""
    metadata = load_model_metadata() if metadata is None else metadata
    attributes = [c for c in metadata.columns if c != cons.COL_MODEL and c not in configs]
    merged = configs.merge(metadata[[cons.COL_MODEL] + attributes], on=cons.COL_MODEL, how='left')
    missing = sorted(merged.loc[merged[attributes[0]].isna(), cons.COL_MODEL].unique()) if attributes else []
    if missing:
        raise ValueError(f"Models without metadata ({cons.DEFAULT_MODELS_METADATA_FILE}): {missing}")
    return merged


# --- configuration table -----------------------------------------------------------------------------------------
def configuration_table(metrics_dir: Path | str, models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per configuration: the configuration metrics, and the response metrics averaged over the responses
    of the configuration (missing values left out)."""
    rows = []
    for config_file in sorted((Path(metrics_dir) / 'config').glob('*.parquet')):
        if models and config_file.stem not in models:
            continue
        config = pd.read_parquet(config_file)
        response = pd.read_parquet(Path(metrics_dir) / 'response' / config_file.name)
        social_file = Path(metrics_dir) / 'social' / config_file.name
        if social_file.exists():                       # social metrics (step 9), per response like the others
            social = pd.read_parquet(social_file)
            social['rid'], response['rid'] = tm.row_id(social, tm.RESPONSE_KEY), tm.row_id(response, tm.RESPONSE_KEY)
            response = response.merge(social[['rid'] + [c for c in SOCIAL_METRICS if c in social]], on='rid', how='left')
        key = tm.CONFIG_KEY
        config['cid'] = tm.row_id(config, key)
        response['cid'] = tm.row_id(response, key)
        per_config = response.groupby('cid')[[c for c in RESPONSE_METRICS + SOCIAL_METRICS if c in response]].mean()
        rows.append(config.join(per_config, on='cid').drop(columns='cid'))
    if not rows:
        raise FileNotFoundError(f"No metrics in {metrics_dir}/config/ (run run_metrics_pipeline.sh)")
    return pd.concat(rows, ignore_index=True)


# --- estimation --------------------------------------------------------------------------------------------------
def _model_draws(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot x K) means of one model's configurations resampled with replacement (NaN left out)."""
    picks = rng.integers(0, len(values), size=(n_boot, len(values)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(values[picks], axis=1)


def estimate(configs: pd.DataFrame, rows: Sequence[RowGroup], metrics: Sequence[Metric],
             n_boot: int = cons.DEFAULT_BOOTSTRAP_SAMPLES, confidence: float = cons.DEFAULT_CONFIDENCE,
             seed: int = cons.DEFAULT_SEED) -> pd.DataFrame:
    """Mean and CI of every metric for every row (long table: group, row, label, metric, mean, ci_low, ci_high,
    n_models, n_configs). A model row bootstraps its configurations; any other row is the mean of its models
    with a hierarchical bootstrap (models, then configurations)."""
    keys = [m.key for m in metrics]
    missing = [k for k in keys if k not in configs]
    if missing:
        raise ValueError(f"Metrics not in the configuration table: {missing}")
    alpha = (1 - confidence) / 2
    out = []
    for group in rows:
        for level, label in zip(group.levels, group.labels):
            subset = configs[configs[group.column] == level]
            per_model = {m: g[keys].to_numpy(dtype=float) for m, g in subset.groupby(cons.COL_MODEL, sort=True)}
            rng = np.random.default_rng(seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                if not per_model:
                    point = low = high = np.full(len(keys), np.nan)
                else:
                    means = np.vstack([np.nanmean(v, axis=0) for v in per_model.values()])     # (M x K)
                    draws = np.stack([_model_draws(v, n_boot, rng) for v in per_model.values()])  # (M x B x K)
                    point = np.nanmean(means, axis=0)
                    if group.column == cons.COL_MODEL or len(per_model) == 1:
                        boot = draws[0]
                    else:
                        picks = rng.integers(0, len(per_model), size=(n_boot, len(per_model)))
                        boot = np.nanmean(draws[picks, np.arange(n_boot)[:, None], :], axis=1)
                    low, high = np.nanquantile(boot, [alpha, 1 - alpha], axis=0)
            for i, key in enumerate(keys):
                out.append({'group': group.title, 'color_key': group.color_key, 'row': level, 'label': label,
                            'metric': key, 'mean': point[i], 'ci_low': low[i], 'ci_high': high[i],
                            'n_models': len(per_model), 'n_configs': len(subset)})
    return pd.DataFrame(out)
