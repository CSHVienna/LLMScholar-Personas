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
from utils.stats import percentile_ci

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
    narrow: str = ''                       # header in wide bar figures (many columns), e.g. 'Dup.'; '' = label

    @property
    def band(self) -> str:
        """Top-level block of columns in the heatmaps: technical (incl. factuality) or social."""
        return cons.BAND_SOCIAL if self.family == cons.METRIC_SOCIAL else cons.BAND_TECHNICAL


METRICS: Dict[str, Metric] = {m.key: m for m in [
    Metric('refusals', 'Refusals', LOWER, cons.METRIC_TECHNICAL, 'refusals', short='Refusals'),
    Metric('validity', 'Validity', HIGHER, cons.METRIC_TECHNICAL, 'validity', short='Validity'),
    Metric('behaviour_consistency', 'Consist.$_{behav}$', NEUTRAL, cons.METRIC_TECHNICAL, 'consistency', short='Behaviour'),
    Metric('recommendation_consistency', 'Consist.$_{recs}$', NEUTRAL, cons.METRIC_TECHNICAL, 'consistency', short='Recs'),
    Metric('duplicates', 'Duplicates', LOWER, cons.METRIC_TECHNICAL, 'duplicates', short='Duplicat.', narrow='Dup.'),
    Metric('k_balance', 'k balance', NEUTRAL, cons.METRIC_TECHNICAL, 'k_balance', limits=(0.0, 2.0), short='k balance'),
    Metric('factuality_name', 'Fact.$_{name}$', HIGHER, cons.METRIC_FACTUALITY, 'factuality', short='Name'),
    Metric('factuality_name_field', 'Fact.$_{field}$', HIGHER, cons.METRIC_FACTUALITY, 'factuality', short='Field'),
    Metric('factuality_name_seniority', 'Fact.$_{seniority}$', HIGHER, cons.METRIC_FACTUALITY, 'factuality', short='Seniority'),
    Metric('factuality_author', 'Fact.$_{author}$', HIGHER, cons.METRIC_FACTUALITY, 'factuality', short='Author'),
]}
# Person attribute of a social metric -> (short: heatmap column label, sub: subscript of the bar header). The metric
# key 'location' is the scholar's COUNTRY (latest affiliation): shown as 'Country' ('Location' = the prompt variable);
# prominence attributes are counts: '# works' / '# citations' / '# coauthors' ('#' only in these short forms)
ATTRIBUTE_NAMES = {cons.ATTR_GENDER: ('Gender', 'gender'), cons.ATTR_RACE: ('Race', 'race'),
                   cons.ATTR_ETHNICITY: ('Ethnicity', 'ethn'), cons.ATTR_LOCATION: ('Country', 'country'),
                   cons.ATTR_PRODUCTIVITY: ('# works', r'{\#}works'), cons.ATTR_INFLUENCE: ('# citations', r'{\#}cit'),
                   cons.ATTR_COLLABORATION: ('# coauthors', r'{\#}coauth')}
_SOCIAL_HEADERS = {cons.PARITY: ('Par.', HIGHER), cons.DIVERSITY: ('Div.', HIGHER), cons.POPULARITY: ('Pop.', NEUTRAL)}
for _kind, (_head, _direction) in _SOCIAL_HEADERS.items():
    _attributes = cons.POPULARITY_ATTRIBUTES if _kind == cons.POPULARITY else cons.SOCIAL_METRIC_ATTRIBUTES
    for _key, _attribute in zip(cons.social_metric_keys(_kind, _attributes), _attributes):
        _short, _sub = ATTRIBUTE_NAMES[_attribute]
        METRICS[_key] = Metric(_key, f"{_head}$_{{{_sub}}}$", _direction, cons.METRIC_SOCIAL, _kind, short=_short)
METRICS['connectedness'] = Metric('connectedness', 'Coh.$_{conn}$', NEUTRAL, cons.METRIC_SOCIAL, 'cohesion', short='Connect.')
METRICS['similarity'] = Metric('similarity', 'Coh.$_{sim}$', NEUTRAL, cons.METRIC_SOCIAL, 'cohesion', short='Similar.')
METRICS['locality'] = Metric('locality', 'Locality', NEUTRAL, cons.METRIC_SOCIAL, 'locality', short='Locality', narrow='Loc.')
# Size-adjusted social metrics (metrics/expected.py): <key>_adj = observed - expected under a random recommender;
# group <group>_adj (so 'parity' still selects the raw ones); differences, so scaled to the data


def adjusted_label(label: str) -> str:
    """Header of a size-adjusted metric: 'Par.$_{gender}$' -> 'Par.$^{adj}_{gender}$', 'Loc.' -> 'Loc.$^{adj}$'."""
    return label.replace('$_{', '$^{adj}_{') if '$_{' in label else label + '$^{adj}$'


for _key in [k for k, m in METRICS.items() if m.family == cons.METRIC_SOCIAL]:
    _m = METRICS[_key]
    METRICS[_key + cons.ADJ_SUFFIX] = Metric(_key + cons.ADJ_SUFFIX, adjusted_label(_m.label), _m.direction, cons.METRIC_SOCIAL,
                                        _m.group + cons.ADJ_SUFFIX, _m.decimals, None, _m.short,
                                        adjusted_label(_m.narrow) if _m.narrow else '')
# Default columns of each family (the figures of the paper); social metrics are added with step 9
COLUMN_SETS = cons.METRIC_COLUMN_SETS          # named column sets: all, technical, factuality, social, parity, paper, ...
_unknown = sorted({k for keys in COLUMN_SETS.values() for k in keys} - set(METRICS))
if _unknown:
    raise ValueError(f"METRIC_COLUMN_SETS (utils/constants.py) names unknown metrics: {_unknown}")
CONFIG_METRICS = ['validity', 'refusals', 'behaviour_consistency', 'format_consistency', 'recommendation_consistency']
RESPONSE_METRICS = ['duplicates', 'k_balance'] + [f"factuality_{level}" for level in tm.LEVELS]
SOCIAL_METRICS = [k for k, m in METRICS.items() if m.family == cons.METRIC_SOCIAL]
SOCIAL_EXPECTED = [k + cons.EXPECTED_SUFFIX for k in SOCIAL_METRICS if not k.endswith(cons.ADJ_SUFFIX)]   # RQ3 adj ranges


def use_study_ethnicity(attribute: str) -> List[str]:
    """Point the named metric sets (COLUMN_SETS = cons.METRIC_COLUMN_SETS) and cons.RQ3_SOCIAL_METRICS at the study's
    perceived-ethnicity attribute (analysis config study.ethnicity_attribute: ethnicity | race | both), in place, so
    every later selection uses it. Call once, right after loading the config. Returns the study attributes."""
    sets = cons.metric_column_sets(attribute)
    COLUMN_SETS.clear()
    COLUMN_SETS.update(sets)
    cons.RQ3_SOCIAL_METRICS[:] = sets['parity']
    return cons.study_ethnicity_attributes(attribute)


def adjusted_keys(keys: Sequence[str]) -> List[str]:
    """The keys with every social metric replaced by its size-adjusted version (<key>_adj); others unchanged."""
    return [k + cons.ADJ_SUFFIX if k in METRICS and METRICS[k].family == cons.METRIC_SOCIAL and not k.endswith(cons.ADJ_SUFFIX)
            and k + cons.ADJ_SUFFIX in METRICS else k for k in keys]


def _adjusted_item(item: str) -> str:
    """One selection item size-adjusted: a set / subgroup name -> '<name>_adj' (if it exists), a key -> '<key>_adj'."""
    if item + cons.ADJ_SUFFIX in COLUMN_SETS or any(m.group == item + cons.ADJ_SUFFIX for m in METRICS.values()):
        return item + cons.ADJ_SUFFIX
    return adjusted_keys([item])[0]


def adjusted_selection(selection, adjusted: bool):
    """A column selection with its social metrics size-adjusted if ``adjusted``: every item (set name, subgroup or key)
    -> its '_adj' version where one exists (None = all -> 'all_adj'); unchanged if not adjusted (notebooks: ADJUSTED)."""
    if not adjusted:
        return selection
    if selection is None:
        return 'all_adj'
    if isinstance(selection, str):
        return _adjusted_item(selection)
    return [_adjusted_item(item) for item in selection]


def selection_name(selection) -> str:
    """File-name part of a column selection, with '_adj' moved to the END so a figure and its size-adjusted twin sort
    together: 'paper' -> 'paper', 'paper_adj' -> 'paper' + '_adj' (the caller appends the rest before it)."""
    base, adjusted = split_adjusted(selection)
    return base + (cons.ADJ_SUFFIX if adjusted else '')


def _items(selection) -> List[str]:
    return [selection] if isinstance(selection, str) else list(selection)


def split_adjusted(selection, exclude=None) -> tuple:
    """(name without _adj, adjusted?) of a selection: None = 'all', else its items joined by '-' (e.g.
    'technical-factuality'); excluded items appended as '-no-<items>' (e.g. 'technical-factuality-no-consistency')."""
    if selection is None:
        return 'all', False
    items = _items(selection)
    adjusted = any(k.endswith(cons.ADJ_SUFFIX) for k in items)
    name = '-'.join(k.removesuffix(cons.ADJ_SUFFIX) for k in items)
    if exclude:
        name += '-no-' + '-'.join(k.removesuffix(cons.ADJ_SUFFIX) for k in _items(exclude))
    return name, adjusted


def _expand(item: str) -> List[str]:
    """The keys of one selection item: a named set, a subgroup, or a metric key."""
    if item in COLUMN_SETS:
        return list(COLUMN_SETS[item])
    return [m.key for m in METRICS.values() if m.group == item] or [item]


def select_metrics(selection: str | Sequence[str], adjusted: bool = False,
                   exclude: Optional[str | Sequence[str]] = None) -> List[Metric]:
    """A named column set (cons.METRIC_COLUMN_SETS: 'all', 'technical', 'social', 'parity', 'paper', ...), a subgroup
    ('consistency', 'parity_adj', ...), one metric key, or a list mixing them (expanded in order, duplicates once).
    adjusted: the social metrics as <key>_adj. exclude: items of the same kinds to leave out; excluding a raw key also
    drops its _adj twin (e.g. exclude='parity_gender' with 'parity_adj')."""
    keys = list(dict.fromkeys(k for item in _items(selection) for k in _expand(item)))
    if adjusted:
        keys = adjusted_keys(keys)
    if exclude:
        drop = {k for item in _items(exclude) for k in _expand(item)}
        keys = [k for k in keys if k not in drop and k.removesuffix(cons.ADJ_SUFFIX) not in drop]
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
    cons.MODEL_ACCESS: ('Access', ['open-weight', 'proprietary'], ['Open', 'Proprietary']),
    cons.MODEL_SIZE: ('Size', ['T', 'S', 'M', 'L', 'XL'], ['Tiny', 'Small', 'Medium', 'Large', 'Extra large']),
    cons.MODEL_REASONING: ('Reasoning', [True, False], ['Yes', 'No']),
    cons.MODEL_ORIGIN: ('Origin', ['USA', 'FRA', 'CHN', 'ARE'], ['USA', 'FRA', 'CHN', 'ARE']),
}
PROMPT_TITLES = {cons.VAR_LANGUAGE: 'Language', cons.VAR_LOCATION: 'Location', cons.VAR_ROLE: 'Role', cons.VAR_SUBFIELD: 'Subfield',
                 cons.VAR_K: 'k', cons.VAR_FIELD: 'Field', cons.VAR_SENIORITY: 'Seniority'}   # display titles
PROMPT_LABELS = {'Director/Recruiter': 'Recruiter', 'PhD student': 'Student', 'Computer Science': 'Computer Sc.'}


def level_label(value) -> str:
    """Display label of a prompt-variable value: PROMPT_LABELS, else lower-case values capitalised ('english' ->
    'English'), others as they are ('Computer Science', 'US')."""
    text = str(value)
    return PROMPT_LABELS.get(text, text.capitalize() if text.islower() else text)


def model_rows(models: Sequence[str], labels: Optional[Dict[str, str]] = None) -> List[RowGroup]:
    """One block with one row per model (in the given order)."""
    labels = labels or {}
    return [RowGroup('', cons.COL_MODEL, list(models), [labels.get(m, m) for m in models], cons.FAMILY_MODEL)]


def model_family_rows(models: Sequence[str], metadata: Optional[pd.DataFrame] = None,
                      labels: Optional[Dict[str, str]] = None) -> List[RowGroup]:
    """One block per model family (metadata 'family', in the order the families first appear in `models`), one row
    per model; every row in the model colour (the blocks are titled and separated by rules)."""
    metadata = load_model_metadata() if metadata is None else metadata
    family_of = dict(zip(metadata[cons.COL_MODEL], metadata[cons.MODEL_FAMILY]))
    labels = labels or {}
    families: Dict[str, List[str]] = {}
    for model in models:
        families.setdefault(family_of.get(model, 'other'), []).append(model)
    return [RowGroup(family, cons.COL_MODEL, members, [labels.get(m, m) for m in members], cons.FAMILY_MODEL)
            for family, members in families.items()]


def attribute_rows(attributes: Sequence[str] = tuple(MODEL_ATTRIBUTES)) -> List[RowGroup]:
    return [RowGroup(MODEL_ATTRIBUTES[a][0], a, MODEL_ATTRIBUTES[a][1], MODEL_ATTRIBUTES[a][2], a) for a in attributes]


def prompt_rows(variables: Sequence[str] = tuple(v for v in cons.CONFIG_VARIABLES if v != cons.VAR_SUBFIELD)
                ) -> List[RowGroup]:
    groups = []
    for v in variables:
        levels = list(cons.VARIABLE_VALUES[v])
        groups.append(RowGroup(PROMPT_TITLES.get(v, v), v, levels, [level_label(x) for x in levels],
                               cons.variable_family(v)))
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
    of the configuration (missing values left out); also <social metric>_expected (SOCIAL_EXPECTED, for the RQ3
    ranges of the adjusted metrics)."""
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
            response = response.merge(social[['rid'] + [c for c in SOCIAL_METRICS + SOCIAL_EXPECTED if c in social]],
                                      on='rid', how='left')
        key = tm.CONFIG_KEY
        config['cid'] = tm.row_id(config, key)
        response['cid'] = tm.row_id(response, key)
        per_config = response.groupby('cid')[[c for c in RESPONSE_METRICS + SOCIAL_METRICS + SOCIAL_EXPECTED
                                              if c in response]].mean()
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
                    low, high = percentile_ci(boot, confidence, axis=0, skipna=True)
            for i, key in enumerate(keys):
                out.append({'group': group.title, 'color_key': group.color_key, 'row': level, 'label': label,
                            'metric': key, 'mean': point[i], 'ci_low': low[i], 'ci_high': high[i],
                            'n_models': len(per_model), 'n_configs': len(subset)})
    return pd.DataFrame(out)
