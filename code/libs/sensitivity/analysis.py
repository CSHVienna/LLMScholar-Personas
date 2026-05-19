"""SensitivityAnalysis: orchestrates the full sensitivity pipeline.

Workflow:
    1. __init__:        declare metrics + a base spec.
    2. prepare_data:    clean, impute, standardize, categoricalize once.
    3. fixed_effects:   fit Model 1 (FixedEffectsModel) per metric.
    4. mixed_effects:   fit Model 2 single-moderator models per metric.
    5. save / load:     portable on-disk persistence (JSON + parquet).
"""

from dataclasses import replace
from pathlib import Path
from typing import Iterable
import json

import pandas as pd

from .spec import ModelSpec
from .prep import (
    standardize, as_categorical,
    handle_nested_missingness, drop_remaining_na,
)
from .fixed_effects import FixedEffectsModel
from .mixed_effects import MixedEffectsModel

FN_ANALYSIS = "analysis.json"
FN_EFFECTS = "effect_sizes.parquet"

class SensitivityAnalysis:
    """Run sensitivity models across multiple metrics from one base spec.

    Each metric gets its own ModelSpec (derived from `base_spec` by swapping
    the `metric` field). Fitted models are stored in `self.results`, keyed
    by metric name and model type.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, all_metrics: Iterable[str], metrics_to_exclude: Iterable[str], base_spec: ModelSpec):
        """
        Args:
            all_metrics: every metric column you intend to model. Drives
                         the default loop in fixed_effects() / mixed_effects(),
                         but specific calls can override via `metrics_to_exclude`.
            metrics_to_exclude: metrics to exclude from modeling.
            base_spec:   a ModelSpec with everything set except `metric`
                         (the metric will be overridden per iteration).
        """
        self.all_metrics = set(all_metrics)
        self.metrics_to_exclude = set(metrics_to_exclude)
        self.available_metrics = self.all_metrics - self.metrics_to_exclude
        self.base_spec = base_spec
        self.df: pd.DataFrame | None = None
        self.results: dict[str, dict] = {m: {} for m in self.available_metrics}
        self.effects : pd.DataFrame | None = None

    def __getitem__(self, metric: str) -> dict:
        return self.results[metric]

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self,
                     data: pd.DataFrame,
                     nested_pairs: dict[str, str] | None = None,
                     nested_strategy: str = 'indicator',
                     categorical_llm_group_refs: dict[str, object] | None = None,
                     categorical_prompt_var_refs: dict[str, object] | None = None,
                     standardize_numeric: bool = True,
                     dropna: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
        """Clean and prepare the dataframe once. Stores the result on self.df.

        Steps:
            1. Resolve structurally-nested NaN (e.g. factuality_field ← factuality_author).
            2. Drop remaining NaN on all variables used in any model.
            3. (Optional) Z-score numeric prompt variables.
            4. Set categorical references for prompt and LLM-attribute variables.

        Args:
            data:                  the raw dataframe.
            nested_pairs:        {child_var: parent_var} for structural NaN handling.
            nested_strategy:     the strategy to use for handling nested missingness.
            categorical_refs:    {column: reference_level} for as_categorical().
            standardize_numeric: whether to z-score numeric prompt vars.
            dropna:              whether to drop remaining NaN values.
            verbose:             print row counts before/after.

        Returns:
            The prepared dataframe (also stored on self.df).
        """
        spec = self.base_spec
        df = data.copy()

        # 1. Structurally-nested NaN → zero-impute
        if nested_pairs:
            df, new_vars = handle_nested_missingness(df, 
                                                    nested_pairs, 
                                                    nested_strategy, 
                                                    self.available_metrics,
                                                    verbose=verbose)
            spec.structural_vars.extend(new_vars)

        # 2. Drop remaining NaN on every column used in any model
        needed_cols = (list(self.available_metrics)
                       + [spec.llm_id]
                       + list(spec.prompt_vars)
                       + list(spec.llm_attrs))
        if dropna:
            df = drop_remaining_na(df, cols=needed_cols)

        # 3. Categorical reference levels
        if categorical_llm_group_refs:
            cat_cols = list(spec.llm_attrs)
            cat_cols = [c for c in cat_cols if c in categorical_llm_group_refs]
            df = as_categorical(df, cat_cols, refs=categorical_llm_group_refs)

        if categorical_prompt_var_refs:
            cat_cols = list(spec.prompt_vars)
            cat_cols = [c for c in cat_cols if c in categorical_prompt_var_refs]
            df = as_categorical(df, cat_cols, refs=categorical_prompt_var_refs)

        # 4. Standardize numeric prompt vars (no-op on categoricals)
        if standardize_numeric:
            df = standardize(df, spec.prompt_vars)

        self.df = df
        if verbose:
            print(f'\nPrepared dataframe: N = {len(df):,} rows, '
                  f'{len(self.available_metrics)} metrics, '
                  f'{len(spec.prompt_vars)} prompt vars '
                  f'{len(spec.structural_vars)} structural vars.'
                  )
        return df

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fixed_effects_model(self,
                            verbose: bool = True) -> dict:
        """Fit Model 1 (OLS with LLM fixed effects) for each metric.

        Stores each fit under self.results[metric]['m1'].
        Also caches the per-metric ModelSpec under self.results[metric]['spec'].

        Args:
            verbose:            print progress per metric.

        Returns:
            {metric: FixedEffectsModel} for the metrics actually fit.
        """
        self._require_data()

        fits = {}
        for metric in self.available_metrics:
            if verbose:
                print(f'  [m1] fitting fixed-effects model for: {metric}')
            spec = replace(self.base_spec, metric=metric)
            m1 = FixedEffectsModel(spec).fit(self.df, verbose=verbose, cluster_se=False)
            self.results[metric]['spec'] = spec
            self.results[metric]['m1'] = m1
            fits[metric] = m1
        return fits

    def mixed_effects_model(self,
                            moderators: list[str] | None = None,
                            single_moderator: bool = True,
                            verbose: bool = True) -> dict:
        """Fit Model 2 (mixed model with random LLM intercept + moderators).

        By default fits one model PER moderator (single_moderator=True) under
        self.results[metric]['m2_by_attr'][attr]. Set single_moderator=False
        to additionally fit a fully-interacted model at self.results[metric]['m2'].
        (Note: full m2 often fails to identify if some attribute combinations
        are missing in the data, e.g. XL × proprietary.)

        Args:
            moderators:         which LLM attributes to use. Defaults to
                                base_spec.llm_attrs.
            single_moderator:   if True, fit one model per moderator (safer).
                                If False, also fit one model with all moderators.
            verbose:            print progress per metric.

        Returns:
            {metric: {'m2_by_attr': {...}, 'm2': ...}}
        """
        self._require_data()
        moderators = moderators or self.base_spec.llm_attrs

        fits = {}
        for metric in self.available_metrics:
            spec = self.results[metric].get('spec') or replace(self.base_spec, metric=metric)
            self.results[metric]['spec'] = spec
            entry = {}

            if single_moderator:
                entry['m2_by_attr'] = {}
                for attr in moderators:
                    if verbose:
                        print(f'  [m2/{attr}] fitting for: {metric}')
                    entry['m2_by_attr'][attr] = (
                        MixedEffectsModel(spec, moderators=[attr]).fit(self.df, verbose=verbose)
                    )
                self.results[metric]['m2_by_attr'] = entry['m2_by_attr']
            else:
                if verbose:
                    print(f'  [m2/full] fitting for: {metric}')
                full = MixedEffectsModel(spec, moderators=moderators).fit(self.df)
                self.results[metric]['m2'] = full
                entry['m2'] = full

            fits[metric] = entry
        return fits

    # ------------------------------------------------------------------
    # Aggregated outputs
    # ------------------------------------------------------------------

    def coefs(self) -> pd.DataFrame:
        """Long-format table of every coefficient across all fitted models.
        Columns: metric, model, term, estimate, se, stat, pval, ci_low,
        ci_high, kind. Use as input to cross-metric plots."""
        rows = []
        for metric, r in self.results.items():
            if 'm1' in r:
                c = r['m1'].coefs()
                c['metric'] = metric; c['model'] = 'm1'
                rows.append(c)
            if 'm2' in r:
                c = r['m2'].coefs()
                c['metric'] = metric; c['model'] = 'm2'
                rows.append(c)
            for attr, m in r.get('m2_by_attr', {}).items():
                c = m.coefs()
                c['metric'] = metric; c['model'] = f'm2_{attr}'
                rows.append(c)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def effect_sizes(self) -> pd.DataFrame:
        """Long-format partial η² per (metric, variable) — for the heatmap.
        Requires FixedEffectsModel.effect_sizes() to be available."""
        if self.effects is None:
            rows = []
            for metric, r in self.results.items():
                if 'm1' not in r or not hasattr(r['m1'], 'effects'):
                    continue
                es = r['m1'].effects
                # es['metric'] = metric
                rows.append(es)
            self.effects = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return self.effects

    # ------------------------------------------------------------------
    # Persistence (portable: JSON + parquet, no pickle)
    # ------------------------------------------------------------------

    def save(self, dirpath: str | Path, verbose: bool = True) -> None:
        """Save the analysis state to disk.

        Layout:
            dirpath/
                analysis.json                 # top-level metadata
                <metric>/m1/                  # per-model directories
                <metric>/m2_<attr>/
                <metric>/m2/                  (if fitted)
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # effcts size
        effects = self.effect_sizes()
        effects.to_parquet(dirpath / FN_EFFECTS, index=False)
        
        
        # Top-level metadata
        with open(dirpath / FN_ANALYSIS, 'w') as f:
            json.dump({
                'all_metrics': list(self.all_metrics),
                'metrics_to_exclude': list(self.metrics_to_exclude),
                'base_spec': {
                    'persona_vars': list(self.base_spec.persona_vars),
                    'context_vars': list(self.base_spec.context_vars),
                    'llm_attrs':    list(self.base_spec.llm_attrs),
                    'llm_id':       self.base_spec.llm_id,
                },
                'metrics_fit': list(self.results.keys()),
                'm1_effect_sizes_file': FN_EFFECTS
            }, f, indent=2)

        # Per-metric models
        for metric, r in self.results.items():
            mdir = dirpath / metric
            if 'm1' in r:
                r['m1'].save(mdir / 'm1')
                r['m1']._save_extras(mdir)

            if 'm2' in r:
                r['m2'].save(mdir / 'm2')
            for attr, m in r.get('m2_by_attr', {}).items():
                m.save(mdir / f'm2_{attr}')

        if verbose:
            print(f'Saved analysis to {dirpath}/')

    @classmethod
    def load(cls,
             dirpath: str | Path,
             base_spec: ModelSpec | None = None,
             df: pd.DataFrame | None = None,
             verbose: bool = True) -> 'SensitivityAnalysis':
        """Restore an analysis from disk.

        Args:
            dirpath:    directory created by .save()
            base_spec:  reconstruct from this if provided; otherwise from JSON.
            df:         original dataframe. If provided, models are refit so
                        all live methods work (.coefs(), .t_test(), etc.).
                        If None, models load 'thin' — only saved tables
                        are accessible via .saved_coefs() / .saved_params().

        Returns:
            A SensitivityAnalysis instance with self.results populated.
        """
        dirpath = Path(dirpath)
        with open(dirpath / FN_ANALYSIS) as f:
            meta = json.load(f)

        # Reconstruct base_spec if not provided
        if base_spec is None:
            bs = meta['base_spec']
            base_spec = ModelSpec(
                metric=None,
                persona_vars=bs['persona_vars'],
                context_vars=bs['context_vars'],
                llm_attrs=bs['llm_attrs'],
                llm_id=bs['llm_id'],
            )

        instance = cls(all_metrics=meta['all_metrics'], metrics_to_exclude=meta['metrics_to_exclude'], base_spec=base_spec)
        if df is not None:
            instance.df = df

        # Load effect sizes
        instance.effects = pd.read_parquet(dirpath / meta['m1_effect_sizes_file'])

        # Per-metric models
        for metric in meta['metrics_fit']:
            mdir = dirpath / metric
            if not mdir.is_dir():
                continue
            entry = {}

            if (mdir / 'm1').exists():
                entry['m1'] = FixedEffectsModel.load(mdir / 'm1', df=df)
                entry['m1']._load_extras(mdir)

            if (mdir / 'm2').exists():
                entry['m2'] = MixedEffectsModel.load(mdir / 'm2', df=df)

            m2_by_attr = {}
            for p in mdir.glob('m2_*'):
                if not p.is_dir() or p.name == 'm2':
                    continue
                attr = p.name[3:]   # strip 'm2_'
                m2_by_attr[attr] = MixedEffectsModel.load(p, df=df)
            if m2_by_attr:
                entry['m2_by_attr'] = m2_by_attr

            # Recover the per-metric spec from any loaded model
            for m in (*entry.get('m2_by_attr', {}).values(),
                      entry.get('m1'), entry.get('m2')):
                if m is not None:
                    entry['spec'] = m.spec
                    break

            instance.results[metric] = entry

        if verbose:
            n_fit = sum(1 for r in instance.results.values() if r)
            print(f'Loaded analysis from {dirpath}/ — '
                  f'{n_fit} metrics, df {"available" if df is not None else "not provided"}')
        return instance

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_data(self) -> None:
        if self.df is None:
            raise RuntimeError(
                "Call .prepare_data(df, ...) before fitting models."
            )