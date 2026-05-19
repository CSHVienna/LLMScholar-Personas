from abc import ABC, abstractmethod
import pandas as pd
from .spec import ModelSpec
from pathlib import Path
import json

class SensitivityModel(ABC):
    """Abstract base. Subclasses define their formula and fit/coefs methods."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.result = None
        self._formula: str | None = None

    @property
    def formula(self) -> str:
        if self._formula is None:
            self._formula = self._build_formula()
        return self._formula

    @abstractmethod
    def _build_formula(self) -> str: ...

    @abstractmethod
    def fit(self, df: pd.DataFrame): ...

    @abstractmethod
    def coefs(self) -> pd.DataFrame: ...

    def _tidy(self, params, ses, stats, pvals, ci) -> pd.DataFrame:
        """Common coefficient table shape with kind tag."""
        out = pd.DataFrame({
            'estimate': params, 'se': ses,
            'stat': stats, 'pval': pvals,
            'ci_low': ci[0], 'ci_high': ci[1],
        }).rename_axis('term').reset_index()
        out['kind'] = out['term'].apply(self.spec.var_kind)
        return out
    
    # ---------- persistence ----------

    def save(self, dirpath: str | Path) -> None:
        """Save model spec + fitted outputs to a directory.

        Layout:
            dirpath/
                spec.json           ← ModelSpec + class name + formula
                coefs.parquet       ← coefficient table
                params.parquet      ← params, SE, p-values, CIs (raw)
                anova.parquet       ← if applicable
                r2.json             ← if applicable
        """
        if self.result is None:
            raise RuntimeError("Call .fit() before .save()")
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # 1. Spec + metadata
        meta = {
            'class':        type(self).__name__,
            'formula':      self.formula,
            'metric':       self.spec.metric,
            'persona_vars': list(self.spec.persona_vars),
            'context_vars': list(self.spec.context_vars),
            'llm_attrs':    list(self.spec.llm_attrs),
            'llm_id':       self.spec.llm_id,
            'moderators':   getattr(self, 'moderators', None),
        }
        with open(dirpath / 'spec.json', 'w') as f:
            json.dump(meta, f, indent=2)

        # 2. Tidy coefficient table
        self.coefs().to_parquet(dirpath / 'coefs.parquet')

        # 3. Raw params table (params, SE, p-values, CIs)
        pt = self.params_table()
        params_df = pd.DataFrame({
            'estimate': pt['params'],
            'se':       pt['se'],
            'pval':     pt['pvalues'],
            'ci_low':   pt['conf_int'][0],
            'ci_high':  pt['conf_int'][1],
        })
        params_df.index.name = 'term'
        params_df.reset_index().to_parquet(dirpath / 'params.parquet')

        # 4. Optional artifacts (subclass-specific)
        self._save_extras(dirpath)

    def _save_extras(self, dirpath: Path) -> None:
        """Subclasses override to save model-specific artifacts."""
        pass

    @classmethod
    def load(cls, dirpath: str | Path, df: pd.DataFrame | None = None):
        """Restore a model from disk.

        Args:
            dirpath: directory created by .save()
            df: original dataframe. If provided, refits the model so all
                methods (.coefs(), .t_test(), marginal_effects, etc.) work.
                If None, returns a 'thin' model with only the saved tables
                accessible via .saved_coefs(), .saved_params(), etc.
        """
        dirpath = Path(dirpath)
        with open(dirpath / 'spec.json') as f:
            meta = json.load(f)

        # Reconstruct spec
        spec = ModelSpec(
            metric=meta['metric'],
            persona_vars=meta['persona_vars'],
            context_vars=meta['context_vars'],
            llm_attrs=meta['llm_attrs'],
            llm_id=meta['llm_id'],
        )

        # Dispatch to subclass
        from .fixed_effects import FixedEffectsModel
        from .mixed_effects import MixedEffectsModel
        klass = {
            'FixedEffectsModel': FixedEffectsModel,
            'MixedEffectsModel': MixedEffectsModel,
        }[meta['class']]

        mods = meta.get('moderators')
        instance = (klass(spec, moderators=mods) if mods is not None
                    else klass(spec))

        # Load saved tables (always available, even without refit)
        instance._saved_coefs  = pd.read_parquet(dirpath / 'coefs.parquet')
        instance._saved_params = pd.read_parquet(dirpath / 'params.parquet')
        instance._saved_dir    = dirpath

        # Refit if data is available — gives you a fully-functional object
        if df is not None:
            instance.fit(df)
        return instance

    # ---------- accessors that work without refit ----------

    def saved_coefs(self) -> pd.DataFrame:
        if not hasattr(self, '_saved_coefs'):
            raise RuntimeError("Model not loaded from disk.")
        return self._saved_coefs

    def saved_params(self) -> pd.DataFrame:
        if not hasattr(self, '_saved_params'):
            raise RuntimeError("Model not loaded from disk.")
        return self._saved_params