import pandas as pd
import statsmodels.formula.api as smf
from typing import List
from scipy.stats import chi2
import json
from pathlib import Path

from .base import SensitivityModel
from .spec import ModelSpec


class MixedEffectsModel(SensitivityModel):
    """
    Model 2: metric ~ (prompt_vars) * (llm_attrs) + (1 | llm)

    LLM enters as a random intercept; prompt x attribute interactions
    test whether each prompt variable's effect varies by model kind.

    Usage:
        moderators=None          -> baseline (random intercept only)
        moderators=spec.llm_attrs -> full heterogeneity model
        moderators=['size']       -> single-moderator robustness model
    """

    def __init__(self, spec: ModelSpec, moderators: List[str] | None = None):
        super().__init__(spec)
        self.moderators = moderators or []
        if self.moderators and not spec.llm_attrs:
            raise ValueError("Set spec.llm_attrs before using moderators.")

    # def _build_formula(self) -> str:
    #     prompt = " + ".join(self.spec.prompt_vars)
    #     if not self.moderators:
    #         return f"{self.spec.metric} ~ {prompt}"
    #     attrs = " + ".join(f"C({a})" for a in self.moderators)
    #     return f"{self.spec.metric} ~ ({prompt}) * ({attrs})"
    
    def _build_formula(self) -> str:
        predictors = " + ".join(self.spec.all_predictors)  # ← was prompt_vars
        if not self.moderators:
            return f"{self.spec.metric} ~ {predictors}"
        attrs = " + ".join(f"C({a})" for a in self.moderators)
        # Only prompt_vars get interacted with moderators; structurals are main-effects only
        prompt = " + ".join(self.spec.prompt_vars)
        return (f"{self.spec.metric} ~ ({prompt}) * ({attrs}) "
                f"+ {' + '.join(self.spec.structural_vars)}"
                if self.spec.structural_vars
                else f"{self.spec.metric} ~ ({prompt}) * ({attrs})")

    def fit(self, df, reml=True, verbose=False):
        needed = [self.spec.metric, self.spec.llm_id] + self.spec.prompt_vars
        if self.moderators:
            needed += self.moderators
        df_clean = df.dropna(subset=needed).reset_index(drop=True)

        model = smf.mixedlm(self.formula, df_clean,
                            groups=df_clean[self.spec.llm_id].values)
        # Try multiple optimizers in sequence
        for m in ['lbfgs', 'bfgs', 'powell', 'cg']:
            try:
                self.result = model.fit(reml=reml, method=m)
                if self.result.converged:
                    self._optimizer_used = m
                    return self
            except Exception:
                continue
        # Fall through with last result even if not converged
        if verbose:
            print("Model did not converge.")
        return self

    def coefs(self) -> pd.DataFrame:
        r = self.result
        return self._tidy(r.fe_params, r.bse_fe, r.tvalues, r.pvalues, r.conf_int())

    def icc(self) -> float:
        """Share of variance attributable to LLM identity (after fixed effects)."""
        v_llm = float(self.result.cov_re.iloc[0, 0])
        v_res = float(self.result.scale)
        return v_llm / (v_llm + v_res)

    def lrt_vs(self, baseline: 'MixedEffectsModel') -> dict:
        """
        Likelihood-ratio test for joint significance of the interaction block.
        Both models must be fit with reml=False for the test to be valid.
        """
        ll_full = self.result.llf
        ll_base = baseline.result.llf
        df_diff = len(self.result.fe_params) - len(baseline.result.fe_params)
        stat = 2 * (ll_full - ll_base)
        return {'chi2': stat, 'df': df_diff, 'pval': 1 - chi2.cdf(stat, df_diff)}
    
    def params_table(self) -> dict:
        r = self.result
        return {
            'params':   r.fe_params,
            'se':       r.bse_fe,
            'pvalues':  r.pvalues,
            'conf_int': r.conf_int(),
        }
    
    def _save_extras(self, dirpath: Path) -> None:
        with open(dirpath / 'mixed_stats.json', 'w') as f:
            json.dump({
                'icc':            self.icc(),
                'random_var':     float(self.result.cov_re.iloc[0, 0]),
                'residual_var':   float(self.result.scale),
                'converged':      bool(getattr(self.result, 'converged', True)),
            }, f, indent=2)