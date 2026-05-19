import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import json
from pathlib import Path

from .base import SensitivityModel

FN_R2 = "r2.json"
FN_PARAMS_TABLE_PARAMS = "pt_params.parquet"
FN_PARAMS_TABLE_SE = "pt_se.parquet"
FN_PARAMS_TABLE_PVALUES = "pt_pvalues.parquet"
FN_PARAMS_TABLE_CONF_INT = "pt_conf_int.parquet"

class FixedEffectsModel(SensitivityModel):
    """
    Model 1: metric ~ social_vars + technical_vars + C(llm)

    LLM enters as fixed-effect dummies. Coefficients on prompt variables
    represent within-LLM effects pooled across the 43 LLMs in your sample.
    Use cluster-robust SEs (default) since observations within an LLM
    are not independent.
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.params = None
        self.effects = None
        self.llm_share = None
        self.r2_values = None

    def _build_formula(self, include_llm_dummies: bool = True) -> str:
        # @TODO: handle numeric vars that shouldn't be categorically encoded
        terms = [f"C({v})" for v in self.spec.prompt_vars]
        structural_terms = ' + '.join(self.spec.structural_vars) if len(self.spec.structural_vars) > 0 else None
        prompt = " + ".join(terms)
        
        formula = f"{self.spec.metric} ~ {prompt}"
        
        if structural_terms is not None:
            formula += " + " + structural_terms

        if include_llm_dummies:
            formula += f" + C({self.spec.llm_id})"

        # print(f"Built formula: {formula}")
        return formula
    

    def fit(self, df: pd.DataFrame, cluster_se: bool = False, verbose: bool = True):
        needed = [self.spec.metric, self.spec.llm_id] + self.spec.prompt_vars
        df_clean = df.dropna(subset=needed).reset_index(drop=True)

        model = smf.ols(self.formula, data=df_clean)
        if cluster_se:
            self.result = model.fit(
                cov_type='cluster',
                cov_kwds={'groups': df_clean[self.spec.llm_id].values},
            )
        else:
            self.result = model.fit()

        if verbose:
            print("Fixed-effects model fitted.")

        # model withuot LLM dummies for R² comparison
        self.llm_share = self.__llm_share_of_r2()
        self.effects = self.__effect_sizes(verbose=verbose)
        self.params = self.params_table()
        self.r2_values = self.r2()
        return self

    def __anova(self, typ: int = 2) -> pd.DataFrame:
        """Type-II ANOVA decomposition with partial η² per term.

        Returns one row per model term (prompt variable + LLM dummies),
        with sum of squares, df, F, p-value, and partial η².
        """
        if self.result is None:
            raise RuntimeError("Call .fit() before .anova()")

        # statsmodels anova_lm needs the fitted model object
        table = anova_lm(self.result, typ=typ)

        # # Partial η² = SS_effect / (SS_effect + SS_residual)
        # ss_residual = table.loc['Residual', 'sum_sq']
        # table = table.drop('Residual')
        # table['partial_eta2'] = table['sum_sq'] / (table['sum_sq'] + ss_residual)
        # table.index.name = 'term'
        return table
    
    def __effect_sizes(self, verbose: bool = False) -> pd.DataFrame:
        """Tidy ANOVA table aligned with your spec.

        One row per prompt variable, plus one row for LLM identity.
        Use this as the input to your heatmap.
        """
        if self.effects is not None:
            if verbose:
                print("Using cached effect sizes...")
            return self.effects
        if verbose:
            print("Calculating effect sizes from ANOVA...")
        anova = self.__anova(typ=2)

        # Partial η² = SS_effect / (SS_effect + SS_residual)
        residual_ss = anova.loc['Residual', 'sum_sq']
        
        rows = []
        for pv in self.spec.prompt_vars + [self.spec.llm_id]:
            term = f"C({pv})"

            rows.append({
                "metric": self.spec.metric,
                "variable": pv,
                'term': term,
                "sum_sq": anova.loc[term, "sum_sq"],
                "df": anova.loc[term, "df"],
                "F": anova.loc[term, "F"],
                "p_value": anova.loc[term, "PR(>F)"],
                "partial_eta2": (
                    anova.loc[term, "sum_sq"] /
                    (anova.loc[term, "sum_sq"] + residual_ss)
                ),
                'kind': ('llm' if pv == self.spec.llm_id
                         else 'persona' if pv in self.spec.persona_vars
                         else 'context'),
            })

        # for pv in self.spec.prompt_vars + [self.spec.llm_id]:
        #     # Match Patsy's term name: 'C(var)' for explicit categorical,
        #     # or just 'var' for numeric / already-categorical columns.
        #     candidates = [pv, f'C({pv})']
        #     match = anova[anova['term'].isin(candidates)]
        #     if match.empty:
        #         continue
        #     r = match.iloc[0].to_dict()
        #     r['variable'] = pv
        #     r['kind']     = ('llm' if pv == self.spec.llm_id
        #                      else 'persona' if pv in self.spec.persona_vars
        #                      else 'context')
        #     rows.append(r)

        self.effects = pd.DataFrame(rows)
        return self.effects
    
    def coefs(self, drop_llm_dummies: bool = True) -> pd.DataFrame:
        r = self.result
        out = self._tidy(r.params, r.bse, r.tvalues, r.pvalues, r.conf_int())
        if drop_llm_dummies:
            out = out[out['kind'] != 'llm_fe'].reset_index(drop=True)
        return out

    def r2(self) -> dict:
        if self.r2_values is not None:
            return self.r2_values
        self.r2_values = {'r2': self.result.rsquared, 'r2_adj': self.result.rsquared_adj}
        return self.r2_values
    
    def params_table(self) -> dict:
        r = self.result
        return {
            'params':   r.params,
            'se':       r.bse,
            'pvalues':  r.pvalues,
            'conf_int': r.conf_int(),
        }

    def __llm_share_of_r2(self) -> float:
        """Share of R² attributable to LLM fixed effects.
        Compared to a model without LLM dummies."""

        if self.llm_share is not None:
            return self.llm_share
        
        r2_full = self.result.rsquared

        # Refit without LLM dummies
        formula_no_llm = self._build_formula(include_llm_dummies=False)
        no_llm = smf.ols(formula_no_llm, data=self.result.model.data.frame).fit()
        self.llm_share = r2_full - no_llm.rsquared
        return self.llm_share

    def _save_extras(self, dirpath: Path) -> None:
        # R²
        if hasattr(self, 'r2_values'):
            with open(dirpath / FN_R2, 'w') as f:
                json.dump(self.r2() | {'llm_share': self.__llm_share_of_r2()}, f, indent=2)

        # params table
        if hasattr(self, 'params'):
            self.params['params'].to_frame(name='value').to_parquet(dirpath / FN_PARAMS_TABLE_PARAMS)
            self.params['se'].to_frame(name='value').to_parquet(dirpath / FN_PARAMS_TABLE_SE)
            self.params['pvalues'].to_frame(name='value').to_parquet(dirpath / FN_PARAMS_TABLE_PVALUES)
            self.params['conf_int'].to_parquet(dirpath / FN_PARAMS_TABLE_CONF_INT)

    def _load_extras(self, dirpath: Path) -> None:
        # R²
        if (dirpath / FN_R2).exists():
            with open(dirpath / FN_R2) as f:
                tmp = json.load(f)
                self.r2_values = {k: tmp[k] for k in ['r2', 'r2_adj']}
                self.llm_share = tmp.get('llm_share', None)
        # params table
        if (dirpath / FN_PARAMS_TABLE_PARAMS).exists():
            self.params = {}
            self.params['params'] = pd.read_parquet(dirpath / FN_PARAMS_TABLE_PARAMS)['value']
            self.params['se'] = pd.read_parquet(dirpath / FN_PARAMS_TABLE_SE)['value']
            self.params['pvalues'] = pd.read_parquet(dirpath / FN_PARAMS_TABLE_PVALUES)['value']
            self.params['conf_int'] = pd.read_parquet(dirpath / FN_PARAMS_TABLE_CONF_INT)