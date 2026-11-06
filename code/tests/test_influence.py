"""
Unit tests for RQ1 (analysis/influence.py): partial omega^2 against statsmodels' type-II ANOVA, the Shapley R^2
decomposition, and the cluster bootstrap over models.

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import itertools
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from analysis import influence as inf                                    # noqa: E402

FACTORS = ['model', 'language', 'k']


def data(seed=0, n_models=6):
    """Configurations of a crossed design: the model and k matter, the language does not."""
    rng = np.random.default_rng(seed)
    rows = []
    for m, lang, k in itertools.product(range(n_models), ['en', 'de', 'es'], [1, 5, 10]):
        for _ in range(4):
            rows.append({'model': f"m{m}", 'language': lang, 'k': k,
                         'y': 0.1 * m + 0.05 * (k == 10) + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


class TestInfluence(unittest.TestCase):
    def test_omega_squared_matches_statsmodels(self):
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        df = data()
        table = anova_lm(smf.ols('y ~ C(model) + C(language) + C(k)', data=df).fit(), typ=2)
        ms_error = table.loc['Residual', 'sum_sq'] / table.loc['Residual', 'df']
        y, blocks = inf._prepare(df, 'y', FACTORS)
        omega = inf.omega_squared(y, blocks)
        for factor, name in zip(FACTORS, ['C(model)', 'C(language)', 'C(k)']):
            ss, dof = table.loc[name, 'sum_sq'], table.loc[name, 'df']
            expected = (ss - dof * ms_error) / (ss + (len(df) - dof) * ms_error)
            self.assertAlmostEqual(omega[factor], expected, places=10)
        self.assertGreater(omega['model'], 0.8)
        self.assertLess(abs(omega['language']), 0.05)                   # no effect: ~0 (can be < 0)

    def test_shapley_sums_to_r2(self):
        df = data()
        y, blocks = inf._prepare(df, 'y', FACTORS)
        shares = inf.shapley_r2(y, blocks)
        r2 = 1 - inf.residual_ss(y, list(blocks.values())) / ((y - y.mean()) ** 2).sum()
        self.assertAlmostEqual(sum(shares.values()), r2, places=10)
        self.assertGreater(shares['model'], shares['k'])
        self.assertGreater(shares['k'], shares['language'])

    def test_bootstrap_resamples_whole_models(self):
        df = data()
        sample = inf.resample_models(df, np.random.default_rng(1))
        self.assertEqual(len(sample), len(df))                          # same number of models x configurations
        self.assertTrue((sample.groupby('model').size() == 36).all())   # a drawn model keeps all its configurations

    def test_influence_table(self):
        table = inf.influence(data(), 'y', FACTORS, n_boot=60, n_boot_shapley=20).set_index('variable')
        self.assertEqual(list(table.index), FACTORS)
        self.assertEqual(table.loc['language', 'family'], 'persona')
        self.assertEqual(table.loc['k', 'family'], 'context')
        self.assertTrue((table['omega2_ci_low'] <= table['omega2_ci_high']).all())
        self.assertGreater(table.loc['model', 'omega2_ci_high'] - table.loc['model', 'omega2_ci_low'], 0)
        self.assertAlmostEqual(table['shapley_r2'].sum(), table['r2_full'].iloc[0])

    def test_single_level_and_missing_values(self):
        df = data()
        df.loc[::5, 'y'] = np.nan                                       # missing values are left out
        df['role'] = 'student'                                          # one level: no effect estimable
        table = inf.influence(df, 'y', FACTORS + ['role'], n_boot=10, n_boot_shapley=5).set_index('variable')
        self.assertTrue(np.isnan(table.loc['role', 'omega2']))
        self.assertEqual(table['n_configs'].iloc[0], df['y'].notna().sum())



class TestRq1Figure(unittest.TestCase):
    def test_rq1_bars(self):
        import matplotlib
        matplotlib.use('Agg')
        from viz import rq_plots
        df = data().rename(columns={'y': 'validity'})
        results = inf.influence(df, 'validity', FACTORS, n_boot=20, n_boot_shapley=5)
        table, rows, metrics = rq_plots.rq1_bars(results)
        self.assertEqual([g.title for g in rows], ['Persona', 'Context', 'LLM'])
        self.assertTrue((table['mean'] >= 0).all())                       # negative omega^2 clipped for plotting
        self.assertIsNone(metrics[0].limits)                              # panel scaled to the data
        fig = rq_plots.plot_rq1(results)
        self.assertEqual(len(fig.axes), 1)



class TestEdgeCases(unittest.TestCase):
    def test_constant_metric_is_not_estimable(self):
        df = data().assign(y=0.0)                                      # e.g. duplicates = 0 everywhere
        table = inf.influence(df, 'y', FACTORS, n_boot=10, n_boot_shapley=5)
        self.assertTrue(table['omega2'].isna().all() and table['shapley_r2'].isna().all())

    def test_empty_cells_use_ranks(self):
        df = data()
        df = df[~((df['model'] == 'm0') & (df['k'] == 10))]            # an empty model x k cell: still estimable
        y, blocks = inf._prepare(df, 'y', FACTORS)
        omega = inf.omega_squared(y, blocks)
        self.assertTrue(all(np.isfinite(v) for v in omega.values()))
        df2 = data().assign(k2=lambda d: d['k'])                       # a factor duplicating another adds no rank
        y2, blocks2 = inf._prepare(df2, 'y', FACTORS + ['k2'])
        self.assertTrue(np.isnan(inf.omega_squared(y2, blocks2)['k2']))



class TestNested(unittest.TestCase):
    @staticmethod
    def fields_data(within_effect: float, seed=0):
        """Fields differ (0.3 apart); subfields of a field differ by `within_effect`."""
        rng = np.random.default_rng(seed)
        rows = []
        for m, (field, subs) in itertools.product(range(5), [('bio', ['b1', 'b2']), ('cs', ['c1', 'c2']),
                                                             ('math', ['m1', 'm2'])]):
            for i, sub in enumerate(subs):
                for _ in range(6):
                    rows.append({'model': f"m{m}", 'field': field, 'subfield': sub,
                                 'y': 0.3 * ['bio', 'cs', 'math'].index(field) + within_effect * i + rng.normal(0, 0.05)})
        return pd.DataFrame(rows)

    def test_field_keeps_its_effect(self):
        factors = ['model', 'field', 'subfield']
        y, blocks = inf._prepare(self.fields_data(0.0), 'y', factors)
        omega = inf.omega_squared(y, blocks)
        self.assertGreater(omega['field'], 0.9)                         # not zeroed by the nested subfield
        self.assertLess(abs(omega['subfield']), 0.05)                   # no difference within fields
        self.assertEqual(blocks['subfield'].shape[1], 3)                # 1 contrast per field (2 subfields each)
        self.assertTrue(np.allclose(blocks['subfield'].sum(axis=0), 0))  # effect coding: balanced -> sums to 0

    def test_within_field_differences_go_to_subfield(self):
        factors = ['model', 'field', 'subfield']
        y, blocks = inf._prepare(self.fields_data(0.2), 'y', factors)
        omega, shares = inf.omega_squared(y, blocks), inf.shapley_r2(y, blocks)
        self.assertGreater(omega['subfield'], 0.8)
        self.assertGreater(shares['subfield'], 0.05)
        y0, blocks0 = inf._prepare(self.fields_data(0.0), 'y', factors)   # same noise, no within-field difference

        def ss_field(y_, blocks_):
            gram = inf.Gram(y_, blocks_)
            return gram.rss(['model']) - gram.rss(['model', 'field'])
        self.assertAlmostEqual(ss_field(y, blocks), ss_field(y0, blocks0), places=8)   # field's own SS unchanged



class TestRq1Heatmap(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use('Agg')
        df = data().rename(columns={'y': 'validity'}).assign(refusals=lambda d: d['validity'] * 0.5)
        self.results = pd.concat([inf.influence(df, m, FACTORS, n_boot=20, n_boot_shapley=5) for m in ['validity', 'refusals']])

    def test_kinds_and_measures(self):
        from viz import rq_plots
        for kind in ['bars', 'heatmap']:
            for measure in ['omega2', 'shapley_r2']:
                self.assertIsNotNone(rq_plots.plot_rq1_figure(self.results, kind, measure))
        with self.assertRaises(ValueError):
            rq_plots.plot_rq1_figure(self.results, 'pie')

    def test_residual_only_for_shapley(self):
        from viz import rq_plots
        texts = lambda fig: [t.get_text() for t in fig.axes[0].texts]
        self.assertIn('Residual', texts(rq_plots.plot_rq1_heatmap(self.results, 'shapley_r2')))
        self.assertNotIn('Residual', texts(rq_plots.plot_rq1_heatmap(self.results, 'omega2')))


if __name__ == '__main__':
    unittest.main()
