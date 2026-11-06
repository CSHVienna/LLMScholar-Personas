"""
Unit tests for RQ2 (analysis/deviations.py) and its figures (viz/rq_plots.plot_rq2 / plot_rq2_heatmap).

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import matplotlib                                                        # noqa: E402
matplotlib.use('Agg')
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from analysis import deviations as dv                                    # noqa: E402
from viz import metric_plots as mp                                       # noqa: E402
from utils import constants as cons                                      # noqa: E402

PARENT = {'s1': 'A', 's2': 'A', 's3': 'B', 's4': 'B', 's5': 'B'}
FACTORS = ['model', 'language', 'field', 'subfield']


def data(n=400, seed=0):
    """Unbalanced design: additive effects + noise; subfield nested in field."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(list(PARENT), n, p=[.1, .3, .2, .2, .2])
    df = pd.DataFrame({'model': rng.choice(list('mnop'), n), 'language': rng.choice(['en', 'de', 'zh'], n, p=[.5, .3, .2]),
                       'subfield': sub, 'field': [PARENT[s] for s in sub]})
    effect = {'en': 0, 'de': .3, 'zh': -.2, 'A': .5, 'B': 0, 's1': .2, 's2': -.1, 's3': 0, 's4': .4, 's5': -.3,
              'm': 0, 'n': .1, 'o': .2, 'p': -.1}
    df['y'] = [sum(effect[v] for v in r) for r in df[['model', 'language', 'field', 'subfield']].itertuples(index=False)]
    df['y'] += rng.normal(0, .1, n)
    return df


class TestDeviations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data()
        cls.out = dv.deviations(cls.df, 'y', FACTORS, n_boot=30)

    def value(self, variable, level, column='deviation'):
        return self.out.query("variable == @variable and level == @level")[column].item()

    def test_matches_direct_counterfactual(self):
        df = self.df
        columns = {c: sorted(df[c].unique()) for c in ['model', 'language', 'subfield']}

        def design(d):
            return np.column_stack([np.ones(len(d))] + [pd.get_dummies(d[c]).reindex(columns=levels, fill_value=0)
                                                        .to_numpy(float)[:, 1:] for c, levels in columns.items()])
        beta = np.linalg.lstsq(design(df), df['y'], rcond=None)[0]

        def emm(**fixed):
            return (design(df.assign(**fixed)) @ beta).mean()
        for level in ['en', 'de', 'zh']:
            self.assertAlmostEqual(self.value('language', level, 'emm'), emm(language=level))
        for s, f in PARENT.items():
            self.assertAlmostEqual(self.value('subfield', s, 'emm'), emm(subfield=s, field=f))
        subs = ['s1', 's2']
        w = np.array([(df['subfield'] == s).sum() for s in subs], float)
        field_a = sum(wi * emm(subfield=s, field='A') for wi, s in zip(w / w.sum(), subs))
        self.assertAlmostEqual(self.value('field', 'A', 'emm'), field_a)

    def test_weighted_deviations_sum_to_zero(self):
        for variable, g in self.out.groupby('variable'):
            self.assertAlmostEqual(float((g['share'] * g['deviation']).sum()), 0.0, places=10)
        self.assertAlmostEqual(self.out['grand_mean'].iloc[0], self.df['y'].mean())

    def test_contrasts(self):
        out = self.out.set_index(['variable', 'level'])
        self.assertEqual(out.loc[('language', 'en'), 'reference'], 'de')        # values not in the design: alphabetical
        for (variable, parent), g in self.out.groupby(['variable', 'parent']):
            ref = g['reference'].iloc[0]
            self.assertEqual(g.loc[g['level'] == ref, 'contrast'].item(), 0.0)
            emm_ref = g.loc[g['level'] == ref, 'emm'].item()
            np.testing.assert_allclose(g['contrast'], g['emm'] - emm_ref, atol=1e-12)
        self.assertEqual(set(self.out.loc[self.out['variable'] == 'subfield', 'reference']), {'s1', 's3'})  # per field
        custom = dv.deviations(self.df, 'y', FACTORS, n_boot=0, references={'language': 'zh', 'subfield': 's5'})
        refs = custom.groupby(['variable', 'parent'])['reference'].first().to_dict()
        self.assertEqual((refs[('language', '')], refs[('subfield', 'A')], refs[('subfield', 'B')]), ('zh', 's1', 's5'))

    def test_fit_statistics(self):
        r = self.out.iloc[0]
        X = np.column_stack([np.ones(len(self.df))] + [pd.get_dummies(self.df[c], drop_first=True).to_numpy(float)
                                                       for c in ['model', 'language', 'subfield']])
        resid = self.df['y'] - X @ np.linalg.lstsq(X, self.df['y'], rcond=None)[0]
        r2 = 1 - (resid ** 2).sum() / ((self.df['y'] - self.df['y'].mean()) ** 2).sum()
        self.assertAlmostEqual(r['r2'], r2)
        self.assertTrue(r['r2_adj'] < r['r2'] and 0 < r['delta_r2_model'] < r['r2'])

    def test_model_not_reported_and_ci(self):
        self.assertNotIn('model', set(self.out['variable']))
        self.assertTrue((self.out['deviation_n_boot'] == 30).all())
        inside = (self.out['deviation_ci_low'] <= self.out['deviation']) & (self.out['deviation'] <= self.out['deviation_ci_high'])
        self.assertGreater(inside.mean(), 0.8)
        self.assertEqual(set(self.out.loc[self.out['variable'] == 'subfield', 'parent']), {'A', 'B'})

    def test_missing_metric_values_left_out(self):
        df = self.df.copy()
        df.loc[:9, 'y'] = np.nan
        out = dv.deviations(df, 'y', FACTORS, n_boot=0)
        self.assertEqual(out['n_configs'].iloc[0], len(df) - 10)
        self.assertTrue(out['deviation_ci_low'].isna().all())


class TestFigures(unittest.TestCase):
    def test_format_signed(self):
        self.assertEqual(mp.format_value(0.034, 2, signed=True), '+0.03')
        self.assertEqual(mp.format_value(-0.12, 2, signed=True), '−0.12')
        self.assertEqual(mp.format_value(-0.001, 2, signed=True), '−<0.01')

    def test_plots(self):
        from viz import rq_plots
        df = data().rename(columns={'y': 'validity'})
        df = df.assign(language=df['language'].map({'en': 'english', 'de': 'german', 'zh': 'spanish'}))
        results = pd.concat([dv.deviations(df, 'validity', FACTORS, n_boot=10),
                             dv.deviations(df.assign(duplicates=df['validity'] / 10), 'duplicates', FACTORS, n_boot=10)])
        rows = rq_plots.rq2_rows(results)
        self.assertEqual([g.column for g in rows], ['language', 'field', 'subfield'])
        self.assertEqual(rows[0].levels, [str(x) for x in cons.LANGUAGES])           # design order, not alphabetical
        for value in rq_plots.RQ2_VALUES:
            rq_plots.plot_rq2(results, value)
        rq_plots.plot_rq2_heatmap(results)
        for value in ('contrast', 'deviation'):
            fig = rq_plots.plot_rq2_forest(results, value, ['validity', 'duplicates'])
            self.assertEqual(len(fig.axes), 2)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        self.assertIn('Language', labels)
        self.assertEqual(rq_plots.rq2_figure_name('heatmap'), 'rq2_deviation_all_heatmap')


if __name__ == '__main__':
    unittest.main()
