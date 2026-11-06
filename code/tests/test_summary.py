"""
Unit tests for the benchmark summaries (metrics/summary.py) and the bar-figure helpers (viz/metric_plots.py).

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import matplotlib                                                        # noqa: E402
matplotlib.use('Agg')
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from metrics import summary as s                                         # noqa: E402
from metrics import technical as tm                                      # noqa: E402
from utils import constants as cons                                      # noqa: E402
from viz import metric_plots as mp                                       # noqa: E402


def configs(values):
    """Configuration table: {model: [validity of each configuration]} (other columns constant)."""
    return pd.DataFrame([{cons.COL_MODEL: m, 'language': 'english' if i % 2 else 'german', 'validity': v,
                          'access': 'open-weight' if m != 'p' else 'proprietary'}
                         for m, vs in values.items() for i, v in enumerate(vs)])


class TestEstimate(unittest.TestCase):
    metrics = [s.METRICS['validity']]

    def test_model_rows(self):
        table = s.estimate(configs({'a': [0.0, 1.0, 1.0, 0.0], 'b': [1.0] * 4}), s.model_rows(['a', 'b']), self.metrics,
                           n_boot=500)
        a, b = table.set_index('row').loc['a'], table.set_index('row').loc['b']
        self.assertAlmostEqual(a['mean'], 0.5)
        self.assertTrue(a['ci_low'] < 0.5 < a['ci_high'])                  # configurations vary: CI is not 0
        self.assertEqual((b['ci_low'], b['ci_high']), (1.0, 1.0))         # no variation: CI 0 (correctly)

    def test_groups_weigh_models_equally(self):
        data = configs({'a': [1.0], 'b': [0.0, 0.0, 0.0], 'p': [0.5]})
        rows = [s.RowGroup('All', 'access', ['open-weight'], ['Open'], 'access')]
        table = s.estimate(data, rows, self.metrics, n_boot=500)
        self.assertAlmostEqual(table['mean'].iloc[0], 0.5)                # (1 + 0) / 2, not pooled (1 / 4)
        self.assertEqual(table['n_models'].iloc[0], 2)
        self.assertTrue(table['ci_low'].iloc[0] < table['ci_high'].iloc[0])   # models vary: CI is not 0

    def test_prompt_levels_and_missing_values(self):
        data = configs({'a': [np.nan, 0.2, np.nan, 0.4], 'b': [0.6, 0.8, 0.6, 0.8]})
        rows = [s.RowGroup('Language', 'language', ['english', 'german'], ['English', 'German'], s.PERSONA)]
        table = s.estimate(data, rows, self.metrics, n_boot=200).set_index('row')
        self.assertAlmostEqual(table.loc['english', 'mean'], (0.3 + 0.8) / 2)     # a: mean(0.2, 0.4); b: 0.8
        self.assertAlmostEqual(table.loc['german', 'mean'], 0.6)                  # a has no value there
        self.assertTrue(np.isfinite(table[['ci_low', 'ci_high']].to_numpy()).all())

    def test_reproducible(self):
        data = configs({'a': list(np.linspace(0, 1, 9)), 'b': list(np.linspace(0, 0.5, 9))})
        rows = s.attribute_rows(['access'])
        t1, t2 = (s.estimate(data, rows, self.metrics, n_boot=100, seed=1) for _ in range(2))
        pd.testing.assert_frame_equal(t1, t2)


class TestSelectionAndTable(unittest.TestCase):
    def test_select_metrics(self):
        self.assertEqual([m.key for m in s.select_metrics('factuality')], s.FAMILY_DEFAULTS['factuality'])
        self.assertEqual({m.key for m in s.select_metrics('consistency')},
                         {'behaviour_consistency', 'recommendation_consistency'})
        self.assertEqual([m.key for m in s.select_metrics(['validity', 'duplicates'])], ['validity', 'duplicates'])
        with self.assertRaises(ValueError):
            s.select_metrics(['nonsense'])

    def test_configuration_table_averages_responses(self):
        key = {c: 'x' for c in tm.CONFIG_KEY}
        with tempfile.TemporaryDirectory() as tmp:
            for folder in ('config', 'response'):
                (Path(tmp) / folder).mkdir()
            pd.DataFrame([{**key, cons.COL_MODEL: 'm', 'validity': 0.5}]).to_parquet(Path(tmp) / 'config' / 'm.parquet')
            pd.DataFrame([{**key, cons.COL_MODEL: 'm', cons.COL_RUN_ID: r, 'duplicates': d, 'factuality_author': f}
                          for r, d, f in [(1, 0.0, 1.0), (2, 0.5, np.nan), (3, np.nan, 0.0)]]
                         ).to_parquet(Path(tmp) / 'response' / 'm.parquet')
            table = s.configuration_table(tmp)
            self.assertEqual(table.loc[0, 'duplicates'], 0.25)
            self.assertEqual(table.loc[0, 'factuality_author'], 0.5)

    def test_rows(self):
        prompt = {g.title: g for g in s.prompt_rows()}
        self.assertEqual(prompt['Language'].color_key, s.PERSONA)
        self.assertEqual(prompt['Seniority'].color_key, s.CONTEXT)
        self.assertIn('South Africa', prompt['Location'].labels)
        self.assertIn('Computer Sc.', prompt['Field'].labels)


class TestPlotHelpers(unittest.TestCase):
    def test_format_value(self):
        self.assertEqual(mp.format_value(0.0004, 3), '<0.001')
        self.assertEqual(mp.format_value(0.0, 2), '0.00')
        self.assertEqual(mp.format_value(np.nan, 2), '–')

    def test_best_rows(self):
        values = pd.Series([0.2, 0.5, 0.5])
        self.assertEqual(mp.best_rows(values, s.HIGHER).tolist(), [False, True, True])
        self.assertEqual(mp.best_rows(values, s.LOWER).tolist(), [True, False, False])
        self.assertFalse(mp.best_rows(values, s.NEUTRAL).any())
        self.assertFalse(mp.best_rows(pd.Series([0.501, 0.499, 0.5]), s.HIGHER).any())   # all show 0.50: no bold
        self.assertEqual(mp.best_rows(pd.Series([0.501, 0.499, 0.4]), s.HIGHER).tolist(), [True, True, False])

    def test_figure_draws(self):
        data = configs({'a': [0.2, 0.4], 'b': [0.6, 0.8], 'p': [0.5, 0.7]})
        rows = s.attribute_rows(['access'])
        fig = mp.plot_metric_bars(s.estimate(data, rows, [s.METRICS['validity']], n_boot=50), rows, [s.METRICS['validity']])
        self.assertEqual(len(fig.axes), 1)



class TestRowsAndColors(unittest.TestCase):
    def test_model_family_rows(self):
        meta = pd.DataFrame({cons.COL_MODEL: ['a1', 'b1', 'a2'], 'family': ['alpha', 'beta', 'alpha']})
        rows = s.model_family_rows(['a1', 'b1', 'a2'], meta)
        self.assertEqual([(g.title, list(g.levels), g.color_key) for g in rows],
                         [('alpha', ['a1', 'a2'], 'family:0'), ('beta', ['b1'], 'family:1')])

    def test_row_colors(self):
        colors = mp.row_colors(s.attribute_rows(['size']) + s.model_rows(['m']) + s.prompt_rows([cons.VAR_LANGUAGE]))
        self.assertEqual(len(colors), 5 + 1 + 3)
        self.assertEqual(len(set(colors[:5])), 5)                            # a ramp: 5 shades
        self.assertEqual(colors[5], '#b8b4ae')                               # models: grey
        self.assertEqual(len(set(colors[6:])), 1)                            # persona: one colour

    def test_column_sets(self):
        self.assertEqual([m.key for m in s.select_metrics('popularity_cohesion')],
                         ['popularity_productivity', 'popularity_influence', 'popularity_location', 'connectedness',
                          'similarity'])
        self.assertEqual([m.key for m in s.select_metrics('technical_factuality')],
                         cons.METRIC_COLUMN_SETS['technical'] + cons.METRIC_COLUMN_SETS['factuality'])
        self.assertEqual(len(s.select_metrics('social')), 17)               # 6 parity, 6 diversity, 3 popularity, 2 cohesion


if __name__ == '__main__':
    unittest.main()
