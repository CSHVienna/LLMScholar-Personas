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

from metrics import summary                                         # noqa: E402
from metrics import technical as tm                                      # noqa: E402
from utils import constants as cons                                      # noqa: E402
from viz import formatting, metric_plots as mp                                       # noqa: E402
from viz import constants as viscons                                     # noqa: E402


def configs(values):
    """Configuration table: {model: [validity of each configuration]} (other columns constant)."""
    return pd.DataFrame([{cons.COL_MODEL: m, 'language': 'english' if i % 2 else 'german', 'validity': v,
                          'access': 'open-weight' if m != 'p' else 'proprietary'}
                         for m, vs in values.items() for i, v in enumerate(vs)])


class TestEstimate(unittest.TestCase):
    metrics = [summary.METRICS['validity']]

    def test_model_rows(self):
        table = summary.estimate(configs({'a': [0.0, 1.0, 1.0, 0.0], 'b': [1.0] * 4}), summary.model_rows(['a', 'b']), self.metrics,
                           n_boot=500)
        a, b = table.set_index('row').loc['a'], table.set_index('row').loc['b']
        self.assertAlmostEqual(a['mean'], 0.5)
        self.assertTrue(a['ci_low'] < 0.5 < a['ci_high'])                  # configurations vary: CI is not 0
        self.assertEqual((b['ci_low'], b['ci_high']), (1.0, 1.0))         # no variation: CI 0 (correctly)

    def test_groups_weigh_models_equally(self):
        data = configs({'a': [1.0], 'b': [0.0, 0.0, 0.0], 'p': [0.5]})
        rows = [summary.RowGroup('All', 'access', ['open-weight'], ['Open'], 'access')]
        table = summary.estimate(data, rows, self.metrics, n_boot=500)
        self.assertAlmostEqual(table['mean'].iloc[0], 0.5)                # (1 + 0) / 2, not pooled (1 / 4)
        self.assertEqual(table['n_models'].iloc[0], 2)
        self.assertTrue(table['ci_low'].iloc[0] < table['ci_high'].iloc[0])   # models vary: CI is not 0

    def test_prompt_levels_and_missing_values(self):
        data = configs({'a': [np.nan, 0.2, np.nan, 0.4], 'b': [0.6, 0.8, 0.6, 0.8]})
        rows = [summary.RowGroup('Language', 'language', ['english', 'german'], ['English', 'German'], cons.FAMILY_PERSONA)]
        table = summary.estimate(data, rows, self.metrics, n_boot=200).set_index('row')
        self.assertAlmostEqual(table.loc['english', 'mean'], (0.3 + 0.8) / 2)     # a: mean(0.2, 0.4); b: 0.8
        self.assertAlmostEqual(table.loc['german', 'mean'], 0.6)                  # a has no value there
        self.assertTrue(np.isfinite(table[['ci_low', 'ci_high']].to_numpy()).all())

    def test_reproducible(self):
        data = configs({'a': list(np.linspace(0, 1, 9)), 'b': list(np.linspace(0, 0.5, 9))})
        rows = summary.attribute_rows(['access'])
        t1, t2 = (summary.estimate(data, rows, self.metrics, n_boot=100, seed=1) for _ in range(2))
        pd.testing.assert_frame_equal(t1, t2)


class TestSelectionAndTable(unittest.TestCase):
    def test_select_metrics(self):
        self.assertEqual([m.key for m in summary.select_metrics('factuality')], summary.COLUMN_SETS['factuality'])
        self.assertEqual({m.key for m in summary.select_metrics('consistency')},
                         {'behaviour_consistency', 'recommendation_consistency'})
        self.assertEqual([m.key for m in summary.select_metrics(['validity', 'duplicates'])], ['validity', 'duplicates'])
        with self.assertRaises(ValueError):
            summary.select_metrics(['nonsense'])

    def test_adjusted_labels(self):
        self.assertEqual(summary.METRICS['parity_gender_adj'].label, 'Par.$^{adj}_{gender}$')
        self.assertEqual(summary.METRICS['locality_adj'].label, 'Locality$^{adj}$')     # no subscript: still marked
        self.assertEqual(summary.METRICS['locality_adj'].narrow, 'Loc.$^{adj}$')
        self.assertEqual(summary.METRICS['similarity_adj'].narrow, '')

    def test_configuration_table_averages_responses(self):
        key = {c: 'x' for c in tm.CONFIG_KEY}
        with tempfile.TemporaryDirectory() as tmp:
            for folder in ('config', 'response'):
                (Path(tmp) / folder).mkdir()
            pd.DataFrame([{**key, cons.COL_MODEL: 'm', 'validity': 0.5}]).to_parquet(Path(tmp) / 'config' / 'm.parquet')
            pd.DataFrame([{**key, cons.COL_MODEL: 'm', cons.COL_RUN_ID: r, 'duplicates': d, 'factuality_author': f}
                          for r, d, f in [(1, 0.0, 1.0), (2, 0.5, np.nan), (3, np.nan, 0.0)]]
                         ).to_parquet(Path(tmp) / 'response' / 'm.parquet')
            table = summary.configuration_table(tmp)
            self.assertEqual(table.loc[0, 'duplicates'], 0.25)
            self.assertEqual(table.loc[0, 'factuality_author'], 0.5)

    def test_rows(self):
        prompt = {g.title: g for g in summary.prompt_rows()}
        self.assertEqual(prompt['Language'].color_key, cons.FAMILY_PERSONA)
        self.assertEqual(prompt['Seniority'].color_key, cons.FAMILY_CONTEXT)
        self.assertIn('South Africa', prompt['Location'].labels)
        self.assertIn('Computer Sc.', prompt['Field'].labels)


class TestPlotHelpers(unittest.TestCase):
    def test_format_value(self):
        self.assertEqual(formatting.format_value(0.0004, 3), '<0.001')
        self.assertEqual(formatting.format_value(0.0, 2), '0.00')
        self.assertEqual(formatting.format_value(np.nan, 2), '–')

    def test_best_rows(self):
        values = pd.Series([0.2, 0.5, 0.5])
        self.assertEqual(mp.best_rows(values, summary.HIGHER).tolist(), [False, True, True])
        self.assertEqual(mp.best_rows(values, summary.LOWER).tolist(), [True, False, False])
        self.assertFalse(mp.best_rows(values, summary.NEUTRAL).any())
        self.assertFalse(mp.best_rows(pd.Series([0.501, 0.499, 0.5]), summary.HIGHER).any())   # all show 0.50: no bold
        self.assertEqual(mp.best_rows(pd.Series([0.501, 0.499, 0.4]), summary.HIGHER).tolist(), [True, True, False])

    def test_figure_draws(self):
        data = configs({'a': [0.2, 0.4], 'b': [0.6, 0.8], 'p': [0.5, 0.7]})
        rows = summary.attribute_rows(['access'])
        fig = mp.plot_metric_bars(summary.estimate(data, rows, [summary.METRICS['validity']], n_boot=50), rows, [summary.METRICS['validity']])
        self.assertEqual(len(fig.axes), 1)


class TestRowsAndColors(unittest.TestCase):
    def test_model_family_rows(self):
        meta = pd.DataFrame({cons.COL_MODEL: ['a1', 'b1', 'a2'], 'family': ['alpha', 'beta', 'alpha']})
        rows = summary.model_family_rows(['a1', 'b1', 'a2'], meta)
        self.assertEqual([(g.title, list(g.levels), g.color_key) for g in rows],
                         [('alpha', ['a1', 'a2'], 'model'), ('beta', ['b1'], 'model')])

    def test_row_colors(self):
        colors = mp.row_colors(summary.attribute_rows(['size']) + summary.model_rows(['m']) + summary.prompt_rows([cons.VAR_LANGUAGE]))
        self.assertEqual(len(colors), 5 + 1 + 3)
        self.assertEqual(len(set(colors[:5])), 5)                            # a ramp: 5 shades
        self.assertEqual(colors[5], viscons.FAMILY_COLORS['model'])          # models: the model grey
        self.assertEqual(set(colors[6:]), {viscons.FAMILY_COLORS['persona']})   # persona: its family colour

    def test_column_sets(self):
        for removed in ('technical_factuality', 'popularity_cohesion', 'popularity_cohesion_locality'):
            self.assertNotIn(removed, summary.COLUMN_SETS)                     # combinations are lists now

    def test_mixed_selection(self):
        keys = lambda *a, **k: [m.key for m in summary.select_metrics(*a, **k)]
        self.assertEqual(keys(['popularity', 'cohesion', 'locality']),
                         ['popularity_productivity', 'popularity_influence', 'popularity_location', 'connectedness',
                          'similarity', 'locality'])
        self.assertEqual(keys(['technical', 'factuality'], exclude=['consistency']),
                         ['refusals', 'validity', 'duplicates', 'k_balance'] + cons.METRIC_COLUMN_SETS['factuality'])
        self.assertEqual(keys(['parity_gender', 'diversity', 'locality', 'parity_gender']),        # duplicates once
                         ['parity_gender'] + cons.METRIC_COLUMN_SETS['diversity'] + ['locality'])
        self.assertEqual(keys('parity_adj', exclude='parity_gender'),                             # raw key drops its twin
                         [f"parity_{a}_adj" for a in ('ethnicity', 'location', 'productivity', 'influence',
                                                       'collaboration')])
        self.assertEqual(summary.adjusted_selection(['popularity', 'cohesion', 'validity', 'locality'], True),
                         ['popularity_adj', 'cohesion_adj', 'validity', 'locality_adj'])
        self.assertEqual(summary.split_adjusted(['technical', 'factuality'], ['consistency']),
                         ('technical-factuality-no-consistency', False))
        self.assertEqual(summary.split_adjusted(['popularity_adj', 'locality_adj']), ('popularity-locality', True))

        self.assertEqual(len(summary.select_metrics('social')), 18)               # 6 parity, 6 diversity, 3 popularity, 2 cohesion, locality
        self.assertEqual(summary.select_metrics('social')[-1].key, 'locality')    # last, after cohesion


class TestStudyEthnicity(unittest.TestCase):
    """study.ethnicity_attribute: the *_ethnicity metrics of every set -> race / both (in place; restored after)."""

    def tearDown(self):
        summary.use_study_ethnicity(cons.DEFAULT_STUDY_ETHNICITY)

    def test_default_sets_unchanged(self):
        self.assertEqual(cons.metric_column_sets(), cons.METRIC_COLUMN_SETS)
        self.assertIn('parity_ethnicity', cons.METRIC_COLUMN_SETS['parity'])
        self.assertNotIn('parity_race', cons.METRIC_COLUMN_SETS['all'])
        self.assertEqual(cons.METRIC_COLUMN_SETS['race'], ['parity_race', 'diversity_race'])

    def test_race(self):
        self.assertEqual(summary.use_study_ethnicity('race'), ['race'])
        keys = [m.key for m in summary.select_metrics('all')]
        self.assertIn('parity_race', keys)
        self.assertFalse(any(k.endswith('_ethnicity') for k in keys))
        self.assertEqual(summary.select_metrics('paper_social_adj')[0].key, 'parity_race_adj')   # order kept
        self.assertEqual(cons.RQ3_SOCIAL_METRICS[1], 'parity_race')
        self.assertEqual(len(cons.RQ3_SOCIAL_METRICS), 6)

    def test_both(self):
        summary.use_study_ethnicity('both')
        parity = cons.METRIC_COLUMN_SETS['parity']
        self.assertEqual(parity[1:3], ['parity_ethnicity', 'parity_race'])                # race right after ethnicity
        self.assertEqual(len(cons.RQ3_SOCIAL_METRICS), 7)
        self.assertEqual(len([m.key for m in summary.select_metrics(['all', 'social_adj'])]), len(set(
            [m.key for m in summary.select_metrics(['all', 'social_adj'])])))                    # no repeats

    def test_unknown_choice(self):
        with self.assertRaises(ValueError):
            summary.use_study_ethnicity('caste')

    def test_every_choice_names_known_metrics(self):
        for choice in cons.STUDY_ETHNICITY_CHOICES:
            unknown = {k for keys in cons.metric_column_sets(choice).values() for k in keys} - set(summary.METRICS)
            self.assertEqual(unknown, set(), choice)


class TestConstants(unittest.TestCase):
    """utils/constants.py: one name per concept (families, social metric keys)."""

    def test_variable_family(self):
        self.assertEqual(cons.variable_family(cons.COL_MODEL), cons.FAMILY_MODEL)
        self.assertEqual(cons.variable_family(cons.MODEL_SIZE), cons.FAMILY_MODEL)
        self.assertEqual(cons.variable_family(cons.VAR_LANGUAGE), cons.FAMILY_PERSONA)
        self.assertEqual(cons.variable_family(cons.VAR_K), cons.FAMILY_CONTEXT)

    def test_social_metric_keys(self):
        from metrics import social as so
        self.assertEqual(cons.social_metric_keys(cons.POPULARITY),
                         ['popularity_productivity', 'popularity_influence', 'popularity_location'])
        self.assertEqual(so.METRIC_KEYS, [k for k, m in summary.METRICS.items()
                                          if m.family == cons.METRIC_SOCIAL and not k.endswith(cons.ADJ_SUFFIX)])
        self.assertEqual(so.ATTRIBUTES[cons.ATTR_LOCATION], cons.ATTR_COUNTRY)



class TestRowId(unittest.TestCase):
    """tm.row_id: one string per row, missing -> '', every value as str(value)."""

    def test_strings(self):
        df = pd.DataFrame({'a': ['m1', 'm2', None], 'b': [5, 10, 7], 'c': [1.5, np.nan, 2.0],
                           'd': pd.Categorical(['Physics', None, 'Biology'])})
        self.assertEqual(list(tm.row_id(df, ['a', 'b', 'c', 'd'])),
                         ['m1\x1f5\x1f1.5\x1fPhysics', 'm2\x1f10\x1f\x1f', '\x1f7\x1f2.0\x1fBiology'])
        self.assertEqual(list(tm.row_id(df, ['b'])), ['5', '10', '7'])


if __name__ == '__main__':
    unittest.main()
