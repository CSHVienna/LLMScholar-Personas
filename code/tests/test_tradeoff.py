"""
Unit tests for the RQ3 trade-off (analysis/tradeoff.py) and its figure (viz/rq_plots.plot_rq3).

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

from analysis import tradeoff as tr                                      # noqa: E402
from utils import constants as cons                                      # noqa: E402

TECH, SOCIAL = ['validity', 'refusals', 'duplicates'], ['parity_gender', 'parity_ethnicity']


def configs():
    """Two configurations per model; model a is good on both axes, d bad on both."""
    values = {'a': (1.0, 0.0, 0.0, 1.0, 1.0), 'b': (0.2, 0.8, 0.6, 0.9, 0.9), 'c': (0.9, 0.1, 0.1, 0.1, 0.2),
              'd': (0.1, 0.9, 0.9, 0.1, 0.1)}
    return pd.DataFrame([{cons.COL_MODEL: m, **dict(zip(TECH + SOCIAL, np.array(v) + d))}
                         for m, v in values.items() for d in (-0.0, 0.0)])


class TestTradeoff(unittest.TestCase):
    def setUp(self):
        self.scores = tr.model_scores(configs(), TECH, SOCIAL).set_index(cons.COL_MODEL)

    def test_defaults(self):
        self.assertEqual(len(cons.RQ3_TECHNICAL_METRICS), 7)
        self.assertEqual(cons.RQ3_SOCIAL_METRICS, [f"parity_{a}" for a in
                                                   ['gender', 'ethnicity', 'location', 'productivity', 'influence', 'collaboration']])

    def test_complements_and_sums(self):
        a = self.scores.loc['a']
        self.assertAlmostEqual(a['technical'], 3.0)                       # 1 + (1 - 0) + (1 - 0)
        self.assertAlmostEqual(self.scores.loc['b', 'technical'], 0.2 + 0.2 + 0.4)
        self.assertEqual((a['technical_max'], a['social_max']), (3, 2))

    def test_quadrants(self):
        s = self.scores.reset_index()
        rel = dict(zip(s[cons.COL_MODEL], tr.quadrants(s, 'relative')))
        self.assertEqual(rel, {'a': 'Q1', 'b': 'Q2', 'c': 'Q4', 'd': 'Q3'})
        self.assertEqual(tr.splits(s, 'absolute'), (1.5, 1.0))
        with self.assertRaises(ValueError):
            tr.splits(s, 'other')

    def test_pareto(self):
        s = self.scores.reset_index()
        self.assertEqual(sorted(s.loc[tr.pareto_front(s), cons.COL_MODEL]), ['a'])

    def test_missing_metric(self):
        with self.assertRaises(ValueError):
            tr.model_scores(configs(), TECH, ['parity_location'])

    def test_axis_label(self):
        self.assertEqual(tr.axis_label(cons.RQ3_TECHNICAL_METRICS),
                         'validity + refusals$^c$ + duplicates$^c$ + $\\sum$ factuality')
        self.assertEqual(tr.axis_label(cons.RQ3_SOCIAL_METRICS), '$\\sum$ parity')

    def test_plot_and_name(self):
        from viz import names, rq_plots
        for mode in cons.RQ3_MODES:
            fig = rq_plots.plot_rq3(self.scores.reset_index(), mode, pareto=True, technical=TECH, social=SOCIAL)
            self.assertEqual(len(fig.axes), 1)
        fig = rq_plots.plot_rq3_inset(self.scores.reset_index(), pareto=True, technical=TECH, social=SOCIAL)
        self.assertEqual(len(fig.axes[0].child_axes), 1)                         # overview + zoomed inset (a child axes)
        inset = fig.axes[0].child_axes[0]
        self.assertEqual(len(inset.texts), len(self.scores))                     # the labels are in the inset
        self.assertEqual(len(fig.axes[0].texts), 1)                              # overview: only the correlation
        self.assertEqual(names.rq3_figure_name('absolute', kind='inset'), 'rq3_absolute_inset')
        with self.assertRaises(ValueError):
            rq_plots.plot_rq3_inset(self.scores.reset_index(), technical=TECH, social=SOCIAL, color_by='nosuch')
        self.assertEqual(names.rq3_figure_name('absolute'), 'rq3_absolute_scatter')
        self.assertEqual(names.rq3_figure_name('relative', TECH, SOCIAL),
                         'rq3_relative_validity-refusals-duplicates-vs-parity-gender-parity-ethnicity_scatter')


class TestTradeoffAdjusted(unittest.TestCase):
    """Size-adjusted social metrics (<key>_adj = observed - E in [-E, 1 - E]): axis from the expected values,
    absolute split at 0, _adj label / file name."""
    def setUp(self):
        from metrics import summary
        self.social = summary.adjusted_keys(SOCIAL)
        self.frame = configs()
        for raw, adj in zip(SOCIAL, self.social):
            self.frame[adj] = self.frame[raw] - 0.5                              # observed - expected, can be < 0
            self.frame[raw + '_expected'] = 0.5
        self.scores = tr.model_scores(self.frame, TECH, self.social)

    def test_range_and_split(self):
        self.assertEqual(self.social, ['parity_gender_adj', 'parity_ethnicity_adj'])
        self.assertEqual(tr.axis_range(self.scores, 'social'), (-1.0, 1.0))     # 2 x [-0.5, 0.5]
        self.assertEqual(tr.axis_range(self.scores, 'technical'), (0.0, 3.0))
        self.assertEqual(tr.splits(self.scores, 'absolute'), (1.5, 0.0))
        d = self.scores.set_index(cons.COL_MODEL).loc['d', 'social']
        self.assertAlmostEqual(d, -0.8)

    def test_range_per_model_from_expected(self):
        frame = self.frame.copy()
        a = frame[cons.COL_MODEL] == 'a'
        frame.loc[a, 'parity_gender_expected'] = 0.8                       # model a: longer lists, E higher
        frame.loc[a, 'parity_gender_adj'] = frame.loc[a, 'parity_gender'] - 0.8
        scores = tr.model_scores(frame, TECH, self.social).set_index(cons.COL_MODEL)
        self.assertAlmostEqual(scores.loc['a', 'social_min'], -1.3)          # -(0.8 + 0.5)
        self.assertAlmostEqual(scores.loc['a', 'social_max'], 0.7)           # 2 - 1.3
        self.assertAlmostEqual(scores.loc['b', 'social_max'], 1.0)
        self.assertEqual(tr.axis_range(scores.reset_index(), 'social'), (-1.3, 1.0))
        self.assertEqual(tr.splits(scores.reset_index(), 'absolute'), (1.5, 0.0))   # still 0, not the middle
        self.assertTrue((scores['social'] <= scores['social_max'] + 1e-12).all())
        self.assertTrue((scores['social'] >= scores['social_min'] - 1e-12).all())

    def test_lower_is_better_adjusted(self):
        from metrics import summary
        metric = summary.Metric('x_adj', 'x', summary.LOWER, 'social', 'x_adj')
        frame = pd.DataFrame({cons.COL_MODEL: ['a', 'a'], 'x_adj': [0.1, 0.2], 'x_expected': [0.3, 0.5]})
        r = tr.metric_range(frame, metric).loc['a']
        self.assertAlmostEqual(r['low'], -0.6)                                # E - 1, E = 0.4
        self.assertAlmostEqual(r['high'], 0.4)

    def test_fallback_without_expected(self):
        frame = self.frame.drop(columns=[k + '_expected' for k in SOCIAL])
        with self.assertWarns(UserWarning):
            scores = tr.model_scores(frame, TECH, self.social)
        self.assertEqual(tr.axis_range(scores, 'social'), (-2.0, 2.0))

    def test_ceilings_drawn(self):
        import matplotlib.pyplot as plt
        from viz import rq_plots
        for fig in (rq_plots.plot_rq3(self.scores, 'absolute', technical=TECH, social=self.social, ceilings=True),
                    rq_plots.plot_rq3_inset(self.scores, technical=TECH, social=self.social, ceilings=True)):
            self.assertGreaterEqual(len(fig.axes[0].lines), 2 + 2 * len(self.scores))   # splits + line and tick per model
            plt.close(fig)

    def test_warning_when_mixed(self):
        import warnings
        frame = configs()
        frame['similarity'] = frame['parity_gender']
        for key in ('parity_gender', 'parity_ethnicity', 'diversity_gender', 'similarity'):
            if key not in frame:
                frame[key] = frame['parity_gender']
            frame[key + '_adj'] = frame[key] - 0.5
            frame[key + '_expected'] = 0.5
        for social in (['parity_gender_adj', 'parity_ethnicity'],              # raw + adj mixed
                       ['parity_gender_adj', 'similarity_adj'], ['parity_gender', 'similarity']):   # no direction
            with self.assertWarns(UserWarning):
                tr.model_scores(frame, TECH, social)
        with warnings.catch_warnings():
            warnings.simplefilter('error')                                       # higher-is-better sums: no warning
            tr.model_scores(frame, TECH, ['parity_gender_adj', 'parity_ethnicity_adj', 'diversity_gender_adj'])
            tr.model_scores(frame, TECH, SOCIAL + ['diversity_gender'])

    def test_label_and_name(self):
        from metrics import summary
        from viz import names, rq_plots
        self.assertEqual(tr.axis_label(summary.adjusted_keys(cons.RQ3_SOCIAL_METRICS)), '$\\sum$ parity$^{adj}$')
        self.assertEqual(tr.axis_label(['parity_gender_adj']), 'parity$^{adj}$')
        self.assertEqual(tr.axis_label(TECH), 'validity + refusals$^c$ + duplicates$^c$')
        self.assertEqual(names.rq3_figure_name('absolute', None, summary.adjusted_keys(cons.RQ3_SOCIAL_METRICS),
                                               'inset', 'access'), 'rq3_absolute_inset-access_adj')
        self.assertEqual(names.rq3_figure_name('relative', TECH, self.social),
                         'rq3_relative_validity-refusals-duplicates-vs-parity-gender-parity-ethnicity_scatter_adj')
        fig = rq_plots.plot_rq3(self.scores, 'absolute', technical=TECH, social=self.social)
        self.assertEqual(fig.axes[0].get_ylim(), (-1.0, 1.0))
        self.assertIn('parity$^{adj}$', fig.axes[0].get_ylabel())
        fig = rq_plots.plot_rq3_inset(self.scores, technical=TECH, social=self.social)
        self.assertEqual(fig.axes[0].get_ylim(), (-1.0, 1.0))


class TestLabelPlacement(unittest.TestCase):
    def test_no_overlaps_and_deterministic(self):
        from viz import rq_plots
        rng = np.random.default_rng(0)
        points = rng.normal([300, 200], [40, 25], size=(40, 2))                 # a dense cluster
        sizes = [(float(w), 9.0) for w in rng.integers(30, 80, size=40)]
        axes = (0.0, 0.0, 700.0, 450.0)
        dot = 3.0
        a = rq_plots._place_labels(sizes, points, axes, [], dot)
        b = rq_plots._place_labels(sizes, points, axes, [], dot)
        np.testing.assert_array_equal(a, b)                                      # deterministic
        boxes = [(x - w / 2, y - h / 2, x + w / 2, y + h / 2) for (x, y), (w, h) in zip(a, sizes)]
        overlap = lambda p, q: p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]
        for i in range(len(boxes)):
            self.assertTrue(axes[0] <= boxes[i][0] and boxes[i][2] <= axes[2] and axes[1] <= boxes[i][1] and boxes[i][3] <= axes[3])
            for j in range(i + 1, len(boxes)):
                self.assertFalse(overlap(boxes[i], boxes[j]), (i, j))           # no label on another label
            for k, (x, y) in enumerate(points):
                self.assertFalse(overlap(boxes[i], (x - dot, y - dot, x + dot, y + dot)), (i, k))   # nor on a point



class TestStudyChoiceAtCallTime(unittest.TestCase):
    """The RQ3 defaults are read when called, so they follow summary.use_study_ethnicity (review §7, R-a)."""

    def test_defaults_follow_the_study_attribute(self):
        from metrics import summary
        from viz import names
        try:
            summary.use_study_ethnicity(cons.STUDY_ETHNICITY_RACE)
            _, social = tr.axis_metrics()
            self.assertIn('parity_race', social)
            self.assertNotIn('parity_ethnicity', social)
            self.assertEqual(names.rq3_figure_name('absolute'), 'rq3_absolute_scatter')     # default sets: no selection
        finally:
            summary.use_study_ethnicity(cons.DEFAULT_STUDY_ETHNICITY)
        self.assertIn('parity_ethnicity', tr.axis_metrics()[1])


if __name__ == '__main__':
    unittest.main()
