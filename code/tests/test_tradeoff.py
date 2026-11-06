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
        from viz import rq_plots
        for mode in cons.RQ3_MODES:
            fig = rq_plots.plot_rq3(self.scores.reset_index(), mode, pareto=True, technical=TECH, social=SOCIAL)
            self.assertEqual(len(fig.axes), 1)
        self.assertEqual(rq_plots.rq3_figure_name('absolute'), 'rq3_absolute')
        self.assertEqual(rq_plots.rq3_figure_name('relative', TECH, SOCIAL),
                         'rq3_relative_validity-refusals-duplicates_parity_gender-parity_ethnicity')


if __name__ == '__main__':
    unittest.main()
