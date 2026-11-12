"""
Unit tests for analysis/multiple_testing.py (Wald p from the bootstrap SE, Holm / BH / BY against statsmodels, the
significance rule of the heatmaps) and for the heatmaps that fade on it (RQ1, RQ2, interaction cells).

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
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from analysis import multiple_testing as mt                              # noqa: E402


class TestCorrections(unittest.TestCase):
    def test_match_statsmodels(self):
        from statsmodels.stats.multitest import multipletests
        p = np.array([0.001, 0.04, 0.03, 0.2, np.nan, 0.01, 0.5])
        ok = np.isfinite(p)
        for method, name in (('holm', 'holm'), ('bh', 'fdr_bh'), ('by', 'fdr_by')):
            got = mt.adjust(p, method)
            self.assertTrue(np.isnan(got[~ok]).all())
            np.testing.assert_allclose(got[ok], multipletests(p[ok], method=name)[1])

    def test_order_of_strictness(self):
        p = np.random.default_rng(0).uniform(0, 0.05, 50)
        self.assertTrue((mt.adjust(p, 'bh') <= mt.adjust(p, 'by') + 1e-15).all())

    def test_wald_p(self):
        self.assertAlmostEqual(float(mt.wald_p(1.96, 1.0, 'two')), 0.05, places=3)
        self.assertAlmostEqual(float(mt.wald_p(1.645, 1.0, 'greater')), 0.05, places=3)
        self.assertAlmostEqual(float(mt.wald_p(-1.96, 1.0, 'two')), 0.05, places=3)
        self.assertTrue(np.isnan(mt.wald_p(1.0, 0.0)))
        self.assertTrue(np.isnan(mt.wald_p(1.0, np.nan)))


class TestSignificant(unittest.TestCase):
    def table(self):
        """One strong estimate (z = 10), one borderline (z = 2.2: CI excludes 0, not after correction), 98 nulls."""
        rng = np.random.default_rng(1)
        est = np.r_[10.0, 2.2, rng.normal(0, 1, 98)]
        return pd.DataFrame({'deviation': est, 'deviation_std': 1.0,
                             'deviation_ci_low': est - 1.96, 'deviation_ci_high': est + 1.96})

    def test_corrected_vs_uncorrected(self):
        t = self.table()
        raw, by = mt.significant(t, 'deviation', None), mt.significant(t, 'deviation', 'by')
        self.assertTrue(raw[0] and raw[1])
        self.assertTrue(by[0])
        self.assertFalse(by[1])                                          # borderline: faded after BY
        self.assertLessEqual(by.sum(), raw.sum())

    def test_one_sided(self):
        t = pd.DataFrame({'omega2': [0.1, -0.01], 'omega2_std': [0.01, 0.01],
                          'omega2_ci_low': [0.08, -0.03], 'omega2_ci_high': [0.12, 0.01]})
        np.testing.assert_array_equal(mt.significant(t, 'omega2', 'by', sided='greater'), [True, False])
        np.testing.assert_array_equal(mt.significant(t, 'omega2', None, sided='greater'), [True, False])

    def test_add_p_values_columns(self):
        out = mt.add_p_values(self.table(), 'deviation')
        for column in ['p_deviation', 'p_deviation_holm', 'p_deviation_bh', 'p_deviation_by']:
            self.assertIn(column, out)


class TestHeatmaps(unittest.TestCase):
    def test_rq1_and_rq2_heatmaps_with_and_without_correction(self):
        from viz import rq_plots
        rows = []
        for variable in ['model', 'language', 'k']:
            for metric in ['validity', 'refusals']:
                v = 0.3 if variable == 'model' else 0.002
                rows.append({'metric': metric, 'variable': variable, 'family': 'model' if variable == 'model' else
                             'persona' if variable == 'language' else 'context', 'r2_full': 0.5,
                             **{f"{m}{s}": x for m in ('omega2', 'shapley_r2')
                                for s, x in (('', v), ('_ci_low', v - 0.001), ('_ci_high', v + 0.001), ('_std', 0.0008))}})
        results = pd.DataFrame(rows)
        for correction in ('by', None):
            plt.close(rq_plots.plot_rq1_heatmap(results, 'omega2', ['validity', 'refusals'], correction=correction))
        rq2 = []
        for level in ['english', 'german']:
            for metric in ['validity', 'refusals']:
                d = 0.05 if level == 'english' else -0.05
                rq2.append({'metric': metric, 'variable': 'language', 'parent': '', 'level': level, 'family': 'persona',
                            'deviation': d, 'deviation_std': 0.01, 'deviation_ci_low': d - 0.02,
                            'deviation_ci_high': d + 0.02})
        for correction in ('by', None):
            plt.close(rq_plots.plot_rq2_heatmap(pd.DataFrame(rq2), ['validity', 'refusals'], correction=correction))


if __name__ == '__main__':
    unittest.main()
