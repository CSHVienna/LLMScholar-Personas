"""
Unit tests for utils/stats.py (percentile bootstrap summaries shared by every analysis).

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from utils import stats                                                  # noqa: E402


class TestPercentile(unittest.TestCase):
    def test_tails_and_ci(self):
        np.testing.assert_allclose(stats.tails(0.95), (0.025, 0.975))
        draws = np.arange(101, dtype=float)
        np.testing.assert_allclose(stats.percentile_ci(draws, 0.9), (5.0, 95.0))
        self.assertTrue(np.isnan(stats.percentile_ci(np.array([1.0, np.nan]), 0.9)[0]))      # NaN kept by default
        self.assertEqual(stats.percentile_ci(np.array([1.0, np.nan]), 0.9, skipna=True), (1.0, 1.0))

    def test_summary_drops_non_finite(self):
        s = stats.percentile_summary([1.0, 2.0, 3.0, np.nan, np.inf], 0.5)
        self.assertEqual((s['ci_low'], s['ci_high'], s['n_boot']), (1.5, 2.5, 3))
        self.assertAlmostEqual(s['std'], 1.0)
        empty = stats.percentile_summary([np.nan], 0.95)
        self.assertTrue(np.isnan(empty['ci_low']) and np.isnan(empty['std']) and empty['n_boot'] == 0)

    def test_grouped(self):
        draws = pd.DataFrame({'g': ['a'] * 3 + ['b'] * 3, 'v': [1.0, 2.0, 3.0, 1.0, np.nan, 3.0]})
        out = stats.grouped_percentile_summary(draws, 'g', 'v', 0.5)
        self.assertEqual(list(out.columns), ['ci_low', 'ci_high', 'std', 'n_boot'])
        self.assertEqual((out.at['a', 'ci_low'], out.at['a', 'n_boot']), (1.5, 3))
        self.assertTrue(np.isnan(out.at['b', 'ci_low']))                                  # a NaN draw: no interval
        skipped = stats.grouped_percentile_summary(draws, 'g', 'v', 0.5, skipna=True)
        self.assertEqual((skipped.at['b', 'ci_low'], skipped.at['b', 'n_boot']), (1.5, 2))


if __name__ == '__main__':
    unittest.main()
