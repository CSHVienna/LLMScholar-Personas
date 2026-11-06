"""
Unit tests for the technical metrics (configuration + response) on a hand-made configuration.

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

from metrics import technical as tm                                      # noqa: E402
from utils import constants as cons                                      # noqa: E402

CONFIG = {cons.COL_MODEL: 'm1', cons.COL_DATE: '2025-01-01', 'language': 'english', 'role': 'x', 'location': 'y',
          'k': 3, 'seniority': 'senior', 'field': 'Physics', 'subfield': None}
VALID_RESPONSES = cons.DEFAULT_VALID_RESPONSE_TAGS
VALID_RECS = cons.DEFAULT_VALID_REC_TAGS

# run: (response tags, number of person records)
RESPONSES = {1: ('correct', 4), 2: ('wrapped', 2), 3: ('refused', 0), 4: ('invalid', 0)}
# (run, rec_index, name_raw, lastname_raw, first_name, last_name, name_key, rec tags, matched name/field/seniority/author)
RECS = [
    (1, 1, 'John', 'Smith', 'John', 'Smith', 'john smith', 'correct', (True, True, True, True)),
    (1, 2, 'Ada', 'Lovelace', 'Ada', 'Lovelace', 'ada lovelace', 'correct', (True, False, False, False)),
    (1, 3, 'John', 'Smith', 'John', 'Smith', 'john smith', 'correct', (True, True, True, True)),     # duplicate
    (1, 4, '[Name]', '[Lastname]', None, None, None, 'invalid', (False, False, False, False)),        # placeholder
    (2, 1, 'Smith', 'John', 'Smith', 'John', 'john smith', 'correct', (True, True, True, True)),     # swapped
    (2, 2, 'Ada', 'Lovelace', 'Ada', 'Lovelace', 'ada lovelace', 'correct', (True, False, False, False)),
]


def frames():
    responses = pd.DataFrame([{**CONFIG, cons.COL_RUN_ID: run, cons.COL_RESPONSE_TAGS: tags, 'n_persons': n}
                              for run, (tags, n) in RESPONSES.items()])
    recs = pd.DataFrame([{**CONFIG, cons.COL_RUN_ID: run, cons.COL_REC_INDEX: i, 'name_raw': nr, 'lastname_raw': lr,
                          'first_name': fn, 'last_name': ln, 'name_key': key, cons.COL_REC_TAGS: tags}
                         for run, i, nr, lr, fn, ln, key, tags, _ in RECS])
    matches = pd.DataFrame([{**CONFIG, cons.COL_RUN_ID: run, cons.COL_REC_INDEX: i,
                             **dict(zip([f"matched_{l}" for l in tm.LEVELS], m))}
                            for run, i, *_, m in RECS])
    return responses, recs, matches


class TestHelpers(unittest.TestCase):
    def test_outcome_order(self):
        tags = pd.Series(['correct', 'wrapped|python_literal', 'refused', 'truncated|wrapped', 'invalid', None])
        self.assertEqual(tm.outcome(tags, VALID_RESPONSES).tolist(),
                         ['valid', 'valid', 'refused', 'invalid', 'invalid', 'invalid'])

    def test_pairwise_agreement(self):
        self.assertEqual(tm.pairwise_agreement([10, 0, 0]), 1.0)
        self.assertAlmostEqual(tm.pairwise_agreement([7, 3, 0]), (21 + 3) / 45)
        self.assertTrue(np.isnan(tm.pairwise_agreement([1, 0, 0])))

    def test_written_order_key(self):
        self.assertEqual(tm.written_order_key('Candidate 1', 'Lastname 2'), 'candidate 1|lastname 2')
        self.assertNotEqual(tm.written_order_key('Candidate 1', 'Lastname 2'), tm.written_order_key('Candidate 2', 'Lastname 1'))


class TestConfigMetrics(unittest.TestCase):
    def setUp(self):
        self.responses, self.recs, _ = frames()

    def metrics(self, **kwargs):
        return tm.config_metrics(self.responses, self.recs, VALID_RESPONSES, **kwargs).iloc[0]

    def test_counts_and_behaviour(self):
        m = self.metrics()
        self.assertEqual((m.n_runs, m.n_valid, m.n_refused, m.n_invalid), (4, 2, 1, 1))
        self.assertEqual((m.validity, m.refusals), (0.5, 0.25))
        self.assertAlmostEqual(m.behaviour_consistency, 1 / 6)          # one pair (the 2 valid runs) of 6
        self.assertEqual(m.format_consistency, 0.0)                     # 4 different tag sets

    def test_recommendation_consistency(self):
        m = self.metrics()                                              # run 1: john smith, ada lovelace, placeholder
        self.assertEqual(m.n_valid_runs, 2)                             # run 2: john smith (swapped), ada lovelace
        self.assertAlmostEqual(m.recommendation_consistency, 2 / 3)     # placeholder counts; duplicates once
        m = self.metrics(order_sensitive_keys=True)                     # 'smith|john' != 'john|smith'
        self.assertAlmostEqual(m.recommendation_consistency, 1 / 4)     # {ada} / {john|smith, smith|john, ada, ph}
        m = self.metrics(consistency_all_runs=True)                     # + two empty sets (refused, invalid)
        self.assertEqual(m.n_valid_runs, 4)
        self.assertAlmostEqual(m.recommendation_consistency, (2 / 3 + 0 + 0 + 0 + 0 + 1) / 6)


class TestResponseMetrics(unittest.TestCase):
    def setUp(self):
        self.responses, self.recs, self.matches = frames()

    def metrics(self, **kwargs):
        out = tm.response_metrics(self.responses, self.recs, self.matches, VALID_RESPONSES, VALID_RECS, **kwargs)
        return out.set_index(cons.COL_RUN_ID)

    def test_k_and_duplicates(self):
        m = self.metrics()
        self.assertEqual((m.loc[1, 'k_ratio'], m.loc[2, 'k_ratio']), (4 / 3, 2 / 3))   # all person records count
        self.assertTrue(np.isnan(m.loc[3, 'k_ratio']))                                  # refused: not valid
        self.assertEqual((m.loc[1, 'n_valid_recs'], m.loc[1, 'n_unique_names']), (3, 2))
        self.assertAlmostEqual(m.loc[1, 'duplicates'], 1 / 3)
        self.assertEqual(m.loc[2, 'duplicates'], 0.0)
        self.assertTrue(np.isnan(m.loc[4, 'duplicates']))

    def test_factuality_on_unique_names(self):
        m = self.metrics()
        self.assertEqual((m.loc[1, 'factuality_name'], m.loc[1, 'factuality_author']), (1.0, 0.5))
        self.assertEqual(m.loc[1, 'n_factual_author'], 1)
        self.assertTrue(np.isnan(m.loc[3, 'factuality_author']))
        kept = self.metrics(keep_duplicates=True)
        self.assertAlmostEqual(kept.loc[1, 'factuality_author'], 2 / 3)               # John Smith counted twice

    def test_missing_matching_fails(self):
        with self.assertRaises(ValueError):
            tm.response_metrics(self.responses, self.recs, self.matches.iloc[1:], VALID_RESPONSES, VALID_RECS)


if __name__ == '__main__':
    unittest.main()
