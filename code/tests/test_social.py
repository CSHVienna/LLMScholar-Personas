"""
Unit tests for the social metrics (metrics/social.py) on a hand-made GT and two responses.

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from metrics import social as so                                         # noqa: E402
from metrics import technical as tm                                      # noqa: E402
from utils import constants as cons                                      # noqa: E402

CONFIG = {cons.COL_MODEL: 'm1', cons.COL_DATE: 'd', 'language': 'english', 'role': 'x', 'location': 'Germany',
          'k': 5, 'seniority': 'senior', 'field': 'Physics', 'subfield': None}


class TestMeasures(unittest.TestCase):
    def test_parity(self):
        p, q = pd.Series({'female': 0.5, 'male': 0.5}), pd.Series({'female': 0.2, 'male': 0.8})
        self.assertAlmostEqual(so.parity(p, q), 0.7)
        self.assertEqual(so.parity(q, q), 1.0)
        self.assertAlmostEqual(so.parity(pd.Series({'a': 1.0}), pd.Series({'b': 1.0})), 0.0)   # disjoint classes

    def test_unknown_handling(self):
        d = pd.Series({'female': 0.25, 'male': 0.25, 'unknown': 0.5})
        self.assertEqual(so.known(d, False).to_dict(), {'female': 0.5, 'male': 0.5})
        self.assertEqual(so.known(d, True).to_dict(), d.to_dict())

    def test_diversity(self):
        self.assertAlmostEqual(so.diversity(pd.Series({'a': 0.5, 'b': 0.5}), 2, 2), 1.0)
        self.assertAlmostEqual(so.diversity(pd.Series({'a': 1.0}), 3, 5), 0.0)
        self.assertTrue(np.isnan(so.diversity(pd.Series({'a': 1.0}), 3, 1)))          # k = 1: undefined

    def test_top_countries(self):
        shares = pd.Series({'US': 0.3, 'CN': 0.2, 'DE': 0.1, 'FR': 0.1, 'unknown': 0.3})
        self.assertEqual(so.top_countries(shares, 1), {'US'})
        self.assertEqual(so.top_countries(shares, 2), {'US', 'CN'})
        self.assertEqual(so.top_countries(shares, 3), {'US', 'CN', 'DE', 'FR'})      # FR ties with the 3rd
        self.assertEqual(so.top_countries(shares, 10), {'US', 'CN', 'DE', 'FR'})     # fewer than k: all
        self.assertNotIn('unknown', so.top_countries(shares, 5))

    def test_connectedness(self):
        self.assertEqual(so.connectedness([1, 2, 3], [(1, 2), (2, 3)]), 1.0)          # one component
        self.assertAlmostEqual(so.connectedness([1, 2, 3], []), 0.0)                  # nobody coauthored
        self.assertTrue(0 < so.connectedness([1, 2, 3, 4], [(1, 2)]) < 1)
        self.assertTrue(np.isnan(so.connectedness([1], [])))

    def test_similarity(self):
        self.assertAlmostEqual(so.similarity(np.array([[1.0, 0], [2.0, 0]])), 1.0)
        self.assertAlmostEqual(so.similarity(np.array([[1.0, 0], [0, 1.0]])), 0.0)
        self.assertTrue(np.isnan(so.similarity(np.array([[1.0, 0]]))))


class TestPipeline(unittest.TestCase):
    """GT: 4 Physics seniors (2 female, 1 male, 1 unknown gender; 3 in DE, 1 in FR); response 1 names authors 1 and 2 (+ a duplicate,
    an unmatched name); response 2 is invalid."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        gt = Path(cls.tmp.name)
        members = pd.DataFrame([{'author_id': a, 'field': 'Physics', 'seniority': 'senior', 'gender': g,
                                 'ethnicity': 'white', 'country': 'FR' if a == 3 else 'DE', 'productivity': b, 'influence': b,
                                 'collaboration': b, 'is_top_productivity': a == 1, 'is_top_influence': False}
                                for a, g, b in [(1, 'female', 'H'), (2, 'female', 'L'), (3, 'male', 'L'), (4, 'unknown', 'M')]])
        members.to_parquet(gt / 'gt_members.parquet')
        shares = []
        for attribute, column in so.ATTRIBUTES.items():
            counts = members[column].value_counts()
            shares += [{'scope': 'field_x_seniority', 'field': 'Physics', 'seniority': 'senior', 'attribute': column,
                        'class': c, 'n': n, 'share': n / 4} for c, n in counts.items()]
        for name in so.POPULARITY:
            share = members[f"is_top_{name}"].mean()
            shares += [{'scope': 'field_x_seniority', 'field': 'Physics', 'seniority': 'senior', 'attribute': f"top_{name}",
                        'class': 'top', 'n': 1, 'share': share}]
        pd.DataFrame(shares).to_parquet(gt / 'gt_shares.parquet')
        pd.DataFrame([{'author_id': a, **{f: float(a) for f in so.PCA_FEATURES}} for a in [1, 2, 3, 4]]
                     ).to_parquet(gt / 'gt_authors.parquet')
        (gt / 'coauthor_edges').mkdir()
        pd.DataFrame({'author_id': [1, 2], 'coauthor_id': [2, 1]}).to_parquet(gt / 'coauthor_edges' / 'bucket-00.parquet')
        cls.gt = so.GroundTruth(gt)

        cls.responses = pd.DataFrame([{**CONFIG, cons.COL_RUN_ID: 1, cons.COL_RESPONSE_TAGS: 'correct', 'n_persons': 4},
                                      {**CONFIG, cons.COL_RUN_ID: 2, cons.COL_RESPONSE_TAGS: 'invalid', 'n_persons': 0}])
        recs = [(1, 1, 'ada lovelace', 1), (1, 2, 'marie curie', 2), (1, 3, 'ada lovelace', 1), (1, 4, 'fake person', None)]
        cls.recs = pd.DataFrame([{**CONFIG, cons.COL_RUN_ID: r, cons.COL_REC_INDEX: i, 'name_key': k, 'first_name': k,
                                  'last_name': k, cons.COL_REC_TAGS: 'correct'} for r, i, k, _ in recs])
        cls.matches = pd.DataFrame([{**CONFIG, cons.COL_RUN_ID: r, cons.COL_REC_INDEX: i, 'matched_author': a is not None,
                                     'author_id': a} for r, i, _, a in recs])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def people(self):
        return so.response_people(self.responses, self.recs, self.matches, cons.DEFAULT_VALID_RESPONSE_TAGS,
                                  cons.DEFAULT_VALID_REC_TAGS)

    def test_people_unique_and_matched(self):
        people = self.people()
        self.assertEqual(sorted(people['author_id']), [1, 2])                # duplicate once, unmatched name out

    def test_metrics(self):
        metrics, gaps = so.social_metrics(self.responses, self.people(), self.gt)
        m = metrics.set_index(cons.COL_RUN_ID)
        self.assertEqual(m.loc[1, 'n_people'], 2)
        # gender: people {female: 1.0}; target without unknown {female: 2/3, male: 1/3} -> 1 - 1/3
        self.assertAlmostEqual(m.loc[1, 'parity_gender'], 2 / 3)
        self.assertAlmostEqual(m.loc[1, 'diversity_gender'], 0.0)             # both female
        self.assertAlmostEqual(m.loc[1, 'popularity_productivity'], 0.5)      # author 1 in the top group
        self.assertEqual(m.loc[1, 'connectedness'], 1.0)                      # 1 and 2 coauthored
        self.assertEqual(m.loc[1, 'popularity_location'], 1.0)                # both in DE, the top country (.75)
        self.assertEqual(m.loc[1, 'locality'], 1.0)                           # both from the prompted Germany
        self.assertAlmostEqual(m.loc[1, 'locality_expected'], 0.75)          # GT share of DE: 3 of 4
        self.assertNotIn('locality_ratio', m.columns)
        self.assertAlmostEqual(m.loc[1, 'locality_adj'], 0.25)
        self.assertTrue(np.isnan(m.loc[2, 'parity_gender']) and m.loc[2, 'n_people'] == 0)
        g = gaps[(gaps[cons.COL_RUN_ID] == 1) & (gaps['attribute'] == 'gender')].set_index('class')
        self.assertAlmostEqual(g.loc['female', 'gap'], 1 - 2 / 3)
        self.assertNotIn('unknown', g.index)

    def test_locality_unmapped_location(self):
        responses = self.responses.assign(location='Atlantis')
        people = self.people().assign(location='Atlantis')
        metrics, _ = so.social_metrics(responses, people, self.gt)
        self.assertTrue(np.isnan(metrics.set_index(cons.COL_RUN_ID).loc[1, 'locality']))   # no country code: NaN

    def test_include_unknown(self):
        metrics, _ = so.social_metrics(self.responses, self.people(), self.gt, include_unknown=True)
        # target with unknown {female: 0.5, male: 0.25, unknown: 0.25}; people {female: 1} -> 1 - 0.5
        self.assertAlmostEqual(metrics.set_index(cons.COL_RUN_ID).loc[1, 'parity_gender'], 0.5)


if __name__ == '__main__':
    unittest.main()
