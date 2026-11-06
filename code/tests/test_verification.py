"""
Unit tests for the verification annotation tasks (annotation/verification.py): sampling on a tiny matching / GT
setup, items shown, and the evaluation (adjudication per label group, weighted rates).

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

from annotation import verification as ve                               # noqa: E402
from annotation.tasks import ITEM_ID, assign_item_ids                    # noqa: E402
from utils import constants as cons                                      # noqa: E402

CONFIG = {cons.COL_MODEL: 'm1', cons.COL_DATE: 'd', 'language': 'english', 'role': 'x', 'location': 'y', 'k': 5,
          'seniority': 'senior', 'field': 'Physics', 'subfield': 'Education', 'run_id': 1}


class TestSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.matching, cls.candidates, cls.authors, cls.meta, cls.gt = (root / d for d in
                                                                      ('match', 'cand', 'authors', 'meta', 'gt'))
        for d in (cls.matching, cls.candidates, cls.authors, cls.meta, cls.gt):
            d.mkdir()
        rows = [  # rec_index, first, last, rec_name, rule, author_id (linked), tags
            (1, 'Ada', 'Lovelace', 'ada|lovelace', 'exact_display', 11, 'correct'),
            (2, 'Yasuhiro', 'Tanaka', 'yasuhiro|tanaka', 'fuzzy', None, 'correct'),     # name only: no pick
            (3, 'Nobody', 'Known', 'nobody|known', None, None, 'correct'),
            (4, 'A', 'P', 'a|p', 'initials', 13, 'initials_only'),                        # invalid recommendation
        ]
        recs = pd.DataFrame([{**CONFIG, 'rec_index': i, 'first_name': f, 'last_name': l, 'rec_name': n,
                              'match_rule': r, 'author_id': a, 'author_id_field': None, 'author_id_seniority': None,
                              'matched_name': r is not None} for i, f, l, n, r, a, _ in rows])
        recs['author_id'] = recs['author_id'].astype(float)
        recs.to_parquet(cls.matching / 'm1.parquet')
        pd.DataFrame([{**CONFIG, 'rec_index': i, cons.COL_REC_TAGS: t} for i, *_, t in rows]
                     ).to_parquet(cls.meta / 'm1.parquet')
        (cls.matching / '_params.json').write_text('{"jw_full": 0.9, "jw_first": 0.9, "jw_last": 0.9, "seed": 1}')
        cand = [('ada|lovelace', 11, True, None, None, None), ('yasuhiro|tanaka', 12, False, 0.95, 0.92, 1.0),
                ('yasuhiro|tanaka', 14, False, 0.91, 0.80, 1.0),                # below jw_first: not shown
                ('nobody|known', 15, False, 0.85, 0.7, 0.9), ('a|p', 13, False, None, None, None)]
        pd.DataFrame([{'rec_name': n, 'author_id': a, 'exact_display': e, 'exact_alternative': False, 'short': False,
                       'initials': n == 'a|p', 'jw_full': f, 'jw_first': fi, 'jw_last': la}
                      for n, a, e, f, fi, la in cand]).to_parquet(cls.candidates / 'bucket-00.parquet')
        pd.DataFrame([{'author_id': a, 'display_name': d, 'orcid': None, 'first_pub_year': 2000, 'last_pub_year': 2024,
                       'n_works': 10, 'n_citations': 100, 'h_index': 5, 'n_works_topic': 10, 'works_physics': 8,
                       'country': 'DE', 'last_known_country': 'DE'}
                      for a, d in [(11, 'Ada Lovelace'), (12, 'Yasushi Tanaka'), (14, 'Y Tanaka'), (15, 'Nob Knowles'),
                                   (13, 'Peter Andrews')]]).to_parquet(cls.authors / 'bucket-00.parquet')
        pd.DataFrame([{'author_id': a, 'field': f, 'field_share': 0.8, 'seniority': s, 'country': 'DE'}
                      for a, f, s in [(11, 'Physics', 'senior'), (12, 'Physics', 'senior'), (14, 'Physics', 'junior'),
                                      (15, 'Biology', 'junior')]]).to_parquet(cls.gt / 'gt_members.parquet')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_matching_sample(self):
        sample = ve.sample_matching(self.matching, self.candidates, self.authors, 5, 100, 1, None, self.meta,
                                    cons.DEFAULT_VALID_REC_TAGS).set_index('rec_name')
        self.assertNotIn('a|p', sample.index)                                       # invalid recommendation left out
        self.assertEqual(sample.loc['ada|lovelace', ve.STRATUM], 'exact_display|linked')
        self.assertEqual(sample.loc['yasuhiro|tanaka', ve.STRATUM], 'fuzzy|name')
        self.assertEqual(sample.loc['yasuhiro|tanaka', 'shown_author_ids'], [12])   # 14 fails jw_first
        self.assertEqual(sample.loc['nobody|known', 'shown_author_ids'], [15])      # closest name, below thresholds
        texts = {n: '\n'.join(s.title + '\n' + s.text for s in row['sections']) for n, row in sample.iterrows()}
        self.assertIn('linked by the pipeline', texts['ada|lovelace'])
        self.assertIn('https://openalex.org/A11', texts['ada|lovelace'])
        self.assertIn('Publications: 2000–2024', texts['ada|lovelace'])
        self.assertIn('Nothing linked', texts['nobody|known'])
        key = ve.key_table(ve.MATCHING, assign_item_ids(sample.reset_index(), 'MAT', 1))
        self.assertTrue({ve.STRATUM, ve.STRATUM_SIZE, 'shown_author_ids'} <= set(key.columns))

    def test_authors_sample(self):
        sample = ve.sample_authors(self.gt, self.authors, 1, 100, 1)
        self.assertEqual(sorted(sample[ve.STRATUM]), ['Biology|junior', 'Physics|junior', 'Physics|senior'])
        self.assertEqual(int(sample.loc[sample[ve.STRATUM] == 'Physics|senior', ve.STRATUM_SIZE].iloc[0]), 2)
        text = sample['sections'].iloc[0][1].text
        self.assertIn('Seniority:', text)


class TestEvaluation(unittest.TestCase):
    def test_codebooks_groups(self):
        self.assertEqual(list(ve.groups(ve.MATCHING)), ['match', 'intended'])
        self.assertEqual(list(ve.groups(ve.AUTHORS)), ['field', 'seniority', 'country', 'profile'])
        for task in ve.VERIFICATION_TASKS:
            ids = [l['id'] for l in ve.codebook(task)['labels']]
            self.assertEqual(len(ids), len(set(ids)))

    def test_adjudication_and_weighted_rates(self):
        key = pd.DataFrame({ITEM_ID: ['a', 'b', 'c', 'd'], ve.STRATUM: ['s1', 's1', 's2', 's2'],
                            ve.STRATUM_SIZE: [900, 900, 100, 100]})
        ann = pd.DataFrame([  # item, annotator, tags, unsure
            ('a', 'x', 'correct_match|likely_intended', False), ('a', 'y', 'correct_match', False),
            ('b', 'x', 'correct_match', False), ('b', 'y', 'false_match', False),       # tie: left out
            ('c', 'x', 'false_match', False), ('d', 'x', 'correct_match|false_match', False),   # two in a group: none
            ('d', 'y', 'false_match', True)], columns=[ITEM_ID, 'annotator', 'tags', 'unsure'])
        answers = ve.group_answers(ann, ve.MATCHING)
        gold = ve.adjudicate_groups(answers)
        match = gold[gold['group'] == 'match'].set_index(ITEM_ID)
        self.assertEqual(match.loc['a', 'label'], 'correct_match')
        self.assertTrue(match.loc['b', 'is_tie'])
        self.assertNotIn('d', match.index)
        rates = ve.label_rates(key, gold, ve.MATCHING)
        r = rates[(rates['group'] == 'match') & (rates['label'] == 'correct_match')].set_index('stratum')
        self.assertEqual((r.loc['s1', 'share'], r.loc['s2', 'share']), (1.0, 0.0))
        self.assertAlmostEqual(r.loc['overall (weighted)', 'share'], 0.9)          # 900 * 1 + 100 * 0
        self.assertTrue(0 <= r.loc['s1', 'ci_low'] < 1)

    def test_scopes(self):
        key = pd.DataFrame({ITEM_ID: ['a', 'b', 'c'], ve.STRATUM: ['exact|linked', 'exact|linked', 'none'],
                            ve.STRATUM_SIZE: [10, 10, 1000]})
        gold = pd.DataFrame({ITEM_ID: ['a', 'b', 'c'], 'group': 'match', 'is_tie': False, 'n_annotators': 1,
                             'label': ['correct_match', 'false_match', 'not_in_oa']})
        r = ve.label_rates(key, gold, ve.MATCHING)
        overall = r[r['stratum'] == 'overall (weighted)'].set_index('label')['share']
        self.assertAlmostEqual(overall['correct_match'], 0.5)                      # precision: the large 'none' left out
        self.assertAlmostEqual(overall['not_in_oa'], 1.0)

    def test_wilson(self):
        low, high = ve.wilson(8, 10)
        self.assertTrue(low < 0.8 < high and 0 <= low and high <= 1)
        self.assertTrue(all(np.isnan(ve.wilson(0, 0))))

    def test_agreement(self):
        answers = pd.DataFrame({ITEM_ID: ['a', 'a', 'b', 'b', 'c', 'c'], 'annotator': ['x', 'y'] * 3, 'group': 'match',
                                'label': ['correct_match', 'correct_match', 'false_match', 'false_match',
                                          'correct_match', 'false_match']})
        alpha = ve.group_agreement(answers).set_index('group').loc['match']
        self.assertEqual(alpha['n_items_multiple_annotators'], 3)
        self.assertTrue(-1 <= alpha['krippendorff_alpha'] < 1)


if __name__ == '__main__':
    unittest.main()
