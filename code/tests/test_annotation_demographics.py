"""
Unit tests for the demographics annotation task (annotation/demographics.py): codebook, stratified sampling from
synthetic A.4 / B.6 outputs, blind items, and the weighted evaluation.

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import numpy as np                                                        # noqa: E402
import pandas as pd                                                       # noqa: E402

from annotation import demographics as dm                                 # noqa: E402
from annotation import verification                                       # noqa: E402
from annotation.tasks import ITEM_ID, assign_item_ids                     # noqa: E402
from utils import constants as cons                                       # noqa: E402

TMP_ROOT = Path(cons.DEFAULT_TMP_DIR) / 'tests'


def synthetic(folder: Path) -> None:
    """20 GT scholars: 12 male|white, 6 female|asian, 2 neutral|unknown; authors 1, 2 and 13 were recommended."""
    gender = ['male'] * 12 + ['female'] * 6 + ['neutral'] * 2
    ethnicity = ['white'] * 12 + ['asian'] * 6 + ['unknown'] * 2
    pd.DataFrame({'author_id': range(1, 21), 'display_name': [f"Name{i} Middle Last{i}" for i in range(1, 21)],
                  'gender': gender, 'ethnicity': ethnicity, 'p_female': np.linspace(0, 1, 20),
                  **{f'p_{c}': 0.2 for c in ['asian', 'black', 'hispanic', 'white', 'other']}}
                 ).to_parquet(folder / 'gt_authors.parquet')
    matching = folder / 'matching'
    matching.mkdir()
    pd.DataFrame({'author_id': [1, 2, None, 13, 1]}).to_parquet(matching / 'model-a.parquet')


class DemographicsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        cls.dir = Path(cls.tmp.name)
        synthetic(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def sample(self, per_stratum=3, max_items=100):
        s = dm.sample_demographics(self.dir, self.dir / 'matching', per_stratum, max_items, seed=1, threads=1,
                                   memory_limit='1GB')
        return assign_item_ids(s, 'DEM', seed=1)

    def test_codebook(self):
        cb = dm.codebook()
        ids = [l['id'] for l in cb['labels']]
        self.assertEqual(len(ids), len(set(ids)))                          # 'unknown' twice, ids unique
        self.assertIn('gender_unknown', ids)
        self.assertIn('ethnicity_unknown', ids)
        self.assertEqual(set(l['group'] for l in cb['labels']), {'gender', 'ethnicity'})
        self.assertEqual(dm.class_of('ethnicity_unknown'), ('ethnicity', 'unknown'))
        self.assertEqual(len(ids), 13)                                     # the tool has 18 keyboard keys
        self.assertIn('ethnicity_mena', ids)
        self.assertNotIn('ethnicity_white', ids)                           # european replaces white
        for mapping in cons.ANNOTATION_ETHNICITY_MAPPINGS.values():          # every fine class is mapped
            self.assertEqual(set(mapping), set(cons.ANNOTATION_ETHNICITY_CLASSES))
            self.assertTrue(set(mapping.values()) <= set(cons.ETHNICITY_CLASSES))

    def test_sample_strata_and_sides(self):
        s = self.sample()
        self.assertEqual(s['author_id'].nunique(), len(s))                 # a person once
        self.assertTrue(set(s['side']) <= {dm.SIDE_GT, dm.SIDE_RECOMMENDED})
        self.assertLessEqual(s.groupby(dm.STRATUM).size().max(), 3)
        sizes = s.groupby(dm.STRATUM)[dm.STRATUM_SIZE].first()
        self.assertEqual(sizes['gt|male|white'], 12)
        self.assertEqual(sizes['recommended|male|white'], 2)               # authors 1, 2 (1 twice in B.6)
        self.assertEqual(sizes['recommended|female|asian'], 1)             # author 13
        self.assertEqual(s.loc[s['author_id'] == 13, 'side'].iloc[0], dm.SIDE_RECOMMENDED)
        capped = self.sample(per_stratum=100, max_items=8)                 # 5 strata: quota lowered to 1
        self.assertLessEqual(len(capped), 8)
        self.assertEqual(capped.groupby(dm.STRATUM).size().max(), 1)

    def test_items_are_blind(self):
        s = self.sample()
        payload = dm.items_payload(cons.ANNOTATION_TASK_DEMOGRAPHICS, s)
        text = json.dumps(payload['items'])
        for word in ['male', 'white', 'asian', 'p_female', 'recommended']:
            self.assertNotIn(word, text)
        item = payload['items'][0]
        self.assertEqual([sec['title'] for sec in item['sections']], ['Name'])
        self.assertIn(item['sections'][0]['text'], {f"Name{i} Middle Last{i}" for i in range(1, 21)})   # one line
        key = dm.key_table(cons.ANNOTATION_TASK_DEMOGRAPHICS, s)
        self.assertIn('gender', key)
        self.assertIn(dm.STRATUM_SIZE, key)

    def test_without_matching(self):
        s = dm.sample_demographics(self.dir, self.dir / 'missing', 2, 100, seed=1, threads=1, memory_limit='1GB')
        self.assertEqual(set(s['side']), {dm.SIDE_GT})

    def test_evaluation(self):
        key = pd.DataFrame({ITEM_ID: ['a', 'b', 'c', 'd'], 'side': 'gt',
                            'gender': ['male', 'male', 'female', 'neutral'],
                            'ethnicity': ['white', 'white', 'asian', 'unknown'],
                            dm.STRATUM: ['gt|male|white', 'gt|male|white', 'gt|female|asian', 'gt|neutral|unknown'],
                            dm.STRATUM_SIZE: [100, 100, 10, 1]})
        sep = cons.TAG_SEPARATOR
        annotations = pd.DataFrame({
            ITEM_ID: ['a', 'b', 'c', 'd', 'a'], 'annotator': ['x', 'x', 'x', 'x', 'y'],
            'tags': [f'gender_male{sep}ethnicity_european', f'gender_female{sep}ethnicity_mena',
                     f'gender_female{sep}ethnicity_east_asian', 'gender_unknown', f'gender_male{sep}ethnicity_hispanic'],
            'unsure': [False, False, False, False, False]})
        answers = verification.group_answers_of(annotations, dm.groups())
        gold = verification.adjudicate_groups(answers)
        items = dm.weighted_items(key, gold)
        g = items[items['group'] == 'gender'].set_index(ITEM_ID)
        self.assertEqual(g.loc['a', 'annotated'], 'male')                  # both annotators agree
        self.assertAlmostEqual(g.loc['a', 'weight'], 50.0)                 # 100 people / 2 items answered
        e = items[items['group'] == 'ethnicity'].set_index(ITEM_ID)
        self.assertNotIn('a', e.index)                                     # european vs hispanic: tie, left out
        self.assertEqual(e.loc['b', 'annotated'], 'mena')                  # fine class kept before mapping
        self.assertNotIn('d', e.index)                                     # ethnicity not answered
        summary = dm.summary(items).set_index('group')
        # gender: a right (w 50), b wrong (w 50), c right (w 10), d wrong (unknown vs neutral, w 1)
        self.assertAlmostEqual(summary.loc['gender', 'accuracy_weighted'], 60 / 111)
        self.assertAlmostEqual(summary.loc['gender', 'annotated_unknown_weighted'], 1 / 111)
        c = dm.confusion(items)
        cell = c[(c['group'] == 'gender') & (c['annotated'] == 'female') & (c['inferred'] == 'male')].iloc[0]
        self.assertAlmostEqual(cell['p_inferred_given_annotated'], 50 / 60)      # annotated female: b (w 50), c (w 10)
        male = c[(c['group'] == 'gender') & (c['inferred'] == 'male')]
        self.assertAlmostEqual(male['p_annotated_given_inferred'].sum(), 1.0)

    def test_mappings(self):
        key = pd.DataFrame({ITEM_ID: ['a', 'b'], 'side': 'gt', 'gender': ['male', 'male'], 'ethnicity': ['white', 'asian'],
                            dm.STRATUM: ['s1', 's2'], dm.STRATUM_SIZE: [10, 10]})
        sep = cons.TAG_SEPARATOR
        annotations = pd.DataFrame({ITEM_ID: ['a', 'b'], 'annotator': 'x', 'unsure': False,
                                    'tags': [f'gender_male{sep}ethnicity_mena', f'gender_male{sep}ethnicity_south_asian']})
        gold = verification.adjudicate_groups(verification.group_answers_of(annotations, dm.groups()))
        out = dm.evaluate(key, gold)
        acc = out['summary'][out['summary']['group'] == 'ethnicity'].set_index('mapping')['accuracy_weighted']
        self.assertAlmostEqual(acc['main'], 1.0)                           # mena -> white, south_asian -> asian
        self.assertAlmostEqual(acc['mena_asian'], 0.5)                     # mena -> asian: item a now wrong
        self.assertAlmostEqual(acc['fine'], 0.0)                           # unmapped: fine classes never equal
        self.assertEqual(set(out['summary']['mapping']), set(dm.mappings()))
        gender_rows = out['summary'][out['summary']['group'] == 'gender']
        self.assertEqual(gender_rows['mapping'].tolist(), ['main'])        # gender reported once


if __name__ == '__main__':
    unittest.main()
