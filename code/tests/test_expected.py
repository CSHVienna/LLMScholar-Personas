"""
Unit tests for the size-adjusted social metrics (metrics/expected.py, social.py, summary.py): expected values under a
random recommender, the <metric>_adj columns and their registry.

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from metrics import expected as ex                                       # noqa: E402
from metrics import social as so                                         # noqa: E402
from metrics import summary as su                                        # noqa: E402
from utils import constants as cons                                      # noqa: E402

TOL = 0.01      # Monte Carlo error with LABEL_DRAWS draws is ~0.002


def exact(population: pd.Series, q: pd.Series, n: int, include_unknown: bool):
    """Exact expected parity / diversity by enumerating every sequence of n labels (small cases only)."""
    parities, diversities = [], []
    for labels in itertools.product(population.index, repeat=n):
        prob = float(np.prod([population[l] for l in labels]))
        counts = pd.Series(labels).value_counts()
        p = so.known(counts / counts.sum(), include_unknown)
        n_known = int(counts.sum() if include_unknown else counts.drop(so.UNKNOWN, errors='ignore').sum())
        parities.append((prob, so.parity(p, q)))
        diversities.append((prob, so.diversity(p, len(q), n_known)))
    mean = lambda pairs: (sum(w * v for w, v in pairs if np.isfinite(v)) / sum(w for w, v in pairs if np.isfinite(v))
                          if any(np.isfinite(v) for _, v in pairs) else np.nan)
    return mean(parities), mean(diversities)


class TestLabels(unittest.TestCase):
    def test_worked_example(self):
        """70 % male / 30 % female: n = 1 -> parity sum(q^2) = 0.58; n = 2 -> parity 0.706, diversity 0.42."""
        q = pd.Series({'male': 0.7, 'female': 0.3})
        rng = np.random.default_rng(0)
        parity, diversity = ex.label_expectation(q, q, 1, False, rng)
        self.assertAlmostEqual(parity, 0.58, delta=TOL)
        self.assertTrue(np.isnan(diversity))                                  # one person: undefined
        parity, diversity = ex.label_expectation(q, q, 2, False, rng)
        self.assertAlmostEqual(parity, 0.706, delta=TOL)
        self.assertAlmostEqual(diversity, 0.42, delta=TOL)

    def test_matches_exact_enumeration(self):
        population = pd.Series({'female': 0.3, 'male': 0.4, 'neutral': 0.1, 'unknown': 0.2})
        rng = np.random.default_rng(1)
        for include_unknown in (False, True):
            q = so.known(population, include_unknown)
            for n in (1, 2, 3, 4):
                got = ex.label_expectation(population, q, n, include_unknown, rng)
                want = exact(population, q, n, include_unknown)
                for g, w in zip(got, want):
                    if np.isnan(w):
                        self.assertTrue(np.isnan(g))
                    else:
                        self.assertAlmostEqual(g, w, delta=TOL, msg=f"n={n}, include_unknown={include_unknown}")

    def test_target_of_another_scope(self):
        """Draws from the field x seniority, compared with a broader target (parity_target robustness)."""
        population, q = pd.Series({'a': 1.0}), pd.Series({'a': 0.5, 'b': 0.5})
        parity, _ = ex.label_expectation(population, q, 3, False, np.random.default_rng(2))
        self.assertAlmostEqual(parity, 0.5)

    def test_expected_parity_grows_with_n(self):
        q = pd.Series({'male': 0.6, 'female': 0.3, 'unknown': 0.1})
        rng = np.random.default_rng(3)
        values = [ex.label_expectation(q, so.known(q, False), n, False, rng)[0] for n in (1, 5, 20)]
        self.assertTrue(values[0] < values[1] < values[2] < 1)


class TestPeople(unittest.TestCase):
    def test_connectedness_curve(self):
        lists = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
        complete = [(a, b) for a, b in itertools.combinations([1, 2, 3, 4], 2)]
        self.assertEqual(ex.connectedness_curve(lists, complete, 4), {2: 1.0, 3: 1.0, 4: 1.0})
        for value in ex.connectedness_curve(lists, [], 4).values():
            self.assertAlmostEqual(value, 0.0)
        curve = ex.connectedness_curve(lists, [(1, 2)], 4)                      # one edge
        self.assertAlmostEqual(curve[2], 0.5)                                   # first list connected, second not
        self.assertAlmostEqual(curve[4], so.connectedness([1, 2, 3, 4], [(1, 2)]))
        self.assertTrue(np.isnan(ex.connectedness_curve(lists[:, :2], [], 4)[3]))   # lists shorter than n

    def test_sample_members_filters_before_sampling(self):
        """DuckDB samples before a WHERE of the same SELECT: the group must be filtered first."""
        import types
        import duckdb
        with tempfile.TemporaryDirectory() as tmp:
            members = pd.DataFrame({'author_id': range(20_000), 'field': ['Physics', 'Biology'] * 10_000,
                                    'seniority': ['senior'] * 10_000 + ['junior'] * 10_000})
            members.to_parquet(Path(tmp) / 'gt_members.parquet')
            gt = types.SimpleNamespace(con=duckdb.connect(), dir=Path(tmp))
            ids = ex.sample_members(gt, 'Physics', 'senior', 2_000, seed=1)
            group = set(members[(members['field'] == 'Physics') & (members['seniority'] == 'senior')]['author_id'])
            self.assertEqual(len(ids), 2_000)
            self.assertEqual(len(set(ids)), 2_000)                              # without replacement
            self.assertTrue(set(ids) <= group)
            self.assertEqual(len(ex.sample_members(gt, 'Physics', 'senior', 50_000, seed=1)), len(group))

    def test_mean_pairwise_cosine_is_similarity(self):
        vectors = np.random.default_rng(4).normal(size=(30, 4))
        vectors[3] = 0.0                                                        # a zero vector counts as cosine 0
        self.assertAlmostEqual(ex.mean_pairwise_cosine(vectors), so.similarity(vectors))
        self.assertTrue(np.isnan(ex.mean_pairwise_cosine(vectors[:1])))


class TestAdjusted(unittest.TestCase):
    """A small GT (Physics seniors, as in test_social) through expected_table, the social metrics and the registry."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / 'groundtruth'
        gt = root / 'population' / 'main'
        gt.mkdir(parents=True)
        members = pd.DataFrame([{'author_id': a, 'field': 'Physics', 'seniority': 'senior', 'gender': g,
                                 'ethnicity': 'white', 'country': 'DE', 'productivity': b, 'influence': b,
                                 'collaboration': b, 'is_top_productivity': a == 1, 'is_top_influence': False}
                                for a, g, b in [(1, 'female', 'H'), (2, 'female', 'L'), (3, 'male', 'L'), (4, 'unknown', 'M')]])
        members.to_parquet(gt / 'gt_members.parquet')
        shares = []
        for column in so.ATTRIBUTES.values():
            counts = members[column].value_counts()
            shares += [{'scope': 'field_x_seniority', 'field': 'Physics', 'seniority': 'senior', 'attribute': column,
                        'class': c, 'n': n, 'share': n / 4} for c, n in counts.items()]
        for name in so.POPULARITY:
            shares += [{'scope': 'field_x_seniority', 'field': 'Physics', 'seniority': 'senior', 'attribute': f"top_{name}",
                        'class': 'top', 'n': 1, 'share': members[f"is_top_{name}"].mean()}]
        pd.DataFrame(shares).to_parquet(gt / 'gt_shares.parquet')
        pd.DataFrame([{'author_id': a, **{f: float(a) for f in so.PCA_FEATURES}} for a in [1, 2, 3, 4]]
                     ).to_parquet(gt / 'gt_authors.parquet')
        (root / 'coauthor_edges').mkdir()                                       # shared, in the GT root
        pd.DataFrame({'author_id': [1, 2], 'coauthor_id': [2, 1]}).to_parquet(root / 'coauthor_edges' / 'bucket-00.parquet')
        cls.gt_dir = gt
        cls.gt = so.GroundTruth(gt)
        cls.table = ex.expected_table(cls.gt, [('Physics', 'senior')], 'field_x_seniority', False, seed=7, n_max=4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_table(self):
        t = self.table
        self.assertEqual(set(t['metric']), set(ex.SIMULATED))
        par = t[(t['metric'] == 'parity_gender')].set_index('n')['expected']
        q = pd.Series({'female': 2 / 3, 'male': 1 / 3})
        self.assertAlmostEqual(par[1], float((q ** 2).sum()), delta=TOL)        # n = 1: sum of squared shares
        conn = t[t['metric'] == 'connectedness'].set_index('n')['expected']
        self.assertTrue(conn.dropna().between(0, 1).all() and conn.notna().any())   # edges found in the GT root
        sim = t[t['metric'] == 'similarity']['expected'].unique()
        self.assertEqual(len(sim), 1)                                           # the same for every n
        again = ex.expected_table(self.gt, [('Physics', 'senior')], 'field_x_seniority', False, seed=7, n_max=4)
        pd.testing.assert_frame_equal(self.table, again)                        # reproducible

    def test_adjusted_columns(self):
        config = {cons.COL_MODEL: 'm1', cons.COL_DATE: 'd', 'language': 'english', 'role': 'x', 'location': 'y',
                  'k': 5, 'seniority': 'senior', 'field': 'Physics', 'subfield': None}
        responses = pd.DataFrame([{**config, cons.COL_RUN_ID: 1, cons.COL_RESPONSE_TAGS: 'correct', 'n_persons': 2}])
        people = pd.DataFrame([{**config, cons.COL_RUN_ID: 1, 'author_id': a} for a in (1, 2)])
        expected = ex.Expected(self.table)
        metrics, _ = so.social_metrics(responses, people, self.gt, expectations=expected)
        row = metrics.iloc[0]
        for key in ex.SIMULATED:
            want = row[key] - expected.value(key, 'Physics', 'senior', 2)
            self.assertTrue(np.isnan(want) and np.isnan(row[key + '_adj']) or np.isclose(row[key + '_adj'], want), key)
        self.assertAlmostEqual(row['popularity_productivity_adj'], 0.5 - 0.25)  # observed 1 of 2, expected 1 of 4
        self.assertEqual(sorted(c for c in metrics if c.endswith('_adj')), sorted(so.ADJUSTED_KEYS))
        self.assertTrue(np.isclose(expected.value('parity_gender', 'Physics', 'senior', 99),
                                   expected.value('parity_gender', 'Physics', 'senior', 4)))   # capped at n_max
        self.assertTrue(np.isnan(expected.value('parity_gender', 'Physics', 'senior', 0)))

    def test_load_checks_the_settings(self):
        out = Path(self.tmp.name) / 'metrics'
        path = ex.expected_file(out)
        path.parent.mkdir(parents=True)
        self.table.to_parquet(path)
        (path.parent / '_params.json').write_text(json.dumps(ex.depends_on(self.gt_dir, False, 'field_x_seniority')))
        self.assertEqual(ex.load(out, self.gt_dir, False, 'field_x_seniority').n_max, 4)
        with self.assertRaises(ValueError):
            ex.load(out, self.gt_dir, True, 'field_x_seniority')               # other unknown handling
        with self.assertRaises(FileNotFoundError):
            ex.load(Path(self.tmp.name) / 'none', self.gt_dir, False, 'field_x_seniority')

    def test_registry(self):
        for key in so.ADJUSTED_KEYS:
            self.assertIn(key, su.METRICS)
            self.assertEqual(su.METRICS[key].group, su.METRICS[key[:-4]].group + '_adj')
        self.assertEqual([m.key for m in su.select_metrics('parity')], cons.METRIC_COLUMN_SETS['parity'])
        self.assertEqual([m.key for m in su.select_metrics('paper', adjusted=True)], cons.METRIC_COLUMN_SETS['paper_adj'])
        self.assertIn('validity', cons.METRIC_COLUMN_SETS['all_adj'])            # technical metrics unchanged
        self.assertEqual(su.adjusted_selection('paper', True), 'paper_adj')
        self.assertEqual(su.adjusted_selection('technical', True), 'technical')         # no social metrics: as is
        self.assertEqual(su.adjusted_selection(None, True), 'all_adj')
        self.assertIsNone(su.adjusted_selection(None, False))
        self.assertEqual(su.adjusted_selection(['validity', 'parity_gender'], True), ['validity', 'parity_gender_adj'])
        from viz import rq_plots
        self.assertEqual(rq_plots.rq1_figure_name('bar', 'omega2', 'paper'), 'rq1_omega2_paper_bar')
        self.assertEqual(rq_plots.rq1_figure_name('bar', 'omega2', 'paper_adj'), 'rq1_omega2_paper_bar_adj')
        self.assertEqual(rq_plots.rq2_figure_name('heatmap', 'deviation', 'all_adj'), 'rq2_deviation_all_heatmap_adj')
        self.assertEqual(rq_plots.rq1_figure_name('heatmap', 'omega2', None), 'rq1_omega2_all_heatmap')
        self.assertEqual(rq_plots.rq1_figure_name('bar', 'omega2', ['validity', 'parity_gender_adj']),
                         'rq1_omega2_validity-parity_gender_bar_adj')
        self.assertEqual(su.split_adjusted('technical'), ('technical', False))
        results = pd.DataFrame({'metric': ['validity', 'parity_gender', 'parity_gender_adj']})
        self.assertEqual(rq_plots.rq1_metric_keys(results), ['validity', 'parity_gender'])   # default: raw only
        self.assertEqual(rq_plots.rq1_metric_keys(results, 'parity_adj'), ['parity_gender_adj'])


if __name__ == '__main__':
    unittest.main()
