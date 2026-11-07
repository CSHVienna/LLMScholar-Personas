"""
Unit tests for the GT descriptives (groundtruth/stats.py) and their figures (viz/gt_plots.py), on tiny synthetic
A.1 / A.4 outputs (same column names as the real files).

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import matplotlib                                                         # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402
import pandas as pd                                                       # noqa: E402

from groundtruth import stats as gs                                       # noqa: E402
from utils import constants as cons                                       # noqa: E402
from viz import gt_plots                                                  # noqa: E402

TMP_ROOT = Path(cons.DEFAULT_TMP_DIR) / 'tests'


def synthetic_gt(folder: Path) -> None:
    """3 scholars in Physics only, 2 in Physics and Biology, 1 in Biology; plus one organisation and one author
    outside the prompt fields in the A.1 authors file."""
    authors = pd.DataFrame({
        'author_id': [1, 2, 3, 4, 5, 6],
        'fields': [['Physics'], ['Physics'], ['Physics'], ['Physics', 'Biology'], ['Physics', 'Biology'], ['Biology']],
        'seniority': ['junior', 'junior', 'senior', 'mid', 'senior', 'junior'],
        'academic_age': [2, 8, 30, 15, 25, 5],
        'n_works': [1, 3, 50, 10, 80, 2], 'n_citations': [0, 5, 900, 40, 1500, 1],
        'n_coauthors': [1, 4, 60, 10, 90, 2], 'h_index': [0, 1, 15, 4, 20, 1],
        'citations_per_paper_per_year': [0.0, 0.2, 0.6, 0.3, 0.75, 0.1],
        'gender_first_name': ['Anna', 'Li', None, 'Kim', 'Bob', 'Xq'],
        'p_female': [0.95, 0.5, np.nan, 0.4, 0.02, np.nan],
        'gender': ['female', 'neutral', 'unknown', 'neutral', 'male', 'unknown'],
        'p_asian': [0.1, 0.8, np.nan, 0.6, 0.1, 0.3], 'p_black': [0.1, 0.05, np.nan, 0.1, 0.1, 0.3],
        'p_hispanic': [0.1, 0.05, np.nan, 0.1, 0.1, 0.2], 'p_white': [0.6, 0.05, np.nan, 0.1, 0.6, 0.1],
        'p_other': [0.1, 0.05, np.nan, 0.1, 0.1, 0.1],
        'ethnicity': ['white', 'asian', 'unknown', 'asian', 'white', 'unknown'],
        'country': ['US', 'DE', 'US', 'US', None, 'JP'],
        'last_known_country': ['US', 'DE', 'FR', 'US', 'DE', 'JP'],
    })
    authors.to_parquet(folder / 'gt_authors.parquet')
    members = authors.explode('fields').rename(columns={'fields': 'field'})[
        ['author_id', 'field', 'seniority', 'gender', 'ethnicity']].reset_index(drop=True)
    # Physics: 1 L, 2 L, 3 H, 4 M, 5 H | Biology: 4 L, 5 H, 6 M; author 5 is the top group everywhere
    members['productivity'] = ['L', 'L', 'H', 'M', 'L', 'H', 'H', 'M']
    members['influence'] = members['productivity']
    members['collaboration'] = ['L', 'L', 'H', 'M', 'L', 'H', 'H', 'unknown']
    members['is_top_productivity'] = members['author_id'] == 5
    members['is_top_influence'] = members['author_id'] == 5
    members.to_parquet(folder / 'gt_members.parquet')
    a1 = pd.DataFrame({'author_id': range(1, 9), 'is_person': [True] * 6 + [False, True]})
    (folder / 'authors').mkdir()
    a1.to_parquet(folder / 'authors' / 'bucket-00.parquet')
    members['country'] = ['US', 'DE', 'FR', 'unknown', 'US', 'unknown', 'US', 'JP']
    shares = []
    for attribute in ['gender', 'ethnicity', 'country', 'productivity', 'influence', 'collaboration']:
        for keys, scope in [(['field', 'seniority'], 'field_x_seniority'), (['seniority'], 'seniority')]:
            for key, g in members.groupby(keys):
                field, seniority = key if len(keys) == 2 else (None, key[0])
                for c, n in g[attribute].value_counts().items():
                    shares.append((scope, field, seniority, attribute, c, n))
    shares = pd.DataFrame(shares, columns=['scope', 'field', 'seniority', 'attribute', 'class', 'n'])
    shares['share'] = shares['n'] / shares.groupby(['scope', 'field', 'seniority', 'attribute'], dropna=False)['n'].transform('sum')
    shares.to_parquet(folder / 'gt_shares.parquet')
    pd.DataFrame({'field': ['Physics'] * 3, 'seniority': ['junior'] * 3, 'attribute': ['productivity'] * 3,
                  'metric': ['n_works'] * 3, 'class': ['L', 'M', 'H'], 'n': [2, 1, 1],
                  'min_value': [1, 2, 4], 'max_value': [1, 3, 9]}).to_parquet(folder / 'gt_cutoffs.parquet')


class GTStatsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT)
        cls.dir = Path(cls.tmp.name)
        synthetic_gt(cls.dir)
        cls.con = gs.connect(threads=1, memory_limit='1GB', temp_dir=str(cls.dir / 'duckdb'))

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def test_funnel(self):
        f = gs.funnel(self.con, self.dir)
        self.assertEqual(f['n'].tolist(), [8, 7, 6, 8])
        self.assertAlmostEqual(f['share_of_previous'].iloc[1], 7 / 8)

    def test_population_order_and_all_fields(self):
        p = gs.population(self.con, self.dir)
        self.assertEqual(p['field'].tolist()[:3], ['Physics'] * 3)          # design order: Physics before Biology
        self.assertEqual(p['field'].iloc[-1], gs.ALL_FIELDS)
        phys = p[p['field'] == 'Physics'].set_index('seniority')['n']
        self.assertEqual(phys.to_dict(), {'junior': 2, 'mid': 1, 'senior': 2})
        self.assertEqual(p.loc[p['field'] == gs.ALL_FIELDS, 'n'].sum(), 8)  # memberships pooled
        self.assertEqual(p[p['field'] == 'Physics']['seniority'].tolist(), ['junior', 'mid', 'senior'])

    def test_population_by_gender(self):
        p = gs.population(self.con, self.dir, by='gender')
        phys = p[p['field'] == 'Physics'].set_index('gender')['n'].to_dict()
        self.assertEqual(phys, {'female': 1, 'neutral': 2, 'unknown': 1, 'male': 1})
        self.assertEqual(p['field'].iloc[-1], gs.ALL_FIELDS)
        self.assertEqual(p.loc[p['field'] == gs.ALL_FIELDS, 'n'].sum(), 8)
        with self.assertRaises(ValueError):
            gs.population(self.con, self.dir, by='country')
        self.assertEqual(gt_plots.count_unit(5e6), (1e6, 'millions'))
        self.assertEqual(gt_plots.count_unit(12), (1.0, ''))

    def test_fields_and_overlap(self):
        self.assertEqual(gs.fields_per_scholar(self.con, self.dir).set_index('n_fields')['n'].to_dict(), {1: 4, 2: 2})
        o = gs.field_overlap(self.con, self.dir)
        self.assertEqual(o.loc['Physics', 'Physics'], 5)
        self.assertEqual(o.loc['Biology', 'Biology'], 3)
        self.assertEqual(o.loc['Physics', 'Biology'], 2)
        self.assertEqual(o.loc['Biology', 'Physics'], 2)

    def test_bibliometric_quantiles(self):
        q = gs.bibliometric_quantiles(self.con, self.dir, metrics=['academic_age', 'n_works'])
        row = q[(q['field'] == 'Physics') & (q['seniority'] == 'junior') & (q['metric'] == 'academic_age')].iloc[0]
        self.assertEqual(row['n'], 2)
        self.assertAlmostEqual(row['mean'], 5.0)
        self.assertTrue(row['q05'] <= row['q50'] <= row['q95'])
        self.assertEqual(q['metric'].astype(str).unique().tolist(), ['academic_age', 'n_works'])

    def test_bibliometrics_by_gender(self):
        q = gs.bibliometric_quantiles(self.con, self.dir, metrics=['n_works'], by='gender')
        row = q[(q['field'] == 'Physics') & (q['seniority'] == 'senior') & (q['gender'] == 'male')].iloc[0]
        self.assertEqual(row['n'], 1)
        self.assertAlmostEqual(row['q50'], 80)
        self.assertIn(gs.ALL_FIELDS, set(q['field']))
        with self.assertRaises(ValueError):
            gs.bibliometric_quantiles(self.con, self.dir, by='country')

    def test_shares_by_prominence(self):
        t = gs.shares_by_prominence(self.con, self.dir, 'gender')
        self.assertNotIn('unknown', set(t['class']))
        # Physics junior, productivity L: authors 1 (female) and 2 (neutral)
        cell = t[(t['field'] == 'Physics') & (t['seniority'] == 'junior') & (t['group'] == 'productivity')
                 & (t['level'] == 'L')].set_index('class')['share']
        self.assertEqual(cell.to_dict(), {'female': 0.5, 'neutral': 0.5})
        top = t[(t['group'] == 'top_influence') & (t['field'] == 'Physics') & (t['seniority'] == 'senior')]
        self.assertEqual(top[top['level'] == 'top'].set_index('class')['share'].to_dict(), {'male': 1.0})
        # 'all' = every known member of the cell (author 3 is unknown): the reference of the top group
        self.assertEqual(top[top['level'] == 'all'].set_index('class')['n'].to_dict(), {'male': 1})
        # collaboration 'unknown' bins are left out
        self.assertNotIn('unknown', set(t.loc[t['group'] == 'collaboration', 'level']))
        sums = t.groupby(['field', 'seniority', 'group', 'level'])['share'].sum()
        self.assertTrue(np.allclose(sums, 1))
        with_unknown = gs.shares_by_prominence(self.con, self.dir, 'ethnicity', include_unknown=True)
        self.assertIn('unknown', set(with_unknown['class']))

    def test_histogram(self):
        h = gs.histogram(self.con, self.dir, 'academic_age', 0, 40, 4)
        self.assertEqual(h['n'].tolist(), [3, 1, 1, 1])                       # [0,10) [10,20) [20,30) [30,40]
        self.assertEqual(h['left'].tolist(), [0, 10, 20, 30])
        clipped = gs.histogram(self.con, self.dir, 'academic_age', 0, 20, 2)   # 25 and 30 go to the last bin
        self.assertEqual(clipped['n'].tolist(), [3, 3])
        by = gs.histogram(self.con, self.dir, 'p_female', 0, 1, 2, by_field=True)
        self.assertEqual(by.groupby('field')['n'].sum().to_dict(), {'Physics': 4, 'Biology': 2})   # NaN left out
        eth = gs.histogram(self.con, self.dir, 'p_ethnicity_max', 0, 1, 10)
        self.assertEqual(eth['n'].sum(), 5)
        asian = gs.histogram(self.con, self.dir, 'p_asian', 0, 1, 10)
        self.assertEqual(asian.loc[asian['n'] > 0, 'bin'].tolist(), [1, 3, 6, 8])   # 0.1 (x2), 0.3, 0.6, 0.8
        self.assertEqual(asian['n'].sum(), 5)
        with self.assertRaises(ValueError):
            gs.histogram(self.con, self.dir, 'display_name; DROP', 0, 1, 2)

    def test_gender_coverage(self):
        g = gs.gender_coverage(self.con, self.dir).set_index('label')['n']
        self.assertEqual(g['unknown: no usable first name'], 1)
        self.assertEqual(g['unknown: first name not in nomquamgender'], 1)
        self.assertEqual(g['neutral'], 2)

    def test_composition(self):
        shares = pd.read_parquet(self.dir / 'gt_shares.parquet')
        c = gs.composition(shares, 'gender')
        self.assertTrue(np.allclose(c.sum(axis=1), 1))
        self.assertEqual(c.index[-1][0], gs.ALL_FIELDS)
        known = gs.composition(shares, 'gender', include_unknown=False)
        self.assertNotIn('unknown', known.columns)
        self.assertTrue(np.allclose(known.sum(axis=1), 1))
        countries = gs.composition(shares, 'country', top=1, keep=['DE'])
        self.assertEqual(set(countries.columns), {'US', 'DE', 'other', 'unknown'})   # top 1 = US; FR, JP pooled
        self.assertIn('DE', countries.columns)                                 # kept (prompt country)
        with self.assertRaises(ValueError):
            gs.composition(shares, 'nothing')

    def test_country_ranking(self):
        r = gs.country_ranking(self.con, self.dir)
        pooled = r[r['field'] == gs.ALL_FIELDS].set_index('country')       # each scholar once
        self.assertEqual(pooled['n_scholars'].to_dict(), {'US': 3, 'DE': 1, 'JP': 1, 'unknown': 1})
        self.assertEqual(pooled.loc['US', 'n_works'], 61)
        self.assertEqual(pooled.loc['US', 'n_citations'], 940)
        self.assertAlmostEqual(pooled.loc['US', 'share_n_scholars'], 3 / 6)  # of all, unknown included
        phys = r[r['field'] == 'Physics'].set_index('country')['n_scholars'].to_dict()
        self.assertEqual(phys, {'US': 3, 'DE': 1, 'unknown': 1})
        last = gs.country_ranking(self.con, self.dir, column='last_known_country')
        self.assertEqual(last[last['field'] == gs.ALL_FIELDS].set_index('country')['n_scholars'].to_dict(),
                         {'US': 2, 'DE': 2, 'FR': 1, 'JP': 1})
        with self.assertRaises(ValueError):
            gs.country_ranking(self.con, self.dir, column='institution')
        order = gs.country_order(r, 'n_scholars')                          # known only, renormalised
        self.assertEqual(order['country'].tolist(), ['US', 'DE', 'JP'])    # tie DE / JP: alphabetical
        self.assertAlmostEqual(order['share'].iloc[0], 3 / 5)
        self.assertAlmostEqual(order['cumulative'].iloc[-1], 1.0)
        self.assertEqual(gs.country_order(r, 'n_works')['country'].tolist(), ['US', 'DE', 'JP'])
        self.assertEqual(gs.country_order(r, 'h_index_mean', min_scholars=2)['country'].tolist(), ['US'])
        top = gs.country_top(r, ['n_scholars', 'h_index_mean'], top=4)
        self.assertEqual(top.loc[1, 'scholars'], 'US 60.0 %')
        self.assertEqual(top.loc[4, 'scholars'], '')                       # fewer countries than top
        fig = gt_plots.plot_gt_countries(r, 'n_scholars', top=2)          # US, DE + other (1)
        self.assertEqual([t.get_text() for t in fig.axes[0].get_xticklabels()], ['US', 'DE', 'other (1)'])
        plt.close(fig)
        fig = gt_plots.plot_gt_countries(r, 'h_index_median', top=2, min_scholars=1)   # no 'other' for h-index
        self.assertEqual(len(fig.axes[0].get_xticklabels()), 2)
        plt.close(fig)

    def test_cached(self):
        calls = []
        compute = lambda: calls.append(1) or pd.DataFrame({'a': [1, 2]})
        source = self.dir / 'gt_members.parquet'
        cache = self.dir / 'cache'
        first = gs.cached(cache, 't', compute, [source])
        second = gs.cached(cache, 't', compute, [source])
        self.assertEqual(len(calls), 1)
        pd.testing.assert_frame_equal(first, second)
        os.utime(source, (time.time() + 5, time.time() + 5))                 # newer source -> recomputed
        gs.cached(cache, 't', compute, [source])
        self.assertEqual(len(calls), 2)
        gs.cached(None, 't', compute)
        self.assertEqual(len(calls), 3)

    def test_cutoffs_table(self):
        t = gs.cutoffs_table(pd.read_parquet(self.dir / 'gt_cutoffs.parquet'))
        self.assertEqual(t.loc[0, 'ranges'], 'L 1-1 | M 2-3 | H 4-9')
        self.assertAlmostEqual(t.loc[0, 'share_L'], 0.5)

    def test_figures_render(self):
        shares = pd.read_parquet(self.dir / 'gt_shares.parquet')
        figures = [
            gt_plots.plot_gt_population(gs.population(self.con, self.dir)),
            gt_plots.plot_gt_population(gs.population(self.con, self.dir, by='gender'), by='gender'),
            gt_plots.plot_gt_population(gs.population(self.con, self.dir, by='gender'), by='gender', normalize=True),
            gt_plots.plot_gt_population(gs.population(self.con, self.dir, by='ethnicity'), by='ethnicity',
                                        include_unknown=False),
            gt_plots.plot_gt_field_overlap(gs.field_overlap(self.con, self.dir)),
            gt_plots.plot_gt_bibliometrics(gs.bibliometric_quantiles(self.con, self.dir)),
            gt_plots.plot_gt_histogram(gs.histogram(self.con, self.dir, 'academic_age', 0, 40, 4), 'age',
                                       {'junior': 10.5, 'senior': 19.5}),
            gt_plots.plot_gt_histogram(gs.histogram(self.con, self.dir, 'p_female', 0, 1, 10, by_field=True), 'p'),
            gt_plots.plot_gt_composition(shares, ['gender', 'country'], ncols=2),
            gt_plots.plot_gt_bibliometrics_by(
                gs.bibliometric_quantiles(self.con, self.dir, metrics=['n_works', 'academic_age'], by='gender'), 'gender'),
            gt_plots.plot_gt_shares_by_prominence(gs.shares_by_prominence(self.con, self.dir, 'gender'), 'gender', 'female'),
        ]
        for fig in figures:
            fig.canvas.draw()
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
