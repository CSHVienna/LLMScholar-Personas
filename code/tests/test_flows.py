"""
Unit tests for the flows (metrics/flows.py) and their figures (viz/flow_plots.py).

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import matplotlib                                                        # noqa: E402
matplotlib.use('Agg')
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from metrics import flows as fl                                          # noqa: E402
from metrics import social as so                                         # noqa: E402
from utils import constants as cons                                      # noqa: E402

CONFIG = {cons.COL_MODEL: 'm1', cons.COL_DATE: 'd', 'language': 'english', 'role': 'x', 'location': 'Germany',
          'k': 5, 'seniority': 'senior', 'field': 'Physics', 'subfield': 'Education'}


def people(n_models=3, seed=0):
    """Synthetic people table: prompting Germany raises the share of DE people."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_models):
        for location in ['Germany', 'Japan', 'Canada']:
            for i in range(60):
                home = {'Germany': 'DE', 'Japan': 'JP', 'Canada': 'CA'}[location]
                country = home if rng.random() < (0.4 if location == 'Germany' else 0.1) else rng.choice(['US', 'DE', 'GB'])
                rows.append({**CONFIG, cons.COL_MODEL: f"m{m}", 'location': location, 'run_id': i, 'author_id': i,
                             'gender': rng.choice(['male', 'female', 'unknown'], p=[.6, .3, .1]),
                             'ethnicity': rng.choice(['white', 'asian']), 'country': country,
                             'productivity': 'H', 'influence': 'M', 'collaboration': 'L', 'author_seniority': 'senior',
                             'is_top_productivity': False, 'is_top_influence': False})
    return pd.DataFrame(rows)


class TestPeopleTable(unittest.TestCase):
    def test_attributes_from_requested_field_with_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt = Path(tmp)
            members = pd.DataFrame([
                {'author_id': 1, 'field': 'Physics', 'gender': 'female', 'productivity': 'H'},
                {'author_id': 1, 'field': 'Biology', 'gender': 'female', 'productivity': 'L'},
                {'author_id': 2, 'field': 'Biology', 'gender': 'male', 'productivity': 'M'}])   # not in Physics
            members['seniority'] = ['senior', 'senior', 'junior']                # the person's own class
            for c in fl.PERSON_ATTRIBUTES:
                if c not in members and c != fl.AUTHOR_SENIORITY:
                    members[c] = False if c.startswith('is_top') else 'DE' if c == 'country' else 'x'
            members.to_parquet(gt / 'gt_members.parquet')
            pd.DataFrame(columns=['scope', 'attribute', 'class', 'share']).to_parquet(gt / 'gt_shares.parquet')
            responses = pd.DataFrame([{**CONFIG, 'run_id': 1, cons.COL_RESPONSE_TAGS: 'correct'}])
            recs = pd.DataFrame([{**CONFIG, 'run_id': 1, cons.COL_REC_INDEX: i, 'name_key': k, 'first_name': k,
                                  'last_name': k, cons.COL_REC_TAGS: 'correct'} for i, k in [(1, 'a'), (2, 'b'), (3, 'c')]])
            matches = pd.DataFrame([{**CONFIG, 'run_id': 1, cons.COL_REC_INDEX: i, 'matched_author': a is not None,
                                     'author_id': a} for i, a in [(1, 1), (2, 2), (3, None)]])
            table = fl.people_table(responses, recs, matches, so.GroundTruth(gt), cons.DEFAULT_VALID_RESPONSE_TAGS,
                                    cons.DEFAULT_VALID_REC_TAGS).set_index('author_id')
            self.assertEqual(sorted(table.index), [1, 2])                       # the unmatched name is not a person
            self.assertEqual(table.loc[1, 'productivity'], 'H')                 # requested field's row
            self.assertEqual((table.loc[2, 'gender'], table.loc[2, 'productivity']), ('male', 'M'))   # fallback row
            self.assertEqual(table.loc[1, 'location'], 'Germany')               # prompt variables kept
            self.assertEqual((table.loc[1, 'seniority'], table.loc[2, fl.AUTHOR_SENIORITY]), ('senior', 'junior'))


class TestFlows(unittest.TestCase):
    def test_flow_table(self):
        t = fl.flow_table(people(), 'location', 'gender')
        self.assertNotIn('unknown', set(t['target']))
        np.testing.assert_allclose(t.groupby('source')['share'].sum(), 1.0)
        with_unknown = fl.flow_table(people(), 'location', 'gender', include_unknown=True)
        self.assertIn('unknown', set(with_unknown['target']))
        top = fl.flow_table(people(), 'location', 'country', top=2)
        self.assertEqual(len(set(top['target'])), 3)                        # 2 countries + other
        self.assertIn(fl.OTHER, set(top['target']))

    def test_drift(self):
        d = fl.drift(people(), n_boot=30).set_index('location')
        self.assertEqual(d.loc['Germany', 'country'], 'DE')
        self.assertGreater(d.loc['Germany', 'lift'], 1.0)                     # prompting Germany raises DE
        self.assertTrue(d.loc['Germany', 'share_ci_low'] <= d.loc['Germany', 'share'] <= d.loc['Germany', 'share_ci_high'])
        with self.assertRaises(ValueError):
            fl.drift(people().assign(location='Atlantis'), n_boot=0)

    def test_plots(self):
        from viz import flow_plots as fp
        p = people().assign(origin=lambda d: np.where(d[cons.COL_MODEL] == 'm0', 'USA', 'CHN'))
        fig = fp.plot_flows(p, [('location', 'gender'), ('k', 'ethnicity'), ('origin', 'country')])
        self.assertEqual(len(fig.axes), 3)
        fp.plot_drift(p, fl.drift(p, n_boot=5))
        for kwargs, n in (({'source': 'location'}, len(fl.TARGETS)), ({'target': 'gender', 'sources': ['location', 'k']}, 2)):
            fig = fp.plot_flow_grid(p, ncols=3, **kwargs)
            self.assertEqual(len(fig.axes), n)
        with self.assertRaises(ValueError):
            fp.plot_flow_grid(p)
        fig = fp.plot_drift_flow(p)
        self.assertEqual(len(fig.axes), 1)
        countries = fp.country_table(p, 'location', 'country', False)
        self.assertTrue({'DE', 'JP', 'CA'} <= set(countries['target']))       # prompt countries never pooled
        order, colors = fp.target_order_colors('country', ['US', 'JP', 'DE', 'GB'])
        self.assertEqual(order[:2], ['DE', 'JP'])                               # prompt countries first (design order)
        order, colors = fp.target_order_colors('gender', ['female', 'male', 'other'], ['male', 'neutral', 'female'])
        self.assertEqual(order, ['male', 'female', 'other'])
        self.assertNotEqual(colors['male'], colors['female'])


if __name__ == '__main__':
    unittest.main()
