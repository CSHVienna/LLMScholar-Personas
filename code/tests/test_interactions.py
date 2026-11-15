"""
Unit tests for the interactions (analysis/interactions.py, viz/interaction_plots.py): block statistics against
statsmodels, recovery of a planted interaction, the decomposition share, the global check, the flags, the sparse Gram, and the figures.

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import itertools
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

from analysis import influence as inf                                    # noqa: E402
from analysis import interactions as ix                                  # noqa: E402

BASE = ['model', 'language', 'k', 'location']
VARIABLES = ['language', 'location', 'k']


def data(seed=0, n_models=8, planted=0.15, size_effect=0.1):
    """Crossed design: language x location interact (zh x CN), and k interacts with the model through its size."""
    rng = np.random.default_rng(seed)
    rows = []
    for m, lang, k, loc in itertools.product(range(n_models), ['de', 'en', 'zh'], [1, 5, 10], ['CN', 'DE', 'US']):
        size = 'S' if m < n_models // 2 else 'L'
        for _ in range(3):
            y = (0.1 * m + 0.05 * (k == 10) + planted * (lang == 'zh') * (loc == 'CN')
                 + size_effect * (size == 'S') * (k == 10) + rng.normal(0, 0.05))
            rows.append({'model': f"m{m}", 'language': lang, 'k': k, 'location': loc, 'size': size, 'y': y})
    return pd.DataFrame(rows)


def run(df, n_boot=40, **kwargs):
    return ix.interactions(df, 'y', base=BASE, screening=ix.screening_blocks(VARIABLES),
                           decompositions={('size', 'k'): ('model', 'k')},
                           prespecified=[('language', 'location'), ('model', 'k')], n_boot=n_boot, **kwargs)


class TestBlockStatistics(unittest.TestCase):
    def test_matches_statsmodels(self):
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        df = data()
        table, _ = run(df, n_boot=0)
        base = smf.ols('y ~ C(model) + C(language) + C(k) + C(location)', data=df).fit()
        for block, term in [('language:location', 'C(language):C(location)'), ('model:k', 'C(model):C(k)')]:
            full = smf.ols(f'y ~ C(model) + C(language) + C(k) + C(location) + {term}', data=df).fit()
            comparison = anova_lm(base, full)
            ss, dof = comparison['ss_diff'].iloc[1], comparison['df_diff'].iloc[1]
            ms_error = full.ssr / full.df_resid
            row = table.set_index('block').loc[block]
            self.assertEqual(row['df'], dof)
            self.assertAlmostEqual(row['delta_r2'], full.rsquared - base.rsquared, places=10)
            self.assertAlmostEqual(row['delta_r2_adj'], full.rsquared_adj - base.rsquared_adj, places=10)
            self.assertAlmostEqual(row['f2'], (full.rsquared - base.rsquared) / (1 - full.rsquared), places=10)
            self.assertAlmostEqual(row['omega2'], (ss - dof * ms_error) / (ss + (len(df) - dof) * ms_error), places=10)
        self.assertAlmostEqual(table['r2_base'].iloc[0], base.rsquared, places=10)

    def test_global_matches_statsmodels(self):
        import statsmodels.formula.api as smf
        df = data()
        table, _ = run(df, n_boot=0)
        terms = ' + '.join(f"C({a}):C({b})" for a, b in ix.screening_blocks(VARIABLES))
        base = smf.ols('y ~ C(model) + C(language) + C(k) + C(location)', data=df).fit()
        full = smf.ols(f'y ~ C(model) + C(language) + C(k) + C(location) + {terms}', data=df).fit()
        glob = table.set_index('block').loc[ix.GLOBAL]
        self.assertAlmostEqual(glob['delta_r2'], full.rsquared - base.rsquared, places=10)
        self.assertEqual(glob['df'], full.df_model - base.df_model)

    def test_sparse_gram_equals_dense(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=50)
        blocks = {'a': (rng.random((50, 3)) > 0.5).astype(float), 'b': (rng.random((50, 2)) > 0.5).astype(float)}
        dense = inf.Gram(y, blocks)
        sparse = inf.Gram(y, {'a': blocks['a'], 'b': ix.sp.csr_matrix(blocks['b'])})
        np.testing.assert_allclose(sparse.G, dense.G)
        self.assertAlmostEqual(sparse.rss(['a', 'b']), dense.rss(['a', 'b']), places=10)


class TestRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, cls.cells = run(data())
        cls.flagged = ix.flag(cls.table).set_index('block')

    def test_planted_interactions_flagged_others_not(self):
        self.assertTrue(self.flagged.loc['language:location', 'flag_prespecified'])
        self.assertTrue(self.flagged.loc['language:location', 'flag_screening'])
        self.assertTrue(self.flagged.loc['model:k', 'flag_prespecified'])
        for block in ['language:k', 'location:k', 'model:language', 'model:location']:
            self.assertFalse(self.flagged.loc[block, 'flag_screening'], block)
        self.assertTrue(self.flagged.loc[ix.GLOBAL, 'flag_global'])

    def test_cell_deviation_is_the_double_centred_interaction(self):
        # balanced 3 x 3: a bump d in one cell -> d (1 - 1/3)(1 - 1/3) in that cell after the additive fit
        cell = self.cells[(self.cells['block'] == 'language:location') & (self.cells['level_a'] == 'zh')
                          & (self.cells['level_b'] == 'CN')].iloc[0]
        self.assertAlmostEqual(cell['deviation'], 0.15 * 4 / 9, delta=0.01)
        self.assertLess(cell['deviation_ci_low'], cell['deviation'])
        self.assertGreater(cell['deviation_ci_low'], 0)
        self.assertAlmostEqual(cell['deviation'], cell['observed'] - cell['additive'], places=12)

    def test_model_cells_without_ci(self):
        model_cells = self.cells[self.cells['block'] == 'model:k']
        self.assertEqual(len(model_cells), 8 * 3)
        self.assertTrue(model_cells['deviation_ci_low'].isna().all())

    def test_decomposition_share(self):
        row = self.flagged.loc['size:k']
        self.assertEqual(row['role'], ix.DECOMPOSITION)
        self.assertEqual(row['parent'], 'model:k')
        self.assertGreater(row['share'], 0.8)          # the k effect differs between models only through size
        self.assertLessEqual(row['share'], 1 + 1e-9)
        self.assertTrue(row['reported'])

    def test_no_interaction_not_flagged(self):
        flagged = ix.flag(run(data(seed=2, planted=0.0, size_effect=0.0), n_boot=40)[0]).set_index('block')
        self.assertFalse(flagged['flag_screening'].any())
        self.assertFalse(flagged['flag_prespecified'].any())
        self.assertFalse(flagged.loc['size:k', 'reported'])


class TestCorrection(unittest.TestCase):
    def test_bh_flags_at_least_as_many_as_by(self):
        flagged = ix.flag(run(data())[0])
        self.assertTrue((flagged['p_bh'].dropna() <= flagged['p_by'].dropna()).all())
        self.assertGreaterEqual(flagged['flag_screening_bh'].sum(), flagged['flag_screening'].sum())

    def test_sesoi_per_metric(self):
        self.assertEqual(ix.sesoi('validity'), 0.05)
        self.assertEqual(ix.sesoi('k_balance'), 0.10)
        self.assertEqual(ix.sesoi('parity_gender_adj'), 0.05)

    def test_screening_blocks(self):
        blocks = ix.screening_blocks()
        self.assertEqual(len(blocks), 21)
        self.assertIn(('seniority', 'k'), blocks)
        self.assertTrue(ix.is_prespecified(('location', 'language')))
        self.assertFalse(ix.is_prespecified(('role', 'k')))


class TestFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = data(n_models=6).rename(columns={'y': 'validity'})
        df['locality_adj'] = np.random.default_rng(3).normal(0, 0.05, len(df))
        tables, cells = [], []
        for metric in ['validity', 'locality_adj']:
            t, c = ix.interactions(df, metric, base=BASE, screening=ix.screening_blocks(VARIABLES),
                                   decompositions={('size', 'k'): ('model', 'k')}, n_boot=10)
            tables.append(t)
            cells.append(c)
        cls.table, cls.cells = ix.flag(pd.concat(tables, ignore_index=True)), pd.concat(cells, ignore_index=True)

    def test_heatmap_and_cells(self):
        from matplotlib.colors import to_rgb
        from viz import constants as viscons
        from viz import interaction_plots as ipl
        fig = ipl.plot_interaction_heatmap(self.table, ['validity', 'locality_adj'])
        self.assertGreater(len(fig.axes), 1)
        plt.close(fig)
        for block in ['language:location', 'model:k', 'size:k']:
            fig = ipl.plot_interaction_cells(self.cells, block, ['validity', 'locality_adj'])
            self.assertEqual(len(fig.axes), 2)                 # 2 panels
            legend = fig.legends[0]                            # better / worse / |deviation|, as the RQ2 heatmap
            self.assertEqual([p.get_facecolor()[:3] for p in legend.get_patches()],
                             [to_rgb(c) for c in (viscons.COLOR_BETTER, viscons.COLOR_WORSE, viscons.COLOR_NO_DIRECTION)])
            plt.close(fig)

    def test_figure_names(self):
        from viz import interaction_plots as ipl
        from viz.names import interaction_figure_name
        self.assertEqual(interaction_figure_name('heatmap', metrics='all_adj'), 'interactions_omega2_all_heatmap_adj')
        self.assertEqual(interaction_figure_name('cells', 'language:location', ['validity', 'duplicates']),
                         'interactions_language-location_validity-duplicates_cells')
        self.assertEqual(interaction_figure_name('cells', 'model:k', 'flagged'), 'interactions_model-k_flagged_cells')
        self.assertEqual(ipl.block_label('model:k', True), '* LLM × k')


class TestCellsGrid(unittest.TestCase):
    def test_grid(self):
        from viz import interaction_plots as ipl
        self.assertEqual(ipl.cells_grid(7, 2), (1, 7))            # 14 columns <= 30, 7 panels: one row
        self.assertEqual(ipl.cells_grid(8, 2), (2, 4))            # > 7 panels: balanced rows
        self.assertEqual(ipl.cells_grid(30, 2), (5, 6))           # 30 panels: 5 rows of 6
        self.assertEqual(ipl.cells_grid(8, 6), (2, 4))            # 48 columns: 5 per row -> balanced 4 + 4
        self.assertEqual(ipl.cells_grid(7, 6), (2, 4))            # 4 + 3
        self.assertEqual(ipl.cells_grid(3, 50), (3, 1))           # wider than the limit: one per row
        self.assertEqual(ipl.cells_grid(5, 2, ncols=2), (3, 2))   # forced


if __name__ == '__main__':
    unittest.main()
