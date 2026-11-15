"""
Unit tests for viz/figures.save_figure (title rule, main vs. si/ folder) and the figure names (viz/names.py).

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

from viz import constants as viscons, figures, names, rq_plots                 # noqa: E402

SCRATCH = PROJECT_ROOT / 'tmp' / 'tests' / 'figures'


def visible_titles_when_saved(fig, fmt):
    """Texts of the visible titles (suptitle + axes titles) at the moment fig is saved as fmt."""
    seen = {}
    original = fig.savefig

    def record(path, *args, **kwargs):
        texts = [t for ax in fig.axes for t in (ax.title, ax._left_title, ax._right_title)]
        if fig._suptitle is not None:
            texts.append(fig._suptitle)
        seen[fmt] = sorted(t.get_text() for t in texts if t.get_text() and t.get_visible())
        return original(path, *args, **kwargs)

    fig.savefig = record
    figures.save_figure(fig, 'test', SCRATCH, [fmt])
    plt.close(fig)
    return seen[fmt]


class TestSaveFigureTitles(unittest.TestCase):
    def test_multi_panel_pdf_keeps_subplot_titles_drops_suptitle(self):
        def make():
            fig, axes = plt.subplots(1, 2)
            for ax, name in zip(axes, ['A', 'B']):
                ax.axis('off')                       # flow panels have no axes
                ax.set_title(name)
            fig.suptitle('Main')
            return fig
        self.assertEqual(visible_titles_when_saved(make(), 'pdf'), ['A', 'B'])
        self.assertEqual(visible_titles_when_saved(make(), 'png'), ['A', 'B', 'Main'])

    def test_single_panel_pdf_drops_its_title(self):
        def make():
            fig, ax = plt.subplots()
            image = ax.imshow([[0, 1], [1, 0]])
            fig.colorbar(image, ax=ax)               # a colorbar is not a panel
            ax.set_title('Only', loc='left')
            return fig
        self.assertEqual(visible_titles_when_saved(make(), 'pdf'), [])
        self.assertEqual(visible_titles_when_saved(make(), 'png'), ['Only'])


class TestSaveFigureFolder(unittest.TestCase):
    def test_paper_main_figures_at_root_others_in_si(self):
        main = viscons.PAPER_MAIN_FIGURES[0]
        for name, folder in [(main, SCRATCH), ('test_si', SCRATCH / viscons.SI_SUBDIR)]:
            fig, _ = plt.subplots()
            paths = figures.save_figure(fig, name, SCRATCH, ['png'])
            plt.close(fig)
            self.assertEqual(paths, [folder / f"{name}.png"])
            self.assertTrue(paths[0].exists())


class TestRQFigureNames(unittest.TestCase):
    def test_names(self):
        self.assertEqual(names.rq1_figure_name('bar', 'omega2', 'paper'), 'rq1_omega2_paper_bar')
        self.assertEqual(names.rq1_figure_name('heatmap', 'shapley_r2'), 'rq1_shapley-r2_all_heatmap')
        self.assertEqual(names.rq2_figure_name('forest', 'contrast', ['validity', 'duplicates']),
                         'rq2_contrast_validity-duplicates_forest')
        with self.assertRaises(ValueError):
            names.rq1_figure_name('bars', 'omega2')                              # only 'bar'

    def test_convention(self):
        """<family>_<what>_<selection>_<kind>[_adj]: '_' only between parts."""
        self.assertEqual(names.figure_name('gt', 'field_overlap'), 'gt_field-overlap')
        self.assertEqual(names.figure_name('gt', 'population', None, 'pct'), 'gt_population_pct')
        self.assertEqual(names.metrics_figure_name('model_families', 'paper_adj'), 'metrics_model-families_paper_bar_adj')
        for name in viscons.PAPER_MAIN_FIGURES:
            self.assertLessEqual(len(name.removesuffix('_adj').split('_')), 5, name)


class TestRQ2ForestLabel(unittest.TestCase):
    def test_text(self):
        self.assertEqual(rq_plots.rq2_forest_xlabel('contrast'), 'Difference from the reference value')
        self.assertEqual(rq_plots.rq2_forest_xlabel('contrast', True),
                         'Difference in (obs. − exp.) from the reference value')
        self.assertEqual(rq_plots.rq2_forest_xlabel('deviation', True), 'Deviation of (obs. − exp.) from the grand mean')

    def test_centred_over_the_panels(self):
        import pandas as pd
        from utils import constants as cons
        rows = []
        for metric in ('parity_gender_adj', 'parity_ethnicity_adj', 'diversity_gender_adj', 'validity'):
            for level, v in (('English', 0.0), ('German', 0.1)):
                rows.append({'variable': cons.VAR_LANGUAGE, 'parent': '', 'level': level, 'reference': 'English',
                             'metric': metric, 'contrast': v, 'contrast_ci_low': v - 0.05, 'contrast_ci_high': v + 0.05})
        fig = rq_plots.plot_rq2_forest(pd.DataFrame(rows), 'contrast', ['parity_gender_adj', 'parity_ethnicity_adj',
                                                                         'diversity_gender_adj', 'validity'])
        axes = [ax for ax in fig.axes if ax.get_visible()]
        center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
        label = fig._supxlabel
        self.assertAlmostEqual(label.get_position()[0], center)
        self.assertNotAlmostEqual(center, 0.5, places=2)              # the panels are not centred in the figure
        self.assertEqual(label.get_text(), 'Difference in (obs. − exp.) from the reference value')
        plt.close(fig)

    def test_emm(self):
        """EMM forest: every value kept (no reference dropped); adj panel signed around 0 (= random), raw panel on
        the metric's scale."""
        import pandas as pd
        from utils import constants as cons
        rows = [{'variable': cons.VAR_LANGUAGE, 'parent': '', 'level': level, 'reference': 'English', 'metric': metric,
                 'emm': v, 'emm_ci_low': v - 0.02, 'emm_ci_high': v + 0.02}
                for metric, values in (('parity_gender_adj', (-0.08, -0.03)), ('validity', (0.86, 0.90)))
                for level, v in zip(('English', 'German'), values)]
        fig = rq_plots.plot_rq2_forest(pd.DataFrame(rows), 'emm', ['parity_gender_adj', 'validity'])
        adj, raw = [ax for ax in fig.axes if ax.get_visible()]
        low, high = adj.get_xlim()
        self.assertAlmostEqual(low, -high)
        self.assertEqual(raw.get_xlim(), (0.0, 1.0))
        self.assertEqual([t.get_text() for t in adj.get_yticklabels()][1:], ['English', 'German'])
        self.assertEqual(fig._supxlabel.get_text(), 'Estimated marginal mean of obs. − exp. (0 = as random)')
        plt.close(fig)

    def test_bars(self):
        """marks='bars': one bar per value, no row shading, same layout as the dots."""
        import pandas as pd
        from utils import constants as cons
        rows = [{'variable': cons.VAR_LANGUAGE, 'parent': '', 'level': level, 'reference': 'English',
                 'metric': 'validity', 'emm': v, 'emm_ci_low': v - 0.02, 'emm_ci_high': v + 0.02}
                for level, v in (('English', 0.86), ('German', 0.90))]
        fig = rq_plots.plot_rq2_forest(pd.DataFrame(rows), 'emm', ['validity'], marks='bars')
        ax = fig.axes[0]
        self.assertEqual(len(ax.containers[0].patches), 1)                  # first container: the first bar
        self.assertEqual(sum(len(c.patches) for c in ax.containers if hasattr(c, 'patches')), 2)
        self.assertEqual(ax.get_xlim(), (0.0, 1.0))
        self.assertFalse([p for p in ax.patches if p.get_width() > 1])     # no full-width row shading
        plt.close(fig)

    @staticmethod
    def _field_rows(value):
        """Two fields with two subfields each (references: Physics; first subfield within each field)."""
        import pandas as pd
        from utils import constants as cons
        rows = [(cons.VAR_FIELD, '', 'Mathematics', 'Physics'), (cons.VAR_FIELD, '', 'Physics', 'Physics'),
                (cons.VAR_SUBFIELD, 'Mathematics', 'Number theory', 'Number theory'),
                (cons.VAR_SUBFIELD, 'Mathematics', 'Topology', 'Number theory'),
                (cons.VAR_SUBFIELD, 'Physics', 'Condensed Matter', 'Condensed Matter'),
                (cons.VAR_SUBFIELD, 'Physics', 'Education', 'Condensed Matter')]
        return pd.DataFrame([{'variable': var, 'parent': parent, 'level': level, 'reference': ref,
                              'metric': 'validity', value: 0.5, f"{value}_ci_low": 0.45, f"{value}_ci_high": 0.55}
                             for var, parent, level, ref in rows])

    def test_nested_subfields(self):
        """nest_subfields (default): each field's subfields right under it, smaller grey labels, lighter rows; no
        Subfield block. False: the Subfield block as before."""
        from viz import constants as viscons
        fig = rq_plots.plot_rq2_forest(self._field_rows('emm'), 'emm', ['validity'])
        ax = fig.axes[0]
        labels = ax.get_yticklabels()
        self.assertEqual([t.get_text() for t in labels],
                         ['Field', 'Mathematics', 'Number theory', 'Topology', 'Physics', 'Condensed Matter',
                          'Education'])
        self.assertEqual(labels[2].get_color(), viscons.RQ2_FOREST_SUB_LABEL_COLOR)
        self.assertNotEqual(labels[1].get_color(), viscons.RQ2_FOREST_SUB_LABEL_COLOR)
        self.assertLess(labels[2].get_fontsize(), labels[1].get_fontsize())
        shades = {round(p.get_y() + p.get_height() / 2): p.get_facecolor() for p in ax.patches}
        self.assertGreater(sum(shades[2][:3]), sum(shades[1][:3]))                # subfield row lighter
        plt.close(fig)
        fig = rq_plots.plot_rq2_forest(self._field_rows('emm'), 'emm', ['validity'], nest_subfields=False)
        self.assertEqual([t.get_text() for t in fig.axes[0].get_yticklabels()],
                         ['Field', 'Mathematics', 'Physics', 'Subfield', 'Number theory', 'Topology',
                          'Condensed Matter', 'Education'])
        plt.close(fig)

    def test_nested_subfields_contrast(self):
        """Contrasts nested: the reference field and the reference subfield of each field are left out."""
        fig = rq_plots.plot_rq2_forest(self._field_rows('contrast'), 'contrast', ['validity'])
        self.assertEqual([t.get_text() for t in fig.axes[0].get_yticklabels()],
                         ['Field [ref: Physics]', 'Mathematics', 'Topology (vs. Number theory)',
                          'Education (vs. Condensed Matter)'])
        plt.close(fig)

    def test_paper_column_sets(self):
        from metrics import summary
        self.assertEqual([m.key for m in summary.select_metrics('paper_social_adj')],
                         ['parity_ethnicity_adj', 'locality_adj', 'parity_gender_adj', 'diversity_location_adj',
                          'diversity_ethnicity_adj', 'parity_location_adj'])
        self.assertEqual(names.rq2_figure_name('forest', 'emm', 'paper_social_adj'),
                         'rq2_emm_paper-social_forest_adj')
        self.assertEqual(names.rq2_figure_name('bar', 'emm', 'paper_technical'), 'rq2_emm_paper-technical_bar')


if __name__ == '__main__':
    unittest.main()


class TestLabels(unittest.TestCase):
    """viz/labels.py: one name per concept in every figure."""

    def test_llm_everywhere(self):
        from analysis.influence import MODEL_FACTOR
        from viz import labels
        self.assertEqual(labels.variable_title(MODEL_FACTOR), 'LLM')
        self.assertEqual(labels.variable_title('family'), 'LLM family')
        self.assertEqual(rq_plots.rq1_rows([MODEL_FACTOR])[0].labels, ['LLM'])
        self.assertEqual(labels.FAMILY_TITLES['model'], 'LLM')

    def test_titles_and_levels(self):
        from utils import constants as cons
        from viz import labels
        self.assertEqual(labels.variable_title(cons.VAR_K), 'k')
        self.assertEqual(labels.variable_title('reasoning'), 'Reasoning')
        self.assertEqual(labels.level_label('reasoning', True), 'Yes')
        self.assertEqual(labels.level_label('size', 'XL'), 'Extra large')
        self.assertEqual(labels.level_label(cons.VAR_ROLE, 'PhD student'), 'Student')
        self.assertEqual(labels.attribute_title('gender'), 'Gender')
        self.assertEqual(labels.attribute_title('author_seniority'), 'Seniority')

    def test_class_labels(self):
        from viz import labels
        self.assertEqual(labels.class_label('ethnicity', 'east_asian'), 'East Asian')
        self.assertEqual(labels.class_label('seniority', 'junior'), 'Junior')
        self.assertEqual(labels.class_label('country', 'US'), 'US')
        self.assertEqual(labels.class_label('country', 'other'), 'Other')

    def test_metric_short_names(self):
        from metrics import summary
        self.assertEqual(summary.METRICS['parity_location'].short, 'Country')
        self.assertEqual(summary.METRICS['parity_productivity'].short, '# works')
        self.assertEqual(summary.METRICS['diversity_influence'].label, r'Div.$_{{\#}cit}$')
        self.assertEqual(summary.METRICS['popularity_influence_adj'].label,
                         r'Pop.$^{adj}_{{\#}cit}$')

    def test_fade_rule(self):
        from viz import labels
        self.assertEqual(labels.fade_rule('by', 0.05), 'BY-adjusted p ≥ 0.05')
        self.assertEqual(labels.fade_rule(None, 0.05), '95% CI includes 0')
