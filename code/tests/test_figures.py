"""
Unit tests for viz/figures.save_figure (title rule) and the RQ figure names (viz/rq_plots.rq_figure_name).

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

from viz import figures, rq_plots                                        # noqa: E402

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


class TestRQFigureNames(unittest.TestCase):
    def test_names(self):
        self.assertEqual(rq_plots.rq1_figure_name('bars', 'omega2', 'paper'), 'rq1_omega2_paper_bar')
        self.assertEqual(rq_plots.rq1_figure_name('heatmap', 'shapley_r2'), 'rq1_shapley_r2_all_heatmap')
        self.assertEqual(rq_plots.rq2_figure_name('forest', 'contrast', ['validity', 'duplicates']),
                         'rq2_contrast_validity-duplicates_forest')


if __name__ == '__main__':
    unittest.main()
