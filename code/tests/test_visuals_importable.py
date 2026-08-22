"""Every module under libs/visuals must be importable.

sensitivity_barchart.py shipped in de9d7ec importing `grouped_heatmap`, a module
that never existed in this repo (the sibling was named sensitivity_heatmap.py),
so the file raised ModuleNotFoundError on any import. Nothing caught it because
nothing imports it. This test does.

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_visuals_importable.py -v
"""

import importlib
import pathlib

import pytest

VISUALS_DIR = pathlib.Path(__file__).resolve().parents[1] / "libs" / "visuals"
MODULES = sorted(
    p.stem for p in VISUALS_DIR.glob("*.py") if not p.stem.startswith("_")
)


def test_the_visuals_package_is_not_empty():
    assert MODULES, f"no modules discovered under {VISUALS_DIR}"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"libs.visuals.{name}")


def test_barchart_shares_the_heatmap_palette_and_names():
    # The two are meant to be companion figures: same colors, same labels.
    from libs.visuals import sensitivity_barchart as bar
    from libs.visuals import sensitivity_heatmap as heat

    assert bar.PROMPT_VAR_COLORS is heat.PROMPT_VAR_COLORS
    assert bar.NAME_MAP is heat.NAME_MAP
