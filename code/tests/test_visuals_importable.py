"""Every module under libs/visuals must be importable.

A module that nothing else imports can break without anyone noticing (as
sensitivity_barchart.py did, now in libs/legacy/). This test imports each one.

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

