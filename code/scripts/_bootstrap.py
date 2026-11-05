"""
Make ``code/libs`` importable for the batch scripts, from any working directory
(no need to ``export PYTHONPATH``). Import it first in every script:

    import _bootstrap  # noqa: F401
"""

import sys
from pathlib import Path

LIBS_DIR = Path(__file__).resolve().parents[1] / 'libs'
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))
