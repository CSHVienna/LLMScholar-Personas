"""
run_pipeline.py — Step 2 of everything: the author stats, the coauthorship graph
and the similarity PCA over the whole ground truth (1_ground_truth/).

  1. 1_author_stats.py   h / i10 / e-index of every OpenAlex author of the ground truth
  2. 2_model.py          coauthorship graph and PCA embeddings (cached)

Usage (from code/, with PYTHONPATH=.):
  python scripts/2_graph/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in ["1_author_stats.py", "2_model.py"]:
    print(f"── {script}", flush=True)
    if subprocess.run([sys.executable, str(HERE / script)]).returncode != 0:
        sys.exit(f"{script} failed")
