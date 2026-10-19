"""
2_consistency.py — Step 2: consistency of every setup.

Takes the valid calls of step 1 and, for every setup (model + params, any
created_at), compares the authors recommended in consecutive runs:

  consistency = mean over consecutive runs (r−1, r) of
                |A_(r−1) ∩ A_r| / |A_(r−1) ∪ A_r|        (paper Eq. 4)

where A_r is the set of authors of run r, compared by "name lastname".
NaN when the setup has fewer than 2 valid runs.

Output: 2_consistency/<model>.csv — setup columns + n_runs, consistency.

Usage (from code/, with PYTHONPATH=.):
  python scripts/pipeline/2_consistency.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CALL, RECOMMENDATIONS, SETUP, author_key, load, read_step, write_step,
)


def consistency(runs: list[set]) -> float:
    pairs = [len(a & b) / len(a | b) for a, b in zip(runs, runs[1:]) if a | b]
    return float(np.mean(pairs)) if pairs else np.nan


def run(recommendations_path: str) -> None:
    valid = read_step(1, CALL + ["validity"])
    valid = valid[valid["validity"]].drop(columns="validity")

    df = load(recommendations_path, ["name", "lastname"])
    df["author"] = author_key(df)
    df = df.merge(valid, on=CALL)          # only the valid calls

    rows = []
    for setup, g in df.groupby(SETUP, dropna=False):
        runs = [set(a.dropna()) for _, a in g.groupby("run_id")["author"]]
        rows.append((*setup, len(runs), consistency(runs)))

    out = pd.DataFrame(rows, columns=SETUP + ["n_runs", "consistency"])
    write_step(out, step=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", default=RECOMMENDATIONS)
    args = parser.parse_args()
    run(args.recommendations)


if __name__ == "__main__":
    main()
