"""
2_field.py — Step 2: is any of the name's candidates in the asked field?

Takes the candidates of step 1 and keeps, for every (name, asked field) the
prompts produced, the candidates whose Field is the asked field.

factuality_field = True if at least one candidate survives.

Output: candidates_field.parquet — same columns as candidates_name.parquet,
only the rows whose Field was asked for that name.

Usage (from code/, with PYTHONPATH=.):
  python scripts/6_factuality/2_field.py \\
      --recommendations ../results/summary/recommendations_valid.csv \\
      --input           ../results/summary/cascade/candidates_name.parquet \\
      --output          ../results/summary/cascade/candidates_field.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_requests  # noqa: E402

from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

# Un candidato sobrevive solo si su Field real coincide con algún field que se pidió para ese nombre.
def run(recommendations_path: str, input_path: str, output_path: str) -> None:
    asked = load_requests(recommendations_path)[["query", "Field"]].drop_duplicates()
    cand = pd.read_parquet(input_path)

    # Keep a candidate only if its Field was asked for its name.
    kept = cand.merge(asked, on=["query", "Field"], how="inner")

    kept.to_parquet(output_path, index=False)
    logger.info("Wrote %s (%d of %d candidates)", output_path, len(kept), len(cand))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.recommendations, args.input, args.output)


if __name__ == "__main__":
    main()
