"""
3_seniority.py — Step 3: is any of the name's candidates of the asked
seniority?

Takes the candidates of step 1 (right name, any field) and keeps, for every
(name, asked seniority) the prompts produced, the candidates whose seniority
is the asked one. Seniority comes from career age (2025 − First_year):
<= 10 → Junior, >= 20 → Senior, 11–19 → neither.

It does not look at the field: field and seniority are checked separately,
each over all the people with that name. factuality_author (step 4) is the one
that asks for both at once.

factuality_seniority = True if at least one candidate survives.

Output: candidates_seniority.parquet — same columns as candidates_name.parquet,
only the rows whose seniority was asked for that name.

Usage (from code/, with PYTHONPATH=.):
  python scripts/4_factuality/3_seniority.py \\
      --recommendations ../results/summary/recommendations_valid.csv \\
      --input           ../results/summary/cascade/candidates_name.parquet \\
      --output          ../results/summary/cascade/candidates_seniority.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_requests  # noqa: E402

from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()


def run(recommendations_path: str, input_path: str, output_path: str) -> None:
    asked = load_requests(recommendations_path)[["query", "bucket"]].drop_duplicates()
    cand = pd.read_parquet(input_path)

    # Keep a candidate only if its seniority was asked for its name.
    # Candidates with 11–19 years have bucket = None and never match.
    kept = cand.merge(asked, on=["query", "bucket"], how="inner")

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
