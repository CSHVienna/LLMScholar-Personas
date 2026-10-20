"""
3_pick.py — Step 3: the same cascade as scripts/factuality/, on OpenAlex, and
one author picked per recommendation.

For every (recommended name, asked field, asked seniority, asked subfield) the
prompts produced, among the OpenAlex authors with that name (step 1):
  n_name       how many there are
  n_field      ... in the asked field: their main topics field OR their main
               concept is the asked field (step 2, fields.py)
  n_seniority  ... with the asked seniority (career age from their first paper:
               <= 10 years Junior, >= 20 Senior)
  n_author     ... with both
  oa_id        the one picked: among those of the deepest check passed
               (author, then field, then seniority, then name), those whose
               top subfields include the asked one go first, then
               --tiebreak decides: "random" (seed 42) or "most_cited"

Output: picks_<tiebreak>.csv — norm, Field, bucket, Subfield, n_* and oa_id.
scripts/enrichment/openalex.py --picks uses it instead of choosing by itself.

Usage (from code/, with PYTHONPATH=.):
  python scripts/openalex_cascade/3_pick.py \\
      --candidates    ../results/summary/openalex_cascade/candidates.parquet \\
      --author_fields ../results/summary/openalex_cascade/author_fields.parquet \\
      --output        ../results/summary/openalex_cascade/picks_<tiebreak>.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "enrichment"))
sys.path.insert(0, str(HERE.parent / "factuality"))
from common import add_request_columns, seniority_bucket  # noqa: E402
from fields import SUBFIELDS  # noqa: E402
from openalex import normalize_name  # noqa: E402

from libs.metrics.constants import REFERENCE_YEAR, SUBFIELD_NORM_MAP  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()
SEED = 42
KEYS = ["norm", "Field", "bucket", "Subfield"]


def asked_requests(recommendations: str) -> pd.DataFrame:
    """The distinct (name, field, seniority, subfield) the prompts asked for."""
    rec = pd.read_csv(recommendations, usecols=["name", "lastname", "field", "target", "subfield"])
    rec = add_request_columns(rec)
    rec["Subfield"] = rec["subfield"].map(SUBFIELD_NORM_MAP).fillna(rec["subfield"])
    rec["norm"] = rec["query"].map(normalize_name)
    return rec[KEYS].drop_duplicates()


def in_subfield(asked: str, author_subfields) -> bool:
    """True when one of the author's top subfields is the asked one (fields.SUBFIELDS)."""
    if author_subfields is None or isinstance(author_subfields, float):    # no topics
        return False
    return bool(SUBFIELDS.get(asked, set()) & set(author_subfields))


def run(recommendations: str, candidates: str, author_fields: str, output: str,
        tiebreak: str = "random") -> None:
    asked = asked_requests(recommendations)
    cand = pd.read_parquet(candidates).merge(pd.read_parquet(author_fields), on="oa_id", how="left")
    cand["oa_bucket"] = seniority_bucket((REFERENCE_YEAR - cand["first_year"]).clip(lower=0))

    # Every candidate against every request for its name.
    df = asked.merge(cand, on="norm")
    df["ok_field"] = (df["field_topics"] == df["Field"]) | (df["field_concepts"] == df["Field"])
    df["ok_seniority"] = df["oa_bucket"] == df["bucket"]
    df["ok_author"] = df["ok_field"] & df["ok_seniority"]
    df["ok_subfield"] = [in_subfield(a, s) for a, s in zip(df["Subfield"], df["subfields"])]

    g = df.groupby(KEYS, sort=False)
    out = pd.DataFrame({
        "n_name": g.size(),
        "n_field": g["ok_field"].sum(),
        "n_seniority": g["ok_seniority"].sum(),
        "n_author": g["ok_author"].sum(),
    })

    # The pick: deepest check passed first, then the asked subfield, then random.
    df["depth"] = np.select([df["ok_author"], df["ok_field"], df["ok_seniority"]], [3, 2, 1], default=0)
    # Within the same depth and subfield, the last tie-break: random or most cited.
    if tiebreak == "random":
        df = df.sample(frac=1, random_state=SEED)
    else:
        df = df.sort_values(["cited_by_count", "oa_id"], ascending=False)
    df = df.sort_values(["depth", "ok_subfield"], ascending=False, kind="stable")
    out["oa_id"] = df.groupby(KEYS, sort=False)["oa_id"].first()
    out["pick_subfield"] = df.groupby(KEYS, sort=False)["ok_subfield"].first()

    out = asked.merge(out.reset_index(), on=KEYS, how="left")
    counts = ["n_name", "n_field", "n_seniority", "n_author"]
    out[counts] = out[counts].fillna(0).astype(int)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    for col in counts:
        logger.info("  %-12s > 0 for %5.1f%%", col, 100 * (out[col] > 0).mean())
    logger.info("  picked in the asked subfield: %.1f%%", 100 * out["pick_subfield"].eq(True).mean())
    logger.info("Wrote %s (%d rows)", output, len(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", default="../results/summary/recommendations.csv")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--author_fields", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tiebreak", choices=["random", "most_cited"], default="random",
                        help="Last tie-break among the candidates that pass the same checks")
    args = parser.parse_args()
    run(args.recommendations, args.candidates, args.author_fields, args.output, args.tiebreak)


if __name__ == "__main__":
    main()
