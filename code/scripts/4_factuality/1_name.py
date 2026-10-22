"""
1_name.py — Step 1: does the recommended name exist?

Finds, for every recommended name, ALL the researchers whose name matches it.
"Matches" is defined in names.py:
  • JW(display name, display name) >= 0.85
  • every token of the query (initials excluded) has a token in the
    researcher's name with JW >= 0.95

A name exists (factuality_name = True) if it has at least one candidate.

Output: candidates_name.parquet, one row per (query, researcher):
  query, Researcher_id, Name, Field, bucket, score

Usage (from code/, with PYTHONPATH=.):
  python scripts/4_factuality/1_name.py \\
      --recommendations ../results/summary/recommendations.csv \\
      --output          ../results/summary/cascade/candidates_name.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.process import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import (  # noqa: E402
    DN_THRESHOLD,
    PER_TOKEN_THRESHOLD,
    _JW,
    _passes_per_token_check,
    block_key,
    normalize,
    parse_name,
)
from common import load_requests, seniority_bucket  # noqa: E402

from libs.metrics.constants import REFERENCE_YEAR  # noqa: E402
from libs.utils.config import config_default  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

# se agrupan por las 2 primeras letras del apellido con parse_name 
def add_name_columns(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """Normalised name + the block it is compared in (first 2 letters of the
    last name)."""
    df["display_name"] = df[name_col].astype(str).map(normalize) #normalizo los nombres para quedarme sin títulos académicos, etc
    df["block"] = df["display_name"].map(lambda dn: block_key(parse_name(dn)[3]))
    return df

# cargo los investigadores y calculo su seniority bucket
def load_researchers(parquet_path: str) -> pd.DataFrame:
    logger.info("Loading researchers: %s", parquet_path)
    ref = pd.read_parquet(
        parquet_path, columns=["Researcher_id", "Name", "Field", "First_year"]
    )
    ref = add_name_columns(ref, "Name")
    career_age = (REFERENCE_YEAR - ref["First_year"]).clip(lower=0)
    ref["bucket"] = seniority_bucket(career_age)
    logger.info("Researchers: %d", len(ref))
    return ref

# se hace el match con 2 filtros
#  Jaro-Winkler del nombre completo ≥ 0.85
# el filtro por token
def find_candidates(queries: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """One row per (query, matching researcher), with the JW score."""
    rows = []
    ref_blocks = dict(tuple(ref.groupby("block")))

    for i, (key, q_block) in enumerate(queries.groupby("block"), 1):
        r_block = ref_blocks.get(key)
        if r_block is None:  # no block with 2 letters: retry with one letter
            r_block = ref_blocks.get(key[:1])
        if r_block is None:
            continue
        r_names = r_block["display_name"].tolist()

        # Compare in chunks so the score matrix stays around 200 MB.
        chunk = max(50, 50_000_000 // len(r_names))
        for start in range(0, len(q_block), chunk):
            q_chunk = q_block.iloc[start : start + chunk]
            scores = cdist(
                q_chunk["display_name"].tolist(), r_names,
                scorer=_JW, dtype=np.float32, workers=-1,
            )
            for qi, ri in zip(*np.nonzero(scores >= DN_THRESHOLD)):
                q_name = q_chunk["display_name"].iat[qi]
                if _passes_per_token_check(q_name, r_names[ri], PER_TOKEN_THRESHOLD):
                    rows.append((q_chunk["query"].iat[qi], r_block.index[ri],
                                 float(scores[qi, ri])))

        if i % 100 == 0:
            logger.info("  block %d — %d candidates so far", i, len(rows))

    cand = pd.DataFrame(rows, columns=["query", "ref_row", "score"])
    cand = cand.join(ref[["Researcher_id", "Name", "Field", "bucket"]], on="ref_row")
    return cand.drop(columns="ref_row")


def run(recommendations_path: str, parquet_path: str, output_path: str) -> None:
    requests = load_requests(recommendations_path)
    queries = pd.DataFrame({"query": requests["query"].unique()})
    queries = add_name_columns(queries[queries["query"] != ""], "query")
    logger.info("Unique names: %d", len(queries))

    ref = load_researchers(parquet_path)
    cand = find_candidates(queries, ref)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(output_path, index=False)
    found = cand["query"].nunique()
    logger.info("Wrote %s (%d candidates)", output_path, len(cand))
    logger.info("Names found: %d / %d (%.1f%%)",
                found, len(queries), 100 * found / max(1, len(queries)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.recommendations, args.parquet, args.output)


if __name__ == "__main__":
    main()
