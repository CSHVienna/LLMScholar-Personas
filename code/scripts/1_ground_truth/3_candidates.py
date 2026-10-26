"""
4_candidates.py — Step 4 (the cascade, for the researchers without a DOI match): every OpenAlex author whose name matches each
recommended name (only the recommendations with factuality_author = True).

Same match as scripts/7_enrichment/1_openalex.py (normalised name equal to the
author's display name or to one of its alternative names), but it keeps ALL the
authors with that name instead of choosing one.

Output: candidates.parquet, one row per (name, OpenAlex author):
  norm, oa_id (the OpenAlex URL, as in factuality_full.csv), display_name,
  works_count, cited_by_count, first_year, last_year

Usage (from code/, with PYTHONPATH=.):
  python scripts/1_ground_truth/4_candidates.py --output ../results/summary/openalex_cascade/candidates.parquet
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from libs.openalex_common import normalize_name  # noqa: E402

from libs.utils.config import config_default  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

QUERY = r"""
WITH names AS (
  SELECT a.id, a.display_name, a.works_count, a.cited_by_count, a.counts_by_year,
         trim(regexp_replace(regexp_replace(lower(strip_accents(n)), '[^a-z ]+', ' ', 'g'),
                             ' +', ' ', 'g')) AS norm
  FROM oa.authors a,
       UNNEST(list_concat([a.display_name], COALESCE(a.display_name_alternatives, []))) AS t(n)
)
SELECT DISTINCT names.norm, 'https://openalex.org/A' || id::VARCHAR AS oa_id, display_name, works_count, cited_by_count,
       list_min([e.year::INT for e in counts_by_year if e.works_count > 0]) AS first_year,
       list_max([e.year::INT for e in counts_by_year if e.works_count > 0]) AS last_year
FROM names JOIN queries q ON q.norm = names.norm
"""


def run(recommendations: str, duckdb_path: str, output: str) -> None:
    rec = pd.read_csv(recommendations, usecols=["name", "lastname"])
    full = (rec["name"].fillna("") + " " + rec["lastname"].fillna("")).map(normalize_name)
    queries = pd.DataFrame({"norm": sorted({n for n in full if len(n.split()) >= 2})})
    logger.info("Names to look up: %d", len(queries))

    con = duckdb.connect()
    con.execute(f"ATTACH '{duckdb_path}' AS oa (READ_ONLY)")
    con.register("queries", queries)
    cand = con.execute(QUERY).df()
    con.close()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(output, index=False)
    logger.info("Wrote %s: %d candidates for %d names", output, len(cand), cand["norm"].nunique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", default="../results/summary/cascade/factual_authors.csv")
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.recommendations, args.duckdb, args.output)


if __name__ == "__main__":
    main()
