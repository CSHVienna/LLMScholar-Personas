"""
3_doi_country.py — Step 3: the country and institution of each Semantic Scholar
researcher, from their own papers in OpenAlex.

Among the researcher's papers found in OpenAlex by DOI (1_ss_papers.py), the
authorships of the OpenAlex author chosen for them (2_jw_match.py) are theirs;
the country and institution are those of the most recent of them with an
institution.

Output: doi_country.parquet — researcher_id, country_code, institution, year
(of the paper they come from), n_papers (their papers with an institution).

Usage (from code/, with PYTHONPATH=.):
  python scripts/5_openalex_cascade/3_doi_country.py
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "6_enrichment"))
from oa_common import WORKS_TMP_DIR  # noqa: E402

from libs.utils.config import config_default  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

QUERY = """
WITH mine AS (
  SELECT p.researcher_id, w.id AS work_id, w.publication_year, w.publication_date,
         a.institutions[1].country_code AS country_code, a.institutions[1].display_name AS institution
  FROM oa.works w
  JOIN papers p ON p.doi = lower(w.doi),
       UNNEST(w.authorships) AS t(a)
  JOIN chosen c ON c.researcher_id = p.researcher_id AND c.author_id = a.author.id
  WHERE w.doi IS NOT NULL AND len(a.institutions) > 0
)
SELECT researcher_id, country_code, institution, publication_year AS year,
       count(*) OVER (PARTITION BY researcher_id) AS n_papers
FROM mine
QUALIFY row_number() OVER (PARTITION BY researcher_id
                           ORDER BY publication_year DESC, publication_date DESC, work_id DESC,
                                    institution) = 1
"""


def run(ss_papers: str, match: str, duckdb_path: str, output: str) -> None:
    chosen = pd.read_parquet(match, columns=["researcher_id", "oa_id"])
    chosen["author_id"] = chosen["oa_id"].str.replace("https://openalex.org/A", "").astype("int64")
    papers = pd.read_parquet(ss_papers, columns=["researcher_id", "doi"]).drop_duplicates()
    papers = papers[papers["researcher_id"].isin(chosen["researcher_id"])]

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{WORKS_TMP_DIR}'")
    con.execute("SET memory_limit = '64GB'")
    con.execute(f"ATTACH '{duckdb_path}' AS oa (READ_ONLY)")
    con.register("papers", papers)
    con.register("chosen", chosen[["researcher_id", "author_id"]])
    out = con.execute(QUERY).df()
    con.close()

    out.to_parquet(output, index=False)
    logger.info("Wrote %s: a country for %d of %d researchers with an OpenAlex author (%.1f%%)", output,
                out["country_code"].notna().sum(), len(chosen),
                100 * out["country_code"].notna().sum() / len(chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ss_papers", default="../results/summary/openalex_cascade/ss_papers.parquet")
    parser.add_argument("--match", default="../results/summary/openalex_cascade/jw_match.parquet",
                        help="the OpenAlex author of each researcher (2_jw_match.py)")
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--output", default="../results/summary/openalex_cascade/doi_country.parquet")
    args = parser.parse_args()
    run(args.ss_papers, args.match, args.duckdb, args.output)


if __name__ == "__main__":
    main()
