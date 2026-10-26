"""
2_doi_match.py — Step 2: the OpenAlex author of each Semantic Scholar researcher,
through their papers.

The papers of the researcher (1_ss_papers.py, by DOI) are looked up in OpenAlex
works; among the OpenAlex authors of those papers, the researcher is the one
whose last name is the researcher's last name and who signs the most of them.
It is the same person in both sources because they wrote the same papers, not
because they share a name.

Output: doi_match.parquet, one row per researcher found —
  researcher_id, oa_id, n_shared (papers in common), n_dois (DOIs of the
  researcher found in OpenAlex), display_name, works_count, cited_by_count,
  first_year, last_year
Researchers with no DOI in OpenAlex, or no author with their last name on those
papers, are left out; 6_enrichment/1_openalex.py uses the cascade (6_pick.py) for them.

Usage (from code/, with PYTHONPATH=.):
  python scripts/5_openalex_cascade/2_doi_match.py \\
      --ss_papers ../results/summary/openalex_cascade/ss_papers.parquet \\
      --factual   ../results/summary/cascade/factual_authors.csv \\
      --output    ../results/summary/openalex_cascade/doi_match.parquet
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "6_enrichment"))
from oa_common import WORKS_TMP_DIR, normalize_name  # noqa: E402

from libs.utils.config import config_default  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

# Normalised name in SQL, the same rule as normalize_name.
NORM = "trim(regexp_replace(regexp_replace(lower(strip_accents({x})), '[^a-z ]+', ' ', 'g'), ' +', ' ', 'g'))"

QUERY = f"""
WITH shared AS (
  SELECT p.researcher_id, a.author.id AS author_id, count(DISTINCT p.doi) AS n_shared
  FROM oa.works w
  JOIN papers p ON p.doi = lower(w.doi),
       UNNEST(w.authorships) AS t(a)
  WHERE w.doi IS NOT NULL
  GROUP BY ALL
),
found AS (
  SELECT researcher_id, count(DISTINCT doi) AS n_dois
  FROM papers WHERE doi IN (SELECT lower(doi) FROM oa.works WHERE doi IS NOT NULL)
  GROUP BY ALL
),
named AS (
  SELECT s.*, au.display_name, au.works_count, au.cited_by_count,
         list_min([e.year::INT for e in au.counts_by_year if e.works_count > 0]) AS first_year,
         list_max([e.year::INT for e in au.counts_by_year if e.works_count > 0]) AS last_year,
         list_transform(list_concat([au.display_name], COALESCE(au.display_name_alternatives, [])),
                        n -> list_last(string_split({NORM.format(x='n')}, ' '))) AS last_names
  FROM shared s JOIN oa.authors au ON au.id = s.author_id
)
SELECT n.researcher_id, 'https://openalex.org/A' || n.author_id::VARCHAR AS oa_id, n.n_shared,
       f.n_dois, n.display_name, n.works_count, n.cited_by_count, n.first_year, n.last_year
FROM named n
JOIN researchers r USING (researcher_id)
JOIN found f USING (researcher_id)
WHERE list_contains(n.last_names, r.last_name)
QUALIFY row_number() OVER (PARTITION BY n.researcher_id
                           ORDER BY n.n_shared DESC, n.works_count DESC, n.author_id) = 1
"""


def run(ss_papers: str, factual: str, duckdb_path: str, output: str) -> None:
    papers = pd.read_parquet(ss_papers, columns=["researcher_id", "doi"]).drop_duplicates()
    names = (pd.read_csv(factual, usecols=["researcher_id", "matched_name"])
             .dropna().drop_duplicates("researcher_id"))
    names["researcher_id"] = names["researcher_id"].astype("int64")
    names["last_name"] = names["matched_name"].map(lambda n: (normalize_name(n).split() or [""])[-1])
    logger.info("%d DOIs of %d researchers", len(papers), papers["researcher_id"].nunique())

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{WORKS_TMP_DIR}'")
    con.execute("SET memory_limit = '64GB'")
    con.execute(f"ATTACH '{duckdb_path}' AS oa (READ_ONLY)")
    con.register("papers", papers)
    con.register("researchers", names[["researcher_id", "last_name"]])
    match = con.execute(QUERY).df()
    con.close()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    match.to_parquet(output, index=False)
    n = papers["researcher_id"].nunique()
    logger.info("Wrote %s: %d of %d researchers matched through their papers (%.1f%%), "
                "median %d papers in common", output, len(match), n, 100 * len(match) / n,
                match["n_shared"].median())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ss_papers", default="../results/summary/openalex_cascade/ss_papers.parquet")
    parser.add_argument("--factual", default="../results/summary/cascade/factual_authors.csv")
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--output", default="../results/summary/openalex_cascade/doi_match.parquet")
    args = parser.parse_args()
    run(args.ss_papers, args.factual, args.duckdb, args.output)


if __name__ == "__main__":
    main()
