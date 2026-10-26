"""
2_jw_match.py — The OpenAlex author of each Semantic Scholar researcher, through
their papers, choosing the author by Jaro-Winkler on the names.

For every researcher (ss_papers.parquet, from 1_ss_papers.py):
  1. their papers with a DOI that is in OpenAlex, from the most recent;
  2. on each paper, every OpenAlex author that signs it gets
       score = JW(name in Semantic Scholar, display_name)
             + JW(name in Semantic Scholar, longest alternative name)
     (the display_name twice when there is no alternative name; names
     normalised: lowercase, no accents, no punctuation);
  3. the best author of each paper gets a vote if score >= MIN_SCORE (1.35,
     chosen with notebooks/analysis/checks/jw_threshold.ipynb: it maximises
     true-person rate − coauthor rate on labels that do not use names);
  4. going from the most recent paper, the first author to get 2 votes is the
     one (so if the 1st and 2nd papers agree, the rest are not needed). The
     first 10 papers are looked at; if no author got 2 votes there and several
     have 1 (a tie), papers 11 to 50 are looked at too. With a single vote in
     all (e.g. the researcher has one paper), that author is the one.
Researchers with no vote, or still tied, are left out; the cascade (4–6) is used for them.

Output: jw_match.parquet — researcher_id, oa_id, n_votes, n_papers (papers
looked at until the decision), score_mean (mean score of its votes), display_name, works_count,
cited_by_count, first_year, last_year.

Usage (from code/, with PYTHONPATH=.):
  python scripts/1_ground_truth/2_jw_match.py \\
      --ss_papers ../results/summary/openalex_cascade/ss_papers.parquet \\
      --output    ../results/summary/openalex_cascade/jw_match.parquet
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from libs.openalex_common import WORKS_TMP_DIR, normalize_name  # noqa: E402

from libs.utils.config import config_default  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

N_FIRST = 10      # papers looked at first
N_MORE = 50       # papers looked at when the first ones end in a tie
MIN_SCORE = 1.35
CHUNK = 10_000    # researchers per DuckDB query, to keep memory bounded (papers with thousands of
                  # authors make some blocks heavy); the result of a researcher does not depend on it

# The authors of each researcher's N most recent papers in OpenAlex.
QUERY = """
WITH found AS (
  SELECT p.researcher_id, w.id AS work_id, w.authorships,
         row_number() OVER (PARTITION BY p.researcher_id
                            ORDER BY w.publication_year DESC NULLS LAST, w.id DESC) AS paper_rank
  FROM oa.works w JOIN papers p ON p.doi = lower(w.doi)
  WHERE w.doi IS NOT NULL
  QUALIFY paper_rank BETWEEN {first} AND {last}
),
signers AS (
  SELECT f.researcher_id, f.work_id, f.paper_rank, a.author.id AS author_id
  FROM found f, UNNEST(f.authorships) AS t(a)
)
SELECT s.researcher_id, s.work_id, s.paper_rank, s.author_id, au.display_name,
       COALESCE(au.display_name_alternatives, []) AS alternatives
FROM signers s JOIN oa.authors au ON au.id = s.author_id
"""

META = """
SELECT 'https://openalex.org/A' || id::VARCHAR AS oa_id, display_name, works_count, cited_by_count,
       list_min([e.year::INT for e in counts_by_year if e.works_count > 0]) AS first_year,
       list_max([e.year::INT for e in counts_by_year if e.works_count > 0]) AS last_year
FROM oa.authors WHERE id IN (SELECT author_id FROM chosen)
"""


def scores(signers: pd.DataFrame, ss_names: pd.Series) -> np.ndarray:
    """JW(SS name, display_name) + JW(SS name, longest alternative name), per row."""
    ss = signers["researcher_id"].map(ss_names).fillna("").to_numpy()
    dn = signers["display_name"].map(normalize_name).to_numpy()
    longest = [normalize_name(max(alts, key=len)) if len(alts) else d
               for alts, d in zip(signers["alternatives"], signers["display_name"])]
    return np.array([JaroWinkler.similarity(a, b) + JaroWinkler.similarity(a, c)
                     for a, b, c in zip(ss, dn, longest)])


def votes(signers: pd.DataFrame) -> pd.DataFrame:
    """The best author of each paper, if over the minimum: one vote."""
    best = (signers.sort_values("score", ascending=False)
                   .drop_duplicates(["researcher_id", "work_id"]))
    return best[best["score"] >= MIN_SCORE][["researcher_id", "paper_rank", "author_id", "score"]]


def choose(v: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """(chosen, tied researcher ids). Chosen: the first author to reach 2 votes,
    from the most recent paper; or the only author with a vote."""
    v = v.sort_values(["researcher_id", "paper_rank"])
    v["n_votes"] = v.groupby(["researcher_id", "author_id"]).cumcount() + 1
    two = v[v["n_votes"] == 2].drop_duplicates("researcher_id")
    rest = v[~v["researcher_id"].isin(two["researcher_id"])]
    n_authors = rest.groupby("researcher_id")["author_id"].nunique()
    one = rest[rest["researcher_id"].isin(n_authors[n_authors == 1].index)].drop_duplicates("researcher_id")
    tied = n_authors[n_authors > 1].index.to_numpy()
    chosen = pd.concat([two, one])
    mean = v.groupby(["researcher_id", "author_id"])["score"].mean().rename("score_mean")
    chosen = chosen.merge(mean, on=["researcher_id", "author_id"])
    return (chosen.rename(columns={"paper_rank": "n_papers"})
                  [["researcher_id", "author_id", "n_votes", "n_papers", "score_mean"]], tied)


def run(ss_papers: str, parquet: str, duckdb_path: str, output: str) -> None:
    papers = pd.read_parquet(ss_papers, columns=["researcher_id", "doi"]).drop_duplicates()
    ids = np.sort(papers["researcher_id"].unique())
    names = pd.read_parquet(parquet, columns=["Researcher_id", "Name"]).drop_duplicates("Researcher_id")
    names = names[names["Researcher_id"].isin(ids)]
    ss_names = pd.Series(names["Name"].map(normalize_name).to_numpy(),
                         index=names["Researcher_id"].astype("int64"))
    logger.info("%d researchers with papers", len(ids))

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{WORKS_TMP_DIR}'")
    con.execute("SET memory_limit = '64GB'")
    con.execute(f"ATTACH '{duckdb_path}' AS oa (READ_ONLY)")
    # Each block is saved when done, so a run that stops resumes at the next one.
    parts = Path(output).with_name(Path(output).stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    chosen, n_tied = [], 0
    # Blocks are saved as range_<first>_<end>.parquet (positions in the sorted ids).
    covered = [tuple(int(x) for x in f.stem.split("_")[1:]) for f in parts.glob("range_*.parquet")]
    for f in parts.glob("range_*.parquet"):
        chosen.append(pd.read_parquet(f))
    for i in range(0, len(ids), CHUNK):
        if any(a <= i and min(i + CHUNK, len(ids)) <= b for a, b in covered):
            continue
        done = parts / f"range_{i:08d}_{min(i + CHUNK, len(ids)):08d}.parquet"
        part = papers[papers["researcher_id"].isin(ids[i:i + CHUNK])]
        con.register("papers", part)
        signers = con.execute(QUERY.format(first=1, last=N_FIRST)).df()
        signers["score"] = scores(signers, ss_names)
        v = votes(signers)
        found, tied = choose(v)
        if len(tied):   # a tie in the first papers: look at more of them
            con.register("papers", part[part["researcher_id"].isin(tied)])
            more = con.execute(QUERY.format(first=N_FIRST + 1, last=N_MORE)).df()
            more["score"] = scores(more, ss_names)
            v_tied = pd.concat([v[v["researcher_id"].isin(tied)], votes(more)])
            found_more, tied = choose(v_tied)
            found = pd.concat([found, found_more])
        con.unregister("papers")
        found.to_parquet(done, index=False)
        chosen.append(found)
        n_tied += len(tied)
        logger.info("researchers %d–%d: %d chosen, %d still tied", i, min(i + CHUNK, len(ids)),
                    len(found), len(tied))
    chosen = pd.concat(chosen, ignore_index=True)

    con.register("chosen", chosen[["author_id"]])
    meta = con.execute(META).df()
    con.close()
    chosen["oa_id"] = "https://openalex.org/A" + chosen["author_id"].astype(str)
    out = chosen.drop(columns="author_id").merge(meta, on="oa_id", how="left")
    out = out[["researcher_id", "oa_id", "n_votes", "n_papers", "score_mean", "display_name",
               "works_count", "cited_by_count", "first_year", "last_year"]]
    out.to_parquet(output, index=False)
    logger.info("Wrote %s: %d of %d researchers matched (%.1f%%), %d left tied; "
                "median %d papers looked at", output, len(out), len(ids), 100 * len(out) / len(ids),
                n_tied, out["n_papers"].median())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ss_papers", default="../results/summary/openalex_cascade/ss_papers.parquet")
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--output", default="../results/summary/openalex_cascade/jw_match.parquet")
    args = parser.parse_args()
    run(args.ss_papers, args.parquet, args.duckdb, args.output)


if __name__ == "__main__":
    main()
