"""
6_oa_metadata.py — Step 6: the ground truth, augmented with OpenAlex.

For every Semantic Scholar researcher, the OpenAlex author is the one found
through their papers (2_jw_match.py) or, without it, the cascade's pick by name,
field and seniority (5_pick.py). Its metadata comes from the OpenAlex database:

  oa_id, oa_match (jw | cascade | "")
  oa_display_name, oa_orcid, oa_works_count, oa_cited_by_count,
  oa_h_index, oa_i10_index, oa_2yr_mean_citedness, oa_first_year, oa_last_year
  oa_field_topics, oa_field_concepts, oa_subfields     (4_author_fields.py)
  oa_country_code, oa_institution, oa_country_source   country and institution
      of the researcher's most recent paper with an institution, signed by that
      author (own_papers), or else OpenAlex's last known institution (profile)

and is joined to the Semantic Scholar parquet: every column of it, one row per
researcher.

Output: ground_truth.parquet.

Usage (from code/, with PYTHONPATH=.):
  python scripts/1_ground_truth/6_oa_metadata.py --out_dir ../results/1_ground_truth
"""

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "6_factuality"))
from common import add_request_columns  # noqa: E402

from libs.metrics.constants import SUBFIELD_NORM_MAP  # noqa: E402
from libs.openalex_common import WORKS_TMP_DIR, normalize_name  # noqa: E402
from libs.utils.config import config_default  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

URL = "https://openalex.org/A"

PROFILE = f"""
SELECT '{URL}' || a.id::VARCHAR AS oa_id, a.display_name AS oa_display_name, a.orcid AS oa_orcid,
       a.works_count AS oa_works_count, a.cited_by_count AS oa_cited_by_count,
       a.summary_stats.h_index AS oa_h_index, a.summary_stats.i10_index AS oa_i10_index,
       a.summary_stats."2yr_mean_citedness" AS oa_2yr_mean_citedness,
       list_min([e.year::INT for e in a.counts_by_year if e.works_count > 0]) AS oa_first_year,
       list_max([e.year::INT for e in a.counts_by_year if e.works_count > 0]) AS oa_last_year,
       a.last_known_institutions[1].country_code AS profile_country,
       a.last_known_institutions[1].display_name AS profile_institution
FROM oa.authors a JOIN chosen c ON c.author_id = a.id
"""

CHUNK = 100_000   # researchers per query of own_papers (all at once fills the disk with DuckDB spills)

# The chosen author's own authorships on the researcher's papers, most recent first.
OWN_PAPERS = """
WITH mine AS (
  SELECT p.researcher_id, w.id AS work_id, w.publication_year, w.publication_date,
         a.institutions[1].country_code AS country_code, a.institutions[1].display_name AS institution
  FROM oa.works w
  JOIN (SELECT DISTINCT researcher_id, doi FROM read_parquet('{papers}')
        WHERE researcher_id BETWEEN {lo} AND {hi}) p ON p.doi = lower(w.doi),
       UNNEST(w.authorships) AS t(a)
  JOIN chosen_jw c ON c.researcher_id = p.researcher_id AND c.author_id = a.author.id
  WHERE w.doi IS NOT NULL AND len(a.institutions) > 0
)
SELECT researcher_id, country_code AS own_country, institution AS own_institution
FROM mine
QUALIFY row_number() OVER (PARTITION BY researcher_id
                           ORDER BY publication_year DESC, publication_date DESC, work_id DESC,
                                    institution) = 1
"""


def chosen_authors(out: Path) -> pd.DataFrame:
    """researcher_id → oa_id, from the Jaro-Winkler match or else the cascade."""
    jw = pd.read_parquet(out / "jw_match.parquet", columns=["researcher_id", "oa_id"])
    jw["oa_match"] = "jw"
    requests = add_request_columns(pd.read_csv(out / "cascade_requests.csv"))
    requests["norm"] = requests["query"].map(normalize_name)
    requests["Subfield"] = requests["subfield"].map(SUBFIELD_NORM_MAP).fillna(requests["subfield"])
    picks = pd.read_csv(out / "picks.csv", usecols=["norm", "Field", "bucket", "Subfield", "oa_id"])
    keys = ["norm", "Field", "bucket", "Subfield"]
    for k in keys:   # the same types on both sides for the merge
        requests[k], picks[k] = requests[k].astype(str), picks[k].astype(str)
    cascade = requests.merge(picks, on=keys).dropna(subset=["oa_id"])[["researcher_id", "oa_id"]]
    cascade["oa_match"] = "cascade"
    chosen = pd.concat([jw, cascade[~cascade["researcher_id"].isin(jw["researcher_id"])]])
    chosen["author_id"] = chosen["oa_id"].str.replace(URL, "").astype("int64")
    return chosen


def own_papers(con, ids: np.ndarray, papers: Path, parts: Path) -> pd.DataFrame:
    """OWN_PAPERS in blocks of CHUNK researchers, each saved when done (a run that stops resumes)."""
    parts.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(ids), CHUNK):
        lo, hi = int(ids[i]), int(ids[min(i + CHUNK, len(ids)) - 1])
        done = parts / f"range_{lo}_{hi}.parquet"
        if not done.exists():
            con.execute(OWN_PAPERS.format(papers=papers, lo=lo, hi=hi)).df().to_parquet(done, index=False)
            logger.info("own papers: researchers %d–%d of %d", i, min(i + CHUNK, len(ids)), len(ids))
    return pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("range_*.parquet"))])


def run(parquet: str, ids: str | None, out_dir: str, duckdb_path: str) -> None:
    out = Path(out_dir)
    ss = pd.read_parquet(parquet).drop_duplicates("Researcher_id")
    if ids:
        keep = pd.read_csv(ids)["researcher_id"].astype("int64")
        ss = ss[ss["Researcher_id"].isin(keep)]
    chosen = chosen_authors(out)

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{WORKS_TMP_DIR}'")
    con.execute("SET memory_limit = '48GB'")   # the 111 M DOIs are read by DuckDB, not loaded in pandas
    con.execute(f"ATTACH '{duckdb_path}' AS oa (READ_ONLY)")
    con.register("chosen", chosen[["researcher_id", "author_id"]])
    profile = con.execute(PROFILE).df()
    con.register("chosen_jw", chosen.loc[chosen["oa_match"] == "jw", ["researcher_id", "author_id"]])
    own = own_papers(con, np.sort(chosen.loc[chosen["oa_match"] == "jw", "researcher_id"].unique()),
                     out / "ss_papers.parquet", out / "own_papers_parts")
    con.close()

    fields = pd.read_parquet(out / "author_fields.parquet").rename(columns={
        "field_topics": "oa_field_topics", "field_concepts": "oa_field_concepts", "subfields": "oa_subfields"})
    meta = (chosen[["researcher_id", "oa_id", "oa_match"]]
            .merge(profile.drop_duplicates("oa_id"), on="oa_id", how="left")
            .merge(fields.drop_duplicates("oa_id"), on="oa_id", how="left")
            .merge(own, on="researcher_id", how="left"))
    has_own = meta["own_country"].notna()
    meta["oa_country_code"] = meta["own_country"].where(has_own, meta["profile_country"])
    meta["oa_institution"] = meta["own_institution"].where(has_own, meta["profile_institution"])
    meta["oa_country_source"] = np.select([has_own, meta["profile_country"].notna()],
                                          ["own_papers", "profile"], default="")
    meta = meta.drop(columns=["own_country", "own_institution", "profile_country", "profile_institution"])

    gt = ss.merge(meta.rename(columns={"researcher_id": "Researcher_id"}), on="Researcher_id", how="left")
    gt["oa_match"] = gt["oa_match"].fillna("")
    gt.to_parquet(out / "ground_truth.parquet", index=False)
    logger.info("Wrote %s: %d researchers | OpenAlex author: jw %.1f%%, cascade %.1f%%, none %.1f%% | "
                "country %.1f%% (own papers %.1f%%)", out / "ground_truth.parquet", len(gt),
                100 * (gt["oa_match"] == "jw").mean(), 100 * (gt["oa_match"] == "cascade").mean(),
                100 * (gt["oa_match"] == "").mean(), 100 * gt["oa_country_code"].notna().mean(),
                100 * (gt["oa_country_source"] == "own_papers").mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--ids", default=None, help="CSV with researcher_id to keep (a subset, to test)")
    parser.add_argument("--out_dir", default="../results/1_ground_truth")
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    args = parser.parse_args()
    run(args.parquet, args.ids, args.out_dir, args.duckdb)


if __name__ == "__main__":
    main()
