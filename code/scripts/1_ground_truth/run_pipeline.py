"""
run_pipeline.py — Step 1 of everything: the ground truth, every researcher of
the Semantic Scholar parquet augmented with their OpenAlex author and metadata.
It is built once, before the LLM answers are looked at; 6_factuality/ and
8_metrics/ read it.

Through their papers (the same person, because they signed the same papers):
  1. 1_ss_papers.py      the researcher's papers with DOI, from the Semantic
                         Scholar API (batch calls only)        → ss_papers.parquet
  2. 2_jw_match.py       the OpenAlex author who signs them, by Jaro-Winkler on
                         the names (threshold 1.35)            → jw_match.parquet
By name, for the researchers without a match (the cascade):
  (here)                 their name, Semantic Scholar field and seniority
                                                               → cascade_requests.csv
  3. 3_candidates.py     every OpenAlex author with that name  → candidates.parquet
  4. 4_author_fields.py  field (topics, concepts) and subfields of those and of
                         the authors of step 2                 → author_fields.parquet
  5. 5_pick.py           name → field → seniority, then one at random → picks.csv
Then:
  6. 6_oa_metadata.py    the OpenAlex metadata of the chosen author, joined to
                         the Semantic Scholar parquet          → ground_truth.parquet

Usage (from code/, with PYTHONPATH=.):
  python scripts/1_ground_truth/run_pipeline.py
  python scripts/1_ground_truth/run_pipeline.py --from_step 2
  # a subset, to test (its own folder, with its ss_papers.parquet already there):
  python scripts/1_ground_truth/run_pipeline.py --ids <csv with researcher_id> \\
      --out_dir ../results/1_ground_truth_test --from_step 2
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "6_factuality"))
from common import seniority_bucket  # noqa: E402

from libs.metrics.constants import REFERENCE_YEAR  # noqa: E402
from libs.utils.config import config_default  # noqa: E402

HERE = Path(__file__).resolve().parent
SENIORITY_LABEL = {"Junior": "Junior Professor", "Senior": "Senior Professor"}   # as in the prompts


def cascade_requests(parquet: str, ids: str | None, jw_match: Path, output: Path) -> None:
    """The researchers without a Jaro-Winkler match, in the shape the cascade reads."""
    ss = pd.read_parquet(parquet, columns=["Researcher_id", "Name", "Field", "First_year"])
    ss = ss.drop_duplicates("Researcher_id")
    if ids:
        ss = ss[ss["Researcher_id"].isin(pd.read_csv(ids)["researcher_id"].astype("int64"))]
    matched = pd.read_parquet(jw_match, columns=["researcher_id"])["researcher_id"]
    ss = ss[~ss["Researcher_id"].isin(matched)]
    bucket = seniority_bucket((REFERENCE_YEAR - ss["First_year"]).clip(lower=0))
    pd.DataFrame({
        "researcher_id": ss["Researcher_id"].astype("int64"),
        "name": ss["Name"], "lastname": "", "field": ss["Field"],
        "target": bucket.map(SENIORITY_LABEL).fillna(""), "subfield": "",
    }).to_csv(output, index=False)


def author_ids(candidates: Path, jw_match: Path, output: Path) -> None:
    """Every OpenAlex author whose fields are needed: the cascade's candidates and the matched ones."""
    ids = pd.concat([pd.read_parquet(candidates, columns=["oa_id"]),
                     pd.read_parquet(jw_match, columns=["oa_id"])]).drop_duplicates()
    ids.to_parquet(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--ids", default=None, help="CSV with researcher_id (default: every researcher)")
    parser.add_argument("--out_dir", default="../results/1_ground_truth")
    parser.add_argument("--from_step", type=int, default=1, choices=range(1, 7))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_args = (["--ids", args.ids] if args.ids
                else ["--ids", args.parquet, "--id_column", "Researcher_id"])
    requests, cand, fields = out / "cascade_requests.csv", out / "candidates.parquet", out / "author_fields.parquet"

    def run(script, *extra):
        print(f"── {script}", flush=True)
        if subprocess.run([sys.executable, str(HERE / script), *map(str, extra)]).returncode != 0:
            sys.exit(f"{script} failed")

    if args.from_step <= 1:
        run("1_ss_papers.py", *ids_args, "--batch_only", "--output", out / "ss_papers.parquet")
    if args.from_step <= 2:
        run("2_jw_match.py", "--ss_papers", out / "ss_papers.parquet", "--output", out / "jw_match.parquet")
    if args.from_step <= 3:
        cascade_requests(args.parquet, args.ids, out / "jw_match.parquet", requests)
        run("3_candidates.py", "--recommendations", requests, "--output", cand)
    if args.from_step <= 4:
        author_ids(cand, out / "jw_match.parquet", out / "author_ids.parquet")
        run("4_author_fields.py", "--candidates", out / "author_ids.parquet", "--output", fields)
    if args.from_step <= 5:
        run("5_pick.py", "--recommendations", requests, "--candidates", cand,
            "--author_fields", fields, "--output", out / "picks.csv")
    run("6_oa_metadata.py", "--parquet", args.parquet, "--out_dir", out,
        *(["--ids", args.ids] if args.ids else []))


if __name__ == "__main__":
    main()
