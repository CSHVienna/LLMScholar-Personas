"""
3_factuality.py — Step 3: factuality of every recommendation of a valid call.

Runs the factuality cascade (scripts/factuality/), which compares each
recommended author with Semantic Scholar using the field and seniority of the
prompt (not the LLM's answer), and keeps the recommendations of the valid calls
of step 1 whose name is complete: at least 2 words in "name lastname" (a
single first name or a single last name is left out).

Per recommendation:
  factuality_name       a researcher with that name exists
  matched_names         how many researchers have that name (0 if none)
  factuality_field      at least one of them is in the asked field
  matched_field         how many of them are
  factuality_seniority  at least one of them has the asked seniority
  matched_seniority     how many of them do
  factuality_author     at least one of them has the asked field AND seniority
  matched_author        how many of them do; researcher_id is one of them at random
  bias_location         the country of the prompt is the OpenAlex country of the
                        author (True / False; empty when the name was not found
                        or a country is missing)

Steps 4 and 5 need who the author is, so the output also carries researcher_id
(Semantic Scholar), oa_id (OpenAlex) and their attributes (gender, ethnicity,
country, works, citations, first/last publication year).

Output: 3_factuality/<model>.csv — call columns + the LLM's answer + the
columns above.

Usage (from code/, with PYTHONPATH=.):
  python scripts/pipeline/3_factuality.py               # runs the cascade
  python scripts/pipeline/3_factuality.py --skip_name   # reuses the name matches (slow step)
  python scripts/pipeline/3_factuality.py --skip_cascade  # reuses the whole cascade
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ANSWER, CALL, RECOMMENDATIONS, has_full_name, load, logger, read_step, write_step,
)

CASCADE = Path(__file__).resolve().parent.parent / "factuality" / "run_pipeline.py"
CASCADE_OUT = "../results/summary/cascade/factuality_full.csv"

MATCHED = {"n_name": "matched_names", "n_field": "matched_field",
           "n_seniority": "matched_seniority", "n_author": "matched_author"}
FACTUALITY = ["factuality_name", "factuality_field",
              "factuality_seniority", "factuality_author"]
AUTHOR = ["researcher_id", "matched_name", "gt_gender", "perceived_ethnicity",
          "oa_id", "location_oa_iso", "oa_works_count", "oa_cited_by_count",
          "oa_first_pub_year", "oa_last_pub_year", "oa_career_age"]


def run(recommendations_path: str, skip_cascade: bool, skip_name: bool) -> None:
    if not skip_cascade:
        cmd = [sys.executable, str(CASCADE), "--recommendations", recommendations_path]
        if skip_name:
            cmd.append("--skip_name")
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit("factuality cascade failed")

    valid = read_step(1, CALL + ["validity"])
    valid = valid[valid["validity"]].drop(columns="validity")

    df = load(CASCADE_OUT, ANSWER + list(MATCHED) + FACTUALITY + AUTHOR + ["location_status"])
    df = df.merge(valid, on=CALL).rename(columns=MATCHED)

    complete = has_full_name(df)
    logger.info("Left out %d recommendations with an incomplete name", (~complete).sum())
    df = df[complete]

    df["bias_location"] = df["location_status"].map(
        {"location_match": True, "location_mismatch": False})

    columns = CALL + ANSWER
    for step in ["name", "field", "seniority", "author"]:
        columns += [f"factuality_{step}", MATCHED[f"n_{step}"]]
    write_step(df[columns + ["bias_location"] + AUTHOR], step=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", default=RECOMMENDATIONS)
    parser.add_argument("--skip_cascade", action="store_true",
                        help="Reuse the existing cascade output")
    parser.add_argument("--skip_name", action="store_true",
                        help="Reuse the cascade's name matches (its slow step)")
    args = parser.parse_args()
    run(args.recommendations, args.skip_cascade, args.skip_name)


if __name__ == "__main__":
    main()
