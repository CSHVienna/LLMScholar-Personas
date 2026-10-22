"""
1_factuality.py — Funnel step 7.1: factuality of every recommendation of a valid call.

Reads the output of the funnel up to OpenAlex (summary/factuality_full.csv: the
factuality cascade of scripts/4_factuality/, which compares each recommended
author with Semantic Scholar using the field and seniority of the prompt, not the
LLM's answer, plus the OpenAlex data of the confirmed authors from
scripts/6_enrichment/), and keeps the recommendations of the valid calls of
scripts/3_validity/ whose name is complete: at least 2 words in "name lastname"
(a single first name or a single last name is left out).

Per recommendation, in Semantic Scholar:
  factuality_name       a researcher with that name exists
  matched_names         how many researchers have that name (0 if none)
  factuality_field      at least one of them is in the asked field
  matched_field         how many of them are
  factuality_seniority  at least one of them has the asked seniority
  matched_seniority     how many of them do
  factuality_author     at least one of them has the asked field AND seniority
  matched_author        how many of them do; researcher_id is one of them at random

and, only for factuality_author = True (the authors looked up in OpenAlex):
  bias_location         the country of the prompt is the OpenAlex country of the
                        author (True / False; empty when the author is not
                        confirmed, not in OpenAlex or a country is missing)

Steps 2 and 3 need who the author is, so the output also carries researcher_id
(Semantic Scholar), oa_id (OpenAlex) and their attributes (gender, ethnicity,
country, works, citations, first/last publication year).

Output: ../results/pipeline/factuality/<model>.csv — call columns + the LLM's
answer + the columns above.

Usage (from code/, with PYTHONPATH=.):
  python scripts/7_metrics/1_factuality.py
"""

from libs.pipeline_common import (
    ANSWER, CALL, has_full_name, load, logger, read_step, write_step,
)

FUNNEL_OUT = "../results/summary/factuality_full.csv"
STEPS = ["name", "field", "seniority", "author"]
MATCHED = {"n_name": "matched_names", "n_field": "matched_field",
           "n_seniority": "matched_seniority", "n_author": "matched_author"}
AUTHOR = ["researcher_id", "matched_name", "gt_gender", "perceived_ethnicity",
          "oa_id", "location_oa_iso", "oa_works_count", "oa_cited_by_count",
          "oa_first_pub_year", "oa_last_pub_year", "oa_career_age"]


def run() -> None:
    valid = read_step("validity", CALL + ["validity"])
    valid = valid[valid["validity"]].drop(columns="validity")

    factuality = [f"factuality_{s}" for s in STEPS]
    df = load(FUNNEL_OUT, ANSWER + list(MATCHED) + factuality + AUTHOR + ["location_status"])
    df = df.merge(valid, on=CALL).rename(columns=MATCHED)

    complete = has_full_name(df)
    logger.info("Left out %d recommendations with an incomplete name", (~complete).sum())
    df = df[complete]

    df["bias_location"] = df["location_status"].map(
        {"location_match": True, "location_mismatch": False})

    columns = CALL + ANSWER
    for s in STEPS:
        columns += [f"factuality_{s}", MATCHED[f"n_{s}"]]
    write_step(df[columns + ["bias_location"] + AUTHOR], step="factuality")


if __name__ == "__main__":
    run()
