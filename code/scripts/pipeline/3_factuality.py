"""
3_factuality.py — Step 3: factuality of every recommendation of a valid call.

Reads the output of the factuality cascade (scripts/factuality/), which compares
each recommended author with Semantic Scholar using the field and seniority of
the prompt (not the LLM's answer), and keeps the recommendations of the valid
calls of step 1 whose name is complete: at least 2 words in "name lastname" (a
single first name or a single last name is left out).

Per recommendation, in Semantic Scholar:
  factuality_name       a researcher with that name exists
  matched_names         how many researchers have that name (0 if none)
  factuality_field      at least one of them is in the asked field
  matched_field         how many of them are
  factuality_seniority  at least one of them has the asked seniority
  matched_seniority     how many of them do
  factuality_author     at least one of them has the asked field AND seniority
  matched_author        how many of them do; researcher_id is one of them at random

The same checks in OpenAlex (scripts/openalex_cascade/), as oa_factuality_* and
oa_matched_* (names, field, seniority, author), and:
  bias_location         the country of the prompt is the OpenAlex country of the
                        author (True / False; empty when the name was not found
                        or a country is missing)

Steps 4 and 5 need who the author is, so the output also carries researcher_id
(Semantic Scholar), oa_id (OpenAlex) and their attributes (gender, ethnicity,
country, works, citations, first/last publication year).

Runs once per OpenAlex variant (common.VARIANTS); only the OpenAlex columns
differ between variants.

Output: 3_factuality/<variant>/<model>.csv — call columns + the LLM's answer +
the columns above.

Usage (from code/, with PYTHONPATH=.):
  python scripts/pipeline/3_factuality.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ANSWER, CALL, VARIANTS, has_full_name, load, logger, read_step, write_step,
)

CASCADE_OUT = "../results/summary/cascade/factuality_full_{variant}.csv"
STEPS = ["name", "field", "seniority", "author"]
MATCHED = {"n_name": "matched_names", "n_field": "matched_field",
           "n_seniority": "matched_seniority", "n_author": "matched_author"}
OA_MATCHED = {f"oa_{k}": f"oa_{v}" for k, v in MATCHED.items()}
AUTHOR = ["researcher_id", "matched_name", "gt_gender", "perceived_ethnicity",
          "oa_id", "location_oa_iso", "oa_works_count", "oa_cited_by_count",
          "oa_first_pub_year", "oa_last_pub_year", "oa_career_age"]


def run() -> None:
    valid = read_step(1, CALL + ["validity"])
    valid = valid[valid["validity"]].drop(columns="validity")

    for variant in VARIANTS:
        factuality = [f"factuality_{s}" for s in STEPS]
        df = load(CASCADE_OUT.format(variant=variant),
                  ANSWER + list(MATCHED) + list(OA_MATCHED) + factuality + AUTHOR + ["location_status"])
        df = df.merge(valid, on=CALL).rename(columns={**MATCHED, **OA_MATCHED})

        complete = has_full_name(df)
        logger.info("%s: left out %d recommendations with an incomplete name", variant, (~complete).sum())
        df = df[complete]

        # The OpenAlex checks, as True / False like the Semantic Scholar ones.
        for s, col in zip(STEPS, OA_MATCHED.values()):
            df[f"oa_factuality_{s}"] = df[col] > 0
        df["bias_location"] = df["location_status"].map(
            {"location_match": True, "location_mismatch": False})

        columns = CALL + ANSWER
        for s in STEPS:
            columns += [f"factuality_{s}", MATCHED[f"n_{s}"]]
        for s in STEPS:
            columns += [f"oa_factuality_{s}", OA_MATCHED[f"oa_n_{s}"]]
        write_step(df[columns + ["bias_location"] + AUTHOR], step=3, variant=variant)


if __name__ == "__main__":
    run()
