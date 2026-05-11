"""
factuality_location.py — Step 4 of the factuality pipeline.

Reads the output of factuality_seniority.py and decides whether the country
the LLM assigned to each author matches the country of the author's actual
last-known institution in OpenAlex.

The LLM `location` column comes in EN/ES/DE for 5 countries — all variants
collapse to a single ISO-3166-1 alpha-2 code.

Output columns added:
  location_llm_country     raw LLM string
  location_llm_iso         {EC | JP | DE | CA | ZA | None}
  location_oa_iso          country_code from OpenAlex (set in step 1)
  location_oa_institution  oa_last_institution
  location_status          {location_match | location_mismatch
                            | location_unknown | not_applicable}

Usage (from code/scripts/):
  python factuality_location.py \\
      --input  ../../../results/summary/factuality_seniority.csv \\
      --output ../../../results/summary/factuality_location.csv
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

STATUS_MATCH          = "location_match"
STATUS_MISMATCH       = "location_mismatch"
STATUS_UNKNOWN        = "location_unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

AUTHOR_HALLUCINATED = "hallucinated"

# Maps every observed `location` value (EN/ES/DE) → ISO alpha-2
LLM_COUNTRY_TO_ISO = {
    "Ecuador":      "EC",
    "Japan":        "JP",
    "Japón":        "JP",
    "Germany":      "DE",
    "Alemania":     "DE",
    "Deutschland":  "DE",
    "Canada":       "CA",
    "Canadá":       "CA",
    "Kanada":       "CA",
    "South Africa": "ZA",
    "Sudáfrica":    "ZA",
    "Südafrika":    "ZA",
}


# ── Per-row decision ───────────────────────────────────────────────────────────

def classify_row(row: pd.Series) -> dict:
    raw_country = str(row.get("location") or "").strip()
    llm_iso     = LLM_COUNTRY_TO_ISO.get(raw_country)
    oa_iso      = row.get("oa_country_code")
    oa_iso      = str(oa_iso).strip().upper() if pd.notna(oa_iso) and str(oa_iso).strip() else None

    base = {
        "location_llm_country":    raw_country or None,
        "location_llm_iso":        llm_iso,
        "location_oa_iso":         oa_iso,
        "location_oa_institution": row.get("oa_last_institution") if pd.notna(row.get("oa_last_institution")) else None,
    }

    if row.get("author_status") == AUTHOR_HALLUCINATED:
        return {**base, "location_status": STATUS_NOT_APPLICABLE}

    if not llm_iso or not oa_iso:
        return {**base, "location_status": STATUS_UNKNOWN}

    return {**base, "location_status": STATUS_MATCH if llm_iso == oa_iso else STATUS_MISMATCH}


# ── Main ───────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str) -> None:
    logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Rows: %d", len(df))

    records = [classify_row(row) for _, row in df.iterrows()]
    for col in ["location_llm_country", "location_llm_iso",
                "location_oa_iso", "location_oa_institution", "location_status"]:
        df[col] = [r[col] for r in records]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("Location status distribution:")
    for status, count in df["location_status"].value_counts().items():
        logger.info("  %-20s %6d  (%.1f%%)", status, count, 100 * count / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: verify the LLM-assigned country matches the author's last-known institution country in OpenAlex"
    )
    parser.add_argument("--input",  required=True, help="Path to factuality_seniority.csv (output of factuality_seniority.py)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
