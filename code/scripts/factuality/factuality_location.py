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

Usage (from code/, with PYTHONPATH=.):
  python scripts/factuality/factuality_location.py \\
      --input  ../results/summary/factuality_seniority.csv \\
      --output ../results/summary/factuality_location.csv
"""

import argparse

import pandas as pd

from libs.utils.cli import add_io_args
from libs.utils.ios import read_input_csv, write_output_csv
from libs.utils.logging import log_value_counts, setup_logging

logger = setup_logging()

# ── Constants ──────────────────────────────────────────────────────────────────
# All factuality status flags and the LLM-country → ISO map are centralised in
# libs.metrics.constants — re-export under the legacy names so the rest of the
# module stays unchanged.

from libs.metrics.constants import (
    FACTUALITY_AUTHOR_HALLUCINATED as AUTHOR_HALLUCINATED,
    FACTUALITY_STATUS_MATCH as STATUS_MATCH,
    FACTUALITY_STATUS_MISMATCH as STATUS_MISMATCH,
    FACTUALITY_STATUS_NOT_APPLICABLE as STATUS_NOT_APPLICABLE,
    FACTUALITY_STATUS_UNKNOWN as STATUS_UNKNOWN,
    LLM_COUNTRY_TO_ISO,
)


# ── Per-row decision ───────────────────────────────────────────────────────────


def classify_row(row: pd.Series) -> dict:
    raw_country = str(row.get("location") or "").strip()
    llm_iso = LLM_COUNTRY_TO_ISO.get(raw_country)
    oa_iso = row.get("oa_country_code")
    oa_iso = (
        str(oa_iso).strip().upper()
        if pd.notna(oa_iso) and str(oa_iso).strip()
        else None
    )

    base = {
        "location_llm_country": raw_country or None,
        "location_llm_iso": llm_iso,
        "location_oa_iso": oa_iso,
        "location_oa_institution": (
            row.get("oa_last_institution")
            if pd.notna(row.get("oa_last_institution"))
            else None
        ),
    }

    if row.get("author_status") == AUTHOR_HALLUCINATED:
        return {**base, "location_status": STATUS_NOT_APPLICABLE}

    if not llm_iso or not oa_iso:
        return {**base, "location_status": STATUS_UNKNOWN}

    return {
        **base,
        "location_status": STATUS_MATCH if llm_iso == oa_iso else STATUS_MISMATCH,
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def run(input_path: str, output_path: str) -> None:
    df = read_input_csv(input_path, logger=logger)

    records = [classify_row(row) for _, row in df.iterrows()]
    for col in [
        "location_llm_country",
        "location_llm_iso",
        "location_oa_iso",
        "location_oa_institution",
        "location_status",
    ]:
        df[col] = [r[col] for r in records]

    write_output_csv(df, output_path, logger=logger)
    log_value_counts(df, "location_status", title="Location status distribution", logger=logger)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: verify the LLM-assigned country matches the author's last-known institution country in OpenAlex"
    )
    add_io_args(
        parser,
        input_help="Path to factuality_seniority.csv (output of factuality_seniority.py)",
    )
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
