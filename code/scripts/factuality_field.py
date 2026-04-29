"""
factuality_field.py — Verify if LLM-recommended authors exist in ground truth.

For each recommendation row produced by batch_parse_results.py, checks the
Semantic Scholar ground truth data (one CSV per field) and classifies the result:

  found_in_field       — author found in the same field the LLM was asked about
  found_in_other_field — author found, but in a different field
  not_found            — author absent from all field files

Output: the recommendations CSV enriched with:
  factuality_status, gt_name, gt_field, gt_gender, gt_career_age, gt_citations

Usage (from code/scripts/):
  export PYTHONPATH="$PYTHONPATH:../libs"
  python factuality_field.py \\
      --recommendations ../../results/summary/recommendations.csv \\
      --data_dir        ../../data/data/semantic_scholar_data \\
      --output          ../../results/summary/factuality_field.csv
"""

import argparse
import logging
import os
import re
import unicodedata

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

GT_FILES = {
    "Biology":         "DataFrameRankings_Genderize_Namsor_Biology.csv",
    "Computer_Science":"DataFrameRankings_Genderize_Namsor_Computer_Science.csv",
    "Mathematics":     "DataFrameRankings_Genderize_Namsor_Mathematics.csv",
    "Physics":         "DataFrameRankings_Genderize_Namsor_Physics.csv",
    "Psychology":      "DataFrameRankings_Genderize_Namsor_Psychology.csv",
    "Sociology":       "DataFrameRankings_Genderize_Namsor_Sociology.csv",
}

# Maps recommendation `field` values → ground truth file keys
REC_FIELD_TO_GT = {
    "Biology":          "Biology",
    "Computer Science": "Computer_Science",
    "Mathematics":      "Mathematics",
    "Physics":          "Physics",
    "Psychology":       "Psychology",
    "Sociology":        "Sociology",
}

STATUS_FOUND_IN_FIELD = "found_in_field"
STATUS_FOUND_OTHER    = "found_in_other_field"
STATUS_NOT_FOUND      = "not_found"

GT_COLS = ["Year", "Researcher_id", "Name", "Combined_gender", "Career_age", "Citations"]
CHUNK_SIZE = 500_000


# ── Name normalisation ─────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Lowercase, strip accents, keep only letters and spaces."""
    if not isinstance(name, str) or not name.strip():
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z\s]", "", name.lower())
    return " ".join(name.split())


def _abbr_key(norm_full: str) -> tuple[str, str] | None:
    """Return (last_word, first_initial) for abbreviated matching, or None."""
    parts = norm_full.split()
    if len(parts) < 2:
        return None
    return (parts[-1], parts[0][0])


# ── Ground truth index ─────────────────────────────────────────────────────────

def _load_field(path: str) -> pd.DataFrame:
    """
    Load a single ground truth file keeping only the latest year per researcher.
    Reads in chunks to limit peak memory on large files.
    """
    deduped_chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, usecols=GT_COLS, low_memory=False, chunksize=CHUNK_SIZE):
        chunk = (
            chunk.sort_values("Year", ascending=False)
                 .drop_duplicates(subset=["Researcher_id"], keep="first")
        )
        deduped_chunks.append(chunk)

    df = pd.concat(deduped_chunks, ignore_index=True)
    df = df.sort_values("Year", ascending=False).drop_duplicates(subset=["Researcher_id"], keep="first")
    df["Name"] = df["Name"].fillna("").astype(str).str.strip()
    return df[df["Name"] != ""]


def _build_field_index(df: pd.DataFrame) -> dict:
    """
    Build two lookup dicts from a field DataFrame:
      full — {normalized_full_name: record}
      abbr — {(last_word, first_initial): record}
    """
    df = df.copy()
    df["norm_name"] = df["Name"].apply(normalize_name)
    df = df[df["norm_name"] != ""]

    parts   = df["norm_name"].str.split()
    df["_last"]  = parts.str[-1].fillna("")
    df["_first"] = parts.str[0].str[0].fillna("")

    def to_record(row: pd.Series) -> dict:
        return {
            "gt_name":       row["Name"],
            "gt_gender":     row["Combined_gender"] if pd.notna(row.get("Combined_gender")) else None,
            "gt_career_age": row["Career_age"]      if pd.notna(row.get("Career_age"))      else None,
            "gt_citations":  row["Citations"]        if pd.notna(row.get("Citations"))        else None,
        }

    # Full-name index (first occurrence wins after outer dedup)
    full_idx: dict[str, dict] = {}
    for _, row in df.drop_duplicates(subset=["norm_name"], keep="first").iterrows():
        full_idx[row["norm_name"]] = to_record(row)

    # Abbreviated index: (last_word, first_initial)
    abbr_idx: dict[tuple, dict] = {}
    multi = df[df["norm_name"].str.contains(r"\s")].drop_duplicates(subset=["_last", "_first"], keep="first")
    for _, row in multi.iterrows():
        key = (row["_last"], row["_first"])
        abbr_idx[key] = to_record(row)

    return {"full": full_idx, "abbr": abbr_idx}


def load_gt_index(data_dir: str) -> dict[str, dict]:
    """
    Return {gt_field_key: {"full": {...}, "abbr": {...}}} for every available field.
    """
    index: dict[str, dict] = {}
    for gt_key, filename in GT_FILES.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            logger.warning("Ground truth file not found, skipping: %s", path)
            continue
        logger.info("Loading %s …", gt_key)
        df = _load_field(path)
        index[gt_key] = _build_field_index(df)
        logger.info("  %s: %d unique researchers", gt_key, len(index[gt_key]["full"]))
    return index


# ── Lookup ─────────────────────────────────────────────────────────────────────

def lookup_author(
    norm_full: str,
    target_gt_key: str | None,
    gt_index: dict[str, dict],
) -> tuple[str, dict | None, str | None]:
    """
    Priority:
      1. Exact full name in target field   → found_in_field
      2. Abbreviated match in target field → found_in_field
      3. Exact full name in any other field   → found_in_other_field
      4. Abbreviated match in any other field → found_in_other_field
      5. → not_found

    Returns (status, record_or_None, matched_gt_key_or_None).
    """
    abbr = _abbr_key(norm_full)

    # --- target field first ---
    if target_gt_key and target_gt_key in gt_index:
        field_data = gt_index[target_gt_key]
        if norm_full in field_data["full"]:
            return STATUS_FOUND_IN_FIELD, field_data["full"][norm_full], target_gt_key
        if abbr and abbr in field_data["abbr"]:
            return STATUS_FOUND_IN_FIELD, field_data["abbr"][abbr], target_gt_key

    # --- other fields ---
    for gt_key, field_data in gt_index.items():
        if gt_key == target_gt_key:
            continue
        if norm_full in field_data["full"]:
            return STATUS_FOUND_OTHER, field_data["full"][norm_full], gt_key
        if abbr and abbr in field_data["abbr"]:
            return STATUS_FOUND_OTHER, field_data["abbr"][abbr], gt_key

    return STATUS_NOT_FOUND, None, None


# ── Main processing ────────────────────────────────────────────────────────────

def run(recommendations_path: str, data_dir: str, output_path: str) -> None:
    logger.info("Loading recommendations: %s", recommendations_path)
    df = pd.read_csv(recommendations_path, low_memory=False)
    logger.info("Recommendations: %d rows", len(df))

    gt_index = load_gt_index(data_dir)

    statuses:     list[str]        = []
    gt_names:     list[str | None] = []
    gt_fields:    list[str | None] = []
    gt_genders:   list             = []
    gt_ages:      list             = []
    gt_citations: list             = []

    for _, row in df.iterrows():
        name     = str(row.get("name",     "") or "").strip()
        lastname = str(row.get("lastname", "") or "").strip()
        field    = str(row.get("field",    "") or "").strip()

        full_name     = f"{name} {lastname}".strip()
        norm_full     = normalize_name(full_name)
        target_gt_key = REC_FIELD_TO_GT.get(field)

        if not norm_full:
            status, record, matched = STATUS_NOT_FOUND, None, None
        else:
            status, record, matched = lookup_author(norm_full, target_gt_key, gt_index)

        statuses.append(status)
        gt_names.append(record["gt_name"]       if record else None)
        gt_fields.append(matched                if matched else None)
        gt_genders.append(record["gt_gender"]   if record else None)
        gt_ages.append(record["gt_career_age"]  if record else None)
        gt_citations.append(record["gt_citations"] if record else None)

    df = df.copy()
    df["factuality_status"] = statuses
    df["gt_name"]           = gt_names
    df["gt_field"]          = gt_fields
    df["gt_gender"]         = gt_genders
    df["gt_career_age"]     = gt_ages
    df["gt_citations"]      = gt_citations

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("Factuality distribution:")
    for status, count in df["factuality_status"].value_counts().items():
        logger.info("  %-25s %6d  (%.1f%%)", status, count, 100 * count / n)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify factuality of LLM recommendations against Semantic Scholar ground truth"
    )
    parser.add_argument("--recommendations", required=True,
                        help="Path to recommendations CSV (output of batch_parse_results.py)")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing the ground truth CSV files")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    run(args.recommendations, args.data_dir, args.output)


if __name__ == "__main__":
    main()
