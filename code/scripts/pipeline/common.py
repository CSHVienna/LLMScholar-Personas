"""
common.py — Definitions shared by the pipeline scripts.

The pipeline works on three units:
  setup           model + params
  call            setup + created_at + run_id          (one LLM response)
  recommendation  call + one author of that response   (name, lastname, ...)

Steps 1, 2, 3 and 5 write one CSV per model into their folder under OUT_DIR:
  1_validity/                per call            validity, refusal, duplicates
  2_consistency/             per setup           consistency
  3_factuality/<variant>/    per recommendation  factuality_*, matched_*, oa_matched_*
  5_socials/<variant>/       per call            diversity, parity, popularity,
                                                 similarity, connectedness
Step 4 writes author stats to [data].oa_author_stats; step 6 writes 6_calls.csv.

Steps 3 and 5 run once per OpenAlex variant (VARIANTS): how the OpenAlex author
of each recommendation was chosen among the ones that pass the same checks
(scripts/openalex_cascade/). Steps 1 and 2 do not depend on OpenAlex.
"""

import re
from pathlib import Path

import pandas as pd

from libs.utils.logging import setup_logging

logger = setup_logging()

RECOMMENDATIONS = "../results/summary/recommendations.csv"
OUT_DIR = Path("../results/pipeline")
FOLDERS = {1: "1_validity", 2: "2_consistency", 3: "3_factuality", 5: "5_socials"}
VARIANTS = ["random", "most_cited"]

# Prompt params: name in recommendations.csv → name in the outputs.
PARAMS = {
    "language": "language",
    "location": "country",
    "role": "role",
    "task": "task",
    "target": "seniority",
    "k": "k",
    "field": "field",
    "subfield": "subfield",
}
SETUP = ["model"] + list(PARAMS.values())
CALL = ["model", "created_at", "run_id"] + list(PARAMS.values())

# What the LLM answered about each recommended author.
ANSWER = ["name", "lastname", "current_affiliations", "areas_of_research_or_work",
          "reason", "source"]


def load(path: str, columns: list[str]) -> pd.DataFrame:
    """Read the call columns plus `columns` from a recommendations-like CSV,
    with the params renamed as in the outputs."""
    raw = ["model", "created_at", "run_id"] + list(PARAMS) + columns
    df = pd.read_csv(path, usecols=raw, low_memory=False)
    return df.rename(columns=PARAMS)


def author_key(df: pd.DataFrame) -> pd.Series:
    """The recommended author as written by the LLM: "name lastname",
    lowercased, with punctuation and extra spaces removed ("J. Smith" and
    "j  smith" are the same author; accents are already removed by the parser).
    NaN when both are empty."""
    full = (df["name"].fillna("").astype(str) + " "
            + df["lastname"].fillna("").astype(str)).str.lower()
    full = full.str.replace(r"[\W\d_]+", " ", regex=True).str.split().str.join(" ")
    return full.where(full != "")


def has_full_name(df: pd.DataFrame) -> pd.Series:
    """True when "name lastname" has at least 2 words. A single word (only a
    first name or only a last name) matches far too many researchers by
    chance. A full name written in one field ("Ana Pérez" as the first name)
    counts as complete."""
    return author_key(df).str.split().str.len().fillna(0) >= 2


def step_folder(step: int, variant: str | None = None) -> Path:
    folder = OUT_DIR / FOLDERS[step]
    return folder / variant if variant else folder


def write_step(df: pd.DataFrame, step: int, variant: str | None = None) -> None:
    """One CSV per model into the step's folder (and variant)."""
    folder = step_folder(step, variant)
    folder.mkdir(parents=True, exist_ok=True)
    for model, part in df.groupby("model"):
        part.to_csv(folder / f"{re.sub(r'[^\w.-]', '_', model)}.csv", index=False)
    logger.info("Wrote %d files (%d rows) to %s", df["model"].nunique(), len(df), folder)


def read_step(step: int, columns: list[str] | None = None, variant: str | None = None) -> pd.DataFrame:
    """All the per-model CSVs of a step (and variant), as one table."""
    folder = step_folder(step, variant)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise SystemExit(f"No files in {folder}: run step {step} first")
    return pd.concat(
        (pd.read_csv(f, usecols=columns, low_memory=False) for f in files),
        ignore_index=True,
    )
