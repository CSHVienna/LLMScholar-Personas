"""
common.py — Definitions shared by the four pipeline scripts.

The pipeline works on three units:
  setup           model + params
  call            setup + created_at + run_id          (one LLM response)
  recommendation  call + one author of that response   (name, lastname, ...)

Every script writes one CSV per model into its own folder under OUT_DIR:
  1_validity/      per call            validity, refusal, duplicates
  2_consistency/   per setup           consistency
  3_factuality/    per recommendation  factuality_* and matched_*
  5_socials/       per call            diversity, parity, popularity, similarity,
                                       connectedness
"""

import re
from pathlib import Path

import pandas as pd

from libs.utils.logging import setup_logging

logger = setup_logging()

RECOMMENDATIONS = "../results/summary/recommendations.csv"
OUT_DIR = Path("../results/pipeline")
FOLDERS = {1: "1_validity", 2: "2_consistency", 3: "3_factuality", 5: "5_socials"}

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
    lowercased. NaN when both are empty."""
    full = (df["name"].fillna("").astype(str) + " "
            + df["lastname"].fillna("").astype(str)).str.strip().str.lower()
    return full.where(full != "")


def has_full_name(df: pd.DataFrame) -> pd.Series:
    """True when "name lastname" has at least 2 words. A single word (only a
    first name or only a last name) matches far too many researchers by
    chance. A full name written in one field ("Ana Pérez" as the first name)
    counts as complete."""
    return author_key(df).str.split().str.len().fillna(0) >= 2


def write_step(df: pd.DataFrame, step: int) -> None:
    """One CSV per model into the step's folder."""
    folder = OUT_DIR / FOLDERS[step]
    folder.mkdir(parents=True, exist_ok=True)
    for model, part in df.groupby("model"):
        part.to_csv(folder / f"{re.sub(r'[^\w.-]', '_', model)}.csv", index=False)
    logger.info("Wrote %d files (%d rows) to %s", df["model"].nunique(), len(df), folder)


def read_step(step: int, columns: list[str] | None = None) -> pd.DataFrame:
    """All the per-model CSVs of a step, as one table."""
    files = sorted((OUT_DIR / FOLDERS[step]).glob("*.csv"))
    if not files:
        raise SystemExit(f"No files in {OUT_DIR / FOLDERS[step]}: run step {step} first")
    return pd.concat(
        (pd.read_csv(f, usecols=columns, low_memory=False) for f in files),
        ignore_index=True,
    )
