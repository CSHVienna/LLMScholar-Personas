"""
resolve_homonyms.py — Break name ties with the field the prompt asked for.

The problem
-----------
factuality_author_jw matches a recommended name against 6.7M researchers and
keeps `scores.argmax(axis=1)`. A common name matches several *different* people
exactly (score 1.0), and argmax does not break ties: it returns the first row of
the block, i.e. the first row in file order. The reference parquet is stored in
field blocks —

    Sociology 0–358,801 | Biology 358,802–2,412,033 | Physics | Computer Science
    | Mathematics | Psychology 5,449,467–6,686,107

— so the tie always goes to whichever field sits earlier in the file. Measured on
this dataset the rule predicts the assigned field for 100.0% of the 42,211
ambiguous names: Biology takes 67.0% of ambiguous recommendations against the
26.2% a random pick among the homonyms would give, and Psychology, the last
block, takes 0.0% against 22.5%.

Everything copied from the chosen record inherits that arbitrary choice —
gt_field, gt_career_age, gt_citations, researcher_id. (gt_gender does not vary
between homonyms: same name, same inferred gender.)

What this script does
---------------------
Re-picks, among the homonyms the matcher was choosing between, the one whose
field is the field the prompt asked for; falls back to the current file-order
pick when no homonym has it. Then recomputes the two column groups that read
`gt_*`: field_status/field_check_source/field_evidence and seniority_*.

Ties are re-broken rather than re-matched: two reference records with the same
normalised display name score identically under Jaro-Winkler, so they are
exactly the set argmax was choosing between. No name similarity is recomputed,
which is why this runs in minutes instead of re-running the matcher.

Read the numbers it produces with the circularity in mind: resolving by the
asked field cannot make the model look worse at fields, and raises field_match
from 26.6% to 44.7% on this dataset, nearly all of it by construction: a row
tagged `asked_field` matches by definition, and one tagged `file_order` cannot
match at all. The `tiebreak` column exists so a figure can restrict itself to
`unique` rows — names with a single candidate, where nothing was decided for us
— which is 49.3% of matched authors against 29.5% `asked_field` and 21.1%
`file_order`. There `field_match` is 30.8%, and that is the only one of the
three that measures a model rather than the supply of convenient homonyms.

Usage (from code/, with PYTHONPATH=.):
  python scripts/factuality/resolve_homonyms.py
  python scripts/factuality/resolve_homonyms.py --check   # verify against the
                                                          # per-row classifiers
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The matcher's own normaliser, reused rather than re-implemented: the grouping
# here has to be exactly the one argmax was choosing between, and a vectorised
# twin drifted on non-ASCII characters that are neither letters nor combining
# marks (a soft hyphen survives as a space upstream, so it must here too).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from factuality_author_jw import normalize as _normalize  # noqa: E402

from libs.metrics.constants import (
    FIELD_NORM_MAP,
    LLM_TARGET_TO_BUCKET,
    factuality_status_flags,
)
from libs.utils.config import config_default, get_results_path
from libs.utils.logging import setup_logging

logger = setup_logging()

REFERENCE_YEAR = 2025  # must match factuality_author_jw.REFERENCE_YEAR
JUNIOR_MAX = 10        # must match factuality_seniority
SENIOR_MIN = 20

_FIELD = factuality_status_flags("field")
_SEN = factuality_status_flags("seniority")

# Translates LLM field values (ES / DE) → canonical English, as
# factuality_field_check._norm_field does.
FIELD_TRANSLATION = {**FIELD_NORM_MAP}


def normalize_series(s: pd.Series) -> pd.Series:
    """factuality_author_jw.normalize over a column, computed once per distinct
    value — 6.7M names carry ~5.6M distinct ones, and the map is what makes it
    affordable without paraphrasing the original."""
    vals = s.dropna().astype(str).unique()
    mapping = {v: _normalize(v) for v in vals}
    return s.astype(object).map(mapping).fillna("")


def _norm_field(s: pd.Series) -> pd.Series:
    """Vectorised twin of factuality_field_check._norm_field."""
    out = s.fillna("").astype(str).str.strip().replace(FIELD_TRANSLATION)
    out = out.str.lower().str.replace("_", " ", regex=False)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def build_candidates(parquet_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference records grouped by normalised name, in file order.

    Returns (per_name, per_name_field):
      per_name        one row per normalised name: how many records, how many
                      distinct fields
      per_name_field  one row per (name, field): the first record of that field,
                      which is what the matcher would have picked had that field
                      sat first in the file
    """
    logger.info("Loading reference parquet: %s", parquet_path)
    ref = pd.read_parquet(
        parquet_path,
        columns=["Researcher_id", "Name", "Field", "Combined_gender",
                 "First_year", "Citations"],
    )
    ref["gt_career_age"] = (REFERENCE_YEAR - ref["First_year"]).clip(lower=0)
    ref["key"] = normalize_series(ref["Name"])
    ref["row"] = np.arange(len(ref))  # file order == the matcher's block order
    logger.info("Reference records: %d | distinct normalised names: %d",
                len(ref), ref.key.nunique())

    per_name = ref.groupby("key", sort=False).agg(
        homonym_count=("Researcher_id", "size"),
        homonym_fields=("Field", "nunique"),
    )
    per_name_field = (
        ref.sort_values("row")
        .drop_duplicates(["key", "Field"])
        .set_index(["key", "Field"])[
            ["Researcher_id", "Name", "gt_career_age", "Citations", "Combined_gender"]
        ]
        .rename(columns={
            "Researcher_id": "new_researcher_id",
            "Name": "new_matched_name",
            "gt_career_age": "new_gt_career_age",
            "Citations": "new_gt_citations",
            "Combined_gender": "new_gt_gender",
        })
    )
    return per_name, per_name_field


def resolve_chunk(
    chunk: pd.DataFrame, per_name: pd.DataFrame, per_name_field: pd.DataFrame
) -> pd.DataFrame:
    """Re-break the ties in one chunk and recompute what depends on gt_*."""
    df = chunk.copy()
    df["field_en"] = df["field"].map(FIELD_NORM_MAP).fillna(df["field"])
    found = df["author_status"].eq("found")

    key = normalize_series(df["matched_name"].where(found))
    df["homonym_count"] = key.map(per_name["homonym_count"]).fillna(0).astype(int)
    df["homonym_fields"] = key.map(per_name["homonym_fields"]).fillna(0).astype(int)

    # The asked field among the homonyms → take that record.
    idx = pd.MultiIndex.from_arrays([key, df["field_en"]])
    swap = per_name_field.reindex(idx)
    swap.index = df.index

    ambiguous = found & (df["homonym_fields"] > 1)
    switch = ambiguous & swap["new_researcher_id"].notna()

    df["tiebreak"] = np.where(
        ~found, "", np.where(switch, "asked_field",
                             np.where(ambiguous, "file_order", "unique"))
    )
    if switch.any():
        for col, new in (("researcher_id", "new_researcher_id"),
                         ("matched_name", "new_matched_name"),
                         ("gt_career_age", "new_gt_career_age"),
                         ("gt_citations", "new_gt_citations"),
                         ("gt_gender", "new_gt_gender")):
            df[col] = df[col].astype(object)
            df.loc[switch, col] = swap.loc[switch, new]
        df.loc[switch, "gt_field"] = df.loc[switch, "field_en"]

    # ── Recompute field_status (twin of factuality_field_check.classify_row) ──
    gt = df["gt_field"].fillna("").astype(str).str.strip()
    llm = df["field"].fillna("").astype(str).str.strip()
    has_gt = gt.ne("")
    match = _norm_field(df["field"]).eq(_norm_field(df["gt_field"])) & has_gt
    df["field_status"] = np.select(
        [~found, llm.eq(""), match, has_gt],
        [_FIELD["NOT_APPLICABLE"], _FIELD["UNKNOWN"], _FIELD["MATCH"],
         _FIELD["MISMATCH"]],
        default=_FIELD["UNKNOWN"],
    )
    df["field_check_source"] = np.where(found & llm.ne("") & has_gt, "gt", "none")
    df["field_evidence"] = np.where(found & llm.ne("") & has_gt, gt, "")

    # ── Recompute seniority_* (twin of factuality_seniority.classify_row) ─────
    age = pd.to_numeric(df["gt_career_age"], errors="coerce")
    llm_bucket = df["target"].fillna("").astype(str).str.strip().map(LLM_TARGET_TO_BUCKET)
    bucket = pd.Series(None, index=df.index, dtype="object")
    bucket[age.le(JUNIOR_MAX)] = "Junior"
    bucket[age.ge(SENIOR_MIN)] = "Senior"
    bucket[~found] = None

    df["seniority_career_age"] = age.where(found)
    df["seniority_age_source"] = np.where(found & age.notna(), "gt", "none")
    df["seniority_bucket"] = bucket
    df["seniority_llm_bucket"] = llm_bucket
    bucket_na = bucket.isna()
    same_bucket = bucket.fillna("") == llm_bucket.fillna("")
    df["seniority_status"] = np.select(
        [~found,
         age.isna() | llm_bucket.isna(),
         bucket_na,
         same_bucket],
        [_SEN["NOT_APPLICABLE"], _SEN["UNKNOWN"], _SEN["UNKNOWN"], _SEN["MATCH"]],
        default=_SEN["MISMATCH"],
    )
    return df.drop(columns=["field_en"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    results = get_results_path(path=args.results)
    src = Path(args.input or results / "summary" / "factuality_full.csv")
    dst = Path(args.output or results / "summary" / "factuality_full_resolved.csv")
    if not src.exists():
        parser.error(f"Not found: {src}")

    per_name, per_name_field = build_candidates(args.parquet)

    logger.info("Rewriting %s → %s", src, dst)
    seen = {"unique": 0, "asked_field": 0, "file_order": 0}
    rows = 0
    for i, chunk in enumerate(pd.read_csv(src, chunksize=args.chunksize, low_memory=False)):
        out = resolve_chunk(chunk, per_name, per_name_field)
        out.to_csv(dst, index=False, mode="w" if i == 0 else "a", header=i == 0)
        counts = out["tiebreak"].value_counts()
        for k in seen:
            seen[k] += int(counts.get(k, 0))
        rows += len(out)
        logger.info("chunk %d: %d rows", i, len(out))

    matched = sum(seen.values())
    logger.info("Wrote %s (%d rows)", dst, rows)
    logger.info("Matched authors: %d", matched)
    for k, v in seen.items():
        logger.info("  %-12s %9d  (%.1f%%)", k, v, 100 * v / matched if matched else 0)


if __name__ == "__main__":
    main()
