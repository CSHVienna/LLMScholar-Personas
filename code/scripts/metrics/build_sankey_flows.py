"""
build_sankey_flows.py — Pre-compute the flow tables behind the Sankey figures.

Same shape and rationale as build_location_flows.py (issue #37): read
factuality_full.csv once in chunks, reduce to long (source, target, count) rows
keeping the slicing dimensions, and let the notebook stay plotting-only. The
source CSV is 6.8 GB, so no figure should ever touch it directly.

Six flows, each written to its own CSV under
<results_dir>/factualities/tables/:

  seniority_flows.csv  asked seniority   -> actual seniority (Semantic Scholar)
                       plus the OpenAlex verdict as a separate `source_gt`,
                       because the two disagree by 5x and neither substitutes
                       for the other
  ethnicity_flows.csv  prompt country    -> perceived ethnicity
  field_flows.csv      prompt field      -> ground-truth field (same level both
                       sides, which is what makes the diagonal readable)
  subfield_flows.csv   prompt subfield   -> ground-truth field; the ground truth
                       has no subfield (Semantic Scholar only carries `Field`),
                       so this one is deliberately cross-level
  language_flows.csv   prompt language   -> author country
  gender_flows.csv     prompt country    -> ground-truth gender of the author

Every flow counts *factual* authors only, under the project's own definition
(``author_status == 'found'`` OR an ``oa_id``), matching what
factuality_author reports.

Coverage differs per flow and is logged, because a Sankey silently hides what
it cannot place: a row whose target is missing contributes to no ribbon. The
percentages belong in the figure captions.

Usage (from code/, with PYTHONPATH=.):
  python scripts/metrics/build_sankey_flows.py
  python scripts/metrics/build_sankey_flows.py --flows seniority ethnicity
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from libs.metrics.constants import (
    FIELD_NORM_MAP,
    LANGUAGE_NORM_MAP,
    LLM_COUNTRY_TO_ISO,
    SUBFIELD_NORM_MAP,
    VALID_FLAGS,
)
from libs.utils.config import get_results_path
from libs.utils.logging import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging()

FLOWS = ("seniority", "ethnicity", "field", "subfield", "language", "gender")

# One read serves every flow, so the column set is the union of what they need.
NEEDED = [
    "valid_flag",
    "author_status",
    "oa_id",
    "location",
    "language",
    "field",
    "subfield",
    "model",
    "seniority_llm_bucket",
    "seniority_bucket",
    "gt_field",
    "gt_gender",
    # Written by resolve_homonyms.py: how the name tie was settled
    # (unique | asked_field | file_order). Kept as a slicing dimension so a
    # figure can restrict itself to names that were never ambiguous.
    "tiebreak",
    "oa_country_code",
    "perceived_ethnicity",
]

SPECS = {
    # (source column, target column, dimensions kept for slicing)
    "seniority": ("seniority_llm_bucket", "seniority_bucket", ["field_en", "language_en", "model", "tiebreak"]),
    "ethnicity": ("src_iso", "perceived_ethnicity", ["field_en", "language_en", "model"]),
    # Same level on both sides — the prompt field against the field the author
    # actually works in. The older subfield -> gt_field version mixed two levels
    # of the taxonomy and is kept separately as `subfield`.
    "field": ("field_en", "gt_field", ["subfield_en", "language_en", "model", "tiebreak"]),
    "subfield": ("subfield_en", "gt_field", ["field_en", "language_en", "model", "tiebreak"]),
    "language": ("language_en", "oa_country_code", ["field_en", "model"]),
    # gt_gender only exists for authors matched in Semantic Scholar, so this
    # flow's coverage is lower than the others' — the log line says by how much.
    "gender": ("src_iso", "gt_gender", ["field_en", "language_en", "model"]),
}


def _factual(chunk: pd.DataFrame) -> pd.DataFrame:
    """Valid responses whose author was matched in either source."""
    valid = chunk[chunk["valid_flag"].isin(VALID_FLAGS)]
    return valid[
        (valid["author_status"] == "found")
        | (valid["oa_id"].notna() & (valid["oa_id"].astype(str) != ""))
    ].copy()


def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised labels, so the figures read the same as every other plot."""
    df["field_en"] = df["field"].map(FIELD_NORM_MAP).fillna(df["field"])
    df["subfield_en"] = df["subfield"].map(SUBFIELD_NORM_MAP).fillna(df["subfield"])
    df["language_en"] = df["language"].map(LANGUAGE_NORM_MAP).fillna(df["language"])
    df["src_iso"] = df["location"].map(LLM_COUNTRY_TO_ISO)
    return df


def build(fact_path: Path, wanted: list[str], chunksize: int = 2_000_000) -> dict[str, pd.DataFrame]:
    parts: dict[str, list] = {f: [] for f in wanted}
    seen = {f: [0, 0] for f in wanted}  # [factual rows, rows placeable in a flow]

    header = pd.read_csv(fact_path, nrows=0).columns
    usecols = [c for c in NEEDED if c in header]
    if "tiebreak" not in usecols:
        logger.warning(
            "%s has no `tiebreak` column — run resolve_homonyms.py first if you "
            "want the homonym-resolved flows", fact_path.name
        )
    for i, chunk in enumerate(
        pd.read_csv(fact_path, usecols=usecols, chunksize=chunksize, low_memory=False)
    ):
        if "tiebreak" not in chunk.columns:
            chunk["tiebreak"] = "unknown"
        factual = _annotate(_factual(chunk))
        for flow in wanted:
            src, dst, dims = SPECS[flow]
            seen[flow][0] += len(factual)
            sub = factual.dropna(subset=[src, dst])
            seen[flow][1] += len(sub)
            if sub.empty:
                continue
            group = [src, dst] + dims
            parts[flow].append(sub.groupby(group, dropna=False).size().rename("n"))
        logger.info("chunk %d: %d factual rows", i, len(factual))

    out = {}
    for flow in wanted:
        src, dst, dims = SPECS[flow]
        if not parts[flow]:
            logger.warning("%s: no rows survived — skipping", flow)
            continue
        df = (
            pd.concat(parts[flow])
            .groupby(level=[src, dst] + dims)
            .sum()
            .reset_index()
            .rename(columns={src: "source", dst: "target"})
            .sort_values("n", ascending=False)
        )
        total, placed = seen[flow]
        logger.info(
            "%-10s %d flows | %d of %d factual rows placeable (%.1f%%) | "
            "%d sources, %d targets",
            flow,
            len(df),
            placed,
            total,
            100 * placed / total if total else 0,
            df["source"].nunique(),
            df["target"].nunique(),
        )
        out[flow] = df
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--flows", nargs="*", choices=FLOWS, default=list(FLOWS))
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument(
        "--input",
        default=None,
        help="File under <results>/summary/ to read. Defaults to the output of "
             "resolve_homonyms.py, which carries the `tiebreak` column; pass "
             "factuality_full.csv to rebuild from the unresolved table.",
    )
    args = parser.parse_args()

    results = get_results_path(path=args.results)
    if args.input:
        fact_path = results / "summary" / args.input
    else:
        fact_path = results / "summary" / "factuality_full_resolved.csv"
        if not fact_path.exists():
            fact_path = results / "summary" / "factuality_full.csv"
    if not fact_path.exists():
        parser.error(f"Not found: {fact_path}")
    out_dir = results / "factualities" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", fact_path)
    for flow, df in build(fact_path, list(args.flows), args.chunksize).items():
        fn = out_dir / f"{flow}_flows.csv"
        df.to_csv(fn, index=False)
        logger.info("Wrote %s (%d rows)", fn, len(df))


if __name__ == "__main__":
    main()
