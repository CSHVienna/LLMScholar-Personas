"""
merge_structural_metrics.py — Write recomputed structural metrics (paper
Eqs. 6-8) into valid_requests_metadata.csv, in place of the stale ones.

The analysis notebooks read valid_requests_metadata.csv, so a fresh
structural_metrics_*.parquet from run_similarity.py changes nothing until its
columns land there. This does only that: it replaces the six structural columns
and leaves every other metric untouched, which is much cheaper than re-running
build_valid_calls.py just to propagate them.

Safety, in order:

  1. the CSV is copied to a .bak_<suffix> sibling before anything is written;
  2. the join is on the ten call keys, and must be 1:1 — a many-to-one join
     would duplicate rows silently;
  3. it refuses to run if any row that loses its value had one to begin with,
     so an incomplete parquet cannot blank out published numbers;
  4. row count and column order are asserted to be unchanged at the end.

Note on the join key: `_cid` in the parquet is a surrogate from
groupby().ngroup(), meaningful only inside the run that made it, so the call
keys are what bridge the two files. Parquets written before run_similarity.py
carried those keys can be repaired with --factuality (it rebuilds the mapping
from factuality_full.csv, applying the same model rename build_valid_calls does).

Usage (from code/, with PYTHONPATH=.):
  python scripts/legacy/merge_structural_metrics.py
  python scripts/legacy/merge_structural_metrics.py --variant oa_ss --dry_run
"""

import argparse
import shutil
import warnings
from pathlib import Path

import pandas as pd

from libs.metrics.constants import CALL_KEYS, VALID_FLAGS
from libs.utils.config import get_results_path
from libs.utils.logging import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging()

STRUCTURAL_COLS = [
    "connectedness",
    "similarity",
    "n_excluded_connectedness",
    "n_excluded_similarity",
    "n_used_connectedness",
    "n_used_similarity",
]


def _normalise_model(s: pd.Series) -> pd.Series:
    """The rename _enrich_model_metadata applies before writing the CSV."""
    return s.astype(str).str.replace(":", "-", regex=False).str.replace(".", "_", regex=False)


def add_call_keys(metrics: pd.DataFrame, factuality: Path) -> pd.DataFrame:
    """Rebuild the _cid -> call-keys mapping for a parquet that lacks it."""
    logger.info("Rebuilding the _cid mapping from %s", factuality)
    df = pd.read_csv(factuality, usecols=CALL_KEYS + ["valid_flag"], low_memory=False)
    df = df[df["valid_flag"].isin(VALID_FLAGS)]
    df["_cid"] = df.groupby(CALL_KEYS, dropna=False).ngroup()
    keys = df.drop_duplicates("_cid").set_index("_cid")[CALL_KEYS].copy()
    keys["model"] = _normalise_model(keys["model"])
    return keys.join(metrics, how="right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--variant", default="oa", help="which structural_metrics_<variant>.parquet")
    parser.add_argument("--suffix", default=None, help="backup suffix (default: bak_<variant>)")
    parser.add_argument("--factuality", default=None, help="factuality_full.csv, to repair a parquet without call keys")
    parser.add_argument("--dry_run", action="store_true", help="report the effect and write nothing")
    args = parser.parse_args()

    results = get_results_path(path=args.results)
    target = results / "factualities" / "tables" / "valid_requests_metadata.csv"
    src = results / "factualities" / "tables" / f"structural_metrics_{args.variant}.parquet"
    for p in (target, src):
        if not p.exists():
            parser.error(f"Not found: {p}")

    metrics = pd.read_parquet(src)
    if not set(CALL_KEYS).issubset(metrics.columns):
        fact = Path(args.factuality) if args.factuality else results / "summary" / "factuality_full.csv"
        metrics = add_call_keys(metrics, fact)
    metrics = metrics.reset_index(drop=True)[CALL_KEYS + STRUCTURAL_COLS]

    logger.info("Reading %s", target)
    df = pd.read_csv(target, low_memory=False)
    n_before, cols_before = len(df), list(df.columns)
    logger.info("Rows: %d   structural columns present: %s", n_before,
                [c for c in STRUCTURAL_COLS if c in df.columns])

    for frame in (df, metrics):
        for c in CALL_KEYS:
            frame[c] = frame[c].astype(str)

    if metrics.duplicated(CALL_KEYS).any():
        parser.error(
            f"{metrics.duplicated(CALL_KEYS).sum()} duplicated call keys in the parquet: "
            "the join would multiply rows."
        )

    merged = df.merge(metrics, on=CALL_KEYS, how="left", suffixes=("", "_new"), indicator=True)
    matched = merged["_merge"] == "both"
    logger.info("Matched %d of %d rows", int(matched.sum()), n_before)

    # A row that had a value and finds no replacement would be silently blanked.
    for col in STRUCTURAL_COLS:
        if col not in df.columns:
            continue
        lost = (~matched) & merged[col].notna()
        if lost.any():
            parser.error(
                f"{int(lost.sum())} rows have a {col} value but no row in the parquet. "
                "Refusing to blank published numbers — check that the parquet covers "
                "the same responses."
            )

    for col in STRUCTURAL_COLS:
        old = merged[col] if col in df.columns else pd.Series(index=merged.index, dtype=float)
        new = merged[f"{col}_new"]
        if col in ("similarity", "connectedness"):
            both = old.notna() & new.notna()
            if both.any():
                delta = (new[both] - old[both]).abs()
                logger.info(
                    "  %-14s mean %.4f -> %.4f   |delta| mean %.4f, >0.05 in %.1f%%",
                    col, old[both].mean(), new[both].mean(),
                    delta.mean(), 100 * (delta > 0.05).mean(),
                )
        merged[col] = new

    out = merged[cols_before]
    assert len(out) == n_before, f"row count changed: {n_before} -> {len(out)}"
    assert list(out.columns) == cols_before, "column order changed"

    if args.dry_run:
        logger.info("Dry run: nothing written.")
        return

    backup = target.with_suffix(f".csv.{args.suffix or f'bak_{args.variant}'}")
    if backup.exists():
        parser.error(f"Backup already exists, refusing to overwrite it: {backup}")
    logger.info("Backing up -> %s", backup)
    shutil.copy2(target, backup)

    out.to_csv(target, index=False)
    logger.info("Wrote %s (%d rows, %d columns)", target, len(out), len(out.columns))


if __name__ == "__main__":
    main()
