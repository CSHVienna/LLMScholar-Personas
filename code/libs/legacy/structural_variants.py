"""
structural_variants.py — Structural metrics (connectedness, similarity) per
population variant, written by scripts/legacy/run_similarity.py. Used by the
pipeline before the factuality cascade; the current pipeline computes them in
scripts/pipeline/5_socials.py.
"""

import logging
from pathlib import Path

import pandas as pd

from libs.metrics import constants


STRUCTURAL_VARIANTS = ("oa", "oa_ss", "oa_inter")


def load_structural_variant(
    results_dir: Path | str,
    variant: str,
    *,
    factuality_path: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Structural metrics (Eqs. 6-8) for one variant, keyed by the call keys.

    Written by scripts/legacy/run_similarity.py, one parquet per variant:

    ``oa``        authors matched in either source (305,893), OpenAlex vector.
    ``oa_ss``     same population plus the Semantic Scholar feature block.
    ``oa_inter``  only authors matched in BOTH sources (137,013).

    The frame is indexed by the ten call keys, so it can be joined onto any
    per-call table. Parquets written before the runner stored those keys carry
    only ``_cid`` — a surrogate from ``groupby().ngroup()`` that means nothing
    outside the run that made it — and are repaired here from
    ``factuality_path``, including the model rename build_valid_calls applies
    (``:`` -> ``-``, ``.`` -> ``_``); without that rename the join silently
    matches nothing.
    """
    log = logger or logging.getLogger(__name__)
    if variant not in STRUCTURAL_VARIANTS:
        raise ValueError(
            f"variant must be one of {STRUCTURAL_VARIANTS}, got {variant!r}"
        )
    results_dir = Path(results_dir)
    fn = results_dir / "factualities" / "tables" / f"structural_metrics_{variant}.parquet"
    if not fn.exists():
        raise FileNotFoundError(
            f"{fn} not found. Build it with "
            f"`python scripts/legacy/run_similarity.py --variants {variant}`."
        )

    metrics = pd.read_parquet(fn)
    if not set(constants.CALL_KEYS).issubset(metrics.columns):
        fact = (
            Path(factuality_path)
            if factuality_path
            else results_dir / "summary" / "factuality_full.csv"
        )
        log.info("%s predates the call keys — rebuilding the mapping from %s", fn.name, fact)
        df = pd.read_csv(
            fact, usecols=constants.CALL_KEYS + ["valid_flag"], low_memory=False
        )
        df = df[df["valid_flag"].isin(constants.VALID_FLAGS)]
        df["_cid"] = df.groupby(constants.CALL_KEYS, dropna=False).ngroup()
        keys = df.drop_duplicates("_cid").set_index("_cid")[constants.CALL_KEYS].copy()
        keys["model"] = (
            keys["model"].astype(str).str.replace(":", "-", regex=False)
            .str.replace(".", "_", regex=False)
        )
        metrics = keys.join(metrics, how="right")

    metrics = metrics.reset_index(drop=True)
    cols = [c for c in metrics.columns if c not in constants.CALL_KEYS]
    log.info(
        "Structural variant %s: %d rows, similarity defined for %d",
        variant,
        len(metrics),
        int(metrics["similarity"].notna().sum()),
    )
    return metrics[constants.CALL_KEYS + cols]


def apply_structural_variant(
    df_valid_calls: pd.DataFrame,
    results_dir: Path | str,
    variant: str,
    *,
    factuality_path: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Swap the structural columns of a per-call table for another variant's.

    Returns a copy with the same rows, in the same order, and the same columns
    — only the structural values change, so every plot downstream picks up the
    variant without knowing about it. A row with no counterpart in the variant
    gets NaN, which is the honest answer: under a narrower population that
    response may have too few evaluable authors for Eq. 8 to have a value.
    """
    log = logger or logging.getLogger(__name__)
    metrics = load_structural_variant(
        results_dir, variant, factuality_path=factuality_path, logger=log
    )
    structural = [c for c in metrics.columns if c not in constants.CALL_KEYS]

    out = df_valid_calls.copy()
    left = out[constants.CALL_KEYS].astype(str)
    right = metrics.copy()
    for c in constants.CALL_KEYS:
        right[c] = right[c].astype(str)

    joined = left.merge(right, on=constants.CALL_KEYS, how="left")
    if len(joined) != len(out):
        raise ValueError(
            f"join changed the row count ({len(out)} -> {len(joined)}); "
            "the variant parquet has duplicated call keys."
        )
    for col in structural:
        if col in out.columns:
            out[col] = joined[col].values
    defined = joined["similarity"].notna().sum()
    log.info(
        "Applied structural variant %s: similarity defined for %d of %d rows",
        variant,
        int(defined),
        len(out),
    )
    return out
