"""
run_similarity.py — Recompute the structural metrics (paper Eqs. 6-8) on their
own, without rebuilding the whole per-call metrics table.

Calls build_valid_calls._compute_structural_metrics directly, so this is the
real code path and not a reimplementation, but writes to its own outputs and
leaves valid_requests_metadata.csv alone.

Runs three variants so each choice is measurable rather than assumed:

  oa        union population (305,893 authors), OpenAlex vector  (8 columns)
  oa_ss     union population, plus the Semantic Scholar block    (8 + 4 columns)
  oa_inter  only authors matched in BOTH sources (137,013), OpenAlex vector

Each variant is checkpointed to its own parquet and skipped when that file
exists, and each gets its own cache directory so the PCA fits cannot overwrite
each other. Every variant also exports its fitted model to HDF5 next to the
joblib, under its own name, so a model can be inspected or reused without the
pickle.

The coauthorship graph is shared. It depends on the author population and not
on the feature vector, so the two union variants reuse one copy; oa_inter needs
a subset of the same nodes, and build_coauthorship_graph slices the cached
matrix rather than rescanning oa.works — an induced subgraph is exact, because
coauthorship between two authors does not depend on who else is in the matrix.

Usage (from code/, with PYTHONPATH=.):
  python scripts/legacy/run_similarity.py
  python scripts/legacy/run_similarity.py --variants oa
"""

import argparse
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from libs.metrics.constants import (
    CALL_KEYS,
    SIMILARITY_FEATURE_COLS_FULL,
    SIMILARITY_FEATURE_COLS_WITH_SS,
    VALID_FLAGS,
)
from libs.utils.config import config_default, get_results_path
from libs.utils.logging import setup_logging
from libs.metrics.similarity import save_similarity_model_h5
from scripts.legacy.build_valid_calls import _compute_structural_metrics

warnings.filterwarnings("ignore")
logger = setup_logging()

# Each variant is a (population, feature-vector) pair. They share every
# expensive artefact: the scholarly stats cover the wider population, and the
# cached coauthorship graph is sliced for the narrower one instead of rebuilt.
VARIANTS = {
    # matched in either source, OpenAlex feature vector — what the paper uses
    "oa": {"population": "union", "with_ss_block": False},
    # same population, plus the Semantic Scholar feature block
    "oa_ss": {"population": "union", "with_ss_block": True},
    # only authors matched in BOTH sources, OpenAlex feature vector
    "oa_inter": {"population": "intersection", "with_ss_block": False},
}

# Everything _compute_structural_metrics and build_author_features touch.
NEEDED = CALL_KEYS + [
    "valid_flag",
    "author_status",
    "oa_id",
    "researcher_id",
    "oa_works_count",
    "oa_cited_by_count",
    "oa_career_age",
    "oa_first_pub_year",
    "oa_last_pub_year",
]


def load_valid_recommendations(fact_path: Path) -> pd.DataFrame:
    """The same frame build_valid_calls.main() assembles, minus what these two
    metrics never read."""
    logger.info("Reading %s", fact_path)
    df = pd.read_csv(fact_path, usecols=NEEDED, low_memory=False)
    logger.info("Recommendations: %d", len(df))

    # author_found is a UNION: an author matched only in OpenAlex still counts.
    df["author_found"] = (df["author_status"] == "found") | (
        df["oa_id"].notna() & (df["oa_id"].astype(str) != "")
    )
    df = df[df["valid_flag"].isin(VALID_FLAGS)].copy()
    df["_cid"] = df.groupby(CALL_KEYS, dropna=False).ngroup()
    df["author_id"] = df.oa_id.apply(
        lambda v: (str(int(v)) if isinstance(v, float) else str(v)).split("/")[-1].strip()
        if pd.notna(v) and v != ""
        else None
    )
    logger.info(
        "Valid recommendations: %d over %d responses, %d distinct authors",
        len(df),
        df["_cid"].nunique(),
        df["author_id"].nunique(),
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--oa_duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--oa_author_stats", default=config_default("oa_author_stats"))
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--variants", nargs="*", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    results = get_results_path(path=args.results)
    fact_path = results / "summary" / "factuality_full.csv"
    out_dir = results / "factualities" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    shared_graph = results / ".cache" / "coauthorship_graph.joblib"

    todo = [
        v for v in args.variants
        if args.rebuild or not (out_dir / f"structural_metrics_{v}.parquet").exists()
    ]
    if not todo:
        logger.info("Nothing to do: every requested variant already has its parquet.")
        return

    df = load_valid_recommendations(fact_path)

    for variant in todo:
        out_fn = out_dir / f"structural_metrics_{variant}.parquet"
        cache_dir = results / ".cache" / f"similarity_{variant}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Share the graph rather than rebuilding it per variant; its cached
        # population_key guards against reusing one built for other authors.
        target = cache_dir / "coauthorship_graph.joblib"
        if shared_graph.exists() and not target.exists():
            shutil.copy2(shared_graph, target)
            logger.info("Reusing coauthorship graph -> %s", target)

        t0 = time.time()
        logger.info("=== variant %s ===", variant)
        spec = VARIANTS[variant]
        out = _compute_structural_metrics(
            df,
            cache_dir=cache_dir,
            oa_duckdb_path=args.oa_duckdb,
            oa_author_stats_dir=args.oa_author_stats,
            ss_parquet=args.parquet,
            with_ss_block=spec["with_ss_block"],
            population=spec["population"],
            rebuild=args.rebuild,
        )
        # _cid is a surrogate from groupby().ngroup(), so it only means anything
        # within the run that produced it. Carry the call keys so the file can be
        # joined to anything later without re-deriving the mapping — including
        # the model rename, which is easy to forget and joins to nothing.
        keys = df.drop_duplicates("_cid").set_index("_cid")[CALL_KEYS].copy()
        keys["model"] = (
            keys["model"].astype(str).str.replace(":", "-", regex=False)
            .str.replace(".", "_", regex=False)
        )
        out = keys.join(out, how="right")
        out.to_parquet(out_fn)
        logger.info("Wrote %s (%d rows, %.0fs)", out_fn, len(out), time.time() - t0)

        # HDF5 alongside the joblib: plain arrays, so it survives a scikit-learn
        # upgrade and can be read outside Python. One file per variant.
        cols = (
            SIMILARITY_FEATURE_COLS_WITH_SS
            if spec["with_ss_block"]
            else SIMILARITY_FEATURE_COLS_FULL
        )
        save_similarity_model_h5(
            cache_dir / "similarity_embeddings.joblib",
            cache_dir / f"pca_similarity_{variant}.h5",
            feature_cols=cols,
            logger=logger,
        )

        for col in out.columns:
            if out[col].dtype.kind in "fi":
                s = out[col].dropna()
                if len(s):
                    logger.info(
                        "  %-28s n=%-7d mean=%.4f median=%.4f",
                        col, len(s), s.mean(), s.median(),
                    )

    # Compare every variant against the baseline. A variant that comes out
    # identical did not actually change the fit — which has happened, and is
    # invisible in the parquet itself.
    base_fn = out_dir / "structural_metrics_oa.parquet"
    if not base_fn.exists():
        return
    base = pd.read_parquet(base_fn)
    for variant in VARIANTS:
        if variant == "oa":
            continue
        fn = out_dir / f"structural_metrics_{variant}.parquet"
        if not fn.exists():
            continue
        other = pd.read_parquet(fn)
        common = base.index.intersection(other.index)
        a, b = base.loc[common, "similarity"], other.loc[common, "similarity"]
        both = a.notna() & b.notna()
        if not both.any():
            logger.info("%-9s no response has similarity in both variants", variant)
            continue
        logger.info(
            "%-9s vs oa: mean %.4f -> %.4f, |delta| mean %.4f, r %.5f, "
            "over %d shared responses (%d vs %d defined)",
            variant,
            a[both].mean(), b[both].mean(),
            (b[both] - a[both]).abs().mean(),
            np.corrcoef(a[both], b[both])[0, 1],
            int(both.sum()),
            int(a.notna().sum()), int(b.notna().sum()),
        )


if __name__ == "__main__":
    main()
