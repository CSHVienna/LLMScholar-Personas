"""
build_valid_calls.py — Pre-compute the per-call metrics table consumed by the
analysis notebooks.

Ports the preprocessing stage of notebooks/analysis/metrics_pipeline.ipynb so
the notebook can stay plotting-only. Reads factuality_full.csv plus ethnicity
ground-truth CSVs and the Semantic Scholar parquet, derives every per-call
metric (factuality, diversity, parity, consistency, duplicates, popularity,
connectedness, similarity), and writes a single CSV that every plotting
notebook then loads.

Output: <results_dir>/factualities/tables/valid_requests_metadata.csv

The two structural metrics (paper Eqs. 6-8) need two things prepared first:

  1. the OpenAlex DuckDB snapshot, for the coauthorship graph;
  2. the per-author scholarly stats, from
     scripts/metrics/extract_oa_author_stats.py — run that BEFORE this script
     or the similarity features cannot be built.

Graph and PCA pipeline are cached under <results_dir>/.cache, so the pass over
oa.works happens once; pass --rebuild_structural to force it again. Without
--oa_duckdb the two metrics are skipped and every other metric is unaffected.

To recompute only the structural metrics, without rewriting the 425 MB
per-call table, use scripts/metrics/run_similarity.py instead — it calls the
same code path and writes its own parquet per feature-vector variant.

Usage (from code/, with PYTHONPATH=.):
  python scripts/metrics/extract_oa_author_stats.py      # once, ~95 min
  python scripts/metrics/build_valid_calls.py
  python scripts/metrics/build_valid_calls.py --results <results_dir> --parquet <path_to_ss_parquet>
  python scripts/metrics/build_valid_calls.py --with_ss_block
  python scripts/metrics/build_valid_calls.py --rebuild_structural

Defaults come from [data] in config.ini.
"""

import argparse
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent

from libs.metrics.aggregators import (
    assign_productivity_tier,
    compute_connectedness,
    compute_similarity,
    productivity_thresholds,
)
from libs.metrics.io import (
    build_author_features,
    load_author_scholarly_stats,
    load_ss_author_features,
    build_coauthorship_graph,
    build_similarity_embeddings,
)
from libs.metrics.constants import (
    SIMILARITY_SS_COLS,
    CALL_KEYS,
    CONNECTEDNESS_METRIC,
    ETHNICITY_ORDER,
    SIMILARITY_METRIC,
    STRUCTURAL_EXCLUSION_COLS,
    STRUCTURAL_USED_COLS,
    FIELD_NORM_MAP,
    GENDER_ORDER,
    LANGUAGE_NORM_MAP,
    LOCATION_NORM_MAP,
    MODEL_ARCHITECTURE_COLS,
    MODEL_EXTRA_COLS,
    NEEDED_COLS,
    PROMPT_KEYS,
    PRODUCTIVITY_OA_FIELDS_MAP,
    PRODUCTIVITY_TIER_LABELS,
    REFUSED_FLAG,
    ROLE_NORM_MAP,
    SUBFIELD_NORM_MAP,
    TARGET_NORM_MAP,
    TASK_NORM_MAP,
    VALID_FLAGS,
)
from libs.utils import ios
from libs.utils.config import config_default, get_results_path
from libs.utils.logging import setup_logging
from libs.visuals.constants import ETHNICITY_MAP, GENDER_MAP

warnings.filterwarnings("ignore")
logger = setup_logging()


def author_uid(df: pd.DataFrame) -> pd.Series:
    """A per-author key that never merges two different people.

    ``author_id`` is the OpenAlex id, and it is None for the 37% of
    recommendations OpenAlex does not resolve. Dropping duplicates on it alone
    folds every unresolved author of one response into a single row, which the
    counts then read as the model having repeated itself: measured that way
    `duplicates` is 0.159, against 0.002 when distinct people stay distinct.

    Falls back the way libs/metrics/io.py already does for the structural
    metrics — ``SS:<researcher_id>`` when Semantic Scholar matched the author —
    and finally to the recommended name itself, so two hallucinated authors in
    one response are two authors, while the same name twice is one.
    """
    name = (
        df["name"].fillna("").astype(str).str.strip()
        + " "
        + df["lastname"].fillna("").astype(str).str.strip()
    ).str.lower()
    return np.where(
        df["author_id"].notna(),
        df["author_id"].astype(str),
        np.where(
            df["researcher_id"].notna(),
            "SS:" + df["researcher_id"].astype(str),
            "NM:" + name.str.strip(),
        ),
    )


def match_rate_per_call(
    eligible: pd.DataFrame,
    status_col: str,
    match_value: str,
    index: pd.Index,
) -> pd.Series:
    """Per-call share of evaluable recommendations that matched.

    `eligible` holds only the rows a metric can be scored on (status in
    {match, mismatch}); `index` is every call, so calls absent from it get NaN.

    A call whose evaluable recommendations all missed scores **0.0, not NaN** —
    counting the matches alone and dividing leaves those calls out of the
    groupby entirely, which silently restricts the metric to calls that got at
    least one hit and inflates it (it put bias_location for Ecuador at 0.454
    against a true 0.040). Only a call with nothing evaluable is NaN.
    """
    evaluated = eligible.groupby("_cid").size().reindex(index)
    matched = (
        eligible[eligible[status_col] == match_value]
        .groupby("_cid")
        .size()
        .reindex(index)
        .fillna(0)
    )
    return matched / evaluated.replace(0, np.nan)


def _load_recommendations(fact_path: Path, cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / f"df_{ios.file_hash(fact_path)}.pkl"

    def _compute():
        logger.info("Loading %s (selected columns only)", fact_path.name)
        return pd.read_csv(fact_path, usecols=NEEDED_COLS, low_memory=False)

    df = ios.cached_pickle(cache, _compute, logger=logger)
    # Cache may pre-date column additions — invalidate if columns are missing.
    missing = [c for c in NEEDED_COLS if c not in df.columns]
    if missing:
        logger.info("Cached df missing %s — reloading from CSV", missing)
        df = _compute()
        ios.save_pickle(df, cache, logger=logger)
    logger.info("Total recommendations: %d", len(df))
    logger.info("Unique calls: %d", df[CALL_KEYS].drop_duplicates().shape[0])
    return df


def _load_ground_truth(
    ss_parquet: Path, eth_glob: str, cache_dir: Path
) -> pd.DataFrame:
    gt_files = sorted(glob.glob(eth_glob))
    if not gt_files:
        raise FileNotFoundError(f"No ground-truth files matched: {eth_glob}")
    cache = cache_dir / f"gt_{ios.file_hash(ss_parquet, *gt_files)}.pkl"

    def _compute():
        logger.info("Loading ground-truth (%d CSVs + parquet)", len(gt_files))
        parts = [
            pd.read_csv(f, usecols=["Researcher_id", "Year", "perceived_ethnicity"])
            for f in gt_files
        ]
        df_gt = pd.concat(parts, ignore_index=True)
        df_gt = (
            df_gt.sort_values("Year").groupby("Researcher_id").last().reset_index()
        )
        gt_parquet = pd.read_parquet(
            ss_parquet, columns=["Researcher_id", "Combined_gender"]
        )
        df_gt = df_gt.merge(gt_parquet, on="Researcher_id", how="left")
        df_gt["gender_clean"] = (
            df_gt["Combined_gender"]
            .str.strip()
            .str.lower()
            .map({"male": "Male", "female": "Female", "unisex": "Neutral"})
        )
        df_gt["ethnicity_clean"] = df_gt["perceived_ethnicity"].map(ETHNICITY_MAP)
        return df_gt

    return ios.cached_pickle(cache, _compute, logger=logger)


def _enrich_model_metadata(
    df: pd.DataFrame, models_metadata: dict, models_map: dict
) -> pd.DataFrame:
    df.loc[:, "model"] = df.model.apply(
        lambda x: x.replace(":", "-").replace(".", "_")
    )
    for col, key in [
        ("model_size", "size"),
        ("model_access", "access"),
        ("model_class", "reasoning"),
        ("model_family", "family"),
    ]:
        df.loc[:, col] = df["model"].map(
            lambda m: models_metadata[m][key] if m in models_metadata else "Unknown"
        )
    df.loc[:, "model_short_name"] = df["model"].map(
        lambda m: models_map[m] if m in models_map else m
    )
    return df


def _compute_consistency(responses_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Jaccard similarity between consecutive runs (paper Eq. 4)."""
    rows = []
    for keys, g in responses_df.groupby(PROMPT_KEYS, dropna=False):
        g_sorted = g.sort_values("run_id")
        run_sets = [
            set(rg["author_id"].dropna())
            for _, rg in g_sorted.groupby("run_id", sort=True)
        ]
        if len(run_sets) < 2:
            continue
        jaccards = []
        for i in range(1, len(run_sets)):
            a, b = run_sets[i - 1], run_sets[i]
            union = a | b
            jaccards.append(len(a & b) / len(union) if union else np.nan)
        row = dict(zip(PROMPT_KEYS, keys if isinstance(keys, tuple) else (keys,)))
        row["n_runs"] = len(run_sets)
        row["consistency"] = float(np.nanmean(jaccards)) if jaccards else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_structural_metrics(
    df_valid_recommendations: pd.DataFrame,
    *,
    cache_dir: Path,
    oa_duckdb_path: str,
    oa_author_stats_dir: Path | str,
    ss_parquet: Path | str | None = None,
    with_ss_block: bool = False,
    population: str = "union",
    rebuild: bool = False,
) -> pd.DataFrame:
    """Per-response connectedness and scholarly similarity (paper Eqs. 6-8).

    Returns a DataFrame indexed by ``_cid`` with the two metric columns plus one
    exclusion count each, ready to be assigned onto ``df_valid_calls``.

    U-hat_i is the set of unique *factual* authors of the response. Which
    authors count is set by ``population``:

    ``"union"`` (default) follows the definition the rest of the pipeline uses,
    ``author_found`` = matched in Semantic Scholar OR in OpenAlex. Both metrics
    need OpenAlex-side data, so authors matched only in Semantic Scholar are
    excluded and counted; 305,893 authors carry an ``oa_id`` and enter.

    ``"intersection"`` keeps only the 137,013 authors matched in BOTH sources,
    which is what you want if every author must carry evidence from each. The
    cost is that it drops 55% of the authors OpenAlex knows and Semantic
    Scholar does not, so Sim_i is measured over fewer than half the population
    and per-response exclusions rise accordingly. It needs no extra passes: the
    scholarly stats already cover the wider population, and the cached
    coauthorship graph is sliced to the subset instead of rebuilt.

    Authors are de-duplicated on a composite id (``oa_id`` when present, else
    ``SS:<researcher_id>``) so that several authors without an ``oa_id`` within
    one response are not collapsed into one.
    """
    if population not in ("union", "intersection"):
        raise ValueError(
            f"population must be 'union' or 'intersection', got {population!r}"
        )
    factual = df_valid_recommendations[df_valid_recommendations["author_found"]].copy()
    if population == "intersection":
        before = factual["author_id"].nunique()
        factual = factual[factual["author_status"] == "found"]
        logger.info(
            "Structural metrics: population 'intersection' — %d of %d authors "
            "kept (matched in both sources)",
            factual["author_id"].nunique(),
            before,
        )

    # Composite uid: without it, drop_duplicates(['_cid','author_id']) treats
    # every author_id-less author as the same row and the exclusion count would
    # under-report.
    factual["_uid"] = np.where(
        factual["author_id"].notna(),
        factual["author_id"],
        "SS:" + factual["researcher_id"].astype(str),
    )
    factual = factual.drop_duplicates(["_cid", "_uid"])

    author_ids = factual["author_id"].dropna().unique().tolist()
    logger.info(
        "Structural metrics: %d unique factual authors with an OpenAlex id",
        len(author_ids),
    )

    adjacency, graph_index = build_coauthorship_graph(
        author_ids,
        cache_path=cache_dir / "coauthorship_graph.joblib",
        oa_duckdb_path=oa_duckdb_path,
        temp_dir=cache_dir / "duckdb_tmp",
        rebuild=rebuild,
        logger=logger,
    )
    # h_index / i10_index / two_year_mean_citedness come straight from
    # OpenAlex's own summary_stats (extracted from the raw author records, since
    # the DuckDB dump omits the field); only e_index has to be computed, which
    # extract_oa_author_stats.py does in its own checkpointed pass over
    # oa.works. Both land in [data].oa_author_stats.
    stats = load_author_scholarly_stats(oa_author_stats_dir, logger=logger)
    # Second block, over the other corpus, mirroring the reference PCA's
    # OpenAlex + APS concatenation. Off by default so the effect of turning it
    # on is measurable against the OpenAlex-only vector.
    ss = (
        load_ss_author_features(ss_parquet, logger=logger)
        if with_ss_block and ss_parquet
        else None
    )
    features = build_author_features(
        factual.dropna(subset=["author_id"]), stats=stats, ss=ss
    )
    if ss is not None:
        covered = features[SIMILARITY_SS_COLS].notna().any(axis=1).mean()
        logger.info(
            "Similarity features: SS block covers %.1f%% of authors; the rest is "
            "median-imputed there",
            100 * covered,
        )
    embeddings, emb_index, _ = build_similarity_embeddings(
        features,
        cache_path=cache_dir / "similarity_embeddings.joblib",
        index=graph_index,
        rebuild=rebuild,
        logger=logger,
    )

    rows = []
    grouped = factual.groupby("_cid", sort=False)["author_id"]
    for cid, ids in tqdm(grouped, total=grouped.ngroups, desc="structural metrics"):
        with_oa = [a for a in ids if pd.notna(a)]
        # Factual authors with no oa_id have neither a node nor features.
        n_without_oa = len(ids) - len(with_oa)

        conn = compute_connectedness(with_oa, adjacency, graph_index, logger=logger)
        sim = compute_similarity(with_oa, embeddings, emb_index, logger=logger)
        rows.append(
            {
                "_cid": cid,
                CONNECTEDNESS_METRIC: conn.value,
                SIMILARITY_METRIC: sim.value,
                STRUCTURAL_EXCLUSION_COLS[CONNECTEDNESS_METRIC]: conn.n_excluded
                + n_without_oa,
                STRUCTURAL_EXCLUSION_COLS[SIMILARITY_METRIC]: sim.n_excluded
                + n_without_oa,
                # The actual denominator of Eqs. 6-8. n_authors_found cannot play
                # that role: it collapses all oa_id-less authors into one entry.
                STRUCTURAL_USED_COLS[CONNECTEDNESS_METRIC]: conn.n_used,
                STRUCTURAL_USED_COLS[SIMILARITY_METRIC]: sim.n_used,
            }
        )

    out = pd.DataFrame(rows).set_index("_cid")
    logger.info(
        "Structural metrics: connectedness defined for %d/%d responses, "
        "similarity for %d/%d",
        int(out[CONNECTEDNESS_METRIC].notna().sum()),
        len(out),
        int(out[SIMILARITY_METRIC].notna().sum()),
        len(out),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=None,
        help="Results directory (default: [data].results_dir in config.ini).",
    )
    parser.add_argument(
        "--parquet",
        default=config_default("ss_parquet"),
        help="Path to the Semantic Scholar Researchers_Deduplicated parquet "
        "(default: [data].ss_parquet).",
    )
    parser.add_argument(
        "--data",
        default=str(HERE.parent.parent.parent / "data"),
        help="Repo data/ directory (defaults to ../data).",
    )
    parser.add_argument(
        "--oa_duckdb",
        default=config_default("oa_duckdb"),
        help="OpenAlex DuckDB snapshot, used to build the coauthorship graph for "
        "the structural metrics (default: [data].oa_duckdb). Without it, "
        "connectedness and similarity are skipped.",
    )
    parser.add_argument(
        "--oa_author_stats",
        default=config_default("oa_author_stats"),
        help="Directory holding summary_stats.parquet and eindex.parquet, as "
        "written by scripts/metrics/extract_oa_author_stats.py "
        "(default: [data].oa_author_stats). Needed for the similarity features.",
    )
    parser.add_argument(
        "--population",
        choices=("union", "intersection"),
        default="union",
        help="Which authors enter the structural metrics: 'union' (matched in "
        "either source, 305,893) or 'intersection' (matched in both, 137,013).",
    )
    parser.add_argument(
        "--with_ss_block",
        action="store_true",
        help="Append the Semantic Scholar feature block to the similarity PCA "
        "(4 extra columns over the SS corpus, mirroring the reference's aps_* "
        "block). Off by default: it refits the PCA and so moves every Sim_i.",
    )
    parser.add_argument(
        "--rebuild_structural",
        action="store_true",
        help="Ignore the cached coauthorship graph / PCA pipeline and rebuild them.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="File under <results>/summary/ to read. Defaults to "
             "factuality_full_resolved.csv (the output of resolve_homonyms.py, "
             "whose gt_* columns no longer depend on the order of the reference "
             "parquet) when it exists, else factuality_full.csv.",
    )
    args = parser.parse_args()

    results = Path(args.results) if args.results else get_results_path()
    data = Path(args.data)

    if args.input:
        fact_path = results / "summary" / args.input
    else:
        fact_path = results / "summary" / "factuality_full_resolved.csv"
        if not fact_path.exists():
            fact_path = results / "summary" / "factuality_full.csv"
            logger.warning(
                "factuality_full_resolved.csv not found — falling back to %s, "
                "whose gt_field/gt_career_age still carry the file-order "
                "tie-break (run scripts/factuality/resolve_homonyms.py)",
                fact_path.name,
            )
    logger.info("Reading %s", fact_path)
    eth_gt_glob = str(
        results / "ethnicity" / "DataFrameRankings_Genderize_Namsor_*_with_ethnicity.csv"
    )
    valid_calls_path = results / "factualities" / "tables" / "valid_requests_metadata.csv"
    valid_calls_path.parent.mkdir(parents=True, exist_ok=True)

    models_metadata_fn = data / "models" / "metadata.json"
    models_fn = data / "models" / "models.txt"

    cache_dir = results / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.parquet is None:
        parser.error(
            "Missing --parquet and no [data].ss_parquet in config.ini."
        )

    # ── 1. Load data ─────────────────────────────────────────────────────────
    df_all_recommendations = _load_recommendations(fact_path, cache_dir)
    df_gt = _load_ground_truth(Path(args.parquet), eth_gt_glob, cache_dir)

    models = ios.read_list_from_file(models_fn)
    models_map = {
        m: m.replace("-2025-04-14", "")
        .replace("b-maverick-128e-instruct-q4_K_M", "B-maverick")
        .replace("b-scout-16e-instruct-q4_K_M", "B-scout")
        .split("b-")[0]
        .lower()
        for m in models
    }
    models_metadata = ios.load_json(models_metadata_fn)
    df_all_recommendations = _enrich_model_metadata(
        df_all_recommendations, models_metadata, models_map
    )

    # ── 2. Author factuality (SS OR OA) ──────────────────────────────────────
    df_all_recommendations["author_found"] = (
        (df_all_recommendations["author_status"] == "found")
        | df_all_recommendations["oa_id"].notna()
        & (df_all_recommendations["oa_id"].astype(str) != "")
    )

    # ── 3. Per-call (all responses) for validity / refusals ──────────────────
    df_all_calls = (
        df_all_recommendations.groupby(
            CALL_KEYS + MODEL_ARCHITECTURE_COLS + MODEL_EXTRA_COLS, dropna=False
        )[["valid_flag"]]
        .first()
        .reset_index()
    )
    df_all_calls["validity"] = df_all_calls["valid_flag"].isin(VALID_FLAGS).astype(int)
    df_all_calls["refusals"] = (df_all_calls["valid_flag"] == REFUSED_FLAG).astype(int)

    # ── 4. Valid recommendations and per-call surrogate id ───────────────────
    df_valid_recommendations = df_all_recommendations[
        df_all_recommendations["valid_flag"].isin(VALID_FLAGS)
    ].copy()
    df_valid_recommendations["_cid"] = df_valid_recommendations.groupby(
        CALL_KEYS, dropna=False
    ).ngroup()
    df_valid_recommendations["ethnicity_clean"] = df_valid_recommendations[
        "perceived_ethnicity"
    ].map(ETHNICITY_MAP)
    df_valid_recommendations["gender_clean"] = (
        df_valid_recommendations["gt_gender"]
        .str.strip()
        .str.lower()
        .map(GENDER_MAP)
    )
    df_valid_recommendations["author_id"] = df_valid_recommendations.oa_id.apply(
        # oa_id may arrive as a URL ("https://openalex.org/A123"), a bare numeric
        # id (5003003822.0, read as float when the column has NaNs), or a plain
        # string. Normalise all three to the trailing id token.
        lambda v: (str(int(v)) if isinstance(v, float) else str(v)).split("/")[-1].strip()
        if pd.notna(v) and v != ""
        else None
    )

    # ── 5. Duplicates and author factuality ──────────────────────────────────
    n_total_recommendations = (
        df_valid_recommendations.groupby("_cid").size().rename("n_total_recommendations")
    )
    df_valid_recommendations["author_uid"] = author_uid(df_valid_recommendations)
    uniq = df_valid_recommendations.drop_duplicates(["_cid", "author_uid"])
    n_unique_authors = uniq.groupby("_cid").size().rename("n_unique_authors")
    df_authors_found = uniq[uniq["author_found"]]
    n_authors_found = df_authors_found.groupby("_cid").size().rename("n_authors_found")

    df_valid_recommendations_cid = (
        df_valid_recommendations[
            CALL_KEYS + ["_cid"] + MODEL_ARCHITECTURE_COLS + MODEL_EXTRA_COLS
        ]
        .drop_duplicates("_cid")
        .set_index("_cid")
    )

    df_valid_calls = (
        df_valid_recommendations_cid.join(n_total_recommendations)
        .join(n_unique_authors, how="left")
        .join(n_authors_found, how="left")
        .fillna({"n_unique_authors": 0, "n_authors_found": 0})
        .astype({"n_unique_authors": int, "n_authors_found": int})
    )
    df_valid_calls["duplicates"] = df_valid_calls.apply(
        lambda r: 1 - r["n_unique_authors"] / r["n_total_recommendations"], axis=1
    )
    df_valid_calls["factuality_author"] = np.where(
        df_valid_calls["n_unique_authors"] > 0,
        df_valid_calls["n_authors_found"] / df_valid_calls["n_unique_authors"],
        np.nan,
    )

    # ── 6. Field / seniority factuality + location bias ──────────────────────
    # Same match-rate computation for the three; location is named a bias and
    # grouped as social representation (issue #36), not as a factuality.
    for status_col, metric_col in [
        ("field_status", "factuality_field"),
        ("seniority_status", "factuality_seniority"),
        ("location_status", "bias_location"),
    ]:
        prefix = status_col.split("_")[0]
        eligible = df_authors_found[
            df_authors_found[status_col].isin([f"{prefix}_match", f"{prefix}_mismatch"])
        ]
        df_valid_calls[metric_col] = match_rate_per_call(
            eligible, status_col, f"{prefix}_match", df_valid_calls.index
        )

    # ── 7. Diversity and parity ──────────────────────────────────────────────
    gt_eth_frac = (
        df_gt["ethnicity_clean"]
        .dropna()
        .value_counts(normalize=True)
        .reindex(ETHNICITY_ORDER, fill_value=0)
    )
    gt_gen_frac = (
        df_gt["gender_clean"]
        .dropna()
        .value_counts(normalize=True)
        .reindex(GENDER_ORDER, fill_value=0)
    )

    def _div_fixed(found_df, cat_col, cats):
        known = found_df[found_df[cat_col].notna()]
        counts = (
            known.groupby(["_cid", cat_col])
            .size()
            .unstack(cat_col, fill_value=0)
            .reindex(columns=cats, fill_value=0)
        )
        total = counts.sum(axis=1)
        p = counts.div(total.replace(0, np.nan), axis=0)
        entropy = -(p * np.log(p.where(p > 0))).sum(axis=1)
        div = entropy / np.log(len(cats))
        div[total < 2] = np.nan
        return div

    def _div_geo(found_df):
        known = found_df[found_df["location_oa_iso"].notna()]
        if known.empty:
            return pd.Series(dtype=float, name="div_location")
        counts = known.groupby(["_cid", "location_oa_iso"]).size()
        totals = counts.groupby(level="_cid").transform("sum")
        p = counts / totals
        entropy = (-(p * np.log(p))).groupby(level="_cid").sum()
        n_cats = counts.groupby(level="_cid").count()
        log_n = np.log(n_cats.where(n_cats >= 2))
        return (entropy / log_n).rename("div_location")

    def _parity(found_df, cat_col, cats, gt_frac):
        known = found_df[found_df[cat_col].notna()]
        counts = (
            known.groupby(["_cid", cat_col])
            .size()
            .unstack(cat_col, fill_value=0)
            .reindex(columns=cats, fill_value=0)
        )
        total = counts.sum(axis=1)
        frac = counts.div(total.replace(0, np.nan), axis=0).fillna(0)
        tv = 0.5 * (frac - gt_frac).abs().sum(axis=1)
        return 1 - tv

    df_valid_calls["div_ethnicity"] = _div_fixed(
        df_authors_found, "ethnicity_clean", ETHNICITY_ORDER
    )
    df_valid_calls["div_gender"] = _div_fixed(
        df_authors_found, "gender_clean", GENDER_ORDER
    )
    df_valid_calls["div_location"] = _div_geo(df_authors_found)
    df_valid_calls["parity_ethnicity"] = _parity(
        df_authors_found, "ethnicity_clean", ETHNICITY_ORDER, gt_eth_frac
    )
    df_valid_calls["parity_gender"] = _parity(
        df_authors_found, "gender_clean", GENDER_ORDER, gt_gen_frac
    )

    # ── 7.5 Structural metrics: connectedness + similarity (Eqs. 6-8) ────────
    # Assigned here, while df_valid_calls is still indexed by _cid — the
    # reset_index() below drops it and later sections have to go through _pk.
    if args.oa_duckdb:
        df_structural = _compute_structural_metrics(
            df_valid_recommendations,
            cache_dir=cache_dir,
            oa_duckdb_path=args.oa_duckdb,
            oa_author_stats_dir=args.oa_author_stats,
            ss_parquet=args.parquet,
            with_ss_block=args.with_ss_block,
            population=args.population,
            rebuild=args.rebuild_structural,
        )
        for col in df_structural.columns:
            df_valid_calls[col] = df_structural[col]
    else:
        logger.warning(
            "Skipping structural metrics: no --oa_duckdb and no [data].oa_duckdb "
            "in config.ini"
        )

    df_valid_calls = df_valid_calls.reset_index(drop=True)

    # ── 8. Productivity tiers ────────────────────────────────────────────────
    prod = df_valid_recommendations[df_valid_recommendations["author_found"]].copy()
    prod["field_en"] = prod["field"].map(FIELD_NORM_MAP).fillna(prod["field"])

    prod_thresholds = productivity_thresholds(prod, field_col="field_en")
    for col, lab in PRODUCTIVITY_OA_FIELDS_MAP.items():
        prod[f"tier_{lab}"] = assign_productivity_tier(
            prod[col], prod["field_en"], prod_thresholds[col]
        )

    def _hash_key(d, cols):
        return d[cols].astype(str).agg("||".join, axis=1)

    prod["_pk"] = _hash_key(prod, CALL_KEYS)

    def _tier_stats(prod_df, tier_col, label):
        counts = (
            prod_df.dropna(subset=[tier_col])
            .groupby(["_pk", tier_col])
            .size()
            .unstack(tier_col, fill_value=0)
            .reindex(columns=PRODUCTIVITY_TIER_LABELS, fill_value=0)
        )
        total = counts.sum(axis=1)
        frac = counts.div(total.replace(0, np.nan), axis=0)
        p = frac.where(frac > 0)
        div = -(p * np.log(p)).sum(axis=1) / np.log(len(PRODUCTIVITY_TIER_LABELS))
        div[total < 2] = np.nan
        return pd.DataFrame(
            {
                f"pct_{label}_low": frac["low"],
                f"pct_{label}_med": frac["med"],
                f"pct_{label}_high": frac["high"],
                f"div_productivity_{label}": div,
            }
        )

    stats_works = _tier_stats(prod, "tier_works", "works")
    stats_cit = _tier_stats(prod, "tier_citations", "citations")

    df_valid_calls["_pk"] = _hash_key(df_valid_calls, CALL_KEYS)
    for tier_df in (stats_works, stats_cit):
        for c in tier_df.columns:
            df_valid_calls[c] = df_valid_calls["_pk"].map(tier_df[c])
    df_valid_calls.drop(columns=["_pk"], inplace=True)

    uniform_tier = 1.0 / len(PRODUCTIVITY_TIER_LABELS)
    for lab in PRODUCTIVITY_OA_FIELDS_MAP.values():
        tv = 0.5 * sum(
            (df_valid_calls[f"pct_{lab}_{t}"].fillna(0) - uniform_tier).abs()
            for t in PRODUCTIVITY_TIER_LABELS
        )
        no_data = df_valid_calls[
            [f"pct_{lab}_{t}" for t in PRODUCTIVITY_TIER_LABELS]
        ].isna().all(axis=1)
        df_valid_calls[f"parity_{lab}"] = (1 - tv).where(~no_data)

    df_valid_calls["popularity_works"] = df_valid_calls["pct_works_high"]
    df_valid_calls["popularity_citations"] = df_valid_calls["pct_citations_high"]

    # ── 9. Expand calls to include invalid/refused (for Wilson CIs) ──────────
    extra_model_cols = MODEL_ARCHITECTURE_COLS + MODEL_EXTRA_COLS + [
        "validity",
        "refusals",
    ]
    df_valid_calls = df_all_calls[CALL_KEYS + extra_model_cols].merge(
        df_valid_calls.drop(
            columns=[c for c in extra_model_cols if c in df_valid_calls.columns]
        ),
        on=CALL_KEYS,
        how="left",
    )

    # ── 10. Consistency ──────────────────────────────────────────────────────
    df_consistency = _compute_consistency(df_valid_recommendations)
    df_valid_calls = df_valid_calls.merge(
        df_consistency[PROMPT_KEYS + ["consistency"]], on=PROMPT_KEYS, how="left"
    )

    # ── 11. English-normalised entity names ─────────────────────────────────
    df_valid_calls["language_en"] = (
        df_valid_calls["language"].map(LANGUAGE_NORM_MAP).fillna(df_valid_calls["language"])
    )
    df_valid_calls["location_en"] = (
        df_valid_calls["location"].map(LOCATION_NORM_MAP).fillna(df_valid_calls["location"])
    )
    df_valid_calls["field_en"] = (
        df_valid_calls["field"].map(FIELD_NORM_MAP).fillna(df_valid_calls["field"])
    )
    df_valid_calls["subfield_en"] = (
        df_valid_calls["subfield"].map(SUBFIELD_NORM_MAP).fillna(df_valid_calls["subfield"])
    )
    df_valid_calls["task_en"] = (
        df_valid_calls["task"].map(TASK_NORM_MAP).fillna(df_valid_calls["task"])
    )
    df_valid_calls["target_en"] = (
        df_valid_calls["target"].map(TARGET_NORM_MAP).fillna(df_valid_calls["target"])
    )
    df_valid_calls["role_en"] = (
        df_valid_calls["role"].map(ROLE_NORM_MAP).fillna(df_valid_calls["role"])
    )

    # ── 12. Complement columns (duplicates_c, refusals_c) ───────────────────
    if "duplicates_c" not in df_valid_calls.columns:
        df_valid_calls["duplicates_c"] = df_valid_calls["duplicates"].apply(
            lambda v: 1 - v if pd.notna(v) else np.nan
        )
    if "refusals_c" not in df_valid_calls.columns:
        df_valid_calls["refusals_c"] = df_valid_calls["refusals"].apply(
            lambda v: 1 - v if pd.notna(v) else np.nan
        )

    # ── 13. Save ─────────────────────────────────────────────────────────────
    df_valid_calls.to_csv(valid_calls_path, index=False)
    logger.info(
        "Saved %d rows × %d columns → %s",
        len(df_valid_calls),
        len(df_valid_calls.columns),
        valid_calls_path,
    )


if __name__ == "__main__":
    main()
