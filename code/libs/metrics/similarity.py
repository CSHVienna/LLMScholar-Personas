"""
similarity.py — Scholarly similarity between authors (paper Eq. 8).

  load_author_scholarly_stats   h-index, i10-index, e-index and 2yr citedness per
                                author (files written by pipeline/4_author_stats.py)
  load_ss_author_features       optional Semantic Scholar block of features
  build_author_features         one feature vector per author
  build_similarity_embeddings   impute → log1p → standardise → PCA → unit vectors
                                (fitted once, then cached)
  compute_similarity            mean pairwise cosine among the authors of one response
  save_similarity_model_h5      export the fitted PCA

_scholarly_stats_from_citations and build_author_scholarly_stats compute the
h/i10/e-index themselves; they are kept for the tests that validate them.
"""

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from libs.metrics import constants
from libs.metrics.coauthorship import (
    StructuralMetric, _population_key, _resolve_indices, author_num, build_author_index,
    chunk_bounds, works_author_key,
)
from libs.metrics.io import validate_path


def _scholarly_stats_from_citations(
    idx: np.ndarray, citations: np.ndarray, n_authors: int
) -> pd.DataFrame:
    """h-index, i10-index and e-index from one (author row, paper citations) pair
    array. Split out from the DuckDB pass so it can be unit-tested directly.

    Everything is computed group-wise without a Python-level loop: sorting by
    (author, -citations) once puts each author's papers in descending citation
    order, and the per-group rank of a row is its offset from that group's start.

    * h-index — largest rank r whose paper still has >= r citations, i.e. the
      largest r for which the descending-sorted c[r-1] >= r.
    * i10-index — count of papers with >= 10 citations.
    * e-index — NOT Zhang's sqrt(h-core excess) but LLMScholarBench's definition
      (GTBuilder/APS/code/libs/scholar.py), the entropy of how an author's
      citations spread over their papers:
          e = -1/N * sum_i c_i * log(c_i / c_total)
      where N counts the author's CITED papers only. Uncited papers contribute
      nothing (the limit of c*log(c) as c -> 0 is 0) and are absent from the
      denominator too; an author with no citations at all scores 0.
    """
    order = np.lexsort((-citations, idx))
    idx_s = idx[order]
    c_s = citations[order].astype(np.float64, copy=False)

    is_start = np.empty(len(idx_s), dtype=bool)
    is_start[0] = True
    np.not_equal(idx_s[1:], idx_s[:-1], out=is_start[1:])
    group = np.cumsum(is_start) - 1
    starts = np.flatnonzero(is_start)
    rank = np.arange(len(idx_s), dtype=np.int64) - starts[group] + 1
    n_groups = len(starts)

    h = np.zeros(n_groups, dtype=np.int64)
    holds = c_s >= rank
    np.maximum.at(h, group[holds], rank[holds])

    i10 = np.bincount(group[c_s >= 10], minlength=n_groups).astype(np.int64)

    c_total = np.bincount(group, weights=c_s, minlength=n_groups)
    cited = c_s > 0
    # N is the number of CITED papers, not of all papers. The reference
    # (scholar.py:compute_e_index) runs over a citation table, where an uncited
    # paper has no rows at all, so its `nunique()` can only ever count cited
    # ones. Dividing by every paper deflates e by the uncited share: measured
    # over the 305,893-author population only 67.1% of papers are cited, and
    # the old denominator understated e by 1.48x at the median (1.73x on
    # average, up to 1520x), for 88.5% of authors.
    n_cited = np.bincount(group[cited], minlength=n_groups)
    contrib = np.zeros(len(c_s), dtype=np.float64)
    contrib[cited] = c_s[cited] * np.log(c_s[cited] / c_total[group[cited]])
    # An author with papers but no citations scores 0, which is where the
    # reference's fillna of e_index to 0 lands them anyway.
    e = np.zeros(n_groups, dtype=np.float64)
    has_cites = n_cited > 0
    e[has_cites] = (
        -np.bincount(group, weights=contrib, minlength=n_groups)[has_cites]
        / n_cited[has_cites]
    )

    # Authors whose every paper fell outside the snapshot never appear in `idx`;
    # reindexing onto the full row range leaves them NaN rather than a fake 0.
    present = idx_s[starts]
    out = pd.DataFrame(
        {"h_index": h, "i10_index": i10, "e_index": e}, index=present
    )
    return out.reindex(np.arange(n_authors))


def build_author_scholarly_stats(
    author_ids: Iterable[str],
    *,
    cache_path: Path | str,
    oa_duckdb_path: Path | str,
    memory_limit: str = "12GB",
    temp_dir: Path | str | None = None,
    n_chunks: int = 32,
    rebuild: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Compute (once) and cache h-index / i10-index / e-index per author.

    Returns a DataFrame indexed by author id with those three columns, ready to
    pass as ``build_author_features(stats=...)``.

    They are computed from first principles, which needs nothing but
    ``works.cited_by_count`` and ``works.authorships``. (The pipeline reads
    h/i10 from OpenAlex's summary_stats instead; this version is kept for the
    tests that check both agree.)

    Note what this does *not* need: the citation graph. LLMScholarBench derives
    the e-index from a publication-to-publication citation table because the APS
    dataset has one, but the formula only ever reads c_i, the citation count of
    each paper — a column OpenAlex already stores. ``referenced_works`` is never
    touched.

    Cost is the same shape as :func:`build_coauthorship_graph`: one chunked pass
    over ``oa.works`` exploding ``authorships``, so budget hours and run it once.
    Both passes are independent and the cache keys differ, so adding this does
    not invalidate an existing coauthorship graph.
    """
    log = logger or logging.getLogger()
    cache_path = Path(cache_path)
    index = build_author_index(author_ids)
    n = len(index)
    population_key = _population_key(index)

    if cache_path.exists() and not rebuild:
        import joblib

        payload = joblib.load(cache_path)
        if payload.get("population_key") == population_key:
            log.info(
                "Scholarly stats: loaded %s (%d authors)",
                cache_path,
                len(payload["stats"]),
            )
            return payload["stats"]
        log.info(
            "Scholarly stats: cache was built for a different author population "
            "(%s != %s) — rebuilding",
            payload.get("population_key"),
            population_key,
        )

    import duckdb
    import joblib

    log.info("Scholarly stats: building over %d authors", n)

    u_df = pd.DataFrame(
        {
            "author_url": [f"https://openalex.org/{a}" for a in index],
            "author_num": [author_num(a) for a in index],
            "idx": list(index.values()),
        }
    )

    # DISTINCT on (work, author): an author listed twice in one work's
    # authorships would otherwise count that paper — and its citations — twice.
    chunk_query = """
        SELECT DISTINCT w.id AS work_id, u.idx AS idx, w.cited_by_count AS c
        FROM oa.works w,
             UNNEST(w.authorships) AS t(a)
             JOIN u ON u.{key} = a.author.id
        WHERE w.id >= ? AND w.id < ?
    """

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        if temp_dir is not None:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute(f"PRAGMA temp_directory='{temp_dir}'")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"ATTACH '{oa_duckdb_path}' AS oa (READ_ONLY)")

        con.register("u_df", u_df)
        con.execute(
            "CREATE TEMP TABLE u AS SELECT author_url::VARCHAR AS author_url, "
            "author_num::BIGINT AS author_num, idx::INTEGER AS idx FROM u_df"
        )
        chunk_query = chunk_query.replace("{key}", works_author_key(con))

        bounds = chunk_bounds(con, n_chunks)
        log.info(
            "Scholarly stats: scanning oa.works in %d quantile chunks",
            len(bounds) - 1,
        )

        idx_parts, cit_parts = [], []
        for c, (start, end) in enumerate(zip(bounds, bounds[1:])):
            result = con.execute(chunk_query, [start, end]).fetchnumpy()
            if len(result["idx"]):
                idx_parts.append(result["idx"].astype(np.int32, copy=False))
                cit_parts.append(result["c"].astype(np.int64, copy=False))
            log.info(
                "Scholarly stats: chunk %d/%d — %d (author, paper) rows",
                c + 1,
                len(bounds) - 1,
                len(result["idx"]),
            )
    finally:
        con.close()

    if idx_parts:
        idx_all = np.concatenate(idx_parts)
        cit_all = np.concatenate(cit_parts)
    else:
        idx_all = np.empty(0, dtype=np.int32)
        cit_all = np.empty(0, dtype=np.int64)
    log.info("Scholarly stats: %d (author, paper) rows collected", len(idx_all))

    stats = _scholarly_stats_from_citations(idx_all, cit_all, n)
    # Back from row positions to author ids.
    stats.index = pd.Index(list(index), name="author_id")
    log.info(
        "Scholarly stats: h-index defined for %d/%d authors (median %.0f)",
        int(stats["h_index"].notna().sum()),
        n,
        stats["h_index"].median(skipna=True),
    )

    validate_path(cache_path)
    joblib.dump(
        {"stats": stats, "index": index, "population_key": population_key},
        cache_path,
        compress=3,
    )
    log.info("Scholarly stats: saved → %s", cache_path)
    return stats


def load_author_scholarly_stats(
    stats_dir: Path | str,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """The four SIMILARITY_STATS_COLS, assembled from the two files that carry them.

    ``summary_stats.parquet`` holds what OpenAlex already publishes — h_index,
    i10_index, 2yr_mean_citedness (authors.summary_stats).
    ``eindex.parquet`` holds the e-index, which OpenAlex does not publish and
    which therefore has to be computed from per-paper citation counts. Both are
    produced by scripts/pipeline/4_author_stats.py.

    Returned indexed by the bare author id token (``A5023888391``), matching
    what :func:`build_author_features` expects, not the URL form the parquets
    store.
    """
    log = logger or logging.getLogger(__name__)
    stats_dir = Path(stats_dir)
    summary_fn = stats_dir / "summary_stats.parquet"
    eindex_fn = stats_dir / "eindex.parquet"

    missing = [str(p) for p in (summary_fn, eindex_fn) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run scripts/pipeline/4_author_stats.py "
            "to build them (the summary_stats stage takes ~15 min, the e-index "
            "pass over oa.works ~80 min; both are checkpointed)."
        )

    def _token(series: pd.Series) -> pd.Series:
        return series.astype(str).str.rsplit("/", n=1).str[-1].str.strip()

    summary = pd.read_parquet(
        summary_fn, columns=["id", "h_index", "i10_index", "two_year_mean_citedness"]
    )
    summary.index = _token(summary.pop("id"))

    eindex = pd.read_parquet(eindex_fn, columns=["author_url", "e_index"])
    eindex.index = _token(eindex.pop("author_url"))

    stats = summary.join(eindex, how="outer")
    log.info(
        "Scholarly stats: loaded %d authors from %s (%d without an e-index)",
        len(stats),
        stats_dir,
        int(stats["e_index"].isna().sum()),
    )
    return stats[list(constants.SIMILARITY_STATS_COLS)]


def load_ss_author_features(
    ss_parquet: Path | str,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """The Semantic Scholar half of the feature vector, indexed by Researcher_id.

    Mirrors the reference's `aps_*` block: the same metric families as the
    OpenAlex side, measured inside the other corpus. Returns
    ``constants.SIMILARITY_SS_COLS``.

    Dirty years are dropped rather than passed through: the snapshot ends in
    2019 yet `Last_year` reaches 2021 and `First_year` goes back to 1605, which
    yields career ages up to 414 years. Those rows become NaN so the imputer
    treats them as unknown, instead of stretching the standard deviation the
    shared scaler learns and tilting every PCA axis.
    """
    log = logger or logging.getLogger(__name__)
    df = pd.read_parquet(
        ss_parquet,
        columns=["Researcher_id", "First_year", "Last_year", "Productivity", "Citations"],
    )
    df = df.drop_duplicates(subset=["Researcher_id"]).set_index("Researcher_id")

    first = pd.to_numeric(df["First_year"], errors="coerce")
    last = pd.to_numeric(df["Last_year"], errors="coerce")
    # The panel stops at 2019; anything outside that window is a data error.
    plausible = (first >= 1900) & (first <= 2019) & (last >= first) & (last <= 2020)
    career_age = (last - first + 1).where(plausible)

    works = pd.to_numeric(df["Productivity"], errors="coerce")
    citations = pd.to_numeric(df["Citations"], errors="coerce")
    citations_per_work = citations / works.clip(lower=1)

    out = pd.DataFrame(index=df.index)
    out["ss_works_count"] = works
    out["ss_cited_by_count"] = citations
    out["ss_career_age"] = career_age
    known = works.notna() & career_age.notna()
    out["ss_citations_per_paper_age"] = (
        (citations_per_work / career_age)
        .where((works > 0) & (career_age > 0), 0.0)
        .where(known)
    )
    log.info(
        "SS features: %d researchers (%d dropped for implausible years)",
        len(out),
        int((~plausible).sum()),
    )
    return out[list(constants.SIMILARITY_SS_COLS)]


def _log1p_clipped(x):
    """``log(1 + x)`` with a non-negativity guard.

    Module-level (not a lambda) so the fitted pipeline stays picklable for
    joblib.
    """
    return np.log1p(np.clip(x, 0, None))


def build_author_features(
    df: pd.DataFrame,
    *,
    id_col: str = "author_id",
    works_col: str = "oa_works_count",
    citations_col: str = "oa_cited_by_count",
    career_age_col: str = "oa_career_age",
    first_year_col: str = "oa_first_pub_year",
    last_year_col: str = "oa_last_pub_year",
    researcher_id_col: str = "researcher_id",
    stats: pd.DataFrame | None = None,
    ss: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive the scholarly-similarity features, one row per unique author.

    Columns are ``constants.SIMILARITY_FEATURE_COLS``, indexed by ``id_col``:
    productivity (``works_count``, ``works_per_year``), citation impact
    (``cited_by_count``, ``citations_per_work``, ``citations_per_paper_age``)
    and career stage (``career_age``) — the three axes named in the paper.

    ``ss`` is the optional output of :func:`load_ss_author_features`. When
    given, its four columns are appended as a second block, mirroring the
    reference PCA's OpenAlex + APS concatenation, and the result carries
    ``SIMILARITY_FEATURE_COLS_WITH_SS``. Authors with no Semantic Scholar match
    get NaN there and are median-imputed, which is what the reference does with
    its `aps_*` block for authors outside APS.

    ``stats`` is the optional output of :func:`build_author_scholarly_stats`
    (``h_index`` / ``i10_index`` / ``e_index``). When given, those three columns
    are joined on and the result carries ``SIMILARITY_FEATURE_COLS_FULL``;
    authors absent from ``stats`` get NaN and are median-imputed downstream like
    any other gap. When omitted the base vector is returned unchanged, which is
    what a run without ``--oa_duckdb`` gets.

    ``career_age`` follows the reference's definition, ``(last_year -
    first_year) + 1``, computed from ``first_year_col`` / ``last_year_col`` when
    both are present and falling back to ``career_age_col`` when they are not
    (which is what small fixtures pass). Both reference blocks use the ``+ 1``
    (batch_user_rankings.py:64 and batch_aps_authors_stats.py:67) and it is not
    cosmetic: ``oa_career_age`` is ``last - first``, which reports 0 for an
    author active in a single year — 1.7% of this population — and a zero
    denominator is what forced the guard below.

    ``citations_per_paper_age`` follows LLMScholarBench's definition,
    ``(cited_by_count / works_count) / career_age``, including its guard
    (batch_user_rankings.py:77): 0 when the author has no works or no career
    span, rather than a clipped divisor. Clipping ``career_age`` to 1 handed a
    single-year author their whole ``citations_per_work``, which is the opposite
    of damping by career length.

    Absolute years (``oa_first_pub_year`` / ``oa_last_pub_year``) are
    deliberately not features: after ``log(1+x)`` a 30-year gap between 1985 and
    2015 collapses to a 1.5% difference, so they contribute noise rather than
    signal. Their information survives as ``career_age``.

    Known data-quality caveat, left uncorrected by decision: OpenAlex publication
    years carry disambiguation errors, so a minority of authors show impossible
    values. Measured over the 305,893-author reference population, ``career_age``
    exceeds 100 years for 2.76% of them (p99 = 139 years, max = 2036, against a
    median of 26), and ``works_per_year`` exceeds one paper per day for 10
    authors. These are passed through as-is rather than being reclassified as
    missing. ``log(1+x)`` compresses them, but they still widen the standard
    deviation the shared StandardScaler learns and tilt the PCA axes, which
    shifts every Sim_i — not only the affected authors'. Revisit here, before the
    pipeline, if the similarity distribution ever looks off.
    """
    out = df.drop_duplicates(subset=[id_col]).set_index(id_col)
    works = pd.to_numeric(out[works_col], errors="coerce")
    citations = pd.to_numeric(out[citations_col], errors="coerce")

    if first_year_col in out.columns and last_year_col in out.columns:
        first = pd.to_numeric(out[first_year_col], errors="coerce")
        last = pd.to_numeric(out[last_year_col], errors="coerce")
        career_age = last - first + 1
    else:
        career_age = pd.to_numeric(out[career_age_col], errors="coerce")

    features = pd.DataFrame(index=out.index)
    features["works_count"] = works
    features["cited_by_count"] = citations
    features["career_age"] = career_age
    # citations_per_work is an intermediate, not a feature: it is not in the
    # reference PCA nor in B6, and emitting it tilts the fitted axes.
    citations_per_work = citations / works.clip(lower=1)
    # The reference guard returns 0 for an author with no works or no career
    # span, but only when those are *known* to be zero. An author absent from
    # OpenAlex has them as NaN, and must stay NaN: a 0 here would read as a
    # real zero-impact author and be median-imputed instead of excluded, which
    # is the distinction test_authors_without_features_get_nan_rows pins.
    known = works.notna() & career_age.notna()
    features["citations_per_paper_age"] = (
        (citations_per_work / career_age)
        .where((works > 0) & (career_age > 0), 0.0)
        .where(known)
    )

    if stats is None:
        return features[constants.SIMILARITY_FEATURE_COLS]

    # `stats` must carry every declared column. Two sources feed it: h_index,
    # i10_index and two_year_mean_citedness are extracted from the raw OpenAlex
    # author records (summary_stats), while e_index is computed by
    # build_author_scholarly_stats because OpenAlex does not publish it. A
    # missing column means one of the two never ran, and silently filling it
    # with NaN would hand the imputer an all-NaN column and shift the PCA.
    absent = [c for c in constants.SIMILARITY_STATS_COLS if c not in stats.columns]
    if absent:
        raise ValueError(
            f"stats is missing {absent}; expected all of "
            f"{constants.SIMILARITY_STATS_COLS}. h_index / i10_index / "
            "two_year_mean_citedness come from the raw-file extraction, e_index "
            "from build_author_scholarly_stats — join both before calling."
        )
    for col in constants.SIMILARITY_STATS_COLS:
        features[col] = pd.to_numeric(
            stats[col].reindex(features.index), errors="coerce"
        )
    if ss is None:
        return features[constants.SIMILARITY_FEATURE_COLS_FULL]

    # Second block, from load_ss_author_features. It is keyed by Researcher_id
    # while this frame is keyed by author_id, so the join goes through the
    # per-row researcher_id. drop_duplicates above already picked one row per
    # author, so an oa_id seen with several researcher_ids contributes the first
    # one — they are the same person under two ids, and the SS metrics of either
    # describe the same fragment of that career.
    if researcher_id_col not in out.columns:
        raise ValueError(
            f"ss features require the {researcher_id_col!r} column to join on; "
            "add it to the frame passed in (it is already in NEEDED_COLS)."
        )
    rid = pd.to_numeric(out[researcher_id_col], errors="coerce")
    joined = ss.reindex(rid.values)
    for col in constants.SIMILARITY_SS_COLS:
        features[col] = pd.to_numeric(joined[col].values, errors="coerce")
    return features[constants.SIMILARITY_FEATURE_COLS_WITH_SS]


def build_similarity_embeddings(
    features: pd.DataFrame,
    *,
    cache_path: Path | str,
    index: Mapping[str, int] | None = None,
    variance_threshold: float = constants.SIMILARITY_PCA_VARIANCE,
    random_state: int = constants.SIMILARITY_RANDOM_STATE,
    rebuild: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, dict[str, int], object]:
    """Fit (once) and cache the preprocessing pipeline, then embed every author.

    Returns ``(embeddings, index, pipeline)``. ``embeddings[i]`` is the unit
    vector z_u of the author at row ``i`` of ``index``; rows whose features were
    entirely missing hold NaN so callers can exclude them.

    Preprocessing follows the paper's order exactly (Eq. 8): median imputation →
    ``log(1+x)`` → standardisation to zero mean / unit variance → PCA retaining
    the fewest components explaining >= ``variance_threshold`` of the variance →
    L2 normalisation of the resulting embedding.

    Scaler and PCA are fitted ONCE over the whole reference population passed in
    ``features`` and persisted; individual responses are only transformed. This
    departs from a literal reading of the paper, which fits within each
    response, and is deliberate: with n≈10 authors and 5 features a per-response
    PCA is numerically unstable, the ">=90% of variance" criterion would be
    measured against the variance of those 10 authors alone (amplifying rounding
    noise when they are alike), and each response would end up in its own basis,
    making Sim_i incomparable across models and tasks — which is precisely what
    the benchmark needs to compare.
    """
    log = logger or logging.getLogger()
    cache_path = Path(cache_path)

    if index is None:
        index = build_author_index(features.index)

    # Fingerprint covers the population *and* the feature values: a refreshed
    # OpenAlex snapshot changes the fitted scaler/PCA even at constant authors.
    feature_digest = int(pd.util.hash_pandas_object(features, index=True).sum())
    population_key = f"{_population_key(index)}:{feature_digest & 0xFFFFFFFF:08x}"

    if cache_path.exists() and not rebuild:
        import joblib

        payload = joblib.load(cache_path)
        if payload.get("population_key") == population_key:
            log.info(
                "Similarity embeddings: loaded %s (%d authors, %d components)",
                cache_path,
                payload["embeddings"].shape[0],
                payload["embeddings"].shape[1],
            )
            return payload["embeddings"], payload["index"], payload["pipeline"]
        log.info(
            "Similarity embeddings: cache was built for different authors or "
            "features (%s != %s) — refitting",
            payload.get("population_key"),
            population_key,
        )

    import joblib
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler

    # Take the columns the caller actually supplied rather than a fixed list:
    # build_author_features returns the base vector without the snapshot pass
    # and SIMILARITY_FEATURE_COLS_FULL with it, so hardcoding either one
    # silently drops features in the other case. Ordered by the canonical list
    # so the fitted pipeline is reproducible.
    # WITH_SS is the widest canonical list, so it orders every valid input:
    # the base vector, the base + stats one, and either of those plus the
    # Semantic Scholar block. Filtering against a narrower list silently drops
    # the SS columns and the PCA then refits on a vector the caller did not ask
    # for — which is exactly what happened once and produced two "different"
    # variants with correlation 1.00000.
    feature_cols = [
        c for c in constants.SIMILARITY_FEATURE_COLS_WITH_SS if c in features.columns
    ]
    unknown = [c for c in features.columns if c not in feature_cols]
    if unknown:
        raise ValueError(
            f"features carries columns absent from the canonical order: {unknown}. "
            "Add them to constants.SIMILARITY_FEATURE_COLS_WITH_SS, or they will "
            "be dropped from the PCA without a trace."
        )
    missing = set(constants.SIMILARITY_FEATURE_COLS) - set(feature_cols)
    if missing:
        raise ValueError(f"features is missing base columns: {sorted(missing)}")

    # Align the feature matrix to the shared index; unknown authors stay NaN.
    matrix = features.reindex(list(index.keys()))[feature_cols].to_numpy(dtype=float)
    all_missing = np.isnan(matrix).all(axis=1)
    log.info(
        "Similarity embeddings: fitting over %d authors (%d without any feature)",
        matrix.shape[0] - int(all_missing.sum()),
        int(all_missing.sum()),
    )

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            # Counts are heavy-tailed; log1p before standardising. Features are
            # non-negative by construction, clip guards against upstream noise.
            (
                "log1p",
                FunctionTransformer(_log1p_clipped, feature_names_out="one-to-one"),
            ),
            ("scale", StandardScaler()),
            # float n_components = smallest k reaching the variance threshold.
            (
                "pca",
                PCA(
                    n_components=variance_threshold,
                    svd_solver="full",
                    random_state=random_state,
                ),
            ),
            ("l2", Normalizer(norm="l2")),
        ]
    )

    fit_rows = matrix[~all_missing]
    if fit_rows.shape[0] < 2:
        raise ValueError(
            f"Need >= 2 authors with features to fit the PCA, got {fit_rows.shape[0]}"
        )
    pipeline.fit(fit_rows)
    n_components = pipeline.named_steps["pca"].n_components_
    explained = float(pipeline.named_steps["pca"].explained_variance_ratio_.sum())
    log.info(
        "Similarity embeddings: PCA kept %d/%d components (%.1f%% of variance)",
        n_components,
        len(feature_cols),
        100 * explained,
    )

    embeddings = np.full((matrix.shape[0], n_components), np.nan, dtype=float)
    embeddings[~all_missing] = pipeline.transform(fit_rows)

    validate_path(cache_path)
    joblib.dump(
        {
            "embeddings": embeddings,
            "index": dict(index),
            "pipeline": pipeline,
            "population_key": population_key,
            # Which columns the fit saw. Not needed to reproject — the pipeline
            # carries that — but without the names an exported model is
            # uninterpretable, and they cannot be recovered from the arrays.
            "feature_cols": list(feature_cols),
        },
        cache_path,
        compress=3,
    )
    log.info("Similarity embeddings: saved → %s", cache_path)
    return embeddings, dict(index), pipeline


def save_similarity_model_h5(
    cache_path: Path | str,
    h5_path: Path | str,
    *,
    feature_cols: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Export a fitted similarity model to HDF5, in LLMScholarBench's layout.

    The joblib cache this repo writes is a pickle, so reading it needs the same
    scikit-learn and numpy versions that wrote it. HDF5 stores plain arrays
    instead, which is what makes the reference's `PCASimilarityModel.load_h5`
    able to read it — and any other tool, in any language.

    Dataset names follow the reference (`ids`, `embedding_`,
    `explained_variance_ratio_`, `pca/components_`, `pca/mean_`, plus a
    `meta_json` attribute) so its loader accepts the file unchanged.

    One addition under `preprocessing/`: the imputer medians and the scaler's
    mean and scale. The reference's own format omits them, which means a file in
    that layout can describe an embedding but cannot reproject a new author —
    every step before the PCA is missing. Ours carries them, and the extra group
    is simply ignored by a reader that does not look for it.
    """
    import h5py
    import joblib

    log = logger or logging.getLogger(__name__)
    payload = joblib.load(Path(cache_path))
    pipeline = payload["pipeline"]
    index = payload["index"]
    embeddings = payload["embeddings"]

    pca = pipeline.named_steps["pca"]
    imputer = pipeline.named_steps["impute"]
    scaler = pipeline.named_steps["scale"]

    # `index` maps author_id -> row, so invert it by row to keep ids aligned
    # with the embedding matrix rather than trusting dict order.
    ids = np.empty(len(index), dtype=object)
    for author_id, row in index.items():
        ids[row] = author_id

    # Which columns the PCA was fitted on is not recoverable from the arrays,
    # and caches written before this was stored do not carry it. Without the
    # names the file still reprojects correctly but nobody can tell what the
    # inputs mean, so require them rather than write an empty list.
    cols = feature_cols or payload.get("feature_cols")
    if not cols:
        raise ValueError(
            "feature_cols is neither in the cache nor passed in. Give the column "
            "names the model was fitted on (len must be "
            f"{scaler.n_features_in_})."
        )
    if len(cols) != scaler.n_features_in_:
        raise ValueError(
            f"feature_cols has {len(cols)} names but the model was fitted on "
            f"{scaler.n_features_in_} columns."
        )

    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ids", data=np.asarray(ids, dtype="S"))
        f.create_dataset("embedding_", data=embeddings)
        f.create_dataset(
            "explained_variance_ratio_", data=pca.explained_variance_ratio_
        )
        f.create_dataset("pca/components_", data=pca.components_)
        f.create_dataset("pca/mean_", data=pca.mean_)
        f.create_dataset("preprocessing/imputer_statistics_", data=imputer.statistics_)
        f.create_dataset("preprocessing/scaler_mean_", data=scaler.mean_)
        f.create_dataset("preprocessing/scaler_scale_", data=scaler.scale_)
        f.attrs["meta_json"] = json.dumps(
            {
                "feature_cols": list(cols),
                "n_components_": int(pca.n_components_),
                "rbf_sigma_": None,
                "embedding_l2_normalized_": True,
                # Not in the reference's meta, but the transform order is not
                # recoverable from the arrays alone.
                "preprocessing_order": ["impute_median", "log1p", "standardize"],
                "population_key": payload.get("population_key"),
            }
        )
    log.info(
        "Similarity model: wrote %s (%d authors, %d components)",
        h5_path,
        len(ids),
        pca.n_components_,
    )
    return h5_path


def compute_similarity(
    author_ids: Sequence[str],
    embeddings: np.ndarray,
    index: Mapping[str, int],
    *,
    logger: logging.Logger | None = None,
) -> StructuralMetric:
    """Mean pairwise scholarly similarity of a response's authors (paper Eq. 8).

    With unit embeddings z_u (see build_similarity_embeddings for the exact
    preprocessing order) and n = |U-hat_i|,

        Sim_i = (2 / (n(n-1))) * sum_{u != v, unordered pairs} z_u^T z_v   (Eq. 8)

    computed via the equivalent closed form. Since every ||z_u|| = 1,

        ||S||^2 = ||sum_u z_u||^2 = n + 2 * sum_{unordered pairs} z_u^T z_v
        =>  Sim_i = (||S||^2 - n) / (n(n-1))

    so the n x n cosine matrix is never materialised: cost is O(n·d) instead of
    O(n²·d), in time and in memory.

    ``np.nan`` is returned when the metric is undefined: n == 0, or n == 1 (no
    pair exists, so Eq. 8 has no value). Authors with no embedding — absent from
    ``index``, or with an all-missing feature vector — are excluded and counted.

    Parameters
    ----------
    author_ids : the members of U-hat_i, as ``author_id`` strings.
    embeddings : ``(n_population, d)`` matrix of unit vectors; NaN rows mark
                 authors whose features were entirely missing.
    index      : ``author_id -> row`` mapping that goes with ``embeddings``.
    """
    log = logger or logging.getLogger()
    rows, n_unique, n_excluded = _resolve_indices(author_ids, index)

    if len(rows):
        vectors = embeddings[rows]
        # The closed form is only valid for unit vectors: drop NaN rows (no
        # features) and any degenerate zero-norm row.
        norms = np.linalg.norm(vectors, axis=1)
        keep = np.isfinite(norms) & np.isclose(norms, 1.0, atol=1e-6)
        n_excluded += int((~keep).sum())
        vectors = vectors[keep]
    else:
        vectors = np.empty((0, embeddings.shape[1]))

    n = vectors.shape[0]
    if n_excluded:
        log.debug(
            "similarity: %d/%d authors without a usable embedding", n_excluded, n_unique
        )
    if n == 0:
        log.debug("similarity: undefined, n == 0 (no factual author with features)")
        return StructuralMetric(np.nan, 0, n_excluded)
    if n == 1:
        log.debug("similarity: undefined, n == 1 (no pair to compare)")
        return StructuralMetric(np.nan, 1, n_excluded)

    total = vectors.sum(axis=0)
    # Clip absorbs float error only: the closed form is mathematically bounded
    # to [-1, 1], but ||S||^2 on collinear embeddings can land at 1 + 4e-16.
    value = float(np.clip((total @ total - n) / (n * (n - 1)), -1.0, 1.0))
    log.debug("similarity: n=%d, value=%.4f", n, value)
    return StructuralMetric(value, n, n_excluded)
