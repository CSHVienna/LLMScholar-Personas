import datetime
import glob
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from libs.metrics import constants
except ImportError:  # PYTHONPATH=code/libs/
    from metrics import constants


def printf(message):
    # Get the current timestamp in the desired format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Print the message with the timestamp prepended
    print(f"[{timestamp}] {message}")


def path_join(*path_segments):
    """
    Joins multiple path components into a single path.

    Args:
        *path_segments (str): Path components to join.

    Returns:
        str: The joined path.
    """
    return os.path.join(*path_segments)


def read_list_of_dicts(file_path):
    """
    Read a list of dictionaries from a text file, where each line is a JSON object.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        list: A list of dictionaries read from the file.
    """
    try:
        with open(file_path, "r") as file:
            data = [json.loads(line) for line in file]
            return data
    except Exception as e:
        printf(f"Error reading list of dicts from {file_path}: {e}")
        return None


def read_json_file(file_path):
    """
    Read a JSON file and return its content.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed content of the JSON file.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        printf(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        printf(f"Error: Failed to decode JSON. Details: {e}")
    except Exception as e:
        printf(f"An unexpected error occurred: {e}")


def read_csv(fn, **kwargs):
    try:
        return pd.read_csv(fn, **kwargs)
    except Exception as e:
        printf(f"Error: {e}")
        return None


def save_csv(df, fn, **kwargs):
    try:
        verbose = kwargs.pop("verbose", True)
        df.to_csv(fn, **kwargs)
        if verbose:
            printf(f"Data successfully saved to {fn}")
    except Exception as e:
        printf(f"Error: {e}")


def exists(fn):
    return os.path.exists(fn)


def get_files(path, pattern):
    return glob.glob(os.path.join(path, pattern))


def validate_path(path):
    """
    Ensures all directories in the given path exist.
    - If the path is a file, ensures its containing directory exists.
    - If the path is a directory, ensures the entire directory path exists.
    """
    # Check if the path is a directory or a file
    if (
        os.path.isfile(path) or os.path.splitext(path)[1]
    ):  # Assume paths with extensions are files
        dir_path = os.path.dirname(path)
    else:  # Otherwise, treat it as a directory path
        dir_path = path

    # Create directories if they do not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_text(text, fn):
    try:
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        printf(f"Error: {e}")


#######################################################################################################################
# STRUCTURAL METRICS — cached builders for the coauthorship graph and the PCA pipeline
#######################################################################################################################
# Consumed by aggregators.compute_connectedness / compute_similarity (paper
# Eqs. 6-8). Both artefacts are built ONCE over the reference author population
# and persisted with joblib; per-response code only slices / transforms them.


def _population_key(index: Mapping[str, int]) -> str:
    """Fingerprint of the author population an artefact was built for.

    Stored inside the joblib payload so a cache built for a different set of
    authors is rebuilt instead of silently returning an index that does not
    cover the current authors — which would show up as unexplained exclusions.
    """
    digest = hashlib.md5()
    for author_id in sorted(index):
        digest.update(author_id.encode())
    return f"{len(index)}:{digest.hexdigest()[:12]}"


def build_author_index(author_ids: Iterable[str]) -> dict[str, int]:
    """Stable ``author_id -> row index`` mapping, sorted for reproducibility.

    The index is shared by the coauthorship matrix and the embedding matrix so
    a single lookup serves both metrics.
    """
    unique = sorted({str(a) for a in author_ids if a is not None and str(a) != ""})
    return {author_id: i for i, author_id in enumerate(unique)}


def build_coauthorship_graph(
    author_ids: Iterable[str],
    *,
    cache_path: Path | str,
    oa_duckdb_path: Path | str,
    memory_limit: str = "12GB",
    temp_dir: Path | str | None = None,
    n_chunks: int = 32,
    rebuild: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[sparse.csr_matrix, dict[str, int]]:
    """Build (once) and cache the coauthorship graph G restricted to ``author_ids``.

    Returns ``(adjacency, index)`` where ``adjacency`` is a symmetric boolean
    CSR matrix and ``index`` maps ``author_id -> row``. Cached with joblib; on a
    cache hit nothing is recomputed.

    Restricting G to the benchmark's author population is exact, not an
    approximation: every response's author set U-hat_i is a subset of
    ``author_ids``, and induced subgraphs compose — ``G[U-hat_i]`` equals
    ``(G[author_ids])[U-hat_i]``. This keeps the cached matrix at
    |author_ids|² sparse instead of (113M)² and is why the whole thing fits in
    memory.

    Two authors are linked when they appear in the authorship list of the same
    OpenAlex work. Edges are unweighted, deduplicated, symmetric, and carry no
    self-loops. networkx is deliberately not used: the graph is only ever
    consumed as a sparse matrix by scipy.sparse.csgraph.

    Cost note: exploding ``authorships`` over all ~492M works yields billions of
    (work, author) rows. Doing it in one statement exhausted 56 GiB of DuckDB
    temp space, so the scan is split into ``n_chunks`` quantile ranges of
    ``works.id``. Chunking is exact — a work lies wholly inside one range, so no
    edge can be split across chunks — and only bounds peak memory.

    ``n_chunks`` trades time against memory, and 32 is measured, not guessed.
    Per-chunk cost is mostly fixed (each chunk rescans the authorships column,
    which the id filter barely prunes), so fewer chunks is faster — but memory
    grows with chunk size. On the central chunk, the worst case at ~3.4 authors
    per work: 64 chunks → 235s each (4.2 h total), 32 → 352s each (3.1 h), and 8
    (~56M works) died with 40 GiB of temp space exhausted. Expect ~3 h, once.
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
                "Coauthorship graph: loaded %s (%d nodes, %d edges)",
                cache_path,
                payload["adjacency"].shape[0],
                payload["adjacency"].nnz // 2,
            )
            return payload["adjacency"], payload["index"]
        log.info(
            "Coauthorship graph: cache was built for a different author "
            "population (%s != %s) — rebuilding",
            payload.get("population_key"),
            population_key,
        )

    import duckdb
    import joblib

    log.info("Coauthorship graph: building over %d authors", n)

    # Join on the full OpenAlex URL rather than parsing it: the snapshot stores
    # author ids as 'https://openalex.org/A123', and running split_part +
    # replace + TRY_CAST on billions of authorship rows dominated the runtime.
    u_df = pd.DataFrame(
        {
            "author_url": [f"https://openalex.org/{a}" for a in index],
            "idx": list(index.values()),
        }
    )

    # Each chunk is self-contained: explode, keep authors in U, drop works with
    # fewer than 2 of them, then pair up whatever is left.
    chunk_query = """
        WITH hit AS (
            SELECT w.id AS work_id, u.idx AS idx
            FROM oa.works w,
                 UNNEST(w.authorships) AS t(a)
                 JOIN u ON u.author_url = a.author.id
            WHERE w.id >= ? AND w.id < ? AND length(w.authorships) >= 2
        ),
        multi AS (
            SELECT work_id FROM hit GROUP BY work_id HAVING count(DISTINCT idx) >= 2
        ),
        h AS (SELECT DISTINCT hit.work_id, hit.idx FROM hit JOIN multi USING (work_id))
        SELECT DISTINCT h1.idx AS src, h2.idx AS dst
        FROM h h1 JOIN h h2
          ON h1.work_id = h2.work_id AND h1.idx < h2.idx
    """

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        if temp_dir is not None:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute(f"PRAGMA temp_directory='{temp_dir}'")
        # Nothing downstream depends on row order, and preserving it is what
        # makes the exploded intermediate buffer without bound.
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"ATTACH '{oa_duckdb_path}' AS oa (READ_ONLY)")

        con.register("u_df", u_df)
        con.execute(
            "CREATE TEMP TABLE u AS "
            "SELECT author_url::VARCHAR AS author_url, idx::INTEGER AS idx FROM u_df"
        )

        # Chunk on id *quantiles*, not on equal-width id ranges: OpenAlex ids
        # are wildly non-uniform, and equal-width slicing put 91M works in one
        # bucket against a median of 787k — the fat bucket alone risks the OOM
        # the chunking exists to prevent.
        lo, hi = con.execute("SELECT min(id), max(id) FROM oa.works").fetchone()
        # Inlined rather than bound: approx_quantile takes its quantile list as
        # a literal. The values are generated here, never user input.
        cut_points = ", ".join(f"{i / n_chunks:.6f}" for i in range(1, n_chunks))
        inner = (
            con.execute(
                f"SELECT approx_quantile(id, [{cut_points}]) FROM oa.works"
            ).fetchone()[0]
            if n_chunks > 1
            else []
        )
        # Deduplicate: repeated quantiles (dense id regions) would yield empty
        # ranges, and an unsorted boundary list would silently skip works.
        bounds = sorted({int(lo), *(int(b) for b in inner), int(hi) + 1})
        log.info(
            "Coauthorship graph: scanning oa.works in %d quantile chunks",
            len(bounds) - 1,
        )

        chunks = []
        for c, (start, end) in enumerate(zip(bounds, bounds[1:])):
            result = con.execute(chunk_query, [start, end]).fetchnumpy()
            if len(result["src"]):
                chunks.append(
                    np.stack(
                        [
                            result["src"].astype(np.int32, copy=False),
                            result["dst"].astype(np.int32, copy=False),
                        ],
                        axis=1,
                    )
                )
            log.info(
                "Coauthorship graph: chunk %d/%d — %d pairs",
                c + 1,
                len(bounds) - 1,
                len(result["src"]),
            )
    finally:
        con.close()

    # The same pair can show up in several chunks (different shared works), so
    # deduplicate globally before building the matrix.
    if chunks:
        pairs = np.unique(np.concatenate(chunks, axis=0), axis=0)
        src, dst = pairs[:, 0], pairs[:, 1]
    else:
        src = dst = np.empty(0, dtype=np.int32)
    log.info("Coauthorship graph: %d unique undirected edges", len(src))

    # Symmetrise: store both directions so csgraph sees an undirected graph.
    data = np.ones(2 * len(src), dtype=bool)
    adjacency = sparse.coo_matrix(
        (data, (np.concatenate([src, dst]), np.concatenate([dst, src]))),
        shape=(n, n),
        dtype=bool,
    ).tocsr()
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()

    validate_path(cache_path)
    joblib.dump(
        {"adjacency": adjacency, "index": index, "population_key": population_key},
        cache_path,
        compress=3,
    )
    log.info("Coauthorship graph: saved → %s", cache_path)
    return adjacency, index


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
) -> pd.DataFrame:
    """Derive the 5 scholarly-similarity features, one row per unique author.

    Columns are ``constants.SIMILARITY_FEATURE_COLS``, indexed by ``id_col``:
    productivity (``works_count``, ``works_per_year``), citation impact
    (``cited_by_count``, ``citations_per_work``) and career stage
    (``career_age``) — the three axes named in the paper.

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
    career_age = pd.to_numeric(out[career_age_col], errors="coerce")

    features = pd.DataFrame(index=out.index)
    features["works_count"] = works
    features["cited_by_count"] = citations
    features["citations_per_work"] = citations / works.clip(lower=1)
    features["career_age"] = career_age
    features["works_per_year"] = works / career_age.clip(lower=1)
    return features[constants.SIMILARITY_FEATURE_COLS]


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

    # Align the feature matrix to the shared index; unknown authors stay NaN.
    matrix = features.reindex(list(index.keys()))[
        constants.SIMILARITY_FEATURE_COLS
    ].to_numpy(dtype=float)
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
        len(constants.SIMILARITY_FEATURE_COLS),
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
        },
        cache_path,
        compress=3,
    )
    log.info("Similarity embeddings: saved → %s", cache_path)
    return embeddings, dict(index), pipeline
