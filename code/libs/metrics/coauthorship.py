"""
coauthorship.py — The coauthorship graph and the connectedness metric (paper Eqs. 6-7).

  build_author_index        OpenAlex id → row number, shared by the graph and the PCA
  build_coauthorship_graph  who has written with whom among the recommended
                            authors (built once from OpenAlex, then cached)
  compute_connectedness     how connected the authors of one response are
"""

import hashlib
import logging
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from libs.metrics.io import validate_path


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

        # A requested population contained in the cached one needs no rescan:
        # coauthorship between two authors does not depend on who else is in the
        # matrix, so the induced subgraph is exact. Slicing it costs seconds
        # against the ~3 h a rebuild takes. This is what makes it cheap to try a
        # narrower population (say, only authors matched in both OpenAlex and
        # Semantic Scholar) without paying for the pass again.
        cached_index = payload.get("index") or {}
        if set(index).issubset(cached_index):
            rows = [cached_index[a] for a in index]
            adjacency = payload["adjacency"].tocsr()[rows][:, rows]
            log.info(
                "Coauthorship graph: %d of %d cached nodes requested — induced "
                "subgraph taken from %s (%d edges), no rescan",
                n,
                len(cached_index),
                cache_path,
                adjacency.nnz // 2,
            )
            return adjacency, index

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


class StructuralMetric(NamedTuple):
    """Result of a structural metric over one response.

    ``value``       the metric, or ``np.nan`` where it is undefined.
    ``n_used``      members of U-hat_i that actually entered the computation.
    ``n_excluded``  members dropped for lack of a graph node / feature vector.
    """

    value: float
    n_used: int
    n_excluded: int


def _resolve_indices(
    author_ids: Sequence[str], index: Mapping[str, int]
) -> tuple[np.ndarray, int, int]:
    """Map U-hat_i to matrix rows, dropping unknown authors.

    Returns ``(rows, n_unique, n_excluded)``. Duplicates are collapsed first:
    U-hat_i is a *set* of authors, so a name repeated within a response must not
    inflate n.
    """
    unique = {str(a) for a in author_ids if a is not None and str(a) != ""}
    rows = np.array(sorted(index[a] for a in unique if a in index), dtype=np.int64)
    return rows, len(unique), len(unique) - len(rows)


def compute_connectedness(
    author_ids: Sequence[str],
    adjacency: sparse.spmatrix,
    index: Mapping[str, int],
    *,
    logger: logging.Logger | None = None,
) -> StructuralMetric:
    """Connectedness of a response's factual authors (paper Eqs. 6-7).

    Takes the induced subgraph G-hat_i = G[U-hat_i], splits it into connected
    components {C_1..C_m} of sizes s_c, and with n = |U-hat_i| computes

        NormEntropy_i  = -(1 / log n) * sum_c (s_c/n) * log(s_c/n)   (Eq. 6)
        Connectedness_i = 1 - NormEntropy_i                          (Eq. 7)

    which lands in [0, 1]: a single component spanning all n authors gives 1.0
    (zero component entropy), n isolated singletons give 0.0 (maximal entropy).

    ``np.nan`` is returned when the metric is undefined: n == 0 (no factual
    recommendation) or n == 1 (log n = 0, so Eq. 6 has no value — no default is
    invented). Authors absent from ``index`` are excluded and counted.

    Parameters
    ----------
    author_ids : the members of U-hat_i, as ``author_id`` strings.
    adjacency  : symmetric CSR adjacency of the cached coauthorship graph.
    index      : ``author_id -> row`` mapping that goes with ``adjacency``.
    """
    log = logger or logging.getLogger()
    rows, n_unique, n_excluded = _resolve_indices(author_ids, index)
    n = len(rows)

    if n_excluded:
        log.debug(
            "connectedness: %d/%d authors not in the coauthorship graph",
            n_excluded,
            n_unique,
        )
    if n == 0:
        log.debug("connectedness: undefined, n == 0 (no factual author in the graph)")
        return StructuralMetric(np.nan, 0, n_excluded)
    if n == 1:
        log.debug("connectedness: undefined, n == 1 (log n = 0)")
        return StructuralMetric(np.nan, 1, n_excluded)

    subgraph = adjacency[rows, :][:, rows]
    _, labels = connected_components(subgraph, directed=False, return_labels=True)
    sizes = np.bincount(labels)

    p = sizes / n
    norm_entropy = -np.sum(p * np.log(p)) / np.log(n)
    # Clip absorbs float error only; the maths already bounds this to [0, 1].
    value = float(np.clip(1.0 - norm_entropy, 0.0, 1.0))
    log.debug("connectedness: n=%d, %d component(s), value=%.4f", n, len(sizes), value)
    return StructuralMetric(value, n, n_excluded)
