"""Tests for the structural benchmark metrics (paper Eqs. 6-8).

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_structural_metrics.py -v
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from libs.metrics.aggregators import compute_connectedness, compute_similarity
from libs.metrics.io import build_author_features, build_author_index

# ── Helpers ───────────────────────────────────────────────────────────────────


def graph_from_edges(n_nodes: int, edges: list[tuple[int, int]]) -> sparse.csr_matrix:
    """Symmetric boolean CSR adjacency over `n_nodes`, as the cached builder emits."""
    if edges:
        src, dst = zip(*edges)
    else:
        src, dst = (), ()
    src, dst = np.array(src, dtype=int), np.array(dst, dtype=int)
    data = np.ones(2 * len(src), dtype=bool)
    adj = sparse.coo_matrix(
        (data, (np.concatenate([src, dst]), np.concatenate([dst, src]))),
        shape=(n_nodes, n_nodes),
        dtype=bool,
    ).tocsr()
    adj.setdiag(False)
    adj.eliminate_zeros()
    return adj


def index_of(n: int) -> dict[str, int]:
    return {f"A{i}": i for i in range(n)}


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row, as the similarity pipeline's final step does."""
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def brute_force_similarity(vectors: np.ndarray) -> float:
    """Mean pairwise cosine over unordered pairs, computed the O(n²·d) way."""
    n = len(vectors)
    total = sum(
        float(vectors[i] @ vectors[j]) for i in range(n) for j in range(i + 1, n)
    )
    return 2.0 * total / (n * (n - 1))


# ── Connectedness: the sanity checks the spec mandates ─────────────────────────


def test_single_component_gives_one():
    """One component spanning all n authors → zero entropy → connectedness 1.0."""
    # Path graph over 5 nodes: a single connected component.
    adj = graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    result = compute_connectedness([f"A{i}" for i in range(5)], adj, index_of(5))
    assert result.value == pytest.approx(1.0)
    assert (result.n_used, result.n_excluded) == (5, 0)


def test_all_singletons_gives_zero():
    """n isolated authors → maximal entropy → connectedness 0.0."""
    adj = graph_from_edges(4, [])
    result = compute_connectedness([f"A{i}" for i in range(4)], adj, index_of(4))
    assert result.value == pytest.approx(0.0)
    assert (result.n_used, result.n_excluded) == (4, 0)


def test_two_unequal_components():
    """Two components of sizes 3 and 1 → check against Eq. 6 by hand."""
    # A0-A1-A2 form a triangle-ish chain; A3 is isolated.
    adj = graph_from_edges(4, [(0, 1), (1, 2)])
    result = compute_connectedness([f"A{i}" for i in range(4)], adj, index_of(4))

    n = 4
    p = np.array([3 / n, 1 / n])
    expected = 1.0 - (-np.sum(p * np.log(p)) / np.log(n))
    assert result.value == pytest.approx(expected)
    # Strictly between the two extremes.
    assert 0.0 < result.value < 1.0


def test_intermediate_value_is_between_extremes():
    """Two equal components sit strictly below a single component."""
    adj = graph_from_edges(4, [(0, 1), (2, 3)])
    two_halves = compute_connectedness([f"A{i}" for i in range(4)], adj, index_of(4))
    # Two components of size 2 each: entropy = log 2, normalised by log 4 = 0.5.
    assert two_halves.value == pytest.approx(0.5)


def test_edges_outside_the_response_are_ignored():
    """G[U-hat_i] keeps only edges *between* members of U-hat_i."""
    # A0-A1 are linked only through A2, which is not in the response.
    adj = graph_from_edges(3, [(0, 2), (1, 2)])
    result = compute_connectedness(["A0", "A1"], adj, index_of(3))
    # Induced subgraph on {A0, A1} has no edge → two singletons → 0.0.
    assert result.value == pytest.approx(0.0)


# ── Similarity: closed form vs brute force ────────────────────────────────────


def test_closed_form_matches_brute_force_pairwise_cosine():
    """The O(n·d) identity must equal the O(n²·d) pairwise mean exactly."""
    rng = np.random.default_rng(0)
    embeddings = unit_rows(rng.normal(size=(6, 4)))
    result = compute_similarity([f"A{i}" for i in range(6)], embeddings, index_of(6))
    assert result.value == pytest.approx(brute_force_similarity(embeddings), abs=1e-12)


@pytest.mark.parametrize("n,d", [(2, 2), (3, 5), (7, 3), (12, 6), (25, 4)])
def test_closed_form_matches_brute_force_across_shapes(n, d):
    """Same equivalence across a range of n and embedding dimensions."""
    rng = np.random.default_rng(n * 100 + d)
    embeddings = unit_rows(rng.normal(size=(n, d)))
    result = compute_similarity([f"A{i}" for i in range(n)], embeddings, index_of(n))
    assert result.value == pytest.approx(brute_force_similarity(embeddings), abs=1e-12)


def test_identical_authors_give_similarity_one():
    """Collinear embeddings → every pairwise cosine is 1."""
    embeddings = unit_rows(np.tile(np.array([[3.0, 4.0]]), (4, 1)))
    result = compute_similarity([f"A{i}" for i in range(4)], embeddings, index_of(4))
    assert result.value == pytest.approx(1.0)


def test_collinear_embeddings_do_not_exceed_one():
    """Float error must not push Sim past 1.0.

    Regression: on real data 3 responses came out at 1.0000000000000004 because
    ||S||^2 on collinear unit vectors overshoots by ~4e-16.
    """
    for n in range(2, 12):
        for d in (2, 3, 5):
            vector = np.arange(1.0, d + 1.0)
            embeddings = unit_rows(np.tile(vector, (n, 1)))
            result = compute_similarity(
                [f"A{i}" for i in range(n)], embeddings, index_of(n)
            )
            assert result.value <= 1.0, (n, d, result.value)
            assert result.value == pytest.approx(1.0)


def test_antipodal_embeddings_do_not_fall_below_minus_one():
    """Symmetric guard at the lower bound."""
    embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]])
    result = compute_similarity(["A0", "A1"], embeddings, index_of(2))
    assert result.value >= -1.0
    assert result.value == pytest.approx(-1.0)


def test_n_used_is_the_formula_denominator():
    """n_used must be the n that entered the formula, not the size of the input.

    Callers need it: n_authors_found is not a valid denominator for these
    metrics, so n_used is what makes the result interpretable.
    """
    embeddings = unit_rows(np.random.default_rng(11).normal(size=(4, 3)))
    embeddings[1] = np.nan  # sin features

    # 5 ids de entrada: 4 conocidos (uno sin embedding) + 1 ausente del index.
    result = compute_similarity(
        ["A0", "A1", "A2", "A3", "A99"], embeddings, index_of(4)
    )
    assert result.n_used == 3
    assert result.n_excluded == 2  # A1 sin embedding + A99 ausente
    # Coincide con el coseno por pares de los tres que sí entraron.
    kept = embeddings[[0, 2, 3]]
    assert result.value == pytest.approx(brute_force_similarity(kept), abs=1e-12)


def test_antipodal_pair_gives_similarity_minus_one():
    """Opposite unit vectors → cosine -1. Sim is not bounded below by 0."""
    embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]])
    result = compute_similarity(["A0", "A1"], embeddings, index_of(2))
    assert result.value == pytest.approx(-1.0)


# ── Edge cases: NaN, never a silent 0.0 ───────────────────────────────────────


def test_n_zero_returns_nan_for_both_metrics():
    adj = graph_from_edges(3, [(0, 1)])
    embeddings = unit_rows(np.random.default_rng(1).normal(size=(3, 2)))

    conn = compute_connectedness([], adj, index_of(3))
    sim = compute_similarity([], embeddings, index_of(3))

    assert np.isnan(conn.value) and conn.n_used == 0
    assert np.isnan(sim.value) and sim.n_used == 0


def test_n_one_returns_nan_for_both_metrics():
    """n == 1: connectedness undefined (log n = 0), similarity undefined (no pair)."""
    adj = graph_from_edges(3, [(0, 1)])
    embeddings = unit_rows(np.random.default_rng(2).normal(size=(3, 2)))

    conn = compute_connectedness(["A0"], adj, index_of(3))
    sim = compute_similarity(["A0"], embeddings, index_of(3))

    assert np.isnan(conn.value) and conn.n_used == 1
    assert np.isnan(sim.value) and sim.n_used == 1


def test_duplicate_authors_are_collapsed():
    """U-hat_i is a set: a repeated author must not inflate n."""
    adj = graph_from_edges(3, [(0, 1)])
    result = compute_connectedness(["A0", "A0", "A1"], adj, index_of(3))
    assert result.n_used == 2
    # Both authors are linked → single component → 1.0.
    assert result.value == pytest.approx(1.0)


# ── Missing authors are excluded and counted ───────────────────────────────────


def test_authors_missing_from_the_graph_are_excluded_and_counted():
    """Authors with no node drop out; the count reaches the caller."""
    adj = graph_from_edges(3, [(0, 1), (1, 2)])
    # A9 / A7 are not in the index (e.g. matched only in Semantic Scholar).
    result = compute_connectedness(["A0", "A1", "A2", "A9", "A7"], adj, index_of(3))
    assert result.n_used == 3
    assert result.n_excluded == 2
    # The three remaining authors form one component.
    assert result.value == pytest.approx(1.0)


def test_authors_missing_features_are_excluded_and_counted():
    """NaN embedding rows (no features at all) are dropped, not imputed here."""
    embeddings = unit_rows(np.random.default_rng(3).normal(size=(4, 3)))
    embeddings[2] = np.nan  # author A2 has no feature vector

    result = compute_similarity([f"A{i}" for i in range(4)], embeddings, index_of(4))
    assert result.n_used == 3
    assert result.n_excluded == 1
    expected = brute_force_similarity(embeddings[[0, 1, 3]])
    assert result.value == pytest.approx(expected, abs=1e-12)


def test_all_authors_missing_degrades_to_nan_not_zero():
    """Everything excluded → NaN, and the exclusion count still reports."""
    adj = graph_from_edges(2, [(0, 1)])
    result = compute_connectedness(["A5", "A6", "A7"], adj, index_of(2))
    assert np.isnan(result.value)
    assert (result.n_used, result.n_excluded) == (0, 3)


def test_exclusions_can_push_n_down_to_one():
    """A response with 4 authors but only 1 in the graph is still undefined."""
    adj = graph_from_edges(3, [(0, 1)])
    result = compute_connectedness(["A0", "A8", "A9", "A10"], adj, index_of(3))
    assert np.isnan(result.value)
    assert (result.n_used, result.n_excluded) == (1, 3)


def test_zero_norm_embedding_row_is_excluded():
    """A degenerate zero vector would break the unit-norm identity."""
    embeddings = unit_rows(np.random.default_rng(4).normal(size=(3, 2)))
    embeddings_with_zero = np.vstack([embeddings, np.zeros((1, 2))])
    index = index_of(4)

    result = compute_similarity(
        [f"A{i}" for i in range(4)], embeddings_with_zero, index
    )
    assert result.n_used == 3
    assert result.n_excluded == 1
    assert result.value == pytest.approx(brute_force_similarity(embeddings), abs=1e-12)


# ── Feature derivation ────────────────────────────────────────────────────────


def test_build_author_features_derives_the_base_columns():
    df = pd.DataFrame(
        {
            "author_id": ["A1", "A2"],
            "oa_works_count": [10.0, 4.0],
            "oa_cited_by_count": [100.0, 0.0],
            "oa_career_age": [20.0, 0.0],
        }
    )
    features = build_author_features(df)

    # Without `stats` the vector stays on what factuality_full.csv can supply;
    # h_index / i10_index / e_index only appear once the snapshot pass has run.
    assert list(features.columns) == [
        "works_count",
        "cited_by_count",
        "citations_per_work",
        "career_age",
        "works_per_year",
        "citations_per_paper_age",
    ]
    assert features.loc["A1", "citations_per_work"] == pytest.approx(10.0)
    # career_age 0 must not divide by zero — clipped to 1.
    assert features.loc["A2", "works_per_year"] == pytest.approx(4.0)
    assert features.loc["A2", "citations_per_work"] == pytest.approx(0.0)


def test_build_author_features_deduplicates_authors():
    """The same author recommended in many responses yields one feature row."""
    df = pd.DataFrame(
        {
            "author_id": ["A1", "A1", "A2"],
            "oa_works_count": [10.0, 10.0, 4.0],
            "oa_cited_by_count": [100.0, 100.0, 8.0],
            "oa_career_age": [20.0, 20.0, 4.0],
        }
    )
    assert len(build_author_features(df)) == 2


def test_build_author_index_is_stable_and_sorted():
    index = build_author_index(["A3", "A1", "A2", "A1", None, ""])
    assert index == {"A1": 0, "A2": 1, "A3": 2}
    # Rebuilding from a different input order gives the same mapping.
    assert build_author_index(["A2", "A3", "A1"]) == index
