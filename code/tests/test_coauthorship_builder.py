"""Tests for the cached coauthorship-graph builder.

Builds a tiny synthetic DuckDB whose `works` table mirrors the shape of the real
OpenAlex snapshot (`authorships` as a list of structs with `author.id` as an
OpenAlex URL), so the SQL is exercised without touching the 492M-row table.

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_coauthorship_builder.py -v
"""

import numpy as np
import pytest

from libs.metrics.io import build_coauthorship_graph

duckdb = pytest.importorskip("duckdb")


def make_snapshot(path, works):
    """Write a DuckDB file with a `works` table shaped like the OA snapshot.

    `works` is ``{work_id: [author_num, ...]}``.
    """
    con = duckdb.connect(str(path))
    try:
        # OR REPLACE so a test can overwrite an existing snapshot in place.
        con.execute(
            "CREATE OR REPLACE TABLE works "
            "(id BIGINT, authorships STRUCT(author STRUCT(id VARCHAR))[])"
        )
        for work_id, authors in works.items():
            listing = ", ".join(
                "{'author': {'id': 'https://openalex.org/A%d'}}" % a for a in authors
            )
            con.execute(f"INSERT INTO works VALUES ({work_id}, [{listing}])")
    finally:
        con.close()


def edge_set(adjacency, index):
    """Undirected edges as a set of frozensets of author_ids."""
    reverse = {i: a for a, i in index.items()}
    coo = adjacency.tocoo()
    return {
        frozenset((reverse[r], reverse[c])) for r, c in zip(coo.row, coo.col) if r != c
    }


def test_builds_edges_from_shared_works(tmp_path):
    db = tmp_path / "oa.duckdb"
    # Work 1: authors 100, 200, 300 → triangle. Work 2: 300, 400 → one edge.
    make_snapshot(db, {1: [100, 200, 300], 2: [300, 400]})

    adjacency, index = build_coauthorship_graph(
        ["A100", "A200", "A300", "A400"],
        cache_path=tmp_path / "graph.joblib",
        oa_duckdb_path=db,
    )

    assert edge_set(adjacency, index) == {
        frozenset(("A100", "A200")),
        frozenset(("A100", "A300")),
        frozenset(("A200", "A300")),
        frozenset(("A300", "A400")),
    }


def test_graph_is_symmetric_and_loop_free(tmp_path):
    db = tmp_path / "oa.duckdb"
    make_snapshot(db, {1: [100, 200], 2: [200, 300]})

    adjacency, _ = build_coauthorship_graph(
        ["A100", "A200", "A300"],
        cache_path=tmp_path / "graph.joblib",
        oa_duckdb_path=db,
    )

    dense = adjacency.toarray()
    assert np.array_equal(dense, dense.T), "adjacency must be symmetric"
    assert not dense.diagonal().any(), "no self-loops"
    assert adjacency.dtype == bool


def test_authors_outside_the_population_are_ignored(tmp_path):
    """A work's other coauthors must not add nodes outside `author_ids`."""
    db = tmp_path / "oa.duckdb"
    # 999 co-authors with 100 but is not part of the requested population.
    make_snapshot(db, {1: [100, 999], 2: [100, 200]})

    adjacency, index = build_coauthorship_graph(
        ["A100", "A200"],
        cache_path=tmp_path / "graph.joblib",
        oa_duckdb_path=db,
    )

    assert set(index) == {"A100", "A200"}
    assert adjacency.shape == (2, 2)
    assert edge_set(adjacency, index) == {frozenset(("A100", "A200"))}


def test_repeated_coauthorship_yields_a_single_edge(tmp_path):
    """Two authors on three shared works still get one unweighted edge."""
    db = tmp_path / "oa.duckdb"
    make_snapshot(db, {1: [100, 200], 2: [100, 200], 3: [100, 200]})

    adjacency, index = build_coauthorship_graph(
        ["A100", "A200"],
        cache_path=tmp_path / "graph.joblib",
        oa_duckdb_path=db,
    )

    assert adjacency.nnz == 2  # one undirected edge, stored both ways
    assert edge_set(adjacency, index) == {frozenset(("A100", "A200"))}


def test_single_author_works_produce_no_edges(tmp_path):
    db = tmp_path / "oa.duckdb"
    make_snapshot(db, {1: [100], 2: [200]})

    adjacency, index = build_coauthorship_graph(
        ["A100", "A200"],
        cache_path=tmp_path / "graph.joblib",
        oa_duckdb_path=db,
    )

    assert adjacency.nnz == 0
    assert adjacency.shape == (2, 2)


def test_cache_is_reused_and_survives_the_snapshot_disappearing(tmp_path):
    """Second call must hit the joblib cache, not the database."""
    db = tmp_path / "oa.duckdb"
    cache = tmp_path / "graph.joblib"
    make_snapshot(db, {1: [100, 200]})

    first_adj, first_index = build_coauthorship_graph(
        ["A100", "A200"], cache_path=cache, oa_duckdb_path=db
    )
    assert cache.exists()

    # Point at a path that does not exist: only a cache hit can succeed.
    second_adj, second_index = build_coauthorship_graph(
        ["A100", "A200"], cache_path=cache, oa_duckdb_path=tmp_path / "gone.duckdb"
    )

    assert second_index == first_index
    assert (second_adj != first_adj).nnz == 0


@pytest.mark.parametrize("n_chunks", [1, 2, 5, 64])
def test_chunking_does_not_change_the_graph(tmp_path, n_chunks):
    """Chunking bounds memory only — the edge set must be chunk-invariant.

    A work lies wholly inside one works.id range, so no edge can be split; the
    same pair reappearing in several chunks must be deduplicated globally.
    """
    db = tmp_path / "oa.duckdb"
    make_snapshot(
        db,
        {
            1: [100, 200, 300],
            2: [300, 400],
            5: [100, 400],
            9: [200, 400],
            12: [100, 200],  # repeats the 100-200 edge from work 1
        },
    )

    adjacency, index = build_coauthorship_graph(
        ["A100", "A200", "A300", "A400"],
        cache_path=tmp_path / f"graph_{n_chunks}.joblib",
        oa_duckdb_path=db,
        n_chunks=n_chunks,
    )

    assert edge_set(adjacency, index) == {
        frozenset(("A100", "A200")),
        frozenset(("A100", "A300")),
        frozenset(("A200", "A300")),
        frozenset(("A300", "A400")),
        frozenset(("A100", "A400")),
        frozenset(("A200", "A400")),
    }
    # Deduplicated: 6 undirected edges stored in both directions.
    assert adjacency.nnz == 12


def test_cache_is_invalidated_when_the_population_changes(tmp_path):
    """A cache built for other authors must not be reused silently.

    Reusing it would return an index missing the new authors, which downstream
    shows up as unexplained exclusions rather than an error.
    """
    db = tmp_path / "oa.duckdb"
    cache = tmp_path / "graph.joblib"
    make_snapshot(db, {1: [100, 200], 2: [200, 300]})

    _, small_index = build_coauthorship_graph(
        ["A100", "A200"], cache_path=cache, oa_duckdb_path=db
    )
    assert set(small_index) == {"A100", "A200"}

    # Same cache path, larger population: must rebuild, not reuse.
    adjacency, big_index = build_coauthorship_graph(
        ["A100", "A200", "A300"], cache_path=cache, oa_duckdb_path=db
    )
    assert set(big_index) == {"A100", "A200", "A300"}
    assert frozenset(("A200", "A300")) in edge_set(adjacency, big_index)


def test_rebuild_flag_bypasses_the_cache(tmp_path):
    db = tmp_path / "oa.duckdb"
    cache = tmp_path / "graph.joblib"
    make_snapshot(db, {1: [100, 200]})
    build_coauthorship_graph(["A100", "A200"], cache_path=cache, oa_duckdb_path=db)

    # Same population, richer snapshot: only a rebuild picks up the new edge.
    make_snapshot(db, {1: [100, 200], 2: [200, 300]})
    adjacency, index = build_coauthorship_graph(
        ["A100", "A200", "A300"],
        cache_path=cache,
        oa_duckdb_path=db,
        rebuild=True,
    )

    assert frozenset(("A200", "A300")) in edge_set(adjacency, index)
