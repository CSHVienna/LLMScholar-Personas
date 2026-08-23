"""End-to-end check of build_author_scholarly_stats against a synthetic snapshot.

The unit tests in test_scholarly_stats.py cover the maths on a pair array; what
they cannot cover is the half that only exists as SQL — the UNNEST of
``authorships``, the join back onto the author population, the quantile chunking
and the cache. A miniature ``oa.works`` exercises all of that in milliseconds,
so a typo in the query surfaces here instead of three hours into a real run.
"""

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from libs.metrics import constants  # noqa: E402
from libs.metrics.io import (  # noqa: E402
    build_author_features,
    build_author_scholarly_stats,
)

# A1: 5 papers, citations 10/8/5/4/3 → h=4, i10=1
# A2: 2 papers, 50 + the shared 8    → h=2, i10=1
# A3: 3 papers, 0 citations each     → h=0, i10=0, e=0
# A9: never appears in the snapshot  → all NaN
WORKS = [
    (1, 10, ["A1"]),
    (2, 8, ["A1", "A2"]),  # shared paper: also the coauthorship case
    (3, 5, ["A1"]),
    (4, 4, ["A1"]),
    (5, 3, ["A1"]),
    (6, 50, ["A2"]),
    (7, 0, ["A3"]),
    (8, 0, ["A3"]),
    (9, 0, ["A3"]),
    (10, 999, ["ZZ_outside_population"]),
]


@pytest.fixture
def snapshot(tmp_path):
    """A miniature OpenAlex snapshot with the columns the query actually reads."""
    path = tmp_path / "oa_mini.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE works ("
        "  id BIGINT,"
        "  cited_by_count BIGINT,"
        "  authorships STRUCT(author STRUCT(id VARCHAR))[]"
        ")"
    )
    for work_id, cites, authors in WORKS:
        payload = ", ".join(
            "{'author': {'id': 'https://openalex.org/" + a + "'}}" for a in authors
        )
        con.execute(f"INSERT INTO works VALUES ({work_id}, {cites}, [{payload}])")
    con.close()
    return path


@pytest.fixture
def populated(snapshot):
    return snapshot


def run_stats(populated, tmp_path, ids=("A1", "A2", "A3", "A9"), **kw):
    return build_author_scholarly_stats(
        list(ids),
        cache_path=tmp_path / "stats.joblib",
        oa_duckdb_path=populated,
        memory_limit="1GB",
        **kw,
    )


def test_stats_match_hand_computed_values(populated, tmp_path):
    stats = run_stats(populated, tmp_path)

    assert stats.loc["A1", "h_index"] == 4
    assert stats.loc["A1", "i10_index"] == 1  # only the 10-citation paper
    assert stats.loc["A2", "h_index"] == 2  # [50, 8]: both clear their rank
    assert stats.loc["A2", "i10_index"] == 1  # 50 clears 10, the shared 8 does not
    assert stats.loc["A3", "h_index"] == 0
    assert stats.loc["A3", "e_index"] == pytest.approx(0.0)


def test_author_outside_the_snapshot_is_nan(populated, tmp_path):
    stats = run_stats(populated, tmp_path)
    assert stats.loc["A9"].isna().all()


def test_authors_outside_the_population_are_ignored(populated, tmp_path):
    """ZZ_outside_population has 999 citations but was never asked for."""
    stats = run_stats(populated, tmp_path)
    assert "ZZ_outside_population" not in stats.index
    assert stats["h_index"].max() <= 4


def test_shared_paper_counts_for_both_authors(populated, tmp_path):
    """Work 2 is A1's and A2's — both must see its 8 citations."""
    stats = run_stats(populated, tmp_path)
    # A2 has papers [8, 50]: h=2 requires two papers with >=2 citations. Both
    # qualify, so h=2 — which is only true if the shared paper was counted.
    assert stats.loc["A2", "h_index"] == 2


def test_chunking_does_not_change_the_result(populated, tmp_path):
    """Chunk boundaries split works, never an author's paper set."""
    one = run_stats(populated, tmp_path, n_chunks=1)
    many = run_stats(populated, tmp_path / "b", n_chunks=8)
    pd.testing.assert_frame_equal(one, many)


def test_cache_is_reused_and_keyed_on_population(populated, tmp_path):
    first = run_stats(populated, tmp_path)
    cache = tmp_path / "stats.joblib"
    assert cache.exists()

    # Same population → served from cache even if the snapshot vanishes.
    populated.unlink()
    again = build_author_scholarly_stats(
        ["A1", "A2", "A3", "A9"],
        cache_path=cache,
        oa_duckdb_path=populated,
        memory_limit="1GB",
    )
    pd.testing.assert_frame_equal(first, again)


def test_feature_vector_carries_the_stats_through(populated, tmp_path):
    stats = run_stats(populated, tmp_path)
    authors = pd.DataFrame(
        {
            "author_id": ["A1", "A2"],
            "oa_works_count": [5, 2],
            "oa_cited_by_count": [30, 58],
            "oa_career_age": [10, 4],
        }
    )
    features = build_author_features(authors, stats=stats)
    assert list(features.columns) == constants.SIMILARITY_FEATURE_COLS_FULL
    assert features.loc["A1", "h_index"] == 4
    assert np.isfinite(features.loc["A1", "e_index"])
