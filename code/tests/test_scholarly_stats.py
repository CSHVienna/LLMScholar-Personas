"""h-index / i10-index / e-index and the extended similarity feature vector.

The e-index reference values are computed here with a direct transcription of
LLMScholarBench's compute_e_index (GTBuilder/APS/code/libs/scholar.py), so the
vectorised implementation is checked against the definition it claims to follow
rather than against numbers baked in by hand.
"""

import math

import numpy as np
import pandas as pd
import pytest

from libs.metrics import constants
from libs.metrics.io import _scholarly_stats_from_citations, build_author_features


def reference_e_index(citations):
    """LLMScholarBench's formula, transcribed literally: e = -1/N * sum c_i
    log(c_i / c_total). Papers with no citations drop out of the sum."""
    n = len(citations)
    c_total = sum(citations)
    if n == 0 or c_total == 0:
        return 0.0
    return -1 / n * sum(c * math.log(c / c_total) for c in citations if c > 0)


def stats_for(per_author):
    """Run the vectorised implementation over {row: [citations, ...]}."""
    idx, cits = [], []
    for row, citations in per_author.items():
        idx.extend([row] * len(citations))
        cits.extend(citations)
    return _scholarly_stats_from_citations(
        np.array(idx, dtype=np.int64),
        np.array(cits, dtype=np.int64),
        n_authors=max(per_author) + 1,
    )


# ── h-index ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "citations,expected",
    [
        ([], 0),
        ([0, 0, 0], 0),
        ([1], 1),
        ([100], 1),  # one paper can never yield h > 1
        ([5, 5, 5, 5, 5], 5),
        ([10, 8, 5, 4, 3], 4),  # classic worked example
        ([3, 3, 3], 3),
        ([2, 2, 2, 2], 2),
    ],
)
def test_h_index(citations, expected):
    if not citations:
        pytest.skip("empty author never reaches the pair array")
    assert int(stats_for({0: citations}).loc[0, "h_index"]) == expected


def test_h_index_is_order_independent():
    """Input order must not matter — the implementation sorts internally."""
    a = stats_for({0: [10, 8, 5, 4, 3]}).loc[0, "h_index"]
    b = stats_for({0: [3, 4, 5, 8, 10]}).loc[0, "h_index"]
    c = stats_for({0: [5, 3, 10, 4, 8]}).loc[0, "h_index"]
    assert a == b == c == 4


# ── i10-index ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "citations,expected",
    [
        ([9, 9, 9], 0),
        ([10, 9, 11], 2),  # boundary: 10 counts, 9 does not
        ([100, 50, 10, 10], 4),
    ],
)
def test_i10_index(citations, expected):
    assert int(stats_for({0: citations}).loc[0, "i10_index"]) == expected


# ── e-index ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "citations",
    [
        [10, 8, 5, 4, 3],
        [100, 1, 1],
        [7, 7, 7, 7],
        [0, 0, 50],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ],
)
def test_e_index_matches_reference_formula(citations):
    got = stats_for({0: citations}).loc[0, "e_index"]
    assert got == pytest.approx(reference_e_index(citations), rel=1e-9)


def test_e_index_zero_when_nothing_is_cited():
    """c_total == 0: every term drops out, so the entropy is 0, not NaN."""
    got = stats_for({0: [0, 0, 0]}).loc[0, "e_index"]
    assert got == pytest.approx(0.0)
    assert not np.isnan(got)


def test_e_index_rewards_spread_over_concentration():
    """The whole point of the metric: same citation total, different shape.

    An author whose citations sit on one paper is less 'spread' than one whose
    citations are even, and the entropy must order them that way.
    """
    concentrated = stats_for({0: [90, 5, 5]}).loc[0, "e_index"]
    even = stats_for({0: [33, 33, 34]}).loc[0, "e_index"]
    assert even > concentrated


# ── grouping ─────────────────────────────────────────────────────────────────


def test_authors_are_computed_independently():
    """Several authors in one array must not bleed into each other."""
    out = stats_for({0: [10, 8, 5, 4, 3], 1: [1], 2: [7, 7, 7, 7]})
    assert int(out.loc[0, "h_index"]) == 4
    assert int(out.loc[1, "h_index"]) == 1
    assert int(out.loc[2, "h_index"]) == 4
    assert out.loc[0, "e_index"] == pytest.approx(reference_e_index([10, 8, 5, 4, 3]))
    assert out.loc[2, "e_index"] == pytest.approx(reference_e_index([7, 7, 7, 7]))


def test_absent_authors_are_nan_not_zero():
    """An author with no rows in the snapshot is unknown, not a zero-impact
    author — median imputation must be able to tell them apart."""
    out = _scholarly_stats_from_citations(
        np.array([0, 0], dtype=np.int64),
        np.array([5, 3], dtype=np.int64),
        n_authors=3,
    )
    assert out.loc[0].notna().all()
    assert out.loc[1].isna().all()
    assert out.loc[2].isna().all()


def test_duplicate_papers_are_the_callers_problem():
    """The pair array is deduplicated in SQL, so a repeated paper here is
    counted twice — pinned so the SQL DISTINCT is never quietly dropped."""
    once = stats_for({0: [5, 5, 5]}).loc[0, "h_index"]
    twice = stats_for({0: [5, 5, 5, 5, 5, 5]}).loc[0, "h_index"]
    assert once == 3
    assert twice == 5


# ── feature vector ───────────────────────────────────────────────────────────


def make_authors():
    return pd.DataFrame(
        {
            "author_id": ["A1", "A2"],
            "oa_works_count": [10, 4],
            "oa_cited_by_count": [100, 8],
            "oa_career_age": [5, 2],
        }
    )


def test_base_vector_without_stats():
    features = build_author_features(make_authors())
    assert list(features.columns) == constants.SIMILARITY_FEATURE_COLS
    assert "h_index" not in features.columns


def test_citations_per_paper_age_definition():
    """(cited_by_count / works_count) / career_age, per LLMScholarBench."""
    features = build_author_features(make_authors())
    assert features.loc["A1", "citations_per_paper_age"] == pytest.approx(
        (100 / 10) / 5
    )
    assert features.loc["A2", "citations_per_paper_age"] == pytest.approx((8 / 4) / 2)


def test_career_age_zero_does_not_divide_by_zero():
    df = make_authors()
    df["oa_career_age"] = [0, 0]
    features = build_author_features(df)
    assert np.isfinite(features["citations_per_paper_age"]).all()
    assert np.isfinite(features["works_per_year"]).all()


def test_full_vector_with_stats():
    stats = pd.DataFrame(
        {"h_index": [7, 2], "i10_index": [5, 0], "e_index": [1.5, 0.3]},
        index=["A1", "A2"],
    )
    features = build_author_features(make_authors(), stats=stats)
    assert list(features.columns) == constants.SIMILARITY_FEATURE_COLS_FULL
    assert features.loc["A1", "h_index"] == 7
    assert features.loc["A2", "e_index"] == pytest.approx(0.3)


def test_authors_missing_from_stats_become_nan():
    """A2 has no stats row: NaN, so the imputer handles it — never a silent 0."""
    stats = pd.DataFrame(
        {"h_index": [7], "i10_index": [5], "e_index": [1.5]}, index=["A1"]
    )
    features = build_author_features(make_authors(), stats=stats)
    assert features.loc["A1", "h_index"] == 7
    assert features.loc[["A2"], constants.SIMILARITY_STATS_COLS].isna().all().all()


def test_full_vector_extends_base_without_reordering():
    """Downstream code slices by name, but a reordered base would still silently
    change the PCA input column order — pin it."""
    assert (
        constants.SIMILARITY_FEATURE_COLS_FULL[: len(constants.SIMILARITY_FEATURE_COLS)]
        == constants.SIMILARITY_FEATURE_COLS
    )


# ── The extended vector must actually reach the PCA ──────────────────────────


def test_pca_consumes_every_column_the_features_frame_carries():
    """Regression: build_similarity_embeddings used to slice the matrix with a
    hardcoded SIMILARITY_FEATURE_COLS, so the three snapshot-derived features
    were dropped on the way in — the pass that computes them ran, and the
    embedding ignored its output."""
    import numpy as np

    from libs.metrics.io import build_similarity_embeddings

    rng = np.random.default_rng(0)
    n = 60
    ids = [f"A{i}" for i in range(n)]
    base = {c: rng.integers(1, 500, n) for c in constants.SIMILARITY_FEATURE_COLS}
    stats = {c: rng.integers(1, 90, n) for c in constants.SIMILARITY_STATS_COLS}
    full = pd.DataFrame({**base, **stats}, index=ids)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        _, _, pipe_full = build_similarity_embeddings(
            full, cache_path=Path(tmp) / "full.joblib"
        )
        _, _, pipe_base = build_similarity_embeddings(
            full[constants.SIMILARITY_FEATURE_COLS], cache_path=Path(tmp) / "base.joblib"
        )

    assert pipe_full.named_steps["impute"].n_features_in_ == len(
        constants.SIMILARITY_FEATURE_COLS_FULL
    )
    assert pipe_base.named_steps["impute"].n_features_in_ == len(
        constants.SIMILARITY_FEATURE_COLS
    )


def test_missing_base_column_is_an_error_not_a_silent_drop():
    import tempfile
    from pathlib import Path

    from libs.metrics.io import build_similarity_embeddings

    incomplete = pd.DataFrame(
        {"works_count": [1.0, 2.0], "cited_by_count": [3.0, 4.0]}, index=["A1", "A2"]
    )
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="missing base columns"):
            build_similarity_embeddings(
                incomplete, cache_path=Path(tmp) / "x.joblib"
            )
