"""Tests for the cached scholarly-similarity PCA pipeline (paper Eq. 8).

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_similarity_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from libs.metrics import constants
from libs.metrics.aggregators import compute_similarity
from libs.metrics.io import (
    build_author_features,
    build_author_index,
    build_similarity_embeddings,
)


def population(n=500, seed=0, missing=0):
    """Synthetic author population with heavy-tailed counts, like OpenAlex."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "author_id": [f"A{i}" for i in range(n)],
            "oa_works_count": rng.lognormal(2.0, 1.2, n).round(),
            "oa_cited_by_count": rng.lognormal(4.0, 1.8, n).round(),
            "oa_career_age": rng.integers(0, 45, n).astype(float),
        }
    )
    if missing:
        cols = ["oa_works_count", "oa_cited_by_count", "oa_career_age"]
        df.loc[df.sample(missing, random_state=seed).index, cols] = np.nan
    return df


def brute_force_similarity(vectors):
    n = len(vectors)
    total = sum(
        float(vectors[i] @ vectors[j]) for i in range(n) for j in range(i + 1, n)
    )
    return 2.0 * total / (n * (n - 1))


def test_embeddings_are_unit_vectors(tmp_path):
    """Step 5 of the preprocessing order: L2 normalisation."""
    features = build_author_features(population())
    embeddings, _, _ = build_similarity_embeddings(
        features, cache_path=tmp_path / "emb.joblib"
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0)


def test_pca_retains_enough_variance(tmp_path):
    """Step 4: fewest components reaching >= 90% of the variance."""
    features = build_author_features(population())
    _, _, pipeline = build_similarity_embeddings(
        features, cache_path=tmp_path / "emb.joblib"
    )
    pca = pipeline.named_steps["pca"]
    explained = pca.explained_variance_ratio_
    assert explained.sum() >= constants.SIMILARITY_PCA_VARIANCE
    assert pca.n_components_ <= len(constants.SIMILARITY_FEATURE_COLS)
    # Minimality: dropping the last component must fall below the threshold.
    if pca.n_components_ > 1:
        assert explained[:-1].sum() < constants.SIMILARITY_PCA_VARIANCE


def test_pipeline_step_order_matches_the_spec(tmp_path):
    """The paper fixes the order: impute → log1p → scale → PCA → L2."""
    features = build_author_features(population())
    _, _, pipeline = build_similarity_embeddings(
        features, cache_path=tmp_path / "emb.joblib"
    )
    assert [name for name, _ in pipeline.steps] == [
        "impute",
        "log1p",
        "scale",
        "pca",
        "l2",
    ]


def test_authors_without_features_get_nan_rows(tmp_path):
    """Rows with no feature at all stay NaN so callers can exclude them."""
    features = build_author_features(population(n=300, missing=30))
    embeddings, index, _ = build_similarity_embeddings(
        features, cache_path=tmp_path / "emb.joblib"
    )
    assert int(np.isnan(embeddings).all(axis=1).sum()) == 30


def test_median_imputation_does_not_leak_into_all_missing_rows(tmp_path):
    """An author with *some* features is imputed; one with none is NaN."""
    df = population(n=200)
    # A0 keeps works_count but loses the rest → imputed, not excluded.
    df.loc[df.author_id == "A0", ["oa_cited_by_count", "oa_career_age"]] = np.nan
    # A1 loses everything → NaN row.
    df.loc[
        df.author_id == "A1",
        ["oa_works_count", "oa_cited_by_count", "oa_career_age"],
    ] = np.nan

    features = build_author_features(df)
    embeddings, index, _ = build_similarity_embeddings(
        features, cache_path=tmp_path / "emb.joblib"
    )
    assert np.isfinite(embeddings[index["A0"]]).all()
    assert np.isnan(embeddings[index["A1"]]).all()


def test_fit_is_reproducible_across_runs(tmp_path):
    """Fixed random_state + no mutable global state → identical embeddings."""
    features = build_author_features(population())
    first, _, _ = build_similarity_embeddings(
        features, cache_path=tmp_path / "a.joblib"
    )
    second, _, _ = build_similarity_embeddings(
        features, cache_path=tmp_path / "b.joblib"
    )
    np.testing.assert_allclose(np.nan_to_num(first), np.nan_to_num(second))


def test_cache_round_trip_preserves_embeddings(tmp_path):
    """The fitted pipeline must survive joblib (no lambdas inside)."""
    features = build_author_features(population())
    cache = tmp_path / "emb.joblib"
    first, first_index, _ = build_similarity_embeddings(features, cache_path=cache)
    second, second_index, pipeline = build_similarity_embeddings(
        features, cache_path=cache
    )

    np.testing.assert_allclose(np.nan_to_num(first), np.nan_to_num(second))
    assert second_index == first_index
    # The reloaded pipeline still transforms.
    assert pipeline.transform(features.head(3).to_numpy(dtype=float)).shape[0] == 3


def test_cache_is_refitted_when_features_change(tmp_path):
    """Same authors, different feature values → the fitted PCA must change."""
    cache = tmp_path / "emb.joblib"
    features = build_author_features(population(n=300, seed=1))
    first, _, _ = build_similarity_embeddings(features, cache_path=cache)

    shifted = build_author_features(population(n=300, seed=2))
    second, _, _ = build_similarity_embeddings(shifted, cache_path=cache)

    assert not np.allclose(np.nan_to_num(first), np.nan_to_num(second))


def test_end_to_end_closed_form_matches_brute_force(tmp_path):
    """Real embeddings from the real pipeline, both formulas must agree."""
    features = build_author_features(population(n=400, seed=5))
    embeddings, index, _ = build_similarity_embeddings(
        features, cache_path=tmp_path / "emb.joblib"
    )

    ids = [f"A{i}" for i in (3, 17, 42, 99, 250, 377)]
    result = compute_similarity(ids, embeddings, index)
    rows = np.array([embeddings[index[a]] for a in ids])
    assert result.value == pytest.approx(brute_force_similarity(rows), abs=1e-12)


def test_fitting_needs_at_least_two_authors(tmp_path):
    features = build_author_features(population(n=1))
    with pytest.raises(ValueError, match=">= 2 authors"):
        build_similarity_embeddings(features, cache_path=tmp_path / "emb.joblib")


def test_index_alignment_puts_each_author_on_its_own_row(tmp_path):
    """Embedding rows must follow `index`, not the feature frame's order."""
    df = population(n=50, seed=9)
    features = build_author_features(df)
    # Shuffle the feature frame: the index must still drive row placement.
    shuffled = features.sample(frac=1.0, random_state=3)
    index = build_author_index(df.author_id)

    from_ordered, _, _ = build_similarity_embeddings(
        features, cache_path=tmp_path / "a.joblib", index=index
    )
    from_shuffled, _, _ = build_similarity_embeddings(
        shuffled, cache_path=tmp_path / "b.joblib", index=index
    )
    np.testing.assert_allclose(from_ordered, from_shuffled)
