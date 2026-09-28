"""Tests for the per-call match rates in build_valid_calls.py.

The rule these pin down: a call whose evaluable recommendations all missed
scores 0.0, not NaN. Counting only the matches and dividing drops those calls
out of the groupby, which restricts the metric to calls that hit at least once
— that bug put bias_location for Ecuador at 0.454 against a true 0.040, and
discarded 221,299 of 437,638 scoreable calls.

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_valid_calls_metrics.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "metrics"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location("bvc", _SCRIPTS / "build_valid_calls.py")
bvc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bvc)

MATCH, MISS = "location_match", "location_mismatch"


def _eligible(pairs):
    """pairs: (call id, status) rows the metric can be scored on."""
    return pd.DataFrame(
        {"_cid": [c for c, _ in pairs], "location_status": [s for _, s in pairs]}
    )


def rate(pairs, calls=(1, 2, 3)):
    return bvc.match_rate_per_call(
        _eligible(pairs), "location_status", MATCH, pd.Index(calls, name="_cid")
    )


def test_a_call_that_missed_everything_scores_zero():
    """The regression this file exists for."""
    out = rate([(1, MISS), (1, MISS), (1, MISS)])
    assert out[1] == 0.0
    assert not np.isnan(out[1])


def test_a_call_with_nothing_evaluable_stays_nan():
    """No evidence is not the same as a miss."""
    assert np.isnan(rate([(1, MISS)])[2])


def test_partial_match_is_the_plain_fraction():
    out = rate([(1, MATCH), (1, MISS), (1, MISS), (1, MATCH)])
    assert out[1] == pytest.approx(0.5)


def test_a_call_that_hit_everything_scores_one():
    assert rate([(1, MATCH), (1, MATCH)])[1] == 1.0


def test_calls_are_scored_independently():
    out = rate([(1, MATCH), (2, MISS), (2, MISS), (3, MATCH), (3, MISS)])
    assert out[1] == 1.0
    assert out[2] == 0.0
    assert out[3] == pytest.approx(0.5)


def test_the_mean_includes_the_zero_scoring_calls():
    """What the bug changed: the average over calls, not over lucky calls."""
    out = rate([(1, MATCH), (2, MISS), (3, MISS)])
    assert out.mean() == pytest.approx(1 / 3)
    # The buggy version averaged only call 1 and reported 1.0.
    assert out.dropna().mean() != 1.0


def test_every_call_in_the_index_is_returned():
    out = rate([(1, MATCH)], calls=(1, 2, 3, 4))
    assert list(out.index) == [1, 2, 3, 4]


def test_empty_input_gives_all_nan():
    out = bvc.match_rate_per_call(
        _eligible([]).astype({"_cid": int}), "location_status", MATCH,
        pd.Index([1, 2], name="_cid"),
    )
    assert out.isna().all()


# ── author_uid ────────────────────────────────────────────────────────────────
# The OpenAlex id is missing for 37% of recommendations. De-duplicating on it
# alone merges every unresolved author of one response into one, which the
# counts read as the model repeating itself: `duplicates` came out at 0.159
# against a true 0.002.


def _recs(rows):
    return pd.DataFrame(rows, columns=["author_id", "researcher_id", "name", "lastname"])


def test_unresolved_authors_stay_distinct():
    """Two different people OpenAlex does not know are two people."""
    df = _recs([[None, None, "Ada", "Lovelace"], [None, None, "Alan", "Turing"]])
    assert len(set(bvc.author_uid(df))) == 2


def test_the_openalex_id_wins_when_present():
    df = _recs([["A1", 7.0, "Ada", "Lovelace"], ["A1", 9.0, "A.", "Lovelace"]])
    assert len(set(bvc.author_uid(df))) == 1


def test_semantic_scholar_id_is_the_first_fallback():
    """Same person, matched in SS only, written two ways — still one author."""
    df = _recs([[None, 7.0, "Ada Byron", "Lovelace"], [None, 7.0, "Ada", "Lovelace"]])
    assert len(set(bvc.author_uid(df))) == 1


def test_the_same_name_twice_is_one_author():
    """A genuine repeat must still count as a duplicate."""
    df = _recs([[None, None, "Ada", "Lovelace"], [None, None, "ADA ", "lovelace"]])
    assert len(set(bvc.author_uid(df))) == 1


def test_a_mixed_response_counts_every_distinct_person():
    df = _recs([
        ["A1", None, "Ada", "Lovelace"],      # OpenAlex
        ["A2", None, "Alan", "Turing"],       # OpenAlex
        [None, 7.0, "Grace", "Hopper"],       # Semantic Scholar only
        [None, None, "Someone", "Invented"],  # neither
        [None, None, "Another", "Invention"], # neither
    ])
    assert len(set(bvc.author_uid(df))) == 5
