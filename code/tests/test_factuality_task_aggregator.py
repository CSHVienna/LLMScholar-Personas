"""Tests for aggregate_factuality_task.

The function was dead and unrunnable: it read a constant
(BENCHMARK_FACTUALITY_FIELD_METRICS_MAP) deleted along with constants_old.py in
789bbb0, whose values pointed at a retired column schema. It now resolves the
status column through constants.FACTUALITY_TASK_STATUS_COLS.

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_factuality_task_aggregator.py -v
"""

import pandas as pd
import pytest

from libs.metrics import constants
from libs.metrics.aggregators import aggregate_factuality_task

ATTEMPT_KEY = {
    "model_access": "open",
    "model_size": "M",
    "model_class": "instruct",
    "model": "llama3",
    "grounded": False,
    "temperature": 0.0,
    "date": "2026-01-01",
    "time": "10:00",
    "task_name": "recommend",
    "task_param": "biology",
    "task_attempt": 1,
}


def records(status_col: str, statuses: list[str]) -> pd.DataFrame:
    """One row per author, all within the same attempt."""
    return pd.DataFrame(
        [
            {**ATTEMPT_KEY, "clean_name": f"author {i}", status_col: s}
            for i, s in enumerate(statuses)
        ]
    )


# ── The map the function depends on ───────────────────────────────────────────


def test_status_column_map_covers_every_task_metric():
    assert constants.FACTUALITY_TASK_STATUS_COLS == {
        "factuality_field": "field_status",
        "factuality_seniority": "seniority_status",
        "bias_location": "location_status",
    }


def test_unknown_metric_raises_keyerror():
    with pytest.raises(KeyError):
        aggregate_factuality_task(records("field_status", ["field_match"]), "validity")


# ── Match rate ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "metric,status_col",
    list(constants.FACTUALITY_TASK_STATUS_COLS.items()),
)
def test_runs_for_every_metric(metric, status_col):
    prefix = status_col.split("_")[0]
    df = records(status_col, [f"{prefix}_match", f"{prefix}_mismatch"])
    out = aggregate_factuality_task(df, metric)
    assert out["metric"].tolist() == [0.5]


def test_match_rate_ignores_unknown_and_not_applicable():
    # 2 match + 1 mismatch = 0.667; the unknown / not_applicable rows are not
    # evaluable and must not be counted as failures.
    df = records(
        "field_status",
        [
            "field_match",
            "field_match",
            "field_mismatch",
            "field_unknown",
            "not_applicable",
        ],
    )
    out = aggregate_factuality_task(df, "factuality_field")
    assert out["metric"].tolist() == pytest.approx([2 / 3])


def test_all_match_is_one_and_all_mismatch_is_zero():
    ones = aggregate_factuality_task(
        records("location_status", ["location_match"] * 3), "bias_location"
    )
    zeros = aggregate_factuality_task(
        records("location_status", ["location_mismatch"] * 3), "bias_location"
    )
    assert ones["metric"].tolist() == [1.0]
    assert zeros["metric"].tolist() == [0.0]


def test_duplicate_authors_within_an_attempt_are_collapsed():
    df = records("field_status", ["field_match", "field_mismatch"])
    doubled = pd.concat([df, df], ignore_index=True)
    assert aggregate_factuality_task(doubled, "factuality_field")["metric"].tolist() == (
        aggregate_factuality_task(df, "factuality_field")["metric"].tolist()
    )


def test_attempts_are_aggregated_separately():
    first = records("field_status", ["field_match", "field_match"])
    second = records("field_status", ["field_mismatch", "field_mismatch"])
    second["task_attempt"] = 2
    out = aggregate_factuality_task(
        pd.concat([first, second], ignore_index=True), "factuality_field"
    )
    assert sorted(out["metric"].tolist()) == [0.0, 1.0]
    assert len(out) == 2
