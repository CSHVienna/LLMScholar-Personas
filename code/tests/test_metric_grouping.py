"""Tests for the metric naming/grouping contract (issue #36).

Location is reported as a bias grouped with social representation, not as a
factuality. Artefacts predating the rename must still load.

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_metric_grouping.py -v
"""

import pandas as pd

from libs.metrics import constants
from libs.visuals import constants as vis_constants

OLD_NAME = "factuality_location"
NEW_NAME = "bias_location"


# ── Naming ────────────────────────────────────────────────────────────────────


def test_old_metric_name_is_gone_from_every_metric_list():
    for name in (
        "ALL_METRICS",
        "FACTUALITY_METRICS",
        "TECHNICAL_METRICS",
        "TECHNICAL_METRICS_NORM",
        "SOCIAL_METRICS",
    ):
        assert OLD_NAME not in getattr(constants, name), name


def test_new_metric_is_declared_once_in_all_metrics():
    assert constants.ALL_METRICS.count(NEW_NAME) == 1


def test_bias_prefix_has_a_tick_group_label():
    # Grouped ticks are resolved by prefix; without this the metric would land
    # in the 'Factuality' bracket or in none at all.
    assert constants.PREFIX_GROUPS_METRICS["bias_"] == "Bias"


# ── Grouping: social, not technical ───────────────────────────────────────────


def test_metric_is_social_not_technical():
    assert NEW_NAME in constants.SOCIAL_METRICS
    assert NEW_NAME not in constants.TECHNICAL_METRICS
    assert NEW_NAME in constants.BIAS_METRICS


def test_evaluation_metric_groups_follow_the_lists():
    groups = constants.EVALUATION_METRIC_GROUPS
    assert NEW_NAME in groups["social"]
    assert NEW_NAME not in groups["technical"]


def test_plot_grouping_matches_metric_grouping():
    assert NEW_NAME in vis_constants.PLOT_SOCIAL_METRICS
    assert NEW_NAME not in vis_constants.PLOT_TECHNICAL_METRICS
    assert NEW_NAME in vis_constants.METRIC_TYPES["social"]


def test_metric_is_plottable_and_labelled():
    # A metric absent from PLOT_LABELS silently renders with its raw column name.
    assert NEW_NAME in vis_constants.PLOT_METRICS
    assert NEW_NAME in vis_constants.PLOT_LABELS
    assert NEW_NAME in vis_constants.METRICS_NAME_MAP


def test_nested_denominator_is_preserved():
    # Renaming must not change what the metric is nested under: the denominator
    # is still the factual-records one.
    assert constants.NESTED_METRIC_PAIRS[NEW_NAME] == "factuality_author"
    assert OLD_NAME not in constants.NESTED_METRIC_PAIRS


# ── Backward compatibility with pre-rename artefacts ──────────────────────────


def test_rename_map_maps_old_to_new():
    assert constants.METRIC_RENAME_MAP == {OLD_NAME: NEW_NAME}


def test_rename_is_idempotent_on_wide_frames():
    df = pd.DataFrame({OLD_NAME: [0.5], "validity": [1.0]})
    once = df.rename(columns=constants.METRIC_RENAME_MAP)
    twice = once.rename(columns=constants.METRIC_RENAME_MAP)
    assert list(once.columns) == [NEW_NAME, "validity"]
    assert list(twice.columns) == list(once.columns)
    assert twice[NEW_NAME].tolist() == [0.5]


def test_rename_is_idempotent_on_long_anova_output():
    # effect_sizes.parquet is long-format: the metric name lives in a column.
    df = pd.DataFrame({"metric": [OLD_NAME, "validity"], "omega2": [0.1, 0.2]})
    once = df.assign(metric=df["metric"].replace(constants.METRIC_RENAME_MAP))
    twice = once.assign(metric=once["metric"].replace(constants.METRIC_RENAME_MAP))
    assert once["metric"].tolist() == [NEW_NAME, "validity"]
    assert twice["metric"].tolist() == once["metric"].tolist()
