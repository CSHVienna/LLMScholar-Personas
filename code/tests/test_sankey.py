"""Tests for the Sankey helpers behind the location figure (issue #37).

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_sankey.py -v
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.patches import PathPatch, Rectangle

from libs.visuals.sankey import collapse_tail, plot_sankey


@pytest.fixture
def flows():
    return pd.DataFrame(
        {
            "source": ["EC", "EC", "EC", "JP", "JP"],
            "target": ["ES", "US", "EC", "JP", "US"],
            "value": [280, 225, 50, 857, 77],
        }
    )


def teardown_function():
    plt.close("all")


# ── collapse_tail ─────────────────────────────────────────────────────────────


def test_collapse_tail_keeps_top_n_and_folds_the_rest(flows):
    # Totals per target: JP 857, US 302 (225+77), ES 280, EC 50.
    out = collapse_tail(flows, top_n=2)
    assert set(out.target) == {"JP", "US", "Other"}


def test_collapse_tail_conserves_the_total(flows):
    """Folding must never drop flow — the grand total is invariant."""
    for n in (1, 2, 3, 10):
        assert collapse_tail(flows, top_n=n).value.sum() == flows.value.sum()


def test_collapse_tail_does_not_mutate_the_input(flows):
    before = flows.copy()
    collapse_tail(flows, top_n=1)
    pd.testing.assert_frame_equal(flows, before)


def test_collapse_tail_is_a_noop_when_top_n_exceeds_targets(flows):
    out = collapse_tail(flows, top_n=99)
    assert set(out.target) == set(flows.target)
    assert "Other" not in set(out.target)


def test_collapse_tail_custom_label(flows):
    out = collapse_tail(flows, top_n=1, other_label="Rest of world")
    assert "Rest of world" in set(out.target)


# ── plot_sankey ───────────────────────────────────────────────────────────────


def test_draws_one_ribbon_per_flow_and_one_bar_per_node(flows):
    fig, ax = plot_sankey(flows)
    ribbons = [p for p in ax.patches if isinstance(p, PathPatch)]
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(ribbons) == len(flows)  # 5 flows
    assert len(bars) == 2 + 4  # 2 sources + 4 targets


def test_node_heights_are_proportional_to_flow(flows):
    """A source carrying twice the flow must be twice as tall."""
    fig, ax = plot_sankey(flows, node_gap=0.0)
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    left = sorted(
        [b for b in bars if b.get_x() == pytest.approx(0.0)],
        key=lambda b: -b.get_height(),
    )
    jp, ec = left  # JP total 934, EC total 555
    assert jp.get_height() / ec.get_height() == pytest.approx(934 / 555, rel=1e-6)


def test_all_node_heights_sum_to_the_full_span(flows):
    """With no gaps each column fills [0, 1] exactly — no flow lost in layout."""
    fig, ax = plot_sankey(flows, node_gap=0.0)
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    for x in (0.0, 1.0):
        col = [b.get_height() for b in bars if b.get_x() == pytest.approx(x)]
        assert sum(col) == pytest.approx(1.0)


def test_zero_and_negative_flows_are_dropped(flows):
    padded = pd.concat(
        [flows, pd.DataFrame({"source": ["EC"], "target": ["ZZ"], "value": [0]})]
    )
    fig, ax = plot_sankey(padded)
    assert len([p for p in ax.patches if isinstance(p, PathPatch)]) == len(flows)


def test_empty_input_raises_rather_than_drawing_nothing(flows):
    with pytest.raises(ValueError, match="No positive flows"):
        plot_sankey(flows.assign(value=0))


def test_explicit_order_is_respected(flows):
    """Source order drives vertical stacking, top-down."""
    fig, ax = plot_sankey(flows, source_order=["EC", "JP"], node_gap=0.0)
    bars = [
        b
        for b in ax.patches
        if isinstance(b, Rectangle) and b.get_x() == pytest.approx(0.0)
    ]
    topmost = max(bars, key=lambda b: b.get_y() + b.get_height())
    # EC is listed first, so it occupies the top slot despite carrying less flow.
    assert topmost.get_height() == pytest.approx(555 / (555 + 934))


def test_labels_and_percentages_reach_the_axes(flows):
    fig, ax = plot_sankey(
        flows,
        source_labels={"EC": "Ecuador"},
        target_labels={"US": "United States"},
        value_fmt="{pct:.1f}%",
    )
    texts = [t.get_text() for t in ax.texts]
    assert "Ecuador" in texts
    assert any(t.startswith("United States") and "%" in t for t in texts)
    # JP target total is 857 of 1489 → 57.6%
    assert any("57.6%" in t for t in texts)


def test_value_fmt_none_leaves_bare_labels(flows):
    fig, ax = plot_sankey(flows, value_fmt=None)
    assert all("%" not in t.get_text() for t in ax.texts)


def test_source_colors_are_applied(flows):
    fig, ax = plot_sankey(flows, source_colors={"EC": "#ff0000", "JP": "#00ff00"})
    ribbons = [p for p in ax.patches if isinstance(p, PathPatch)]
    facecolors = {tuple(round(c, 3) for c in p.get_facecolor()[:3]) for p in ribbons}
    assert (1.0, 0.0, 0.0) in facecolors
    assert (0.0, 1.0, 0.0) in facecolors


def test_accepts_custom_column_names(flows):
    renamed = flows.rename(
        columns={"source": "src_iso", "target": "dst_iso", "value": "n"}
    )
    fig, ax = plot_sankey(
        renamed, source_col="src_iso", target_col="dst_iso", value_col="n"
    )
    assert len([p for p in ax.patches if isinstance(p, PathPatch)]) == len(flows)


def test_duplicate_pairs_are_aggregated(flows):
    """The same (source, target) split across rows is one ribbon, not two."""
    doubled = pd.concat([flows, flows], ignore_index=True)
    fig, ax = plot_sankey(doubled)
    assert len([p for p in ax.patches if isinstance(p, PathPatch)]) == len(flows)


def test_collapse_tail_pins_requested_targets(flows):
    """EC is the smallest target, so only `keep` saves it from the tail."""
    assert "EC" not in set(collapse_tail(flows, top_n=2).target)
    out = collapse_tail(flows, top_n=2, keep=["EC"])
    assert "EC" in set(out.target)
    assert out.value.sum() == flows.value.sum()


def test_pinned_targets_count_against_top_n(flows):
    """Pinning must not quietly widen the figure."""
    out = collapse_tail(flows, top_n=2, keep=["EC"])
    assert len(set(out.target)) == 3  # EC + the biggest one + Other


def test_collapse_tail_ignores_unknown_pins(flows):
    out = collapse_tail(flows, top_n=2, keep=["ZZ"])
    assert set(out.target) == {"JP", "US", "Other"}


def test_flow_labels_are_shares_of_their_source(flows):
    """EC sends 280 of its 555 to ES — 50.5%, not 18.8% of the grand total."""
    fig, ax = plot_sankey(flows, flow_fmt="{pct:.1f}%", value_fmt=None)
    texts = [t.get_text() for t in ax.texts]
    assert "50.5%" in texts  # 280 / 555
    assert "40.5%" in texts  # 225 / 555
    assert "91.8%" in texts  # 857 / 934


def test_flow_label_min_pct_suppresses_the_thin_ribbons(flows):
    fig, ax = plot_sankey(
        flows, flow_fmt="{pct:.1f}%", flow_label_min_pct=10.0, value_fmt=None
    )
    texts = [t.get_text() for t in ax.texts]
    assert "50.5%" in texts
    assert "9.0%" not in texts  # EC -> EC, 50 / 555


def test_source_value_fmt_labels_the_left_column(flows):
    fig, ax = plot_sankey(flows, source_value_fmt="{pct:.1f}%", value_fmt=None)
    # EC carries 555 of 1489 → 37.3%
    assert any(t.get_text().startswith("EC") and "37.3%" in t.get_text() for t in ax.texts)


def test_source_labels_can_sit_outside_the_diagram(flows):
    fig, ax = plot_sankey(flows, source_label_side="left")
    ec = next(t for t in ax.texts if t.get_text() == "EC")
    assert ec.get_position()[0] < 0
    assert ec.get_horizontalalignment() == "right"
    assert ax.get_xlim()[0] < -0.2  # room was made for them


def test_crowded_flow_labels_are_thinned_out():
    """Five equal ribbons out of one source cannot all carry a legible label."""
    crowded = pd.DataFrame(
        {
            "source": ["EC"] * 5,
            "target": list("ABCDE"),
            "value": [100] * 5,
        }
    )
    fig, ax = plot_sankey(
        crowded, flow_fmt="{pct:.0f}%", flow_label_min_gap=0.5, value_fmt=None
    )
    assert len([t for t in ax.texts if t.get_text() == "20%"]) == 2  # spans 1.0


def test_min_gap_zero_keeps_every_flow_label(flows):
    fig, ax = plot_sankey(flows, flow_fmt="{pct:.1f}%", flow_label_min_gap=0.0,
                          value_fmt=None)
    assert len([t for t in ax.texts if t.get_text().endswith("%")]) == len(flows)


def test_target_colors_tint_only_the_nodes_they_name(flows):
    """The five prompted countries keep their colour on the right too."""
    fig, ax = plot_sankey(flows, target_colors={"JP": "#c0392b"})
    bars = [b for b in ax.patches if isinstance(b, Rectangle)
            and b.get_x() == pytest.approx(1.0)]
    faces = {tuple(round(c, 3) for c in b.get_facecolor()[:3]) for b in bars}
    assert (0.753, 0.224, 0.169) in faces      # JP, tinted
    assert len(faces) > 1                       # the rest stay grey
