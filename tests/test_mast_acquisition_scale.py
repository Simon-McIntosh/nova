"""Whether a channel recorded at two settings is told apart from one that drifted.

The distinction the whole module exists for is between a scale that is a property of
a channel and a scale that is a property of a block of shots, and getting it wrong is
costly in both directions: a stepping channel handed one number gets an average that
describes no shot, and a steady channel split into blocks loses a real calibration
defect.  Both failure modes are given their own case here.

The concurrency control gets its own tests too, because it is the only thing standing
between "this channel's acquisition changed" and "our coil model is wrong" -- and the
two are indistinguishable from a single channel's numbers.
"""

from __future__ import annotations

import json
import math

import pytest

from nova.imas.mast_acquisition_scale import (
    CONCURRENCY_SHARE,
    LADDER_TOLERANCE,
    MINIMUM_HISTORY_SHOTS,
    ROUTE_AGREEMENT,
    SPLIT_HALF_TOLERANCE,
    STEP_RATIO,
    AcquisitionScaleError,
    ChannelScaleHistory,
    acquisition_record,
    channel_histories,
    nearest_rung,
    promote_scales,
    scale_blocks,
    scale_steps,
    split_half_check,
    steady_channels,
    step_concurrency,
    stepping_channels,
)


def series(*levels: tuple[int, int, float]) -> dict[int, list[float]]:
    """Build a per-shot series from (first shot, count, scale) runs."""

    rows: dict[int, list[float]] = {}
    for first, count, scale in levels:
        for offset in range(count):
            rows[first + offset] = [scale, scale * 1.005, scale * 0.995]
    return rows


# --- telling a step from scatter ----------------------------------------


def test_a_channel_recorded_at_two_settings_yields_two_blocks():
    """A factor of two that persists is a block boundary, not scatter."""

    blocks = scale_blocks("obr11", series((14061, 8, 1.0), (19219, 10, 0.5)))
    assert len(blocks) == 2
    assert blocks[0].scale == pytest.approx(1.0, abs=1e-3)
    assert blocks[1].scale == pytest.approx(0.5, abs=1e-3)
    assert blocks[0].last_shot == 14068
    assert blocks[1].first_shot == 19219
    step = scale_steps(blocks)[0]
    assert step.ratio == pytest.approx(0.5, abs=1e-3)
    assert step.rung == 0.5
    assert step.on_ladder


def test_a_channel_that_only_scatters_yields_one_block():
    """Few-percent reproducibility must not be read as a range change."""

    rows = {shot: [1.0 + 0.03 * ((shot % 5) - 2)] for shot in range(14061, 14081)}
    blocks = scale_blocks("ccbv35", rows)
    assert len(blocks) == 1
    history = ChannelScaleHistory("ccbv35", blocks, len(rows))
    assert history.steady
    assert history.scale == pytest.approx(1.0, abs=0.05)


def test_two_blocks_at_the_same_scale_are_coalesced():
    """One bad shot between two runs of one setting is not two settings.

    Without coalescing the channel reads as stepping, and a real steady
    factor-of-two defect would be refused a promotion for an artefact of the split.
    """

    rows = series((14061, 6, 0.50))
    rows[14067] = [0.20]
    rows.update(series((14068, 6, 0.50)))
    blocks = scale_blocks("obr17", rows)
    assert len(blocks) == 1
    history = ChannelScaleHistory("obr17", blocks, len(rows))
    assert history.steady
    assert history.scale == pytest.approx(0.5, abs=0.02)


def test_a_stepping_channel_refuses_to_report_one_scale():
    """Averaging two settings would describe no shot, so asking is an error."""

    history = ChannelScaleHistory(
        "obv12",
        scale_blocks("obv12", series((14061, 6, 1.0), (19219, 6, 0.5))),
        12,
    )
    assert not history.steady
    with pytest.raises(AcquisitionScaleError, match="no single scale"):
        history.scale
    assert history.scale_for(14063) == pytest.approx(1.0, abs=1e-3)
    assert history.scale_for(19221) == pytest.approx(0.5, abs=1e-3)
    with pytest.raises(AcquisitionScaleError, match="no measured scale"):
        history.scale_for(30000)


def test_too_few_shots_is_neither_steady_nor_stepping():
    """A channel with no history is reported unmeasured rather than guessed."""

    rows = series((14061, MINIMUM_HISTORY_SHOTS - 1, 1.0))
    history = channel_histories({"obv09": rows})[0]
    assert not history.measured
    assert not history.steady
    assert history not in steady_channels([history])
    assert history not in stepping_channels([history])


# --- the ladder ---------------------------------------------------------


def test_the_ladder_rungs_are_recognised_and_the_distance_reported():
    """A step is snapped to a rung only when it is close to one."""

    rung, distance = nearest_rung(2.0)
    assert rung == 2.0 and distance == pytest.approx(0.0)
    rung, distance = nearest_rung(1.96)
    assert rung == 2.0 and distance == pytest.approx(0.02)
    assert distance <= LADDER_TOLERANCE


def test_a_step_off_the_ladder_is_reported_as_off_it():
    """The ladder is a hypothesis, so a step that refutes it must stay visible."""

    blocks = scale_blocks("obv07", series((14061, 6, 1.0), (19219, 6, 3.0)))
    step = scale_steps(blocks)[0]
    assert step.ratio == pytest.approx(3.0, abs=1e-2)
    assert not step.on_ladder
    assert step.ladder_distance > LADDER_TOLERANCE


def test_a_ratio_that_is_not_a_positive_number_is_refused():
    """A zero or negative scale cannot be a range factor."""

    with pytest.raises(AcquisitionScaleError, match="positive number"):
        nearest_rung(0.0)


# --- the concurrency control -------------------------------------------


def test_a_step_most_channels_hold_through_is_per_channel():
    """The control that separates an acquisition change from a model error."""

    rows = {
        "obr11": series((14061, 4, 1.0), (19219, 4, 0.5)),
        **{
            f"ccbv{index:02d}": series((14061, 4, 1.0), (19219, 4, 1.0))
            for index in range(1, 11)
        },
    }
    histories = channel_histories(rows)
    steps = [step for row in stepping_channels(histories) for step in row.steps]
    control = step_concurrency(rows, steps)
    assert len(control) == 1
    assert control[0].moved == ("obr11",)
    assert len(control[0].held) == 10
    assert control[0].held_share > CONCURRENCY_SHARE
    assert control[0].per_channel


def test_a_step_every_channel_takes_together_is_not_per_channel():
    """If the whole array moves, the cause is upstream of the channels."""

    rows = {
        f"ccbv{index:02d}": series((14061, 4, 1.0), (19219, 4, 0.5))
        for index in range(1, 11)
    }
    histories = channel_histories(rows)
    steps = [step for row in stepping_channels(histories) for step in row.steps]
    control = step_concurrency(rows, steps)
    assert control[0].held == ()
    assert control[0].held_share == pytest.approx(0.0)
    assert not control[0].per_channel


def test_a_channel_absent_on_one_side_is_not_counted_as_holding():
    """Silence is not agreement; an unrecorded channel leaves the comparison."""

    rows = {
        "obr11": series((14061, 4, 1.0), (19219, 4, 0.5)),
        "ccbv01": series((14061, 4, 1.0)),
    }
    histories = channel_histories(rows)
    steps = [step for row in stepping_channels(histories) for step in row.steps]
    control = step_concurrency(rows, steps)
    assert control[0].shared == 1
    assert control[0].held == ()


# --- promotion ----------------------------------------------------------


def test_a_steady_channel_two_routes_agree_on_is_promoted():
    """Off unity, corroborated independently, and holding on both halves."""

    rows = {"obr17": series((14061, 12, 0.501))}
    histories = channel_histories(rows)
    decision = promote_scales(histories, rows, {"obr17": 0.5037})[0]
    assert decision.corroborated
    assert decision.split_half.holds
    assert decision.promoted
    lower, upper = decision.interval
    assert lower <= 0.501 <= upper and upper <= 0.5037


def test_a_channel_the_second_route_disagrees_on_is_withheld():
    """Agreement between independent routes is what stops a tuned promotion."""

    rows = {"obv11": series((14061, 12, 0.925))}
    decision = promote_scales(channel_histories(rows), rows, {"obv11": 0.996})[0]
    assert decision.route_disagreement > ROUTE_AGREEMENT
    assert not decision.corroborated
    assert not decision.promoted


def test_a_channel_whose_halves_disagree_is_withheld():
    """A scale that drifted inside its own block is not one scale."""

    rows = {"obv11": series((14061, 6, 0.87), (14200, 6, 1.06))}
    rows = {"obv11": {**rows["obv11"]}}
    check = split_half_check("obv11", rows["obv11"])
    assert check.disagreement > SPLIT_HALF_TOLERANCE
    assert not check.holds


def test_a_channel_at_unity_is_not_promoted():
    """A channel the data finds correct gets no record rather than a scale of one."""

    rows = {"ccbv01": series((14061, 12, 1.002))}
    decision = promote_scales(channel_histories(rows), rows, {"ccbv01": 1.004})[0]
    assert decision.corroborated and decision.split_half.holds
    assert not decision.promoted


def test_a_stepping_channel_is_never_a_promotion_candidate():
    """It has no single scale, so there is nothing to promote."""

    rows = {"obv12": series((14061, 6, 1.0), (19219, 6, 0.5))}
    assert promote_scales(channel_histories(rows), rows, {"obv12": 1.2273}) == ()


# --- the record ---------------------------------------------------------


def test_the_record_carries_the_ladder_the_control_and_both_classes():
    """A reader must be able to recheck the conclusion, not just read it."""

    rows = {
        "obr11": series((14061, 4, 1.0), (19219, 4, 0.5)),
        **{
            f"ccbv{index:02d}": series((14061, 4, 1.0), (19219, 4, 1.0))
            for index in range(1, 11)
        },
    }
    histories = channel_histories(rows)
    steps = [step for row in stepping_channels(histories) for step in row.steps]
    record = acquisition_record(histories, step_concurrency(rows, steps))
    assert record["stepping_channels"] == ["obr11"]
    assert len(record["steady_channels"]) == 10
    assert record["step_count"] == 1
    assert record["steps_on_ladder"] == 1
    assert record["steps_per_channel_controlled"] == 1
    assert record["step_ratio"] == STEP_RATIO
    assert json.loads(json.dumps(record, sort_keys=True)) == record


def test_an_infinite_span_serializes_as_a_null():
    """A block at zero scale must not put an unserializable value in the record."""

    history = ChannelScaleHistory(
        "obv09",
        scale_blocks("obv09", {shot: [0.0] for shot in range(14061, 14071)}),
        10,
    )
    row = history.as_dict()
    assert row["span"] is None or math.isfinite(row["span"])
    assert json.loads(json.dumps(row, sort_keys=True)) == row
