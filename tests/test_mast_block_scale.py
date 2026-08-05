"""Whether a read removes an acquisition setting without removing anything else.

The correction this module applies is one division, so almost every way of getting it
wrong is silent.  Four failure modes get their own cases, each on a channel's real
measured blocks so the case cannot pass on a shape the archive does not have.

Dividing by the fitted ratio instead of the ladder rung would launder the
description's own error into the data and make the residual improve for the wrong
reason.  A channel that halves and returns must be divided by exactly two even though
its measured ratios read 0.4993 and 0.5006, and a channel whose step misses every
rung -- 0.886 to 0.518 is a ratio of 0.585, seventeen percent off the nearest -- must
be refused rather than rounded onto one.  Which block anchors the channel is part of
that: anchoring on the commonest setting rather than the one reading nearest the
described field would divide a channel's ordinary blocks by two.

Claiming a measurement that does not exist.  A block spanning five thousand shots
rests on the few dozen inside it that were read, and a shot in one of its gaps is
bracketed, not measured; a shot between two blocks is unmeasured however narrow the
gap, because that is where the switch is.

Scaling on no warrant.  A refusal and an applied unity leave identical arrays, so the
disposition is the only thing that can tell them apart, and every consumer decides
what to trust from it.

Losing the bracket widths.  The whole reason blocks carry shot lists is so a reader
can see which switches the archive pins and which it does not; a bisecting sweep
reads that back to choose its next shot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nova.imas.mast_acquisition_scale import ChannelScaleHistory, ScaleBlock
from nova.imas.mast_block_scale import (
    BRACKETED,
    MEASURED,
    REFUSED,
    UNMEASURED,
    BlockScale,
    BlockScaleError,
    BlockScaleTable,
    ScaleBracket,
    ScaleCorrection,
    bracket_probe,
    channel_blocks,
    pinning_summary,
    promoted_block_scales,
)


def block(channel, scale, shots, rung=1.0):
    """Build one block over an explicit shot list."""

    return BlockScale(channel=channel, scale=scale, shots=tuple(shots), rung=rung)


def history(channel, *runs):
    """Build a classifier history from (first shot, last shot, scale, count) runs."""

    return ChannelScaleHistory(
        channel=channel,
        blocks=tuple(
            ScaleBlock(channel, first, last, scale, count)
            for first, last, scale, count in runs
        ),
        shot_count=sum(row[3] for row in runs),
    )


# --- the rung is what a read divides by, never the fitted ratio ----------


def halved_channel():
    """Return obr11's measured history, whose steps are clean factors of two."""

    return (
        history(
            "obr11",
            (14061, 14127, 0.9992, 15),
            (19219, 19258, 0.4989, 21),
            (24938, 24965, 0.9988, 5),
            (25722, 25828, 0.5002, 3),
        ),
        list(range(14061, 14076))
        + list(range(19219, 19240))
        + list(range(24938, 24943))
        + [25722, 25800, 25828],
    )


def off_ladder_channel():
    """Return obv03's measured history, whose one step misses every rung by 17%."""

    return (
        history("obv03", (14061, 19258, 0.8860, 36), (19995, 24965, 0.5183, 2)),
        list(range(14061, 14097)) + [19995, 24965],
    )


def test_the_reference_block_is_the_one_reading_nearest_the_described_field():
    """The ordinary setting anchors the channel, not the commonest one."""

    rows = channel_blocks(*halved_channel())
    # the halved block rests on the most shots, and anchoring there would divide
    # the ordinary blocks by two
    assert [row.shot_count for row in rows] == [15, 21, 5, 3]
    assert [row.unchanged for row in rows] == [True, False, True, False]
    assert [row.rung < 1.0 for row in rows] == [False, True, False, True]


def test_a_factor_of_two_step_divides_by_exactly_two_not_by_the_ratio():
    """The ladder rung is discrete, so no description error rides in on it."""

    rows = channel_blocks(*halved_channel())
    reference = rows[0].scale
    assert [row.rung for row in rows] == pytest.approx([1.0, 0.5, 1.0, 0.5])
    # every rung is exact while the ratios it replaces are not, and the difference
    # is the description error that would otherwise be divided into the data
    assert rows[1].scale / reference == pytest.approx(0.4993, abs=1.0e-4)
    assert rows[3].scale / reference == pytest.approx(0.5006, abs=1.0e-4)


def test_a_step_that_misses_every_rung_is_refused_not_rounded_on():
    """A real step that is not a range factor is not a correction."""

    rows = channel_blocks(*off_ladder_channel())
    assert rows[0].unchanged
    assert not rows[1].on_ladder
    assert math.isnan(rows[1].rung)
    # it misses the nearest rung by seventeen percent, twice the tolerance
    assert rows[1].scale / rows[0].scale == pytest.approx(0.585, abs=1.0e-3)


def test_a_refused_block_reads_the_channel_exactly_as_published():
    """Refusing a step means dividing by nothing, and saying so."""

    table = BlockScaleTable.create(channel_blocks(*off_ladder_channel()))
    correction = table.correction("obv03", 19995)
    assert correction.disposition == REFUSED
    assert not correction.applied
    assert correction.flagged
    assert correction.normalise([2.0, 4.0]).tolist() == [2.0, 4.0]


# --- a block claims only the shots it was measured on --------------------


def test_a_shot_the_block_was_measured_on_reads_measured():
    table = BlockScaleTable.create([block("obr11", 0.999, [14061, 14070, 14127])])
    assert table.correction("obr11", 14070).disposition == MEASURED
    assert not table.correction("obr11", 14070).flagged


def test_a_shot_in_a_gap_inside_one_block_is_bracketed_not_measured():
    """The block spans five thousand shots and rests on three of them."""

    table = BlockScaleTable.create([block("ccbv06", 0.975, [14061, 16000, 19258])])
    correction = table.correction("ccbv06", 17000)
    assert correction.disposition == BRACKETED
    assert correction.applied and correction.flagged


def test_a_shot_between_two_blocks_is_unmeasured_and_carries_both_candidates():
    """The switch is in there somewhere, so neither side may be assumed."""

    table = BlockScaleTable.create(
        [
            block("obr11", 0.999, [14061, 14127], rung=1.0),
            block("obr11", 0.499, [19219, 19258], rung=0.5),
        ]
    )
    correction = table.correction("obr11", 17000)
    assert correction.disposition == UNMEASURED
    assert not correction.applied
    assert correction.candidates == (1.0, 0.5)


def test_a_shot_outside_the_measured_span_is_unmeasured():
    """A stepping channel's setting on an unvisited campaign is not knowable."""

    table = BlockScaleTable.create(
        [
            block("obr11", 0.999, [14061, 14127], rung=1.0),
            block("obr11", 0.499, [19219, 19258], rung=0.5),
        ]
    )
    assert table.correction("obr11", 28000).disposition == UNMEASURED
    assert table.correction("obr11", 11000).disposition == UNMEASURED


def test_a_channel_the_table_never_measured_reads_unmeasured():
    """Named for a channel the archive does not carry, so the case cannot go stale.

    Asking about a real channel would make this pass or fail on whether the promoted
    table happens to have measured that one, which is a fact about the sweep rather
    than about the reader.
    """

    table = promoted_block_scales()
    assert "ccbv99" not in table.channels
    assert table.correction("ccbv99", 14061).disposition == UNMEASURED


# --- the division itself -------------------------------------------------


def test_normalising_a_doubled_block_halves_the_samples():
    table = BlockScaleTable.create([block("obr02", 1.985, [14061, 14127], rung=2.0)])
    probes = {"obr02": np.asarray([2.0, 4.0, 6.0])}
    values, corrections = table.normalise(14061, probes)
    assert values["obr02"].tolist() == [1.0, 2.0, 3.0]
    assert corrections[0].scale == pytest.approx(2.0)


def test_an_unmeasured_channel_passes_through_untouched_beside_a_scaled_one():
    """One shot's channels are corrected independently, as the defect is."""

    table = BlockScaleTable.create([block("obr02", 1.985, [14061], rung=2.0)])
    probes = {"obr02": np.asarray([4.0]), "obv06": np.asarray([4.0])}
    values, corrections = table.normalise(14061, probes)
    assert values["obr02"].tolist() == [2.0]
    assert values["obv06"].tolist() == [4.0]
    assert {row.channel: row.disposition for row in corrections} == {
        "obr02": MEASURED,
        "obv06": UNMEASURED,
    }


def test_an_empty_table_is_the_raw_archive():
    table = BlockScaleTable()
    probes = {"obv06": np.asarray([1.0, 2.0])}
    values, corrections = table.normalise(14061, probes)
    assert values["obv06"].tolist() == [1.0, 2.0]
    assert all(row.disposition == UNMEASURED for row in corrections)


def test_a_refusal_and_an_applied_unity_differ_only_in_the_disposition():
    """Which is why nothing downstream may infer the correction from the array."""

    applied = ScaleCorrection("obr11", 14061, 1.0, MEASURED)
    refused = ScaleCorrection("obr11", 15295, 1.0, REFUSED)
    assert applied.normalise([3.0]).tolist() == refused.normalise([3.0]).tolist()
    assert applied.applied and not refused.applied


# --- brackets and what pins them ----------------------------------------


def test_a_bracket_reports_its_own_width_rather_than_a_boundary_shot():
    table = BlockScaleTable.create(
        [
            block("ccbv02", 0.981, [14061, 14131], rung=1.0),
            block("ccbv02", 1.925, [19219, 19258], rung=2.0),
        ]
    )
    (bracket,) = table.brackets()
    assert (bracket.before_shot, bracket.after_shot) == (14131, 19219)
    assert bracket.width == 5088


def test_a_bracket_is_pinned_when_no_readable_shot_lies_inside_it():
    adjacent = ScaleBracket("ccbv02", 14131, 14132, 1.0, 2.0)
    assert adjacent.pinned([14061, 14131, 14132, 19219])
    wide = ScaleBracket("ccbv02", 14131, 19219, 1.0, 2.0)
    assert not wide.pinned([14131, 14135, 19219])
    assert wide.unresolved([14131, 14135, 19219]) == (14135,)


def test_the_next_probe_bisects_the_widest_open_bracket():
    """Reading the middle costs reads growing with the log of the width."""

    narrow = ScaleBracket("obr11", 100, 110, 1.0, 0.5)
    wide = ScaleBracket("ccbv02", 1000, 2000, 1.0, 2.0)
    shots = list(range(100, 111)) + list(range(1000, 2001, 10))
    assert bracket_probe([narrow, wide], shots) == 1500


def test_a_probe_already_read_is_not_offered_again():
    wide = ScaleBracket("ccbv02", 1000, 2000, 1.0, 2.0)
    shots = [1400, 1500, 1700]
    assert bracket_probe([wide], shots, measured=[1500]) == 1400


def test_a_fully_pinned_table_asks_for_no_further_shot():
    bracket = ScaleBracket("ccbv02", 14131, 14132, 1.0, 2.0)
    assert bracket_probe([bracket], [14131, 14132]) is None


def test_the_pinning_summary_counts_switches_and_their_widest_bracket():
    table = BlockScaleTable.create(
        [
            block("ccbv02", 0.981, [14061, 14131], rung=1.0),
            block("ccbv02", 1.925, [19219, 19258], rung=2.0),
            block("obr11", 0.999, [14061, 14127], rung=1.0),
            block("obr11", 0.499, [14128, 14140], rung=0.5),
        ]
    )
    summary = pinning_summary(table, [14061, 14127, 14128, 14131, 16000, 19219, 19258])
    assert summary["switch_count"] == 2
    assert summary["widest_width"] == 5088
    # obr11's switch is pinned to adjacent shots; ccbv02's still holds shot 16000
    assert summary["pinned"] == 1
    assert summary["stepping_channels"] == ["ccbv02", "obr11"]
    assert summary["corrected_channels"] == ["ccbv02", "obr11"]


# --- what the table refuses to be ---------------------------------------


def test_a_block_resting_on_no_shot_is_refused():
    with pytest.raises(BlockScaleError, match="rests on no shot"):
        block("obr11", 1.0, []).validate()


def test_overlapping_blocks_are_refused_because_a_shot_would_have_two_settings():
    with pytest.raises(BlockScaleError, match="overlap"):
        BlockScaleTable.create(
            [
                block("obr11", 1.0, [14061, 14127]),
                block("obr11", 0.5, [14100, 14140]),
            ]
        )


def test_a_zero_rung_is_refused_because_it_would_erase_the_signal():
    with pytest.raises(BlockScaleError, match="erase"):
        block("obr11", 1.0, [14061], rung=0.0).validate()


def test_a_table_round_trips_through_its_json_representation():
    table = BlockScaleTable.create(
        [
            block("ccbv02", 0.981, [14061, 14131], rung=1.0),
            block("ccbv02", 1.925, [19219, 19258], rung=2.0),
            block("ccbv07", 0.684, [15295, 15296], rung=math.nan),
        ],
        route="far-field response ratio on plasma-free shots",
    )
    restored = BlockScaleTable.from_dict(table.as_dict())
    assert restored.as_dict() == table.as_dict()
    assert not restored.blocks["ccbv07"][0].on_ladder
    assert restored.route == table.route
