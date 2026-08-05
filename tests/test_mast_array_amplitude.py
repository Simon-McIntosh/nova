"""Whether an amplitude taken against the array sees a range setting and nothing else.

This observable exists to place boundaries on shots the fitted route cannot read, so
the risk it carries is the opposite of the fitted route's.  The fitted route refuses
too much; this one could accept too much, and a boundary invented inside a block splits
a run of shots a read would otherwise get right.  So the cases here are mostly about
what must NOT move it.

A shot driven twice as hard must leave every ratio alone, because the drive is in the
numerator and the denominator alike -- that is the whole reason for dividing by the
array.  A handful of channels stepping together must leave the reference alone, because
the reference is a median and the steppers are a minority.  A channel too quiet to
measure must be left out rather than handed a ratio built from its noise.

And the gate the module exists behind gets its own cases: the agreement report has to
call a route that invents a boundary on a steady channel a failure, not a near miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.imas.mast_array_amplitude import (
    narrow_bracket,
    narrowing_summary,
    MINIMUM_AMPLITUDE,
    MINIMUM_CHANNELS,
    agreement,
    agreement_summary,
    channel_amplitudes,
)

SAMPLES = 256
"""Samples per channel, enough that a root-mean-square means something."""


def array(count=40, *, drive=1.0, doubled=(), quiet=()):
    """Build one shot's probe set, with named channels doubled or silenced."""

    time = np.linspace(0.0, 1.0, SAMPLES)
    shape = np.sin(2.0 * np.pi * time)
    probes = {}
    for index in range(count):
        channel = "ccbv%02d" % (index + 1)
        level = drive * (1.0 + 0.01 * index)
        if channel in quiet:
            level = 0.1 * MINIMUM_AMPLITUDE
        if channel in doubled:
            level *= 2.0
        probes[channel] = level * shape
    return probes


def test_driving_the_shot_harder_leaves_every_ratio_alone():
    """The reason for dividing by the array: the common factor cancels."""

    weak = channel_amplitudes(array(drive=1.0))
    strong = channel_amplitudes(array(drive=7.0))
    assert set(weak) == set(strong)
    for channel in weak:
        assert weak[channel] == pytest.approx(strong[channel])


def test_a_doubled_channel_reads_twice_the_array():
    """To within the reference's own small walk, which the next case bounds."""

    doubled = channel_amplitudes(array(doubled=("ccbv06",)))
    plain = channel_amplitudes(array())
    assert doubled["ccbv06"] / plain["ccbv06"] == pytest.approx(2.0, rel=0.02)


def test_a_minority_stepping_together_barely_moves_the_reference():
    """Not exactly, and the bound is what makes the observable usable.

    Doubling a channel walks it out of the lower half of the ordering, so the median
    moves a little.  Six moving together is the most the archive shows at one boundary,
    and the bias it leaves has to stay far inside the factor a step must reach before
    it counts -- otherwise a boundary on one channel would manufacture boundaries on
    its neighbours.
    """

    steppers = tuple("ccbv%02d" % index for index in range(6, 12))
    stepped = channel_amplitudes(array(count=76, doubled=steppers))
    plain = channel_amplitudes(array(count=76))
    held = [
        stepped[channel] / plain[channel]
        for channel in plain
        if channel not in steppers
    ]
    moved = [stepped[channel] / plain[channel] for channel in steppers]
    assert max(abs(value - 1.0) for value in held) < 0.05
    assert min(moved) > 1.9


def test_a_channel_too_quiet_to_measure_is_left_out_rather_than_given_a_ratio():
    rows = channel_amplitudes(array(quiet=("ccbv06",)))
    assert "ccbv06" not in rows
    assert len(rows) == 39


def test_a_shot_recording_too_few_channels_yields_nothing():
    """A median over a handful is moved by the handful that stepped."""

    assert channel_amplitudes(array(count=MINIMUM_CHANNELS - 1)) == {}
    assert channel_amplitudes(array(count=MINIMUM_CHANNELS)) != {}


def test_a_standing_offset_is_removed_before_the_amplitude_is_taken():
    """An offset is not signal, and a channel with one would read too large."""

    probes = array()
    baseline = np.zeros(SAMPLES, dtype=bool)
    baseline[:32] = True
    offset = {channel: values + 5.0 for channel, values in probes.items()}
    plain = channel_amplitudes(probes, baseline=baseline)
    shifted = channel_amplitudes(offset, baseline=baseline)
    for channel in plain:
        assert shifted[channel] == pytest.approx(plain[channel], rel=1.0e-6)


# --- the gate the route sits behind --------------------------------------


def fitted(channel, *rows):
    return {channel: {shot: [value] for shot, value in rows}}


def test_the_two_routes_agree_when_they_place_the_same_boundary():
    rows = agreement(
        fitted("obr11", (100, 1.0), (101, 1.0), (102, 0.5), (103, 0.5)),
        {"obr11": {100: 1.0, 101: 1.0, 102: 0.5, 103: 0.5}},
    )
    assert len(rows) == 1
    assert rows[0].agrees
    assert (rows[0].fitted_steps, rows[0].matched) == (1, 1)


def test_a_boundary_the_array_route_misses_is_a_disagreement():
    rows = agreement(
        fitted("obr11", (100, 1.0), (101, 1.0), (102, 0.5), (103, 0.5)),
        {"obr11": {100: 1.0, 101: 1.0, 102: 1.0, 103: 1.0}},
    )
    assert not rows[0].agrees
    assert rows[0].matched == 0


def test_a_boundary_invented_on_a_steady_channel_is_a_failure_not_a_near_miss():
    """Splitting a block a read would get right is the costlier error."""

    rows = agreement(
        fitted("ccbv35", (100, 1.0), (101, 1.0), (102, 1.0)),
        {"ccbv35": {100: 1.0, 101: 1.0, 102: 2.0}},
    )
    assert not rows[0].agrees
    summary = agreement_summary(rows)
    assert summary["invented_on_steady"] == 1
    assert summary["steady_reproduced"] == 0


def test_only_the_shots_both_routes_read_enter_the_comparison():
    """So the gate measures the observable rather than the coverage."""

    rows = agreement(
        fitted("obr11", (100, 1.0), (101, 1.0), (500, 0.5)),
        {"obr11": {100: 1.0, 101: 1.0}},
    )
    assert rows[0].shared_shots == 2
    assert rows[0].fitted_steps == 0
    assert rows[0].agrees


# --- narrowing a bracket without touching its value ----------------------


def test_a_bracket_narrows_onto_the_pair_the_amplitude_crossed_at():
    """The fitted route left five thousand shots; one crossing names two."""

    row = narrow_bracket(
        "ccbv02",
        14131,
        19219,
        2.0,
        {14131: 1.0, 15000: 1.02, 16000: 0.99, 17000: 1.98, 18000: 2.01, 19219: 1.99},
    )
    assert (row.before_shot, row.after_shot) == (16000, 17000)
    assert row.width == 1000
    assert row.fitted_width == 5088
    assert row.narrowed
    assert row.crossing == pytest.approx(2.0, rel=0.02)


def test_narrowing_cannot_reach_outside_the_fitted_bracket():
    """It may divide the gap the fitted route left, never move a placed boundary."""

    row = narrow_bracket(
        "ccbv02",
        14131,
        19219,
        2.0,
        {13000: 1.0, 13500: 2.0, 14131: 1.0, 17000: 1.99, 19219: 2.0},
    )
    assert (row.before_shot, row.after_shot) == (14131, 17000)


def test_the_crossing_must_run_the_way_the_fitted_rungs_did():
    """A bracket holds scatter as well as the switch, and only one moves right."""

    row = narrow_bracket(
        "obr11", 100, 200, 0.5, {100: 1.0, 120: 2.1, 140: 1.0, 160: 0.49, 200: 0.5}
    )
    assert (row.before_shot, row.after_shot) == (140, 160)
    assert row.crossing < 1.0


def test_a_bracket_the_array_route_cannot_resolve_stays_exactly_as_wide():
    """Reporting nothing is the honest outcome, and leaves the fitted bracket alone."""

    assert (
        narrow_bracket(
            "ccbv02", 14131, 19219, 2.0, {14131: 1.0, 17000: 1.01, 19219: 1.0}
        )
        is None
    )
    assert narrow_bracket("ccbv02", 14131, 19219, 2.0, {17000: 1.5}) is None


def test_the_narrowing_summary_reports_what_it_tightened():
    rows = [
        narrow_bracket(
            "ccbv02", 14131, 19219, 2.0, {14131: 1.0, 17000: 2.0, 19219: 2.0}
        ),
        narrow_bracket("obr11", 100, 200, 0.5, {100: 1.0, 150: 0.5, 200: 0.5}),
    ]
    summary = narrowing_summary(rows)
    assert summary["placed"] == 2
    assert summary["narrowed"] == 2
    assert summary["median_fitted_width"] == pytest.approx(2594.0)
    assert summary["widest_width"] == 2869


def test_the_summary_separates_reproduced_steps_from_reproduced_steadiness():
    rows = agreement(
        {
            **fitted("obr11", (100, 1.0), (101, 0.5)),
            **fitted("ccbv35", (100, 1.0), (101, 1.0)),
        },
        {"obr11": {100: 1.0, 101: 0.5}, "ccbv35": {100: 1.0, 101: 1.0}},
    )
    summary = agreement_summary(rows)
    assert summary["stepping_channels"] == 1
    assert summary["stepping_reproduced"] == 1
    assert summary["steady_channels"] == 1
    assert summary["steady_reproduced"] == 1
    assert summary["invented_on_steady"] == 0
