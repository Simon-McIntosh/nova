"""What a flux loop reads against what the description says it should read.

A loop's gain is a scale, and the way a scale goes wrong is by being fitted where
it means nothing: onto a channel the excitation never moved, across an
acquisition step, or against a prediction only one route has ever seen.  Each of
those is a separate refusal here, and the verdict a channel carries has to name
which one it failed.

The other thing pinned here is the convention.  A flux loop links the total flux
through its own contour while a reconstruction's flux function is per radian, so
the store and the kernel could differ by two pi and a single unnoticed factor
would corrupt every current inverted through a loop.  That question is settled by
measurement rather than by reading a unit string, and the measurement is a gain
of one -- which is exactly what these tests hold the coupling to.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from nova.catalog.mast_geometry import physical_snapshot
from nova.imas.mast_flux_loop_adjudication import (
    DESCRIPTION_AGREEMENT,
    MAXIMUM_GAIN_SCATTER,
    MINIMUM_GAIN_SHOTS,
    ROUTE_AGREEMENT,
    LoopGainDisposition,
    LoopGainVerdict,
    LoopShotGain,
    archive_loop_gains,
    loop_channel_positions,
    loop_shot_gains,
    pool_loop_gains,
)
from nova.imas.mast_solve_inputs import reconstruction_loop_positions
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.imas.mast_vacuum_response import (
    ResponseError,
    coil_response_matrix,
    loop_response_matrix,
)

REPRESENTATIVE_SHOT = 11766
"""Shot the described loop set and the reconstruction table are read from."""

SINGLE_COIL_SHOTS = (14098, 14099, 14070, 14084)
"""Sustained shots that held one coil alone, one per side of two coil sets."""

_needs_store = pytest.mark.skipif(
    not Path(SHOT_STORE).is_dir(),
    reason=f"MAST level-1 store not present at {SHOT_STORE}",
)


@pytest.fixture(scope="module")
def geometry():
    """Return the repaired physical description the loops are coupled through."""

    return physical_snapshot(REPRESENTATIVE_SHOT)


@pytest.fixture(scope="module")
def targets(geometry):
    """Return the position each loop channel is read at."""

    return loop_channel_positions(
        geometry, reconstruction_loop_positions(REPRESENTATIVE_SHOT)
    )


def verdict(**overrides) -> LoopGainVerdict:
    """Build one channel's pooled verdict with every input stated outright."""

    row = {
        "channel": "fl_p4l_1",
        "gain": 1.0,
        "standard_error": 0.004,
        "shot_count": 120,
        "block_count": 1,
        "block_span": 1.0,
        "archive_gain": 1.0,
        "shape_agreement": 0.99,
        "described": True,
    }
    row.update(overrides)
    return LoopGainVerdict(**row)


# --- the flux column ---------------------------------------------------------


@_needs_store
def test_a_loop_column_is_the_kernel_flux_and_carries_no_orientation(geometry):
    """A closed loop links the total flux, so no sensitive axis enters."""

    positions = np.array([[1.5984, -1.04443], [0.1785, 0.6249]])
    response = loop_response_matrix(geometry, positions, families=["p4_lower"])

    assert response.shape == (2, 1)
    assert np.all(np.isfinite(response))
    assert abs(response[0, 0]) > abs(response[1, 0])


@_needs_store
def test_a_loop_and_a_probe_column_share_one_coil_model(geometry):
    """Two routes that disagreed about a coil would make their gains incomparable."""

    families = ["p4_lower", "p3_lower"]
    positions = np.array([[1.5984, -1.04443]])
    flux = loop_response_matrix(geometry, positions, families=families)
    field = coil_response_matrix(
        geometry,
        [
            type(
                "Target",
                (),
                {
                    "r": 1.5984,
                    "z": -1.04443,
                    "radial_cosine": 1.0,
                    "axial_sine": 0.0,
                },
            )()
        ],
        families=families,
    )

    assert flux.shape == field.shape == (1, 2)


@_needs_store
def test_a_loop_on_the_axis_is_refused_rather_than_evaluated(geometry):
    """A toroidal loop of zero radius encircles nothing and links no flux."""

    with pytest.raises(ResponseError, match="positive radius"):
        loop_response_matrix(geometry, np.array([[0.0, 0.0]]), families=["p4_lower"])


# --- where a channel is read -------------------------------------------------


@_needs_store
def test_a_served_channel_is_read_at_the_described_loop(targets):
    """A consumer evaluates the description, so the gain must answer for it."""

    r, z, described = targets["fl_p4l_1"]

    assert described
    assert (round(r, 4), round(z, 4)) == (1.5984, -1.0444)


@_needs_store
def test_a_refused_channel_is_still_measurable_at_its_own_position(targets):
    """Measurable and admissible are different, and the flag keeps them apart."""

    r, z, described = targets["fl_p6u_1"]

    assert not described
    assert (round(r, 4), round(z, 4)) == (1.4025, 0.889)


# --- the convention, measured ------------------------------------------------


@_needs_store
@pytest.mark.parametrize("shot", SINGLE_COIL_SHOTS)
def test_a_loop_reads_the_flux_the_description_predicts(geometry, targets, shot):
    """Gain one is the whole convention verdict: no two pi separates the two."""

    rows = [
        row
        for row in loop_shot_gains(geometry, targets, shot, _multipliers(), stride=8)
        if row.shape_agreement >= 0.9
    ]
    gains = np.array([row.gain for row in rows])

    assert gains.size >= 15
    assert abs(float(np.median(gains)) - 1.0) < 0.1
    assert float(np.median(gains)) < 2.0
    assert float(np.median(gains)) > 0.5


@_needs_store
@pytest.mark.parametrize("shot", SINGLE_COIL_SHOTS[:2])
def test_the_archive_forward_prediction_reads_the_same_channels(targets, shot):
    """A route sharing no geometry and no estimator is what a promotion needs."""

    ratios = archive_loop_gains(shot, sorted(targets))

    assert len(ratios) >= 15
    assert abs(float(np.median(list(ratios.values()))) - 1.0) < 0.05


@_needs_store
def test_the_restored_positions_are_what_the_measurement_prefers(geometry, targets):
    """The repair is falsifiable, and the falsifier is consistency, not amplitude.

    A gain absorbs any constant error, so no single shot can refute a position: a
    wrong one merely fits a different scale.  What a wrong position cannot do is
    fit the SAME scale to every shot, because the flux it gets wrong depends on
    which coil is driven.  Across shots that drove four different coils the
    copied positions need gains from a half to nearly two while the restored ones
    need one number, and that is the whole argument.
    """

    copied = dict(targets)
    for channel, twin in (("fl_p4l_1", "fl_p3l_1"), ("fl_p4l_4", "fl_p3l_4")):
        copied[channel] = targets[twin]

    restored: dict[str, list[float]] = {}
    wrong: dict[str, list[float]] = {}
    weights = _multipliers()
    for shot in SINGLE_COIL_SHOTS:
        for target, collected in ((targets, restored), (copied, wrong)):
            for row in loop_shot_gains(geometry, target, shot, weights, stride=8):
                if row.channel in ("fl_p4l_1", "fl_p4l_4"):
                    collected.setdefault(row.channel, []).append(row.gain)

    for channel in ("fl_p4l_1", "fl_p4l_4"):
        assert len(restored[channel]) == len(SINGLE_COIL_SHOTS)
        assert abs(float(np.mean(restored[channel])) - 1.0) < 0.05
        assert float(np.std(restored[channel])) < 0.05
        assert float(np.std(wrong[channel])) > 10.0 * float(np.std(restored[channel]))


def _multipliers() -> dict[str, float]:
    """Return what multiplies each excitation channel to give ampere turns."""

    from nova.imas.mast_fitted_parameters import VACUUM_FITTED_TURNS
    from nova.imas.mast_vacuum_cohort import COIL_DRIVES

    weights = {
        row.family: row.turns / row.turns_per_multiplier
        for row in VACUUM_FITTED_TURNS
        if row.identified
    }
    for drive in COIL_DRIVES:
        if drive.reports_ampere_turns:
            weights[drive.family] = 1.0
    return weights


# --- the verdict -------------------------------------------------------------


def test_a_gain_the_description_already_reproduces_needs_no_correction():
    """Promoting 1.002 would write the description's own residual into the data."""

    row = verdict(gain=1.0 + DESCRIPTION_AGREEMENT / 2.0)

    assert row.disposition is LoopGainDisposition.ADMITTED
    assert "they agree with each other to" in row.cause


def test_a_departure_both_routes_see_is_promoted():
    """An independent route seeing the same number is what makes it the channel."""

    row = verdict(gain=0.5011, archive_gain=0.5023)

    assert row.departure > DESCRIPTION_AGREEMENT
    assert row.corroborated
    assert row.disposition is LoopGainDisposition.PROMOTED
    assert "forward prediction" in row.cause


def test_a_departure_only_this_route_sees_is_not_a_channel_state():
    """One route disagreeing with the description is a model error until confirmed."""

    row = verdict(gain=0.80, archive_gain=1.00)

    assert not row.corroborated
    assert row.disposition is LoopGainDisposition.EXCLUDED
    assert "one route separates it and the other does not" in row.cause


def test_a_channel_recorded_at_two_settings_has_no_single_gain():
    """An average across an acquisition step describes no shot in either block."""

    row = verdict(block_count=2, block_span=2.0)

    assert not row.steady
    assert row.disposition is LoopGainDisposition.EXCLUDED
    assert "acquisition blocks" in row.cause


def test_a_gain_that_scatters_past_its_own_bound_is_excluded():
    """A number that moves between shots is not one number."""

    row = verdict(standard_error=MAXIMUM_GAIN_SCATTER * 2.0)

    assert not row.steady
    assert row.disposition is LoopGainDisposition.EXCLUDED
    assert "scatter" in row.cause


def test_a_channel_too_few_shots_measured_is_not_read_at_all():
    """A pooled scale over three shots has a spread, not a scatter."""

    row = verdict(shot_count=MINIMUM_GAIN_SHOTS - 1)

    assert row.disposition is LoopGainDisposition.EXCLUDED
    assert "below the" in row.cause


def test_an_unmeasured_archive_ratio_never_corroborates():
    """Missing evidence is not agreement, however close the fitted gain looks."""

    row = verdict(gain=0.5, archive_gain=math.nan)

    assert not row.corroborated
    assert row.disposition is LoopGainDisposition.EXCLUDED


def test_the_route_agreement_is_what_separates_the_two_predictions():
    """The bound is on the routes' disagreement, not on either one's error."""

    assert verdict(gain=0.50, archive_gain=0.50 + ROUTE_AGREEMENT / 2).corroborated
    assert not verdict(gain=0.50, archive_gain=0.50 + ROUTE_AGREEMENT * 2).corroborated


# --- pooling -----------------------------------------------------------------


def shot_gain(shot: int, gain: float, agreement: float = 0.99) -> LoopShotGain:
    """Build one shot's scale for one channel."""

    return LoopShotGain(
        shot=shot,
        channel="fl_p4l_1",
        gain=gain,
        shape_agreement=agreement,
        residual=1.0e-4,
        signal=1.0e-2,
        standoff=5.0,
        sample_count=2000,
    )


def test_a_shot_whose_prediction_has_the_wrong_shape_does_not_vote():
    """The best scale onto the wrong waveform is a projection, not a gain."""

    rows = [shot_gain(index, 1.0) for index in range(20)]
    rows.append(shot_gain(99, 40.0, agreement=0.01))

    pooled = pool_loop_gains(rows)

    assert len(pooled) == 1
    assert pooled[0].shot_count == 20
    assert pooled[0].gain == pytest.approx(1.0)


def test_the_bound_is_the_shots_own_disagreement():
    """A sample-count error would shrink by two orders and mean nothing."""

    rows = [shot_gain(index, 1.0 + 0.01 * (-1) ** index) for index in range(16)]

    pooled = pool_loop_gains(rows)[0]

    assert pooled.gain == pytest.approx(1.0, abs=1e-9)
    assert pooled.standard_error == pytest.approx(0.01 / math.sqrt(16), rel=0.1)


def test_a_channel_with_no_admissible_shot_is_absent_rather_than_zero():
    """Reporting a gain of nothing would read as a dead channel."""

    assert pool_loop_gains([shot_gain(1, 1.0, agreement=0.0)]) == ()


def test_what_every_loop_shares_is_not_any_loop_s_own_gain():
    """A factor the whole set carries belongs to whatever predicts them all."""

    rows = [shot_gain(index, 0.96, agreement=0.99) for index in range(30)]
    for index in range(30):
        rows.append(
            LoopShotGain(
                shot=index,
                channel="fl_cc03",
                gain=0.96,
                shape_agreement=0.99,
                residual=1.0e-4,
                signal=1.0e-2,
                standoff=5.0,
                sample_count=2000,
            )
        )

    pooled = pool_loop_gains(rows)

    assert {row.channel for row in pooled} == {"fl_p4l_1", "fl_cc03"}
    for row in pooled:
        assert row.gain == pytest.approx(0.96)
        assert row.reference == pytest.approx(0.96)
        assert row.departure == pytest.approx(0.0, abs=1e-9)
        assert row.disposition is LoopGainDisposition.ADMITTED


def test_a_channel_apart_from_the_set_is_the_one_the_departure_names():
    """The outlier is what a per-channel gain can be, and only the outlier."""

    rows: list[LoopShotGain] = []
    for channel, gain in (("fl_cc01", 0.96), ("fl_cc03", 0.96), ("fl_p4l_1", 0.48)):
        rows.extend(
            LoopShotGain(
                shot=index,
                channel=channel,
                gain=gain,
                shape_agreement=0.99,
                residual=1.0e-4,
                signal=1.0e-2,
                standoff=5.0,
                sample_count=2000,
            )
            for index in range(30)
        )

    pooled = {row.channel: row for row in pool_loop_gains(rows)}

    assert pooled["fl_cc01"].departure == pytest.approx(0.0, abs=1e-9)
    assert pooled["fl_p4l_1"].departure == pytest.approx(0.5, abs=1e-9)
