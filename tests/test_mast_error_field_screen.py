"""What the non-axisymmetric screen refuses, and the case each refusal answers.

The screen exists because an axisymmetric forward model cannot represent a coil
that is not axisymmetric, so whatever such a coil put on a probe would be
attributed to some other term.  Three things it has to get right are tested here
against cases built to break them: a channel named one way on some campaigns and
another way on others, a channel that looks like the strongest such coil in the
store and is in fact a monitor on an axisymmetric supply, and a channel whose
response to the excitation is real but too local to be a field.

The last one carries the screen's whole design.  A coupled channel and a
misdescribed field look identical in one channel's numbers and completely
different across an array, so the tests insist the screen refuse that channel
without refusing the shot.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nova.imas.mast_error_field_screen import (
    DRIVEN_CURRENT,
    PAIR_CURRENT_TOLERANCE,
    ERROR_FIELD_ALIASES,
    ERROR_FIELD_CHANNELS,
    MINIMUM_COUPLING_SHOTS,
    NEIGHBOUR_INCOHERENCE,
    QUIESCENT_CURRENT,
    SUPPLY_MONITOR_CORRELATION,
    ChannelCoupling,
    ErrorFieldDrive,
    ErrorFieldError,
    ErrorFieldScreen,
    SupplyMonitor,
    matched_pairs,
    measure_error_field_coupling,
    probe_response_to_drive,
    screen_shot_set,
)

FLOOR = 8.0e-5
"""Tesla of quiescent scatter the synthetic channels are given."""


def unmeasured_drive(shot: int, *, samples: int = 800) -> ErrorFieldDrive:
    """Build a shot from a campaign that recorded none of these channels."""

    time = np.linspace(-0.4, 1.2, samples)
    return ErrorFieldDrive(
        shot=shot,
        time=time,
        waveforms={},
        absent=tuple(sorted(ERROR_FIELD_CHANNELS + ERROR_FIELD_ALIASES)),
        quiescent_mask=np.ones(time.shape, dtype=bool),
    )


def drive(
    shot: int,
    peaks: dict[str, float],
    *,
    samples: int = 800,
    absent: tuple[str, ...] = (),
) -> ErrorFieldDrive:
    """Build an excitation that idles, rises to its peak, and falls back.

    The excitation is rectangular so that no sample is partially driven: every
    sample in the offset window carries exactly zero current, the estimator's zero
    is then exactly the integrator offset, and a slope it returns can be compared
    with the planted one to machine precision.  A ramp would leave one sample just
    below the quiescent threshold, and that sample's signal biases the zero by
    about a part in ten thousand.
    """

    time = np.linspace(-0.4, 1.2, samples)
    shape = ((time >= 0.1) & (time <= 0.9)).astype(float)
    recorded = dict.fromkeys(ERROR_FIELD_CHANNELS, 0.0) | dict(peaks)
    waveforms = {channel: peak * shape for channel, peak in recorded.items()}
    quiescent = np.ones(time.shape, dtype=bool)
    for values in waveforms.values():
        quiescent &= np.abs(values) < QUIESCENT_CURRENT
    return ErrorFieldDrive(
        shot=shot,
        time=time,
        waveforms=waveforms,
        absent=absent,
        quiescent_mask=quiescent,
    )


def coupling(
    channel: str,
    response: float,
    *,
    neighbour: float,
    shots: int = 6,
    driver: str = "error_field_05",
) -> ChannelCoupling:
    """Build one channel's coupling with its neighbourhood stated outright."""

    return ChannelCoupling(
        channel=channel,
        driver=driver,
        shot_count=shots,
        response=response,
        scatter=abs(response) * 0.2,
        noise_floor=FLOOR,
        neighbour_response=neighbour,
    )


# --- what counts as the excitation --------------------------------------


def test_both_naming_schemes_are_screened():
    """A campaign whose channels are named the other way is not silently clean.

    The earlier scheme carries the strongest excitation in the archive, so a
    screen that knows only the later names reports those shots as having driven
    nothing.
    """

    assert set(ERROR_FIELD_CHANNELS).isdisjoint(ERROR_FIELD_ALIASES)
    early = drive(1, {"error_field_a": 1.2e4}, absent=())
    late = drive(2, {"error_field_05": 3.2e3}, absent=ERROR_FIELD_ALIASES)
    assert early.driven and late.driven
    assert early.peak > late.peak
    assert early.strongest_channel == "error_field_a"
    assert set(late.absent) == set(ERROR_FIELD_ALIASES)
    assert not early.unmeasured and not late.unmeasured


def test_a_shot_nobody_screened_is_not_a_shot_that_passed():
    """A missing drive record is reported, never counted as quiescent."""

    screen = ErrorFieldScreen(couplings=(coupling("obr17", 1.1e-7, neighbour=2.0e-9),))
    outcome = screen_shot_set("training", [1, 2, 3], screen, {2: drive(2, {})})
    assert outcome.unscreened_shots == (1, 3)
    assert not outcome.clean
    assert outcome.refusals == {}


def test_a_channel_below_the_excitation_floor_drove_nothing():
    """The threshold that says "driven" is the one the coils are judged by."""

    quiet = drive(1, {"error_field_05": DRIVEN_CURRENT * 0.5})
    loud = drive(2, {"error_field_05": DRIVEN_CURRENT * 1.5})
    assert not quiet.driven
    assert loud.driven


# --- the channel that is not an error-field channel ----------------------


def test_a_monitor_on_an_axisymmetric_supply_is_identified_by_measurement():
    """A copy of another coil's waveform is a monitor, whatever it is called."""

    monitor = SupplyMonitor(
        channel="efps_current",
        shot_count=62,
        axisymmetric_share=1.0,
        correlation=0.9999,
        amplitude_ratio=0.125,
        best_channel="p2ol_coil_current",
    )
    assert monitor.identified
    assert abs(monitor.correlation) >= SUPPLY_MONITOR_CORRELATION
    assert not monitor.best_channel.startswith("error_field")
    assert json.loads(json.dumps(monitor.as_dict(), sort_keys=True))["identified"]


def test_a_channel_correlating_with_an_error_field_coil_is_not_a_monitor():
    """The test is on which channel it copies, not on how well it copies."""

    coil = SupplyMonitor(
        channel="error_field_05",
        shot_count=40,
        axisymmetric_share=0.0,
        correlation=0.99,
        amplitude_ratio=1.0,
        best_channel="error_field_02",
    )
    assert not coil.identified


def test_too_few_shots_cannot_identify_a_monitor():
    """One shot's correlation is a coincidence, not an identity."""

    thin = SupplyMonitor(
        channel="efps_current",
        shot_count=MINIMUM_COUPLING_SHOTS - 1,
        axisymmetric_share=1.0,
        correlation=0.9999,
        amplitude_ratio=0.125,
        best_channel="p2ol_feed_current",
    )
    assert not thin.identified


# --- the coupling and the threshold it derives ---------------------------


def test_the_threshold_is_the_current_that_reaches_the_channels_own_floor():
    """A screen limit is derived from a measurement, not chosen."""

    row = coupling("obr17", 1.1236e-7, neighbour=2.1e-9)
    assert row.measured
    assert row.threshold == pytest.approx(FLOOR / 1.1236e-7)
    assert row.threshold < DRIVEN_CURRENT


def test_a_channel_the_excitation_never_reaches_has_no_limit():
    """An unmeasurably small coupling gives an infinite threshold, not a small one."""

    row = coupling("ccbv20", 0.0, neighbour=2.0e-9)
    assert math.isinf(row.threshold)
    assert row.as_dict()["threshold"] is None


def test_an_unmeasured_coupling_does_not_set_a_threshold():
    """A slope from fewer shots than the criterion asks for constrains nothing."""

    row = coupling("obr17", 1.1e-7, neighbour=2.0e-9, shots=MINIMUM_COUPLING_SHOTS - 1)
    assert not row.measured
    assert math.isinf(row.threshold)


def test_a_response_far_above_its_neighbours_is_not_a_field():
    """A field varies smoothly across an array; a shared conductor does not."""

    coupled = coupling("obr17", 1.1236e-7, neighbour=2.135e-9)
    smooth = coupling("obr03", 6.0e-9, neighbour=5.4e-9)
    assert coupled.neighbour_ratio > NEIGHBOUR_INCOHERENCE
    assert coupled.shares_a_conductor
    assert smooth.neighbour_ratio < NEIGHBOUR_INCOHERENCE
    assert not smooth.shares_a_conductor


def test_a_channel_without_a_measured_floor_cannot_be_screened():
    """The floor is a measurement, so its absence is an error and not a default."""

    with pytest.raises(ErrorFieldError, match="noise floor"):
        ChannelCoupling(
            channel="obr17",
            driver="error_field_05",
            shot_count=6,
            response=1.0e-7,
            scatter=1.0e-8,
            noise_floor=0.0,
            neighbour_response=2.0e-9,
        ).threshold


def test_the_median_over_shots_is_what_sets_a_threshold():
    """One misbehaving shot must not set the limit for every later fit."""

    steady = [
        ("error_field_05", {"obr17": 1.1e-7, "obr16": 2.0e-9, "obr18": 2.2e-9})
        for _ in range(5)
    ]
    outlier = [("error_field_05", {"obr17": 5.0e-6, "obr16": 2.0e-9, "obr18": 2.2e-9})]
    rows = measure_error_field_coupling(
        steady + outlier, {"obr17": FLOOR, "obr16": FLOOR, "obr18": FLOOR}
    )
    fitted = next(row for row in rows if row.channel == "obr17")
    assert fitted.response == pytest.approx(1.1e-7)
    assert fitted.shot_count == 6
    assert fitted.shares_a_conductor


# --- what the screen does to a shot -------------------------------------


def test_the_screen_removes_a_channel_and_keeps_the_shot():
    """A shot losing one coupled channel is not a shot to discard.

    The archive's error-field drive reaches one channel's floor and nobody
    else's, so refusing the shot would cost the cohort most of its coverage for
    a drive seventy-six channels cannot see.
    """

    screen = ErrorFieldScreen(
        couplings=(
            coupling("obr17", 1.1236e-7, neighbour=2.1e-9),
            coupling("obr16", 2.0e-9, neighbour=1.1e-7),
            coupling("ccbv20", 2.2e-9, neighbour=2.0e-9),
        )
    )
    loud = drive(1, {"error_field_05": 3.2e3})
    assert screen.refused(loud) == ("obr17",)
    assert not screen.passes(loud)
    outcome = screen_shot_set("training", [1], screen, {1: loud})
    assert outcome.driven_shots == (1,)
    assert outcome.refusals == {1: ("obr17",)}
    assert outcome.shot_count == 1


def test_a_quiescent_shot_loses_nothing():
    """With the excitation off, no channel is refused."""

    screen = ErrorFieldScreen(
        couplings=(coupling("obr17", 1.1236e-7, neighbour=2.1e-9),)
    )
    assert screen.refused(drive(1, {"error_field_05": 10.0})) == ()
    assert screen.passes(drive(1, {}))
    outcome = screen_shot_set("noise", [1], screen, {1: drive(1, {})})
    assert outcome.clean


def test_the_strictest_measured_limit_is_the_one_applied():
    """A channel coupled to two coils is screened on whichever bites first."""

    screen = ErrorFieldScreen(
        couplings=(
            coupling("obr17", 1.1236e-7, neighbour=2.1e-9, driver="error_field_05"),
            coupling("obr17", 1.4566e-8, neighbour=9.2e-10, driver="error_field_02"),
        )
    )
    limit = screen.threshold("obr17")
    assert limit == pytest.approx(FLOOR / 1.1236e-7)
    assert screen.coupled_channels == ("obr17",)


def test_the_screen_record_round_trips_through_json():
    """The screen is evidence, so it has to serialize exactly."""

    screen = ErrorFieldScreen(
        couplings=(coupling("obr17", 1.1236e-7, neighbour=2.1e-9),)
    )
    record = screen.as_dict()
    assert json.loads(json.dumps(record, sort_keys=True)) == record
    assert record["quiescent_current"] == QUIESCENT_CURRENT


# --- reading a probe against the excitation ------------------------------


def test_the_slope_is_measured_against_a_zero_taken_while_the_coil_was_off():
    """A standing integrator offset must not become a coupling."""

    excitation = drive(1, {"error_field_05": 4.0e3})
    truth = 1.1e-7
    signals = {
        "obr17": truth * excitation.waveforms["error_field_05"] + 3.0e-4,
        "obr16": 2.0e-9 * excitation.waveforms["error_field_05"] - 5.0e-4,
    }
    slopes = probe_response_to_drive(excitation, signals)
    assert slopes["obr17"] == pytest.approx(truth, rel=1e-9)
    assert slopes["obr16"] == pytest.approx(2.0e-9, rel=1e-9)


def test_a_shot_with_no_such_channel_cannot_be_regressed_on():
    """Asking for a coupling where the channel was never recorded is an error."""

    with pytest.raises(ErrorFieldError, match="drove no error-field channel"):
        probe_response_to_drive(unmeasured_drive(14061), {"obr17": np.zeros(800)})


def test_a_shot_recorded_at_zero_yields_no_slope_rather_than_an_error():
    """A channel present and quiet is a valid shot that simply constrains nothing."""

    assert probe_response_to_drive(drive(1, {}), {"obr17": np.zeros(800)}) == {}


# --- the matched-pair isolation set --------------------------------------


def test_a_pair_is_matched_on_currents_and_not_on_a_family_label():
    """Two shots can share an excitation label and differ by a factor in drive.

    Matching on the label would pair them anyway and the difference would carry
    the mismatched poloidal coil rather than the error-field one.
    """

    peaks = {
        1: {"p4_upper": 9.0e3, "p5_upper": 8.0e3},
        2: {"p4_upper": 9.1e3, "p5_upper": 8.05e3},
        3: {"p4_upper": 4.0e3, "p5_upper": 8.0e3},
    }
    families = {1: "P4+P5", 2: "P4+P5", 3: "P4+P5"}
    field = {1: 3.2e3, 2: 12.0, 3: 15.0}
    pairs = matched_pairs(peaks, families, field)
    assert len(pairs) == 1
    assert pairs[0].driven_shot == 1
    assert pairs[0].quiet_shot == 2
    assert pairs[0].usable
    assert pairs[0].agreement <= PAIR_CURRENT_TOLERANCE


def test_a_partner_driving_a_coil_the_other_did_not_is_refused():
    """An extra coil on one side of the difference does not cancel."""

    peaks = {
        1: {"p4_upper": 9.0e3},
        2: {"p4_upper": 9.0e3, "sol": 1.5e4},
    }
    pairs = matched_pairs(peaks, {1: "P4", 2: "P1+P4"}, {1: 3.2e3, 2: 12.0})
    assert pairs == ()


def test_a_driven_shot_with_no_close_partner_yields_no_pair():
    """A pair too far apart in poloidal drive is refused, not merely flagged."""

    peaks = {1: {"p4_upper": 9.0e3}, 2: {"p4_upper": 6.0e3}}
    assert matched_pairs(peaks, {1: "P4", 2: "P4"}, {1: 3.2e3, 2: 12.0}) == ()


def test_each_driven_shot_keeps_only_its_closest_partner():
    """The set is one pair per experiment rather than every combination."""

    peaks = {
        1: {"p4_upper": 9.0e3},
        2: {"p4_upper": 9.2e3},
        3: {"p4_upper": 9.01e3},
    }
    families = {shot: "P4" for shot in peaks}
    pairs = matched_pairs(peaks, families, {1: 3.2e3, 2: 12.0, 3: 15.0})
    assert len(pairs) == 1
    assert pairs[0].quiet_shot == 3


def test_a_shot_whose_partner_also_drove_the_error_field_is_not_a_partner():
    """The quiet side has to be quiet in the excitation being isolated."""

    peaks = {1: {"p4_upper": 9.0e3}, 2: {"p4_upper": 9.0e3}}
    assert matched_pairs(peaks, {1: "P4", 2: "P4"}, {1: 3.2e3, 2: 2.9e3}) == ()


def test_a_pair_serializes_with_its_own_bound_on_the_cancellation():
    """The agreement is what bounds the difference, so it travels with the pair."""

    peaks = {1: {"p4_upper": 9.0e3}, 2: {"p4_upper": 9.1e3}}
    pair = matched_pairs(peaks, {1: "P4", 2: "P4"}, {1: 3.2e3, 2: 12.0})[0]
    row = pair.as_dict()
    assert json.loads(json.dumps(row, sort_keys=True)) == row
    assert row["agreement"] == pytest.approx(0.1e3 / 9.1e3)


def test_a_campaign_that_never_recorded_these_channels_is_refused_wholesale():
    """An absent channel is not a quiet one, and cannot be vouched for.

    The earliest campaigns predate the recording of the error-field coils, so a
    screen that reads their absence as quiescence passes exactly the shots it knows
    nothing about.
    """

    screen = ErrorFieldScreen(
        couplings=(
            coupling("obr17", 1.1236e-7, neighbour=2.1e-9),
            coupling("obr16", 2.0e-9, neighbour=1.1e-7),
        )
    )
    blind = unmeasured_drive(14061)
    assert blind.unmeasured
    assert not blind.driven
    assert blind.peak == 0.0
    assert set(screen.refused(blind)) == {"obr16", "obr17"}
    assert not screen.passes(blind)
    outcome = screen_shot_set("training", [14061], screen, {14061: blind})
    assert outcome.unmeasured_shots == (14061,)
    assert outcome.driven_shots == ()
    assert not outcome.clean
    assert blind.as_dict()["unmeasured"] is True


def test_a_recorded_and_quiet_shot_is_distinguished_from_an_unrecorded_one():
    """The two look identical in a peak and must not look identical in the screen."""

    screen = ErrorFieldScreen(
        couplings=(coupling("obr17", 1.1236e-7, neighbour=2.1e-9),)
    )
    quiet = drive(19995, {})
    assert not quiet.unmeasured
    assert screen.refused(quiet) == ()
    assert screen_shot_set("noise", [19995], screen, {19995: quiet}).clean
    assert screen.refused(unmeasured_drive(14061)) == ("obr17",)
