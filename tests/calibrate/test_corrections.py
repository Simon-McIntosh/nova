"""Whether a chain of corrections puts a manufactured signal back the way it was.

A correction is one multiplication or one subtraction, and a chain of them is only
right if every step is right and they happen in the right order.  Manufacturing the
corrupted signal is what makes both testable at once: take a clean waveform, apply
the corruptions an instrument applies -- a gain, a pickup state, an acquisition range
setting, a standing offset, an integrator ramp -- and the chain has to give the clean
waveform back to floating-point precision or it has the order wrong.

The order is the part a test earns its keep on, because every order produces numbers
and only one produces the right ones.  A signal corrupted by a gain and an offset and
then corrected offset-last comes back wrong by the offset divided by the gain, which
is a small number that looks like a residual rather than like a bug.

The refusals get the same treatment as the arithmetic.  Each one exists because the
alternative is silent: a channel drawing two corrections onto one stage, a value that
describes no pulse because the channel flips between two, an interval bounded in a
coordinate the read never named, and a channel the record says to drop.  Every one of
those would otherwise return a plausible array.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.calibrate.correction_model import (
    ApplicationStage,
    CorrectionKind,
    CorrectionSet,
)
from nova.calibrate.correction_set import APPLICATION_ORDER, validate_correction_set
from nova.calibrate.corrections import (
    CorrectionApplicationError,
    apply_chain,
    apply_corrections,
    build_chain,
)

PROVENANCE = {
    "method": "synthetic truth for the application engine",
    "evidence_uri": "tests/calibrate/test_corrections.py",
}

LADDER = {
    "name": "acquisition_range",
    "kind": "acquisition_scale",
    "rungs": [0.5, 0.7071067811865475, 1.0, 1.4142135623730951, 2.0],
    "tolerance": 0.08,
}

RUNGS = (0.5, 0.7071067811865475, 1.4142135623730951, 2.0)
"""The exact factors a binary acquisition ladder offers, one half through two."""

SAMPLES = 64
"""Samples in a manufactured waveform; the engine reads no minimum count."""


def correction(**overrides):
    """Return one promoted correction, overridden slot by slot."""

    row = {
        "channel": "probe01",
        "kind": "gain",
        "status": "promoted",
        "value": 0.5,
        "validity": [{}],
        "provenance": dict(PROVENANCE),
    }
    row.update(overrides)
    return row


def document(*corrections):
    """Return a validated document carrying the given corrections."""

    parsed = CorrectionSet.model_validate(
        {
            "machine": "synthetic",
            "diagnostic_system": "magnetics",
            "schema_version": "1.0.0",
            "set_version": "1.0.0",
            "generated_by": "tests/calibrate/test_corrections.py",
            "ladders": [dict(LADDER)],
            "corrections": list(corrections),
        }
    )
    validate_correction_set(parsed)
    return parsed


def waveform():
    """Return a clean signal and the time base it was sampled on."""

    time = np.linspace(0.0, 0.5, SAMPLES)
    return np.sin(6.0 * time) + 0.25 * time**2, time


def test_the_whole_chain_returns_the_signal_it_was_manufactured_from():
    truth, time = waveform()
    gain, pair, rung, offset, drift = 1.37, 0.5, 2.0, -0.083, 0.019
    recorded = truth * gain * pair * rung + offset + drift * (time - time[0])
    parsed = document(
        correction(kind="gain", value=gain),
        correction(kind="pair_state", value=pair, state="single_member"),
        correction(
            kind="acquisition_scale", value=rung, ladder="acquisition_range"
        ),
        correction(kind="offset", value=offset, unit="T"),
        correction(kind="drift_rate", value=drift, unit="T/s"),
    )
    corrected, chain = apply_corrections(parsed, "probe01", recorded, time=time)
    assert np.allclose(corrected, truth, rtol=0.0, atol=1e-12)
    assert [step.stage for step in chain.steps] == [
        ApplicationStage.offset,
        ApplicationStage.drift,
        ApplicationStage.acquisition_scale,
        ApplicationStage.pair_state,
        ApplicationStage.gain,
    ]


def test_the_declared_order_is_the_one_the_chain_walks():
    parsed = document(
        correction(kind="gain", value=1.1),
        correction(kind="offset", value=0.2),
        correction(kind="acquisition_scale", value=2.0, ladder="acquisition_range"),
    )
    chain = build_chain(parsed, "probe01")
    stages = [step.stage for step in chain.steps]
    assert stages == sorted(stages, key=APPLICATION_ORDER.index)


def test_removing_the_offset_after_the_gain_returns_a_different_signal():
    """The order is load-bearing, not decorative.

    Correcting a gain before an offset leaves the offset divided by the gain behind,
    which is small enough to be read as a residual rather than as a fault.
    """

    truth, _ = waveform()
    gain, offset = 1.6, 0.4
    recorded = truth * gain + offset
    parsed = document(
        correction(kind="gain", value=gain), correction(kind="offset", value=offset)
    )
    corrected, _ = apply_corrections(parsed, "probe01", recorded)
    reversed_order = recorded / gain - offset
    assert np.allclose(corrected, truth, atol=1e-12)
    assert np.allclose(reversed_order - truth, -offset + offset / gain, atol=1e-12)
    assert not np.allclose(reversed_order, truth, atol=1e-6)


@pytest.mark.parametrize("rung", RUNGS)
def test_an_exact_acquisition_rung_divides_out_exactly(rung):
    truth, _ = waveform()
    parsed = document(
        correction(kind="acquisition_scale", value=rung, ladder="acquisition_range")
    )
    corrected, chain = apply_corrections(parsed, "probe01", truth * rung)
    assert np.allclose(corrected, truth, rtol=0.0, atol=1e-15)
    assert chain.multiplier == rung


def test_a_drift_ramp_is_removed_against_the_time_it_accumulated_over():
    truth, time = waveform()
    slope = 0.31
    parsed = document(correction(kind="drift_rate", value=slope, unit="T/s"))
    corrected, _ = apply_corrections(
        parsed, "probe01", truth + slope * (time - time[0]), time=time
    )
    assert np.allclose(corrected, truth, atol=1e-13)


def test_a_drift_with_no_time_base_is_refused():
    parsed = document(correction(kind="drift_rate", value=0.1, unit="T/s"))
    chain = build_chain(parsed, "probe01")
    with pytest.raises(CorrectionApplicationError, match="no time base"):
        apply_chain(chain, np.zeros(SAMPLES))


def test_each_pulse_era_carries_its_own_state():
    """A pickup that loses an element halves, and the two eras correct differently."""

    truth, _ = waveform()
    parsed = document(
        correction(
            kind="pair_state",
            value=1.0,
            state="both_members",
            validity=[{"pulse_start": 100, "pulse_end": 199}],
        ),
        correction(
            kind="pair_state",
            value=0.5,
            state="single_member",
            validity=[{"pulse_start": 200, "pulse_end": 299}],
        ),
    )
    early, _ = apply_corrections(parsed, "probe01", truth, pulse=150)
    late, _ = apply_corrections(parsed, "probe01", truth * 0.5, pulse=250)
    assert np.allclose(early, truth, atol=1e-15)
    assert np.allclose(late, truth, atol=1e-15)


def test_a_pulse_outside_every_era_draws_no_correction():
    parsed = document(
        correction(value=0.5, validity=[{"pulse_start": 100, "pulse_end": 199}])
    )
    assert build_chain(parsed, "probe01", pulse=500).steps == ()


def test_an_interval_bounded_in_a_coordinate_the_read_never_named_is_refused():
    parsed = document(
        correction(value=0.5, validity=[{"pulse_start": 100, "pulse_end": 199}])
    )
    with pytest.raises(CorrectionApplicationError, match="names no pulse"):
        build_chain(parsed, "probe01")


def test_a_channel_and_a_group_it_belongs_to_cannot_both_own_one_stage():
    parsed = document(
        correction(kind="gain", value=0.5),
        correction(channel=None, channel_group="outboard", kind="gain", value=0.9),
    )
    with pytest.raises(CorrectionApplicationError, match="two corrections onto the"):
        build_chain(parsed, "probe01", groups={"outboard": ["probe01", "probe02"]})


def test_a_group_correction_reaches_only_its_members():
    parsed = document(
        correction(channel=None, channel_group="flux_loop", kind="convention", value=2.0)
    )
    groups = {"flux_loop": ["loop01"]}
    assert build_chain(parsed, "loop01", groups=groups).multiplier == 2.0
    assert build_chain(parsed, "probe01", groups=groups).steps == ()


def test_a_channel_flipping_between_two_states_is_refused_not_averaged():
    parsed = document(
        correction(
            kind="pair_state",
            status="recorded",
            value=None,
            state="indeterminate",
            candidate_values=[1.25, 0.5],
        )
    )
    with pytest.raises(CorrectionApplicationError, match=r"\[1.25, 0.5\]"):
        build_chain(parsed, "probe01", statuses=["recorded"])


def test_an_unresolved_state_applies_once_the_caller_states_which_one_held():
    truth, _ = waveform()
    parsed = document(
        correction(
            kind="pair_state",
            status="recorded",
            value=None,
            state="indeterminate",
            candidate_values=[1.25, 0.5],
        )
    )
    corrected, chain = apply_corrections(
        parsed,
        "probe01",
        truth * 1.25,
        statuses=["recorded"],
        resolution={("probe01", CorrectionKind.pair_state): 1.25},
    )
    assert np.allclose(corrected, truth, atol=1e-15)
    assert chain.steps[0].resolved


def test_a_resolution_of_zero_is_refused():
    parsed = document(
        correction(status="recorded", value=None, candidate_values=[1.25, 0.5])
    )
    with pytest.raises(CorrectionApplicationError, match="multiplier of zero"):
        build_chain(
            parsed,
            "probe01",
            statuses=["recorded"],
            resolution={("probe01", CorrectionKind.gain): 0.0},
        )


def test_an_excluded_channel_refuses_to_be_corrected():
    parsed = document(
        correction(kind="exclusion", status="promoted", value=None, cause="reads noise")
    )
    chain = build_chain(parsed, "probe01")
    assert chain.excluded and chain.exclusions == ("reads noise",)
    with pytest.raises(CorrectionApplicationError, match="is excluded"):
        apply_chain(chain, np.zeros(SAMPLES))
    assert np.allclose(
        apply_chain(chain, np.ones(SAMPLES), allow_excluded=True), np.ones(SAMPLES)
    )


def test_a_quality_state_describes_the_channel_and_scales_nothing():
    truth, _ = waveform()
    parsed = document(
        correction(
            kind="quality",
            status="promoted",
            value=None,
            quality_status="suspect",
            cause="disagrees with its own neighbours",
        )
    )
    corrected, chain = apply_corrections(parsed, "probe01", truth)
    assert chain.steps == () and not chain.excluded
    assert chain.quality[0][1] == "disagrees with its own neighbours"
    assert np.allclose(corrected, truth)


def test_a_read_inside_a_span_says_whether_the_span_measured_it():
    parsed = document(
        correction(
            value=0.5,
            validity=[
                {
                    "pulse_start": 100,
                    "pulse_end": 900,
                    "measured_pulses": [100, 400, 900],
                }
            ],
        )
    )
    assert build_chain(parsed, "probe01", pulse=400).steps[0].measured
    assert not build_chain(parsed, "probe01", pulse=500).steps[0].measured
    assert build_chain(parsed, "probe01", pulse=500).extrapolated


def test_only_promoted_corrections_reach_the_chain_by_default():
    parsed = document(correction(status="recorded", value=0.5))
    assert build_chain(parsed, "probe01").steps == ()
    assert build_chain(parsed, "probe01", statuses=["recorded"]).multiplier == 0.5
