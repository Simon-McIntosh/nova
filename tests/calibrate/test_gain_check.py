"""Synthetic vacuum pulses recover gain histories and physical pickup states.

The manufactured response rows are Green's columns in miniature: each entry is the
field one recorded current column contributes at a channel. No store, machine
adapter, or correction instance participates, so every asserted transition was
chosen before the kernel saw it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.calibrate.gain_check import (
    GainCheckError,
    fit_gain_check,
    fit_pulse_gain_checks,
)
from nova.calibrate.scale_step import scale_blocks, scale_steps
from nova.calibrate.windows import PulseWindow, WindowKind

SAMPLES = 640
DRIVEN = PulseWindow(WindowKind.driven, 0.1, 0.9, 64, 576)
CHANNEL = "probe"


def recorded_currents(seed: int = 0) -> np.ndarray:
    """Return independently shaped current columns inside one driven window."""

    clock = np.linspace(0.0, 1.0, SAMPLES)
    currents = np.zeros((SAMPLES, 3))
    local = clock[DRIVEN.indices]
    currents[DRIVEN.indices, 0] = 1.4 + np.sin((2.3 + 0.1 * seed) * np.pi * local)
    currents[DRIVEN.indices, 1] = 0.8 + 0.6 * np.cos((4.1 + 0.2 * seed) * local)
    currents[DRIVEN.indices, 2] = 0.5 + local ** (1.2 + 0.05 * seed)
    return currents


def pulse_signal(
    currents: np.ndarray,
    response: np.ndarray,
    gain: float,
    *,
    noise: float = 0.0,
) -> np.ndarray:
    """Return one channel with additive instrument walk still present."""

    clock = np.linspace(0.0, 1.0, SAMPLES)
    generator = np.random.default_rng(710)
    return (
        gain * (currents @ response)
        + 0.03
        + 0.004 * clock
        + generator.normal(0.0, noise, SAMPLES)
    )


def instrument_walk(_: str, clock: np.ndarray) -> np.ndarray:
    return 0.03 + 0.004 * clock


def test_an_injected_gain_error_is_recovered_within_five_parts_in_ten_thousand():
    clock = np.linspace(0.0, 1.0, SAMPLES)
    currents = recorded_currents()
    response = np.asarray([0.8, -0.35, 0.55])
    injected = 1.083
    result = fit_pulse_gain_checks(
        clock,
        currents,
        [DRIVEN],
        [CHANNEL],
        lambda _: pulse_signal(currents, response, injected, noise=2.0e-4),
        lambda _: response,
        instrument_for=instrument_walk,
        pulse=4100,
    )

    check = result.for_channel(CHANNEL)
    assert not result.rejected
    assert check.gain == pytest.approx(injected, rel=5.0e-4)
    assert check.shape_agreement > 0.999
    assert check.sample_count == DRIVEN.sample_count
    assert check.pulse == 4100


def test_an_injected_scale_step_starts_on_the_exact_pulse_that_changed():
    clock = np.linspace(0.0, 1.0, SAMPLES)
    response = np.asarray([0.7, -0.2, 0.45])
    measured: dict[int, list[float]] = {}
    for pulse in range(5100, 5108):
        currents = recorded_currents(pulse - 5100)
        injected = 1.0 if pulse < 5104 else 2.0
        check = fit_pulse_gain_checks(
            clock,
            currents,
            [DRIVEN],
            [CHANNEL],
            lambda _, current=currents, gain=injected: pulse_signal(
                current, response, gain
            ),
            lambda _: response,
            instrument_for=instrument_walk,
            pulse=pulse,
        ).for_channel(CHANNEL)
        measured[pulse] = [check.gain]

    step = scale_steps(scale_blocks(CHANNEL, measured))[0]
    assert step.before_shot == 5103
    assert step.after_shot == 5104
    assert step.ratio == pytest.approx(2.0, rel=1.0e-12)


def test_response_shape_distinguishes_pair_state_from_a_gain_change():
    clock = np.linspace(0.0, 1.0, SAMPLES)
    currents = recorded_currents()
    responses = {
        "both_members": np.asarray([0.9, -0.4, 0.3]),
        "single_member": np.asarray([0.25, -0.55, 0.8]),
    }

    nominal = fit_pulse_gain_checks(
        clock,
        currents,
        [DRIVEN],
        [CHANNEL],
        lambda _: pulse_signal(currents, responses["both_members"], 1.0),
        lambda _: responses,
        instrument_for=instrument_walk,
        pulse=6100,
    ).for_channel(CHANNEL)
    gain_change = fit_pulse_gain_checks(
        clock,
        currents,
        [DRIVEN],
        [CHANNEL],
        lambda _: pulse_signal(currents, responses["both_members"], 1.2),
        lambda _: responses,
        instrument_for=instrument_walk,
        pulse=6101,
    ).for_channel(CHANNEL)
    state_change = fit_pulse_gain_checks(
        clock,
        currents,
        [DRIVEN],
        [CHANNEL],
        lambda _: pulse_signal(currents, responses["single_member"], 1.2),
        lambda _: responses,
        instrument_for=instrument_walk,
        pulse=6102,
    ).for_channel(CHANNEL)

    assert nominal.response_state == gain_change.response_state == "both_members"
    assert gain_change.gain == pytest.approx(1.2, rel=1.0e-12)
    assert state_change.response_state == "single_member"
    assert state_change.gain == pytest.approx(gain_change.gain, rel=1.0e-12)
    assert min(gain_change.state_separation, state_change.state_separation) > 0.05


def test_scale_only_response_states_are_refused_as_gain_confounded():
    clock = np.linspace(0.0, 1.0, SAMPLES)
    currents = recorded_currents()
    response = np.asarray([0.9, -0.4, 0.3])
    with pytest.raises(GainCheckError, match="pair state and gain are confounded"):
        fit_gain_check(
            clock,
            currents,
            pulse_signal(currents, response, 1.1) - instrument_walk(CHANNEL, clock),
            {"both_members": response, "single_member": 0.5 * response},
            [DRIVEN],
            channel=CHANNEL,
        )


def test_a_non_driven_window_is_refused():
    clock = np.linspace(0.0, 1.0, SAMPLES)
    currents = recorded_currents()
    response = np.asarray([0.9, -0.4, 0.3])
    quiet = PulseWindow(WindowKind.quiet, 0.0, 0.1, 0, 64)
    with pytest.raises(GainCheckError, match="quiet window"):
        fit_gain_check(
            clock,
            currents,
            currents @ response,
            response,
            [quiet],
        )


def test_one_bad_channel_is_reported_without_losing_the_other_checks():
    clock = np.linspace(0.0, 1.0, SAMPLES)
    currents = recorded_currents()
    response = np.asarray([0.9, -0.4, 0.3])
    result = fit_pulse_gain_checks(
        clock,
        currents,
        [DRIVEN],
        ["working", "missing"],
        lambda channel: (
            pulse_signal(currents, response, 0.97) if channel == "working" else np.nan
        ),
        lambda _: response,
        instrument_for=instrument_walk,
        pulse=7100,
    )

    assert result.for_channel("working").gain == pytest.approx(0.97, rel=1.0e-12)
    assert [row.channel for row in result.rejected] == ["missing"]
    assert "signal has shape" in result.rejected[0].reason


def test_too_few_finite_driven_samples_are_reported_as_rejected():
    clock = np.linspace(0.0, 1.0, SAMPLES)
    currents = recorded_currents()
    response = np.asarray([0.9, -0.4, 0.3])
    signal = currents @ response
    signal[DRIVEN.start_index + 100 :] = np.nan
    result = fit_pulse_gain_checks(
        clock,
        currents,
        [DRIVEN],
        [CHANNEL],
        lambda _: signal,
        lambda _: response,
        pulse=8100,
    )

    assert not result.checks
    assert "100 finite driven samples" in result.rejected[0].reason
