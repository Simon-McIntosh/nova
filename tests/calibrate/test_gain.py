"""Whether a manufactured gain, baseline and drift come back out of the fits.

Every case here builds a signal from a known scale and asks for the scale back.  That
is the only assertion a gain fit can be held to: recovery of injected truth, to a
stated tolerance, on data whose answer was decided before the fit ran.

Three of the screens get cases of their own because each was put in to refuse a
specific fit that returns a plausible number.  A drive contributing a tenth of the
predicted power has a scale, and it is not the channel's scale.  Two drives cancelling
at a channel make every drive's own power a large multiple of what the channel reads,
so the leverage test passes for the wrong reason and the coherence test is what
catches it.  A scale onto a waveform of the wrong shape exists and is a projection,
which the shape-agreement test refuses.

The joint scale-and-orientation solve gets the case that motivates its whole
construction: on one pulse driving one circuit the two columns carry the same
waveform, any scale trades against any angle, and the pair is not recoverable however
clean the data.  Add pulses whose circuits present different ratios of the two field
components and the same normal equations, summed, recover both.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nova.calibrate.gain import (
    GainError,
    axis_fit,
    baseline_offset,
    drift_fit,
    drive_gains,
    pool_normal_systems,
    pool_scalar_gains,
    score_axis_correction,
    through_origin_fit,
)

SAMPLES = 512
"""Samples per manufactured pulse, comfortably over the fits' own minimum."""


def drive_waveform(seed: int, count: int = SAMPLES) -> np.ndarray:
    """Return a smooth, non-repeating current ramp for one circuit."""

    time = np.linspace(0.0, 1.0, count)
    return np.sin((seed + 1) * 2.1 * time) * (1.0 + 0.3 * seed) + 0.5 * seed * time


def test_a_known_scale_comes_back_exactly():
    predictor = drive_waveform(0)
    fit = through_origin_fit(predictor, 1.37 * predictor)
    assert fit.slope == pytest.approx(1.37, rel=1e-12)
    assert fit.variance_explained == pytest.approx(1.0, abs=1e-12)
    assert fit.residual == pytest.approx(0.0, abs=1e-12)


def test_the_fit_carries_no_intercept_so_a_baseline_must_go_first():
    """A standing offset left in the target biases the scale.

    Which is why the offset is its own correction kind with its own stage: a fit that
    absorbed it would report the bias inside a number read as a gain.
    """

    predictor = drive_waveform(1)
    truth, offset = 0.9, 4.0
    biased = through_origin_fit(predictor, truth * predictor + offset)
    assert biased.slope != pytest.approx(truth, rel=1e-3)
    quiet = np.zeros(predictor.size, dtype=bool)
    quiet[:50] = True
    corrupted = truth * predictor + offset
    removed = corrupted - baseline_offset(corrupted, quiet)
    assert through_origin_fit(
        predictor - predictor[:50].mean(), removed
    ).slope == pytest.approx(truth, rel=1e-12)


def test_a_baseline_is_measured_only_over_the_finite_samples_of_its_window():
    signal = np.full(SAMPLES, 3.0)
    quiet = np.zeros(SAMPLES, dtype=bool)
    quiet[:20] = True
    signal[5] = np.nan
    assert baseline_offset(signal, quiet) == pytest.approx(3.0)


def test_a_quiet_window_holding_nothing_finite_is_refused():
    signal = np.full(SAMPLES, np.nan)
    quiet = np.ones(SAMPLES, dtype=bool)
    with pytest.raises(GainError, match="not measurable"):
        baseline_offset(signal, quiet)


def test_a_drift_ramp_is_recovered_and_the_scatter_survives_it():
    time = np.linspace(0.0, 2.0, SAMPLES)
    slope, level, noise = 0.37, -1.2, 0.05
    generator = np.random.default_rng(11)
    signal = level + slope * time + generator.normal(0.0, noise, SAMPLES)
    fit = drift_fit(time, signal)
    assert fit.slope == pytest.approx(slope, abs=0.01)
    assert fit.intercept == pytest.approx(level, abs=0.01)
    assert fit.scatter == pytest.approx(noise, rel=0.15)


def test_a_scatter_about_the_mean_would_read_the_drift_as_noise():
    time = np.linspace(0.0, 2.0, SAMPLES)
    signal = 0.37 * time
    assert drift_fit(time, signal).scatter == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(signal)) > 0.2


def test_the_dominant_drive_owns_the_scale_and_the_others_are_not_reported():
    """A reported scale is the dominant drive's, carrying the faint one's leakage.

    Which is what the leverage bar is for.  The scale fitted against one column
    absorbs whatever the columns it omits contributed, so the recovered number is the
    truth to about the share the omitted columns hold -- tolerable where one drive
    carries all but a few parts in a thousand of the predicted power, and meaningless
    where it carries a tenth.
    """

    drive = np.column_stack([drive_waveform(0), 1.0e-3 * drive_waveform(3)])
    response = np.asarray([2.5e-6, 3.0e-6])
    truth = 0.83
    observed = truth * (drive * response).sum(axis=1)
    gains = drive_gains(observed, drive, response, ["strong", "faint"], channel="p01")
    assert [row.drive for row in gains] == ["strong"]
    assert gains[0].leverage > 0.99
    assert gains[0].slope == pytest.approx(truth, rel=1.0 - gains[0].leverage)


def test_a_drive_standing_alone_recovers_its_scale_exactly():
    drive = drive_waveform(0).reshape(-1, 1)
    response = np.asarray([2.5e-6])
    truth = 0.83
    gains = drive_gains(truth * drive[:, 0] * response[0], drive, response, ["only"])
    assert gains[0].slope == pytest.approx(truth, rel=1e-12)
    assert gains[0].shape_agreement == pytest.approx(1.0, abs=1e-12)


def test_two_drives_cancelling_at_a_channel_report_nothing():
    """The coherence screen, and the case the leverage screen alone lets through.

    Equal and opposite contributions make the prediction nearly vanish while each
    drive's own power stays large, so each one clears any leverage bar while the scale
    fitted to it is a ratio of two nearly cancelling numbers.
    """

    column = drive_waveform(2)
    drive = np.column_stack([column, column])
    response = np.asarray([1.0e-6, -1.0e-6 * (1.0 - 1.0e-6)])
    observed = (drive * response).sum(axis=1)
    assert drive_gains(observed, drive, response, ["a", "b"]) == ()


def test_a_scale_onto_the_wrong_shape_is_not_reported_as_a_gain():
    drive = drive_waveform(0).reshape(-1, 1)
    response = np.asarray([1.0e-6])
    unrelated = drive_waveform(5)
    assert drive_gains(unrelated, drive, response, ["only"]) == ()
    assert drive_gains(0.7 * drive[:, 0] * 1.0e-6, drive, response, ["only"])


def test_a_pulse_shorter_than_the_minimum_says_nothing():
    drive = drive_waveform(0, 50).reshape(-1, 1)
    response = np.asarray([1.0e-6])
    assert drive_gains(drive[:, 0] * 1.0e-6, drive, response, ["only"]) == ()


def test_pooling_takes_the_scatter_across_pulses_not_across_samples():
    values = [0.98, 1.02, 1.00, 1.04, 0.96]
    pooled = pool_scalar_gains(values, channel="p01", drive="sol")
    assert pooled.slope == pytest.approx(1.0)
    assert pooled.standard_error == pytest.approx(
        float(np.std(values, ddof=1) / math.sqrt(len(values)))
    )
    assert pooled.identified


def test_one_pulse_carries_no_error_bar():
    pooled = pool_scalar_gains([1.0])
    assert pooled.slope == 1.0
    assert math.isinf(pooled.standard_error)
    assert not pooled.identified


def tilted_pulse(seed: int, gain: float, tilt: float, ratio: float):
    """Return one pulse of a tilted channel beside the channel measuring the other axis.

    ``ratio`` is how much of the other field component this pulse's circuit produces
    at the channel, relative to the component the channel is described to read.  It is
    what differs between pulses and what makes the pooled solve separable.
    """

    own = drive_waveform(seed)
    other = ratio * own + 0.15 * drive_waveform(seed + 7)
    observed = gain * math.cos(tilt) * own + gain * math.sin(tilt) * other
    return observed, own, other


def test_one_pulse_driving_one_circuit_cannot_divide_scale_from_orientation():
    observed, own, other = tilted_pulse(0, 1.2, 0.1, ratio=0.6)
    other = 0.6 * own
    observed = 1.2 * math.cos(0.1) * own + 1.2 * math.sin(0.1) * other
    fit, _ = axis_fit(observed, own, other, channel="p01", partner_channel="p02")
    assert not fit.separable
    assert abs(fit.collinearity) == pytest.approx(1.0, abs=1e-9)


def test_pulses_presenting_different_field_ratios_recover_both_parameters():
    gain, tilt = 1.24, 0.17
    systems = []
    for seed, ratio in enumerate((0.2, 1.1, -0.7, 2.4)):
        observed, own, other = tilted_pulse(seed, gain, tilt, ratio)
        systems.append(axis_fit(observed, own, other, channel="p01")[1])
    pooled = pool_normal_systems(systems)
    assert len(pooled) == 1
    assert pooled[0].identified
    assert pooled[0].gain == pytest.approx(gain, rel=1e-9)
    assert pooled[0].tilt == pytest.approx(tilt, abs=1e-9)
    assert pooled[0].residual == pytest.approx(0.0, abs=1e-9)


def test_the_jackknife_widens_when_one_pulse_disagrees():
    systems, rogue = [], []
    for seed, ratio in enumerate((0.2, 1.1, -0.7, 2.4)):
        systems.append(axis_fit(*tilted_pulse(seed, 1.2, 0.15, ratio), channel="p")[1])
        gain = 1.2 if seed < 3 else 1.8
        rogue.append(axis_fit(*tilted_pulse(seed, gain, 0.15, ratio), channel="p")[1])
    assert pool_normal_systems(rogue)[0].gain_error > (
        10.0 * pool_normal_systems(systems)[0].gain_error
    )


def test_a_correction_is_scored_on_a_pulse_it_never_saw():
    gain, tilt = 1.3, 0.22
    observed, own, other = tilted_pulse(9, gain, tilt, ratio=0.9)
    corrected, reference = score_axis_correction(
        observed, own, other, (gain, tilt)
    )
    assert corrected == pytest.approx(0.0, abs=1e-12)
    assert reference > 1.0e-3
