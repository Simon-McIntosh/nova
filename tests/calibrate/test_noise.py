"""Noise and drift kernels recover manufactured waveform truth."""

from dataclasses import dataclass

import numpy as np
import pytest

from nova.calibrate.noise import measure_noise, measure_noise_envelope
from nova.imas import mast_sensor_noise


@dataclass(frozen=True)
class Waveform:
    """Array-only record matching the numerical kernel protocol."""

    shot: int
    time: np.ndarray
    drives: dict[str, np.ndarray]
    probes: dict[str, np.ndarray]
    sample_mask: np.ndarray
    baseline_mask: np.ndarray


def drifting_waveform(seed: int, *, scatter: float, drift: float) -> Waveform:
    generator = np.random.default_rng(seed)
    time = np.linspace(-2.0, 2.0, 4000)
    signal = 0.4 + drift * time + generator.normal(0.0, scatter, time.size)
    return Waveform(
        shot=seed,
        time=time,
        drives={},
        probes={"probe": signal},
        sample_mask=np.ones(time.size, dtype=bool),
        baseline_mask=time < -1.0,
    )


def test_injected_drift_ramp_and_noise_floor_are_recovered_separately():
    waveform = drifting_waveform(7, scatter=2.0e-4, drift=1.3e-3)

    fit = measure_noise(waveform.time, waveform.probes["probe"])

    assert fit.scatter == pytest.approx(2.0e-4, rel=0.04)
    assert fit.drift_rate == pytest.approx(1.3e-3, rel=0.02)


def test_envelope_pools_array_waveforms_and_the_mast_adapter_reexports_it():
    waveforms = [
        drifting_waveform(11, scatter=1.0e-4, drift=7.0e-4),
        drifting_waveform(12, scatter=2.0e-4, drift=-9.0e-4),
    ]

    envelope = measure_noise_envelope(waveforms)

    assert envelope.channel("probe").scatter == pytest.approx(
        np.sqrt((1.0e-8 + 4.0e-8) / 2.0), rel=0.05
    )
    assert envelope.channel("probe").drift_rate == pytest.approx(8.0e-4, rel=0.06)
    assert mast_sensor_noise.measure_noise_envelope is measure_noise_envelope
