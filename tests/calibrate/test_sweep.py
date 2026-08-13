"""Chunked pulse signatures remain complete, resumable, and pulse-addressable."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from nova.calibrate.correction_model import CorrectionStatus
from nova.calibrate.sweep import (
    ChunkSpec,
    ResponseInputs,
    SweepError,
    chunk_spec,
    measure_waveforms,
    read_chunks,
    recorded_corrections,
    refine_expected_transitions,
    sweep_chunk,
    write_chunk,
    write_series,
)


@dataclass
class Waveforms:
    shot: int
    time: np.ndarray
    drives: dict[str, np.ndarray]
    probes: dict[str, np.ndarray]
    plasma_current: np.ndarray


def manufactured_waveforms(shot: int = 100, gain: float = 1.2) -> Waveforms:
    time = np.linspace(0.0, 1.0, 800)
    first = np.zeros(time.size)
    second = np.zeros(time.size)
    driven = (time >= 0.2) & (time < 0.7)
    local = (time[driven] - 0.2) / 0.5
    first[driven] = 2.0e3 * (1.0 + 0.2 * np.sin(np.pi * local))
    second[driven] = 1.5e3 * (1.0 + 0.25 * np.sin(2.0 * np.pi * local))
    response = np.asarray([2.0e-6, -0.8e-6])
    instrument = 1.0e-4 + 2.0e-5 * time + 3.0e-6 * time**2
    signal = gain * (np.column_stack([first, second]) @ response) + instrument
    return Waveforms(
        shot,
        time,
        {"first": first, "second": second},
        {"probe": signal},
        np.zeros(time.size),
    )


def response_inputs() -> ResponseInputs:
    return ResponseInputs(
        channels=("probe",),
        families=("first", "second"),
        response=np.asarray([[2.0e-6, -0.8e-6]]),
        standoff=np.asarray([[4.0, 3.0]]),
        weights=np.ones(2),
    )


def pulse_record(shot: int, gain: float = 1.0) -> dict:
    return {
        "gains": [
            {
                "channel": "probe",
                "gain": gain,
                "residual": 0.0,
                "sample_count": 300,
                "shape_agreement": 1.0,
                "signal": 1.0,
            }
        ],
        "shot": shot,
        "terms": [
            {
                "channel": "probe",
                "drift_curvature": 3.0e-6,
                "drift_rate": 2.0e-5,
                "offset": 1.0e-4,
                "rate_fit_error": 1.0e-7,
                "sample_count": 160,
                "scatter": 1.0e-8,
            }
        ],
    }


def test_chunk_partition_covers_every_pulse_once():
    shots = list(range(11))
    chunks = [chunk_spec(shots, index, 4) for index in range(3)]

    assert [shot for chunk in chunks for shot in chunk.shots] == shots
    assert [(row.first_shot, row.last_shot) for row in chunks] == [
        (0, 3),
        (4, 7),
        (8, 10),
    ]
    with pytest.raises(SweepError, match="outside"):
        chunk_spec(shots, 3, 4)


def test_sweep_retains_an_unreadable_pulse_without_losing_the_chunk():
    def measure(shot: int) -> dict:
        if shot == 102:
            raise OSError("missing group")
        return pulse_record(shot)

    result = sweep_chunk(ChunkSpec(0, 3, 1, (101, 102, 103)), measure)

    assert [row["shot"] for row in result["pulses"]] == [101, 103]
    assert result["failures"] == [{"error": "OSError: missing group", "shot": 102}]


def test_chunk_write_is_complete_and_duplicate_pulses_are_refused(tmp_path):
    result = sweep_chunk(ChunkSpec(0, 1, 2, (101,)), pulse_record)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_chunk(first, result)
    write_chunk(second, result)

    assert json.loads(first.read_text())["pulses"][0]["shot"] == 101
    assert not (tmp_path / "first.json.partial").exists()
    with pytest.raises(SweepError, match="more than one chunk"):
        read_chunks([first, second])


def test_waveform_measurement_recovers_instrument_terms_and_gain():
    result = measure_waveforms(
        manufactured_waveforms(),
        response_inputs(),
        settling_time=0.0,
    )

    assert len(result["terms"]) == 1
    assert len(result["gains"]) == 1
    assert result["terms"][0]["offset"] == pytest.approx(1.0e-4, abs=1.0e-12)
    assert result["terms"][0]["drift_rate"] == pytest.approx(2.0e-5, abs=1.0e-12)
    assert result["gains"][0]["gain"] == pytest.approx(1.2, rel=1.0e-10)


def test_near_field_channel_keeps_instrument_terms_but_not_a_gain():
    inputs = response_inputs()
    near = ResponseInputs(
        inputs.channels,
        inputs.families,
        inputs.response,
        np.asarray([[1.0, 3.0]]),
        inputs.weights,
    )
    result = measure_waveforms(manufactured_waveforms(), near, settling_time=0.0)

    assert len(result["terms"]) == 1
    assert not result["gains"]


def test_series_bank_preserves_pulse_and_channel_axes(tmp_path):
    path = tmp_path / "series.npz"
    write_series(path, [pulse_record(101), pulse_record(102, 2.0)])

    with np.load(path, allow_pickle=False) as bank:
        assert bank["term_shot"].tolist() == [101, 102]
        assert bank["gain_shot"].tolist() == [101, 102]
        assert bank["gain"].tolist() == [1.0, 2.0]


def test_established_transition_is_narrowed_to_the_injected_pulse(tmp_path):
    expected = {
        "histories": [
            {
                "channel": "probe",
                "steps": [
                    {
                        "after_scale": 2.0,
                        "after_shot": 105,
                        "before_scale": 1.0,
                        "before_shot": 100,
                        "channel": "probe",
                    }
                ],
            }
        ]
    }
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(expected))
    pulses = [
        pulse_record(shot, 1.0 if shot < 103 else 2.0) for shot in range(100, 106)
    ]

    result = refine_expected_transitions(pulses, path)

    assert result["expected_count"] == result["exact_count"] == 1
    assert result["transitions"][0]["before_shot"] == 102
    assert result["transitions"][0]["after_shot"] == 103


def test_emitted_corrections_validate_and_remain_recorded():
    pulses = [pulse_record(shot, 1.0 if shot < 104 else 2.0) for shot in range(8)]

    document = recorded_corrections(pulses, "bank/signature_series.npz")

    assert document.corrections
    assert {CorrectionStatus(row.status) for row in document.corrections} == {
        CorrectionStatus.recorded
    }
    assert {row.kind for row in document.corrections} >= {
        "offset",
        "drift_rate",
        "gain",
    }
