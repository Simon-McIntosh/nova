"""Run the calibration kernels against a non-MAST IMAS pulse."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import imas
import numpy as np
import pytest

from nova.calibrate.instrument import InstrumentTerms, fit_instrument_terms
from nova.calibrate.windows import PulseTimeline, WindowKind, classify_pulse


@dataclass(frozen=True)
class PulseArrays:
    """Arrays adapted from one IMAS pulse for the calibration kernels."""

    time: np.ndarray
    drive: np.ndarray
    plasma: np.ndarray
    signals: dict[str, np.ndarray]
    dd_version: str


def _source_path() -> Path:
    """Return the locally mounted TCV pulse used for the portability proof."""

    imas_home = Path(os.environ.get("IMAS_HOME", Path.home() / "public"))
    return imas_home / "imasdb" / "tcv" / "3" / "77549" / "1"


def _written_version(path: Path) -> str:
    """Read the stored dictionary version before opening the pulse semantically."""

    with imas.DBEntry(f"imas:hdf5?path={path}", "r") as entry:
        magnetics = entry.get("magnetics", autoconvert=False)
    return str(magnetics.ids_properties.version_put.data_dictionary)


def _read_arrays(path: Path) -> PulseArrays:
    """Adapt TCV IDS nodes to plain arrays without changing either kernel."""

    version = _written_version(path)
    with imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version=version) as entry:
        pf_active = entry.get("pf_active", autoconvert=False)
        magnetics = entry.get("magnetics", autoconvert=False)
        summary = entry.get("summary", autoconvert=False)

    assert str(pf_active.ids_properties.version_put.data_dictionary) == version
    assert str(magnetics.ids_properties.version_put.data_dictionary) == version
    assert str(summary.ids_properties.version_put.data_dictionary) == version

    time = np.array(pf_active.coil[0].current.time, dtype=float)
    drive = np.column_stack(
        [
            np.array(pf_active.coil[index].current.data, dtype=float)
            for index in range(len(pf_active.coil))
        ]
    )
    summary_time = np.array(summary.time, dtype=float)
    summary_current = np.array(summary.global_quantities.ip.value, dtype=float)
    plasma = np.interp(time, summary_time, summary_current, left=0.0, right=0.0)
    signals = {
        f"bpol:{str(probe.name).strip()}": np.array(probe.field.data, dtype=float)
        for probe in magnetics.bpol_probe
    }
    signals.update(
        {
            f"flux:{str(loop.name).strip()}": np.array(loop.flux.data, dtype=float)
            for loop in magnetics.flux_loop
        }
    )
    return PulseArrays(time, drive, plasma, signals, version)


def _classify(arrays: PulseArrays) -> PulseTimeline:
    """Apply the TCV acquisition thresholds to the generic classifier."""

    return classify_pulse(
        arrays.time,
        arrays.drive,
        plasma=arrays.plasma,
        drive_threshold=100.0,
        plasma_threshold=10_000.0,
        minimum_samples=32,
    )


def _instrument_terms(
    arrays: PulseArrays, timeline: PulseTimeline
) -> tuple[InstrumentTerms, ...]:
    """Fit every magnetic signal over the uncontaminated leading quiet window."""

    window = timeline.leading_quiet
    assert window is not None
    return tuple(
        fit_instrument_terms(
            arrays.time,
            signal,
            window,
            channel=channel,
            pulse=77549,
            reference_time=0.0,
        )
        for channel, signal in sorted(arrays.signals.items())
    )


def test_non_mast_imas_pulse_runs_generic_calibration_kernels() -> None:
    """Classify TCV windows and extract its terms through the public kernels."""

    path = _source_path()
    if not path.exists():
        pytest.skip(f"TCV portability source is not mounted at {path}")

    arrays = _read_arrays(path)
    timeline = _classify(arrays)
    terms = _instrument_terms(arrays, timeline)

    assert arrays.dd_version == "3.41.0"
    assert arrays.time.size == 60_968
    assert arrays.drive.shape == (60_968, 29)
    assert len(arrays.signals) == 76
    assert [window.kind for window in timeline.windows] == [
        WindowKind.quiet,
        WindowKind.driven,
        WindowKind.plasma,
        WindowKind.driven,
        WindowKind.quiet,
    ]
    assert [window.sample_count for window in timeline.windows] == [
        19_426,
        2_734,
        16_212,
        5_930,
        16_666,
    ]
    assert len(terms) == 76
    assert all(term.sample_count == 19_426 for term in terms)
    assert all(
        np.isfinite(
            [
                term.offset,
                term.drift_rate,
                term.drift_curvature,
                term.scatter,
            ]
        ).all()
        for term in terms
    )
    first_probe = next(term for term in terms if term.channel == "bpol:001")
    assert first_probe.offset == pytest.approx(1.217304265e-3, abs=1e-11)
    assert first_probe.drift_rate == pytest.approx(2.704760302e-3, abs=1e-11)
    assert first_probe.scatter == pytest.approx(2.833661837e-4, abs=1e-11)
    assert classify_pulse.__module__ == "nova.calibrate.windows"
    assert fit_instrument_terms.__module__ == "nova.calibrate.instrument"
