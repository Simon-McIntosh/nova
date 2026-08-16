"""Chunk, bank, and combine per-pulse vacuum calibration signatures.

The numerical kernels measure one pulse at a time.  An archive sweep adds three
pieces of engineering around them: deterministic chunks that fit a scheduler wall,
durable partial results that can be resumed, and a merge that keeps measured pulse
identity all the way into time series and recorded correction documents.

This module owns no data store convention in its core functions.  The MAST command
adapter is deliberately thin and entered only by the command-line path: it reads raw
level-1 waveforms, contracts the recorded conductor currents with a staged response
matrix, and delegates classification, instrument fits, closure scoring, and gain
checks to their array kernels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nova.calibrate.correction_model import (
    ChannelCorrection,
    CorrectionKind,
    CorrectionSet,
    CorrectionStatus,
    Provenance,
    ValidityInterval,
)
from nova.calibrate.correction_set import validate_correction_set, write_correction_set
from nova.calibrate.gain_check import GainCheckError, fit_gain_check
from nova.calibrate.instrument import (
    InstrumentError,
    InstrumentTerms,
    PooledInstrumentTerms,
    closure_defect,
    fit_instrument_terms,
    instrument_corrections,
)
from nova.calibrate.scale_step import ChannelScaleHistory, scale_blocks
from nova.calibrate.transition import (
    evaluate_transition_discrimination,
    refine_established_transitions,
)
from nova.calibrate.windows import PulseWindow, classify_pulse

DRIVE_THRESHOLD = 1.0e3
"""Absolute current that labels a conductor as driven, in amperes."""

PLASMA_THRESHOLD = 1.0e4
"""Absolute plasma current above which a sample is not a vacuum sample."""

WINDOW_FLOOR = 64
"""Samples a classified interval must retain before it is banked."""

GAIN_SAMPLE_FLOOR = 200
"""Finite driven samples required for one channel's gain check."""

MINIMUM_STANDOFF = 2.0
"""Winding-pack widths a probe must clear for its gain to describe the channel."""

FORMAT_VERSION = 1


class SweepError(ValueError):
    """Raised when chunked signatures cannot be measured or combined safely."""


@dataclass(frozen=True)
class ResponseInputs:
    """The staged Green response and its channel/current identities."""

    channels: tuple[str, ...]
    families: tuple[str, ...]
    response: np.ndarray
    standoff: np.ndarray
    weights: np.ndarray

    @classmethod
    def read(cls, path: Path | str) -> ResponseInputs:
        """Read and validate a HOME-visible response bundle."""

        with np.load(path, allow_pickle=False) as bank:
            inputs = cls(
                channels=tuple(str(row) for row in bank["channels"]),
                families=tuple(str(row) for row in bank["families"]),
                response=np.asarray(bank["response"], dtype=float),
                standoff=np.asarray(bank["standoff"], dtype=float),
                weights=np.asarray(bank["weights"], dtype=float),
            )
        inputs.validate()
        return inputs

    def validate(self) -> None:
        """Refuse an input bundle whose axes do not describe the same operator."""

        expected = (len(self.channels), len(self.families))
        if self.response.shape != expected:
            raise SweepError(
                f"response shape {self.response.shape} does not match {expected}"
            )
        if self.standoff.shape != expected:
            raise SweepError(
                f"standoff shape {self.standoff.shape} does not match {expected}"
            )
        if self.weights.shape != (len(self.families),):
            raise SweepError(
                f"weight shape {self.weights.shape} does not match "
                f"({len(self.families)},)"
            )
        if len(set(self.channels)) != len(self.channels):
            raise SweepError("response channel names must be unique")
        if len(set(self.families)) != len(self.families):
            raise SweepError("response family names must be unique")
        if not np.all(np.isfinite(self.response)):
            raise SweepError("response matrix must be finite")
        if not np.all(np.isfinite(self.standoff)):
            raise SweepError("standoff matrix must be finite")
        if not np.all(np.isfinite(self.weights)):
            raise SweepError("drive weights must be finite")

    @property
    def weighted_response(self) -> np.ndarray:
        """Return field per recorded conductor ampere for every channel and drive."""

        return self.response * self.weights[None, :]


@dataclass(frozen=True)
class ChunkSpec:
    """One deterministic slice of an ordered pulse list."""

    index: int
    size: int
    total: int
    shots: tuple[int, ...]

    @property
    def first_shot(self) -> int | None:
        return self.shots[0] if self.shots else None

    @property
    def last_shot(self) -> int | None:
        return self.shots[-1] if self.shots else None


def chunk_spec(shots: Sequence[int], index: int, size: int) -> ChunkSpec:
    """Return one zero-based chunk without overlap or implicit wraparound."""

    ordered = tuple(int(shot) for shot in shots)
    if size <= 0:
        raise SweepError("chunk size must be positive")
    total = math.ceil(len(ordered) / size)
    if index < 0 or index >= total:
        raise SweepError(f"chunk index {index} lies outside zero to {total - 1}")
    start = index * size
    return ChunkSpec(index, size, total, ordered[start : start + size])


def _window(row: PulseWindow) -> dict[str, Any]:
    return {
        "guarded": bool(row.guarded),
        "kind": row.kind.value,
        "samples": int(row.sample_count),
        "start": float(row.start),
        "stop": float(row.stop),
    }


def _term(row: InstrumentTerms) -> dict[str, Any]:
    return {
        "channel": row.channel,
        "drift_curvature": float(row.drift_curvature),
        "drift_rate": float(row.drift_rate),
        "offset": float(row.offset),
        "rate_fit_error": float(row.rate_fit_error),
        "sample_count": int(row.sample_count),
        "scatter": float(row.scatter),
    }


def _instrument_wave(row: InstrumentTerms, axis: np.ndarray) -> np.ndarray:
    elapsed = axis - row.reference_time
    return (
        row.offset + row.drift_rate * elapsed + 0.5 * row.drift_curvature * elapsed**2
    )


def _drive_matrix(waveforms: Any, families: Sequence[str], samples: int) -> np.ndarray:
    drive = np.zeros((samples, len(families)), dtype=float)
    for column, family in enumerate(families):
        values = waveforms.drives.get(family)
        if values is not None:
            drive[:, column] = np.asarray(values, dtype=float)
    return drive


def _far_field_channels(
    drive: np.ndarray,
    inputs: ResponseInputs,
    threshold: float,
) -> set[str]:
    peaks = np.nanmax(np.abs(drive), axis=0)
    energised = peaks >= threshold
    if not energised.any():
        return set(inputs.channels)
    keep = np.all(inputs.standoff[:, energised] >= MINIMUM_STANDOFF, axis=1)
    return {
        channel
        for channel, admitted in zip(inputs.channels, keep, strict=True)
        if admitted
    }


def measure_waveforms(
    waveforms: Any,
    inputs: ResponseInputs,
    *,
    settling_time: float,
    drive_threshold: float = DRIVE_THRESHOLD,
    plasma_threshold: float = PLASMA_THRESHOLD,
    window_floor: int = WINDOW_FLOOR,
) -> dict[str, Any]:
    """Measure one pulse's vacuum signatures from array-like waveforms."""

    axis = np.asarray(waveforms.time, dtype=float)
    drive = _drive_matrix(waveforms, inputs.families, axis.size)
    timeline = classify_pulse(
        axis,
        drive,
        plasma=np.asarray(waveforms.plasma_current, dtype=float),
        drive_threshold=drive_threshold,
        plasma_threshold=plasma_threshold,
        decay_time=settling_time,
        settling_periods=1.0,
        minimum_samples=window_floor,
    )
    leading = timeline.leading_quiet
    trailing = timeline.trailing_quiet
    terms: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    rejected_terms: list[dict[str, str]] = []
    rejected_gains: list[dict[str, str]] = []
    fitted: dict[str, InstrumentTerms] = {}
    reference = float(axis[0]) if axis.size else 0.0

    if leading is not None:
        for channel, signal in sorted(waveforms.probes.items()):
            try:
                row = fit_instrument_terms(
                    axis,
                    signal,
                    leading,
                    channel=channel,
                    pulse=int(waveforms.shot),
                    reference_time=reference,
                )
            except InstrumentError as error:
                rejected_terms.append({"channel": channel, "reason": str(error)})
                continue
            fitted[channel] = row
            terms.append(_term(row))
            if trailing is None or trailing is leading:
                continue
            try:
                last = fit_instrument_terms(
                    axis,
                    signal,
                    trailing,
                    channel=channel,
                    pulse=int(waveforms.shot),
                    reference_time=reference,
                )
            except InstrumentError:
                continue
            defect = closure_defect(row, last)
            closures.append(
                {
                    "channel": channel,
                    "closes": bool(defect.closes),
                    "curved_defect": float(defect.curved_defect),
                    "defect": float(defect.defect),
                    "significance": float(defect.significance),
                }
            )

    if timeline.driven_windows:
        channel_index = {name: row for row, name in enumerate(inputs.channels)}
        allowed = _far_field_channels(drive, inputs, drive_threshold)
        response = inputs.weighted_response
        for channel, signal in sorted(waveforms.probes.items()):
            row = channel_index.get(channel)
            instrument = fitted.get(channel)
            if row is None or channel not in allowed or instrument is None:
                continue
            try:
                check = fit_gain_check(
                    axis,
                    drive,
                    signal,
                    response[row],
                    timeline.driven_windows,
                    channel=channel,
                    pulse=int(waveforms.shot),
                    instrument=_instrument_wave(instrument, axis),
                    minimum_samples=GAIN_SAMPLE_FLOOR,
                )
            except GainCheckError as error:
                rejected_gains.append({"channel": channel, "reason": str(error)})
                continue
            gains.append(
                {
                    "channel": channel,
                    "gain": float(check.gain),
                    "residual": float(check.residual),
                    "sample_count": int(check.sample_count),
                    "shape_agreement": float(check.shape_agreement),
                    "signal": float(check.signal),
                }
            )

    return {
        "closures": closures,
        "driven_windows": len(timeline.driven_windows),
        "gains": gains,
        "plasma_windows": len(timeline.plasma_windows),
        "probe_count": len(waveforms.probes),
        "rejected_gain_count": len(rejected_gains),
        "rejected_gains": rejected_gains,
        "rejected_term_count": len(rejected_terms),
        "rejected_terms": rejected_terms,
        "samples": int(axis.size),
        "shot": int(waveforms.shot),
        "terms": terms,
        "windows": [_window(row) for row in timeline.windows],
    }


def sweep_chunk(
    spec: ChunkSpec,
    measure: Callable[[int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure a deterministic pulse chunk, retaining every refusal."""

    started = time.perf_counter()
    pulses: list[Mapping[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for shot in spec.shots:
        try:
            pulses.append(dict(measure(shot)))
        except Exception as error:  # noqa: BLE001 - an unreadable pulse is evidence
            failures.append(
                {"error": f"{type(error).__name__}: {error}", "shot": int(shot)}
            )
    return {
        "chunk": asdict(spec),
        "duration_seconds": float(time.perf_counter() - started),
        "failures": failures,
        "format_version": FORMAT_VERSION,
        "pulses": pulses,
    }


def write_chunk(path: Path | str, result: Mapping[str, Any]) -> None:
    """Write one complete chunk atomically so interruption never looks complete."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_suffix(destination.suffix + ".partial")
    staged.write_text(json.dumps(result, separators=(",", ":"), sort_keys=True))
    staged.replace(destination)


def read_chunks(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Read complete chunks and refuse duplicate pulse results."""

    chunks = [json.loads(Path(path).read_text()) for path in paths]
    chunks.sort(key=lambda row: int(row["chunk"]["index"]))
    seen: set[int] = set()
    for chunk in chunks:
        if int(chunk.get("format_version", -1)) != FORMAT_VERSION:
            raise SweepError("chunk format version does not match this reader")
        for pulse in chunk["pulses"]:
            shot = int(pulse["shot"])
            if shot in seen:
                raise SweepError(f"pulse {shot} occurs in more than one chunk")
            seen.add(shot)
    return chunks


def _column(
    pulses: Sequence[Mapping[str, Any]], section: str, field: str, dtype: Any
) -> np.ndarray:
    return np.asarray(
        [row[field] for pulse in pulses for row in pulse[section]], dtype=dtype
    )


def write_series(path: Path | str, pulses: Sequence[Mapping[str, Any]]) -> None:
    """Bank dense per-channel time series without losing pulse identity."""

    term_shot = np.asarray(
        [pulse["shot"] for pulse in pulses for _ in pulse["terms"]], dtype=np.int64
    )
    gain_shot = np.asarray(
        [pulse["shot"] for pulse in pulses for _ in pulse["gains"]], dtype=np.int64
    )
    np.savez_compressed(
        path,
        term_shot=term_shot,
        term_channel=_column(pulses, "terms", "channel", "U16"),
        offset=_column(pulses, "terms", "offset", float),
        drift_rate=_column(pulses, "terms", "drift_rate", float),
        drift_curvature=_column(pulses, "terms", "drift_curvature", float),
        scatter=_column(pulses, "terms", "scatter", float),
        gain_shot=gain_shot,
        gain_channel=_column(pulses, "gains", "channel", "U16"),
        gain=_column(pulses, "gains", "gain", float),
        gain_shape=_column(pulses, "gains", "shape_agreement", float),
        gain_residual=_column(pulses, "gains", "residual", float),
    )


def _scale_series(
    pulses: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, list[float]]]:
    series: dict[str, dict[int, list[float]]] = {}
    for pulse in pulses:
        shot = int(pulse["shot"])
        for row in pulse["gains"]:
            series.setdefault(str(row["channel"]), {}).setdefault(shot, []).append(
                float(row["gain"])
            )
    return series


def measured_transitions(
    pulses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return every persistent scale block change measured by the sweep."""

    result: list[dict[str, Any]] = []
    for channel, series in sorted(_scale_series(pulses).items()):
        history = ChannelScaleHistory(
            channel,
            scale_blocks(channel, series),
            shot_count=len(series),
        )
        for step in history.steps:
            result.append(step.as_dict())
    return result


def _expected_steps(path: Path | str) -> list[dict[str, Any]]:
    bank = json.loads(Path(path).read_text())
    return [step for history in bank["histories"] for step in history.get("steps", [])]


def refine_expected_transitions(
    pulses: Sequence[Mapping[str, Any]], expected_path: Path | str
) -> dict[str, Any]:
    """Narrow established scale transitions with pulse-by-pulse measurements."""

    shot = np.asarray(
        [pulse["shot"] for pulse in pulses for _ in pulse["gains"]], dtype=np.int64
    )
    channel = _column(pulses, "gains", "channel", "U16")
    gain = _column(pulses, "gains", "gain", float)
    return refine_established_transitions(
        shot, channel, gain, _expected_steps(expected_path)
    )


def rebank_transition_catalogue(
    series_path: Path | str,
    catalogue_path: Path | str,
    expected_path: Path | str,
) -> dict[str, Any]:
    """Attach the simultaneous-channel discrimination and precision bound."""

    with np.load(series_path, allow_pickle=False) as series:
        discrimination = evaluate_transition_discrimination(
            series["gain_shot"],
            series["gain_channel"],
            series["gain"],
            _expected_steps(expected_path),
            shape_agreement=series["gain_shape"],
        )
    destination = Path(catalogue_path)
    catalogue = json.loads(destination.read_text())
    catalogue["discrimination"] = discrimination
    destination.write_text(json.dumps(catalogue, indent=1, sort_keys=True))
    return discrimination


def _pool(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.inf
    if finite.size == 1:
        return float(finite[0]), math.inf
    return float(finite.mean()), float(finite.std(ddof=1) / math.sqrt(finite.size))


def _pooled_terms(pulses: Sequence[Mapping[str, Any]]) -> list[PooledInstrumentTerms]:
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for pulse in pulses:
        for row in pulse["terms"]:
            grouped.setdefault(str(row["channel"]), []).append(
                (int(pulse["shot"]), row)
            )
    result = []
    for channel, rows in sorted(grouped.items()):
        offset, offset_error = _pool(np.asarray([row[1]["offset"] for row in rows]))
        rate, rate_error = _pool(np.asarray([row[1]["drift_rate"] for row in rows]))
        curvature, curvature_error = _pool(
            np.asarray([row[1]["drift_curvature"] for row in rows])
        )
        scatter = float(
            np.sqrt(np.mean([float(row[1]["scatter"]) ** 2 for row in rows]))
        )
        result.append(
            PooledInstrumentTerms(
                channel=channel,
                offset=offset,
                offset_error=offset_error,
                drift_rate=rate,
                drift_rate_error=rate_error,
                drift_curvature=curvature,
                drift_curvature_error=curvature_error,
                scatter=scatter,
                pulses=tuple(row[0] for row in rows),
                fit_count=len(rows),
            )
        )
    return result


def recorded_corrections(
    pulses: Sequence[Mapping[str, Any]], evidence_uri: str
) -> CorrectionSet:
    """Emit pooled instrument terms and measured gain blocks as recorded evidence."""

    provenance = Provenance(
        method=(
            "per-pulse vacuum-window fits over the raw archive; additive instrument "
            "terms from leading quiet windows and scalar gains from far-field "
            "vacuum-driven predictions"
        ),
        evidence_uri=evidence_uri,
        fitted_at=dt.date.today(),
        fitted_by="nova.calibrate.sweep",
        statement=(
            "every value is measured from recorded currents and plasma-free windows; "
            "none is promoted to the read path"
        ),
    )
    corrections = list(
        instrument_corrections(
            _pooled_terms(pulses),
            provenance=provenance,
            unit="T",
            status=CorrectionStatus.recorded,
        )
    )
    series = _scale_series(pulses)
    for channel, values in sorted(series.items()):
        for block in scale_blocks(channel, values):
            measured = [shot for shot in sorted(values) if block.covers(shot)]
            corrections.append(
                ChannelCorrection(
                    channel=channel,
                    kind=CorrectionKind.gain,
                    status=CorrectionStatus.recorded,
                    value=float(block.scale),
                    validity=[
                        ValidityInterval(
                            pulse_start=block.first_shot,
                            pulse_end=block.last_shot,
                            measured_pulses=measured,
                        )
                    ],
                    provenance=provenance,
                    notes=(
                        f"per-pulse gain block supported by {block.shot_count} "
                        "vacuum-driven measurements"
                    ),
                )
            )
    document = CorrectionSet(
        machine="mast",
        diagnostic_system="magnetics",
        schema_version="1.0.0",
        set_version="0.1.0",
        generated_by="nova.calibrate.sweep",
        description=(
            "Archive-harvested offsets, integrator drift, and opportunistic gains. "
            "All records are measured evidence and are not applied by the read path."
        ),
        corrections=corrections,
    )
    validate_correction_set(document)
    return document


def merge_chunks(
    paths: Sequence[Path | str],
    output: Path | str,
    *,
    expected_transitions: Path | str | None = None,
) -> dict[str, Any]:
    """Combine chunks into the signature bank, series, catalogue, and corrections."""

    chunks = read_chunks(paths)
    pulses = sorted(
        [pulse for chunk in chunks for pulse in chunk["pulses"]],
        key=lambda row: int(row["shot"]),
    )
    failures = [row for chunk in chunks for row in chunk["failures"]]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    series_path = root / "signature_series.npz"
    write_series(series_path, pulses)
    measured = measured_transitions(pulses)
    refinement = (
        None
        if expected_transitions is None
        else refine_expected_transitions(pulses, expected_transitions)
    )
    catalogue = {
        "measured": measured,
        "refinement": refinement,
    }
    (root / "transition_catalogue.json").write_text(
        json.dumps(catalogue, indent=1, sort_keys=True)
    )
    discrimination = (
        None
        if expected_transitions is None
        else rebank_transition_catalogue(
            series_path,
            root / "transition_catalogue.json",
            expected_transitions,
        )
    )
    document = recorded_corrections(
        pulses, "~/.cache/nova-mast/vacuum-signatures/archive/signature_series.npz"
    )
    write_correction_set(root / "recorded_corrections.yaml", document)
    summary = {
        "chunk_count": len(chunks),
        "correction_records": len(document.corrections),
        "failed_shots": len(failures),
        "format_version": FORMAT_VERSION,
        "gain_measurements": sum(len(row["gains"]) for row in pulses),
        "maximum_chunk_duration_seconds": max(
            float(row["duration_seconds"]) for row in chunks
        ),
        "measured_shots": len(pulses),
        "measured_transitions": len(measured),
        "requested_shots": sum(len(row["chunk"]["shots"]) for row in chunks),
        "term_measurements": sum(len(row["terms"]) for row in pulses),
        "transition_refinement": refinement,
        "transition_discrimination": discrimination,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    (root / "signature_store.json").write_text(
        json.dumps(
            {
                "chunks": [str(Path(path).resolve()) for path in paths],
                "failures": failures,
                "series": str(series_path.resolve()),
                "summary": summary,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return summary


def draw_summary_figures(
    series_path: Path | str,
    catalogue_path: Path | str,
    output: Path | str,
) -> tuple[Path, Path, Path]:
    """Draw archive coverage and channel histories as compact evidence figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(series_path, allow_pickle=False) as series:
        term_shot = series["term_shot"]
        term_channel = series["term_channel"]
        offset = series["offset"]
        drift = series["drift_rate"]
        gain_shot = series["gain_shot"]
        gain_channel = series["gain_channel"]
        gain = series["gain"]
    catalogue = json.loads(Path(catalogue_path).read_text())

    figure, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), constrained_layout=True)
    bins = np.linspace(float(term_shot.min()), float(term_shot.max()), 80)
    axes[0].hist(term_shot, bins=bins, color="#355f8a", linewidth=0)
    axes[0].set(ylabel="channel fits", title="Per-pulse quiet-window coverage")
    channels = sorted(set(str(row) for row in term_channel))
    for channel in channels[:: max(1, len(channels) // 8)]:
        take = term_channel == channel
        axes[1].plot(
            term_shot[take],
            np.abs(drift[take]),
            ".",
            markersize=1.0,
            alpha=0.45,
            label=channel,
        )
    axes[1].set_yscale("log")
    axes[1].set(xlabel="pulse", ylabel="|drift rate| [T/s]")
    axes[1].legend(ncol=4, fontsize=7, frameon=False)
    coverage = destination / "archive-coverage.png"
    figure.savefig(coverage, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), constrained_layout=True)
    shown = ["obv03"]
    refinement = catalogue.get("refinement") or {}
    shown.extend(
        row["channel"]
        for row in refinement.get("transitions", [])
        if row.get("pulse_width") == 1 and row["channel"] != "obv03"
    )
    for channel in list(dict.fromkeys(shown))[:8]:
        take = gain_channel == channel
        axes[0].plot(gain_shot[take], gain[take], ".", markersize=2.2, label=channel)
    axes[0].set(ylabel="measured gain", title="Vacuum-driven scale histories")
    axes[0].legend(ncol=4, fontsize=7, frameon=False)
    for channel in sorted(set(str(row) for row in term_channel))[::10]:
        take = term_channel == channel
        axes[1].plot(term_shot[take], offset[take], ".", markersize=1.0, alpha=0.45)
    axes[1].set(xlabel="pulse", ylabel="offset [T]")
    histories = destination / "signature-histories.png"
    figure.savefig(histories, dpi=160)
    plt.close(figure)

    discrimination = catalogue.get("discrimination") or {}
    counts = discrimination.get("cause_counts") or {}
    stages = [
        ("raw scalar", discrimination.get("raw_apparent_transitions", 0)),
        (
            "simultaneous cohort",
            discrimination.get("raw_cohort_apparent_transitions", 0),
        ),
        (
            "common response removed",
            discrimination.get("corrected_apparent_transitions", 0),
        ),
        ("established switches", discrimination.get("expected_switches", 0)),
    ]
    causes = [
        ("interleaved", counts.get("interleaved_or_unclassified_states", 0)),
        ("ordered, not adjacent", counts.get("ordered_but_nonadjacent", 0)),
        ("no adjacent pair", counts.get("no_adjacent_observation_pair", 0)),
        ("no ratio samples", counts.get("no_ratio_observations", 0)),
        ("adjacent", counts.get("adjacent_transition", 0)),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    axes[0].bar(
        [row[0] for row in stages],
        [row[1] for row in stages],
        color=["#456990", "#6f8faf", "#9ab4c9", "#c5604f"],
    )
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].set(ylabel="apparent transitions", title="Common-mode discrimination")
    axes[1].barh(
        [row[0] for row in causes],
        [row[1] for row in causes],
        color=["#c5604f", "#d59a65", "#8c8c8c", "#aaaaaa", "#4c956c"],
    )
    axes[1].set(xlabel="established switches", title="Adjacent-precision outcome")
    discrimination_path = destination / "transition-discrimination.png"
    figure.savefig(discrimination_path, dpi=160)
    plt.close(figure)
    return coverage, histories, discrimination_path


def stage_mast_inputs(path: Path | str, weights_path: Path | str) -> Path:
    """Build the response/standoff bundle compute chunks read from HOME."""

    from nova.catalog.mast_geometry import MachineGeometryRegistry
    from nova.imas.mast_vacuum_cohort import probe_channels
    from nova.imas.mast_vacuum_response import ResponseModel

    selection = MachineGeometryRegistry.default().select(21978)
    geometry = selection.configuration.geometry
    probes = geometry["magnetics"]["poloidal_probes"]
    model = ResponseModel.build(geometry, probes, probe_channels(probes))
    weights = json.loads(Path(weights_path).read_text())["drive_weights"]
    ordered = np.asarray([float(weights[name]) for name in model.families])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        channels=np.asarray([row.channel for row in model.targets]),
        families=np.asarray(model.families),
        response=model.response,
        standoff=model.standoff,
        weights=ordered,
    )
    return destination


def _mast_measure(
    inputs: ResponseInputs, settling_time: float
) -> Callable[[int], dict[str, Any]]:
    from nova.imas.mast_vacuum_cohort import RAW_ARCHIVE

    def measure(shot: int) -> dict[str, Any]:
        return measure_waveforms(
            RAW_ARCHIVE.read_shot_waveforms(shot),
            inputs,
            settling_time=settling_time,
        )

    return measure


def _shots(path: Path | str) -> list[int]:
    return [int(row) for row in json.loads(Path(path).read_text())]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage-inputs")
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--weights", type=Path, required=True)
    chunk = commands.add_parser("chunk")
    chunk.add_argument("--shots", type=Path, required=True)
    chunk.add_argument("--response", type=Path, required=True)
    chunk.add_argument("--output", type=Path, required=True)
    chunk.add_argument("--index", type=int, required=True)
    chunk.add_argument("--size", type=int, required=True)
    chunk.add_argument("--settling-time", type=float, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--chunks", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--expected-transitions", type=Path)
    figures = commands.add_parser("figures")
    figures.add_argument("--series", type=Path, required=True)
    figures.add_argument("--catalogue", type=Path, required=True)
    figures.add_argument("--output", type=Path, required=True)
    transitions = commands.add_parser("transitions")
    transitions.add_argument("--series", type=Path, required=True)
    transitions.add_argument("--catalogue", type=Path, required=True)
    transitions.add_argument("--expected-transitions", type=Path, required=True)
    transitions.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run input staging, one compute chunk, the merge, or figure generation."""

    arguments = _parser().parse_args(argv)
    if arguments.command == "stage-inputs":
        print(stage_mast_inputs(arguments.output, arguments.weights))
    elif arguments.command == "chunk":
        shots = _shots(arguments.shots)
        spec = chunk_spec(shots, arguments.index, arguments.size)
        result = sweep_chunk(
            spec,
            _mast_measure(
                ResponseInputs.read(arguments.response), arguments.settling_time
            ),
        )
        write_chunk(arguments.output, result)
        print(
            json.dumps(
                {
                    "duration_seconds": result["duration_seconds"],
                    "failed": len(result["failures"]),
                    "measured": len(result["pulses"]),
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "merge":
        print(
            json.dumps(
                merge_chunks(
                    arguments.chunks,
                    arguments.output,
                    expected_transitions=arguments.expected_transitions,
                ),
                sort_keys=True,
            )
        )
    elif arguments.command == "figures":
        for path in draw_summary_figures(
            arguments.series, arguments.catalogue, arguments.output
        ):
            print(path)
    else:
        report = rebank_transition_catalogue(
            arguments.series,
            arguments.catalogue,
            arguments.expected_transitions,
        )
        if arguments.report is not None:
            arguments.report.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
