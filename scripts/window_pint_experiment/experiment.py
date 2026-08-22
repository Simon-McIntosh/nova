"""Measure a two-level Parareal composition over coupled transport windows.

The experiment keeps the public ``solve_window`` call as the fine propagator
and composes the public fixed-step transport sweep into a frozen-geometry
coarse propagator.  Continuous corrections are evaluated on one declared
normalised radial grid.  Branch class and immutable selection history are
threaded causally and are never included in the algebraic correction.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import os
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import jax
import numpy as np

from nova.equilibrium import SelectionHistory, SelectionPolicy
from nova.transport import (
    ExchangeSweepResult,
    TransportState,
    Waveform,
    WaveformSample,
    WindowConfig,
    solve_window,
)
from scripts.window_demonstration import run_window as demonstration


WINDOW_COUNT = 8
WINDOW_LENGTH = 2.5e-3
OUTER_TOLERANCE = 5.0e-3
MAXIMUM_CORRECTIONS = 2
CORRECTION_GRID = np.linspace(0.0, 1.0, demonstration.SOURCE_SAMPLES)
OUTPUT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_PATH = OUTPUT_DIRECTORY / "results.tsv"
REPORT_PATH = OUTPUT_DIRECTORY / "report.md"
RESULT_FIELDS = (
    "tree_sha",
    "lane",
    "correction",
    "window",
    "kind",
    "field",
    "value",
    "unit",
    "status",
    "provenance",
)


@dataclass(frozen=True)
class BoundaryState:
    """Continuous window boundary plus causally threaded branch metadata."""

    transport: TransportState
    geometry: WaveformSample
    source: WaveformSample
    equilibrium_flux: np.ndarray
    plasma_current: float
    history: SelectionHistory
    policy: SelectionPolicy | None


@dataclass(frozen=True)
class FineResult:
    """One fully converged fine propagation and all nested evidence."""

    initial: BoundaryState
    boundary: BoundaryState
    receipt: Any
    wall_seconds: float
    extraction_records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CoarseResult:
    """One frozen-geometry fixed-step transport advance."""

    initial: BoundaryState
    boundary: BoundaryState
    receipt: Any
    wall_seconds: float


@dataclass
class PreparedFixture:
    """Static fixture ingredients shared by every strictly serial call."""

    profile: Any
    extraction_lattice: Any
    fixture_sources: Any
    model: Any
    initial: BoundaryState
    preparation_seconds: float


def _tree_sha() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _format(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return str(bool(value)).lower()
    if isinstance(value, int | np.integer):
        return str(int(value))
    return f"{float(np.asarray(value)):.17g}"


def _hash_boundary(state: BoundaryState) -> str:
    digest = hashlib.sha256()
    for value in dataclasses.astuple(state.transport):
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    for sample in (state.geometry, state.source):
        digest.update(np.asarray(sample.radial_grid, dtype=np.float64).tobytes())
        for coordinate in (
            sample.phi_boundary,
            sample.axis_reference,
            sample.boundary_reference,
        ):
            digest.update(np.asarray(coordinate, dtype=np.float64).tobytes())
        for name in sorted(sample.values):
            digest.update(name.encode())
            digest.update(np.asarray(sample.values[name]).tobytes())
    digest.update(np.asarray(state.equilibrium_flux, dtype=np.float64).tobytes())
    digest.update(np.asarray(state.plasma_current, dtype=np.float64).tobytes())
    digest.update(repr(state.history).encode())
    return digest.hexdigest()


def _constant_waveform(sample: WaveformSample) -> Waveform:
    time_grid = np.asarray((0.0, WINDOW_LENGTH), dtype=np.float64)
    return Waveform(
        time=time_grid,
        radial_grid=np.stack((sample.radial_grid, sample.radial_grid)),
        phi_boundary=np.full(2, sample.phi_boundary),
        axis_reference=np.full(2, sample.axis_reference),
        boundary_reference=np.full(2, sample.boundary_reference),
        values={
            name: np.stack((value, value)) for name, value in sample.values.items()
        },
    )


def _sample_source(source, coordinate: WaveformSample) -> WaveformSample:
    waveform = demonstration._source_waveform(
        np.asarray((0.0, WINDOW_LENGTH)),
        (source, source),
        (coordinate, coordinate),
    )
    return waveform.sample(0.0, radial_grid=CORRECTION_GRID)


def _prepare_fixture() -> PreparedFixture:
    demonstration.configure_dtypes()
    started = time.perf_counter()
    profile, seed, _vacuum = demonstration._fixture_machine()
    extraction_lattice = demonstration._extraction_lattice(profile)
    fixture_sources = demonstration._fixture_sources(profile)
    equilibrium = profile.solve(
        seed,
        route="anderson",
        evaluations=demonstration.EVALUATIONS,
    )
    geometry, extraction = demonstration._geometry_from_equilibrium(
        equilibrium,
        profile.source,
        extraction_lattice,
        fixture_sources,
    )
    if not extraction["record_valid"]:
        raise RuntimeError("the baseline extraction record is not valid")
    geometry_waveform = Waveform.from_geometries(
        np.asarray((0.0, WINDOW_LENGTH)),
        (geometry, geometry),
    )
    geometry_sample = geometry_waveform.sample(0.0)
    source_sample = _sample_source(profile.source, geometry_sample)
    initial = BoundaryState(
        transport=demonstration._initial_state(geometry, profile.source),
        geometry=geometry_sample,
        source=source_sample,
        equilibrium_flux=np.asarray(equilibrium.flux),
        plasma_current=float(equilibrium.moments.plasma_current),
        history=SelectionHistory(),
        policy=None,
    )
    return PreparedFixture(
        profile=profile,
        extraction_lattice=extraction_lattice,
        fixture_sources=fixture_sources,
        model=demonstration._torax_model(WINDOW_LENGTH),
        initial=initial,
        preparation_seconds=time.perf_counter() - started,
    )


def _window_config() -> WindowConfig:
    time_grid = np.asarray((0.0, WINDOW_LENGTH), dtype=np.float64)
    return WindowConfig(
        length=WINDOW_LENGTH,
        equilibrium_grid=time_grid,
        transport_grid=time_grid,
        iteration_cap=demonstration.ITERATION_CAP,
        tolerance=demonstration.CONVERGENCE_TOLERANCE,
        contraction_threshold=demonstration.CONTRACTION_THRESHOLD,
        hard_iteration_ceiling=demonstration.HARD_ITERATION_CEILING,
        damping_floor=demonstration.DAMPING_FLOOR,
    )


def _scaled_source(
    incoming: WaveformSample,
    geometry: Waveform,
    receipt: Any,
) -> Waveform:
    incoming_source = demonstration._source_from_sample(incoming)
    evolved_source = demonstration.forward_source_from_receipt(
        receipt.receipts[-1],
        geometry.sample(float(receipt.geometry_time[-1])).geometry(),
        ion_density_per_electron=demonstration.ION_DENSITY_PER_ELECTRON,
    )
    return demonstration._scaled_source_waveform(
        geometry,
        np.asarray((0.0, WINDOW_LENGTH)),
        incoming_source,
        (incoming_source, evolved_source),
        0.5,
    )


def _fine_propagate(fixture: PreparedFixture, initial: BoundaryState) -> FineResult:
    geometry_waveform = _constant_waveform(initial.geometry)
    source_waveform = _constant_waveform(initial.source)
    extraction_records: list[Mapping[str, Any]] = []
    latest_history = initial.history
    latest_policy = initial.policy

    def transport_update(geometry: Waveform, sample_grid: np.ndarray):
        receipt = demonstration.transport_sweep(
            geometry,
            initial.transport,
            sample_grid,
            np.full(sample_grid.shape, initial.plasma_current),
            fixture.model,
        )
        return ExchangeSweepResult(
            waveform=_scaled_source(initial.source, geometry, receipt),
            receipt=receipt,
        )

    def equilibrium_update(source: Waveform, sample_grid: np.ndarray):
        nonlocal latest_history, latest_policy
        receipt = demonstration.equilibrium_sweep(
            fixture.profile,
            initial.equilibrium_flux,
            source,
            sample_grid,
            demonstration._source_from_sample,
            route="anderson",
            solve_options={
                "evaluations": demonstration.EVALUATIONS,
                "tolerance": demonstration.EQUILIBRIUM_SOLVE_TOLERANCE,
            },
            selection_history=initial.history,
            selection_policy=initial.policy,
        )
        latest_history = receipt.branch_receipts[-1].selection.next_history
        latest_policy = receipt.branch_receipts[-1].selection.policy
        geometries = []
        for sample_index, (sample, equilibrium) in enumerate(
            zip(receipt.source_samples, receipt.equilibria, strict=True)
        ):
            geometry, extraction = demonstration._geometry_from_equilibrium(
                equilibrium,
                demonstration._source_from_sample(sample),
                fixture.extraction_lattice,
                fixture.fixture_sources,
            )
            extraction_records.append(
                {
                    **extraction,
                    "sample": sample_index,
                    "selected_class": receipt.branch_receipts[
                        sample_index
                    ].selection.selected_class.name.lower(),
                    "selection_reason": receipt.branch_receipts[
                        sample_index
                    ].selection.reason.value,
                    "limited_core_cells": receipt.branch_receipts[
                        sample_index
                    ].core_cell_counts[0],
                    "diverted_core_cells": receipt.branch_receipts[
                        sample_index
                    ].core_cell_counts[1],
                }
            )
            geometries.append(geometry)
        return ExchangeSweepResult(
            waveform=Waveform.from_geometries(sample_grid, geometries),
            receipt=receipt,
        )

    started = time.perf_counter()
    receipt = solve_window(
        geometry_waveform,
        source_waveform,
        _window_config(),
        equilibrium_update,
        transport_update,
        damping=demonstration.DAMPING,
    )
    wall_seconds = time.perf_counter() - started
    equilibrium = receipt.equilibrium_receipt.equilibria[-1]
    boundary = BoundaryState(
        transport=receipt.transport_receipt.state,
        geometry=receipt.geometry_waveform.sample(WINDOW_LENGTH),
        source=receipt.source_waveform.sample(
            WINDOW_LENGTH, radial_grid=CORRECTION_GRID
        ),
        equilibrium_flux=np.asarray(equilibrium.flux),
        plasma_current=float(equilibrium.moments.plasma_current),
        history=latest_history,
        policy=latest_policy,
    )
    return FineResult(
        initial=initial,
        boundary=boundary,
        receipt=receipt,
        wall_seconds=wall_seconds,
        extraction_records=tuple(extraction_records),
    )


def _coarse_propagate(fixture: PreparedFixture, initial: BoundaryState) -> CoarseResult:
    geometry = _constant_waveform(initial.geometry)
    time_grid = np.asarray((0.0, WINDOW_LENGTH), dtype=np.float64)
    started = time.perf_counter()
    receipt = demonstration.transport_sweep(
        geometry,
        initial.transport,
        time_grid,
        np.full(2, initial.plasma_current),
        fixture.model,
    )
    source = _scaled_source(initial.source, geometry, receipt).sample(
        WINDOW_LENGTH, radial_grid=CORRECTION_GRID
    )
    boundary = BoundaryState(
        transport=receipt.state,
        geometry=initial.geometry,
        source=source,
        equilibrium_flux=np.asarray(initial.equilibrium_flux),
        plasma_current=float(receipt.receipts[-1].boundary.plasma_current),
        history=initial.history,
        policy=initial.policy,
    )
    return CoarseResult(
        initial=initial,
        boundary=boundary,
        receipt=receipt,
        wall_seconds=time.perf_counter() - started,
    )


def _correct_profile(
    base_grid: np.ndarray,
    base_value: np.ndarray,
    fine_grid: np.ndarray,
    fine_value: np.ndarray,
    coarse_grid: np.ndarray,
    coarse_value: np.ndarray,
) -> np.ndarray:
    base_common = np.interp(CORRECTION_GRID, base_grid, base_value)
    fine_common = np.interp(CORRECTION_GRID, fine_grid, fine_value)
    coarse_common = np.interp(CORRECTION_GRID, coarse_grid, coarse_value)
    corrected_common = base_common + fine_common - coarse_common
    return np.interp(base_grid, CORRECTION_GRID, corrected_common)


def _correct_state(
    base: TransportState,
    fine: TransportState,
    coarse: TransportState,
) -> TransportState:
    corrected = {}
    for field in (
        "psi",
        "ion_temperature",
        "electron_temperature",
        "electron_density",
    ):
        corrected[field] = _correct_profile(
            base.rho,
            np.asarray(getattr(base, field)),
            fine.rho,
            np.asarray(getattr(fine, field)),
            coarse.rho,
            np.asarray(getattr(coarse, field)),
        )
    if np.any(corrected["ion_temperature"] <= 0.0):
        raise RuntimeError("Parareal correction produced non-positive ion temperature")
    if np.any(corrected["electron_temperature"] <= 0.0):
        raise RuntimeError(
            "Parareal correction produced non-positive electron temperature"
        )
    if np.any(corrected["electron_density"] <= 0.0):
        raise RuntimeError("Parareal correction produced non-positive electron density")
    return TransportState(rho=base.rho, **corrected)


def _correct_sample(
    base: WaveformSample,
    fine: WaveformSample,
    coarse: WaveformSample,
) -> WaveformSample:
    values: dict[str, np.ndarray] = {}
    for name in base.values:
        base_value = np.asarray(base.values[name])
        fine_value = np.asarray(fine.values[name])
        coarse_value = np.asarray(coarse.values[name])
        if (
            base_value.shape != fine_value.shape
            or base_value.shape != coarse_value.shape
        ):
            raise RuntimeError(f"Parareal field shape changed for {name}")
        if np.issubdtype(base_value.dtype, np.bool_) or np.issubdtype(
            base_value.dtype, np.integer
        ):
            if not np.array_equal(fine_value, coarse_value):
                raise RuntimeError(
                    f"discrete Parareal field disagrees between fine and coarse: {name}"
                )
            values[name] = fine_value
        elif base_value.ndim == 1 and base_value.size == base.radial_grid.size:
            values[name] = _correct_profile(
                base.radial_grid,
                base_value,
                fine.radial_grid,
                fine_value,
                coarse.radial_grid,
                coarse_value,
            )
        else:
            values[name] = base_value + fine_value - coarse_value
    return WaveformSample(
        time=WINDOW_LENGTH,
        radial_grid=base.radial_grid,
        phi_boundary=(base.phi_boundary + fine.phi_boundary - coarse.phi_boundary),
        axis_reference=(
            base.axis_reference + fine.axis_reference - coarse.axis_reference
        ),
        boundary_reference=(
            base.boundary_reference
            + fine.boundary_reference
            - coarse.boundary_reference
        ),
        values=values,
    )


def _correct_boundary(
    base: BoundaryState,
    fine: FineResult,
    coarse: CoarseResult,
) -> BoundaryState:
    if fine.initial.history.selected_class is not coarse.initial.history.selected_class:
        raise RuntimeError("fine and coarse inputs disagree on branch class")
    selected_class = fine.boundary.history.selected_class
    if (
        base.history.selected_class is not None
        and selected_class is not base.history.selected_class
    ):
        raise RuntimeError("fine branch class disagrees with the causal predecessor")
    history_increment = (
        fine.boundary.history.sequence_index - fine.initial.history.sequence_index
    )
    history = dataclasses.replace(
        fine.boundary.history,
        sequence_index=base.history.sequence_index + history_increment,
    )
    boundary = BoundaryState(
        transport=_correct_state(
            base.transport,
            fine.boundary.transport,
            coarse.boundary.transport,
        ),
        geometry=_correct_sample(
            base.geometry,
            fine.boundary.geometry,
            coarse.boundary.geometry,
        ),
        source=_correct_sample(
            base.source,
            fine.boundary.source,
            coarse.boundary.source,
        ),
        equilibrium_flux=(
            np.asarray(base.equilibrium_flux)
            + np.asarray(fine.boundary.equilibrium_flux)
            - np.asarray(coarse.boundary.equilibrium_flux)
        ),
        plasma_current=(
            base.plasma_current
            + fine.boundary.plasma_current
            - coarse.boundary.plasma_current
        ),
        history=history,
        policy=fine.boundary.policy,
    )
    boundary.geometry.geometry()
    return boundary


def _continuous_arrays(state: BoundaryState) -> Iterable[tuple[str, np.ndarray]]:
    for field in (
        "psi",
        "ion_temperature",
        "electron_temperature",
        "electron_density",
    ):
        yield (
            f"transport.{field}",
            np.interp(
                CORRECTION_GRID,
                state.transport.rho,
                np.asarray(getattr(state.transport, field)),
            ),
        )
    for prefix, sample in (("geometry", state.geometry), ("source", state.source)):
        yield f"{prefix}.phi_boundary", np.asarray(sample.phi_boundary)
        yield f"{prefix}.axis_reference", np.asarray(sample.axis_reference)
        yield f"{prefix}.boundary_reference", np.asarray(sample.boundary_reference)
        for name, value in sample.values.items():
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(
                array.dtype, np.integer
            ):
                continue
            if array.ndim == 1 and array.size == sample.radial_grid.size:
                array = np.interp(CORRECTION_GRID, sample.radial_grid, array)
            yield f"{prefix}.{name}", array
    yield "equilibrium.flux", np.asarray(state.equilibrium_flux)
    yield "equilibrium.plasma_current", np.asarray(state.plasma_current)


def _relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.max(np.abs(left), initial=0.0)),
        float(np.max(np.abs(right), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    return float(np.max(np.abs(left - right), initial=0.0)) / scale


def _boundary_residuals(
    previous: Sequence[BoundaryState],
    current: Sequence[BoundaryState],
) -> dict[str, float]:
    residuals: dict[str, float] = {}
    for window, (left, right) in enumerate(zip(previous, current, strict=True)):
        left_fields = dict(_continuous_arrays(left))
        right_fields = dict(_continuous_arrays(right))
        for name in left_fields:
            residuals[f"window.{window}.{name}"] = _relative_difference(
                left_fields[name], right_fields[name]
            )
    return residuals


def _trajectory_difference(
    reference: Sequence[BoundaryState],
    candidate: Sequence[BoundaryState],
) -> dict[str, float]:
    return _boundary_residuals(reference, candidate)


def _append(
    rows: list[dict[str, str]],
    tree_sha: str,
    lane: str,
    kind: str,
    field: str,
    value: Any,
    unit: str,
    *,
    correction: int | str = "",
    window: int | str = "",
    status: str = "measured",
    provenance: str = "experiment.py",
) -> None:
    rows.append(
        {
            "tree_sha": tree_sha,
            "lane": lane,
            "correction": str(correction),
            "window": str(window),
            "kind": kind,
            "field": field,
            "value": _format(value),
            "unit": unit,
            "status": status,
            "provenance": provenance,
        }
    )


def _record_fine(
    rows: list[dict[str, str]],
    tree_sha: str,
    lane: str,
    correction: int | str,
    window: int,
    result: FineResult,
) -> None:
    receipt = result.receipt
    convergence = receipt.convergence
    conservation = receipt.conservation
    for field, value, unit in (
        ("wall_time", result.wall_seconds, "s"),
        ("input_hash", _hash_boundary(result.initial), "sha256"),
        ("output_hash", _hash_boundary(result.boundary), "sha256"),
        ("iterations_used", convergence.iterations_used, "count"),
        ("contraction_estimate", convergence.contraction_estimate, "ratio"),
        ("gating_norm", convergence.gating_norm, "relative"),
        ("all_field_norm", convergence.all_field_norm, "relative"),
        ("damping_applied", convergence.damping_applied, "fraction"),
        ("flux_closure_error", conservation.flux_closure_error, "Wb"),
        ("flux_closure_residual", conservation.flux_closure_residual, "relative"),
        ("current_continuity_error", conservation.current_continuity_error, "A"),
        (
            "current_continuity_residual",
            conservation.current_continuity_residual,
            "relative",
        ),
        ("selection_history_sequence", result.boundary.history.sequence_index, "count"),
        (
            "selection_history_class",
            result.boundary.history.selected_class.name.lower(),
            "text",
        ),
    ):
        _append(
            rows,
            tree_sha,
            lane,
            "fine_receipt",
            field,
            value,
            unit,
            correction=correction,
            window=window,
        )
    for iteration, residual in enumerate(convergence.gating_norm_trace, start=1):
        _append(
            rows,
            tree_sha,
            lane,
            "window_residual_trace",
            "gating_norm",
            residual,
            "relative",
            correction=correction,
            window=window,
            provenance=f"exchange={iteration}",
        )
    for field, value in convergence.exit_residual.items():
        _append(
            rows,
            tree_sha,
            lane,
            "window_exit_residual",
            field,
            value,
            "relative",
            correction=correction,
            window=window,
        )
    valid = sum(bool(item["record_valid"]) for item in result.extraction_records)
    _append(
        rows,
        tree_sha,
        lane,
        "extraction_receipt",
        "valid_records",
        valid,
        "count",
        correction=correction,
        window=window,
        provenance=f"total={len(result.extraction_records)}",
    )
    terminal = result.extraction_records[-1]
    for field, unit in (
        ("selected_class", "text"),
        ("selection_reason", "text"),
        ("limited_core_cells", "count"),
        ("diverted_core_cells", "count"),
    ):
        _append(
            rows,
            tree_sha,
            lane,
            "branch_receipt",
            field,
            terminal[field],
            unit,
            correction=correction,
            window=window,
        )


def _write_results(rows: Sequence[Mapping[str, str]]) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=RESULT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _report(
    tree_sha: str,
    fixture: PreparedFixture,
    serial_results: Sequence[FineResult],
    correction_results: Sequence[Sequence[FineResult]],
    correction_residuals: Sequence[float],
    coarse_wall: float,
    correction_wall: float,
    scientific_difference: float,
    branch_match: bool,
    ledger_match: bool,
    outer_converged: bool,
) -> str:
    serial_wall = sum(result.wall_seconds for result in serial_results)
    parareal_wall = coarse_wall + correction_wall
    speedup = serial_wall / parareal_wall
    corrections_used = len(correction_results)
    performance_pass = speedup >= 2.0 and corrections_used <= 2
    scientific_pass = (
        outer_converged
        and scientific_difference <= OUTER_TOLERANCE
        and branch_match
        and ledger_match
    )
    verdict = "PASS" if performance_pass and scientific_pass else "QUALIFIED NEGATIVE"
    lines = [
        "# Eight-window Parareal experiment",
        "",
        (
            f"Tree: `{tree_sha}`. Backend: `{jax.default_backend()}`. "
            f"Verdict: **{verdict}**."
        ),
        "",
        "## Frozen contract",
        "",
        (
            "Eight contiguous 2.5 ms windows use the repository free-boundary "
            "fixture (25 x 25 solve, exact evaluation on 49 x 49), eight TORAX "
            "radial cells, 14 surface bins, auxiliary multiplier 0.5, tolerance "
            "0.005 and a ten-iteration ordinary cap. Fine propagation is unchanged "
            "`solve_window`; coarse propagation is one public fixed-step TORAX "
            "transport sweep over the incoming geometry held constant."
        ),
        (
            f"The live plan's independently locked adaptive-damping decision is "
            f"the current-tree authority, so every fine call starts at damping "
            f"`{_format(demonstration.DAMPING)}` rather than restoring the research "
            "document's earlier 0.5 default. No experiment result was used to make "
            "that choice."
        ),
        (
            "All calls ran strictly serially on the login-node CPU. Continuous "
            "Parareal corrections were formed on the declared 25-point normalised "
            "grid before projection back to each fixed native grid. Discrete branch "
            "class, selector state and history were passed causally and never added."
        ),
        f"One-time fixture preparation: `{_format(fixture.preparation_seconds)}` s.",
        "",
        "## Outcome",
        "",
        "| measure | result | gate |",
        "|---|---:|---|",
        f"| serial fine wall | `{_format(serial_wall)}` s | baseline |",
        f"| Parareal end-to-end wall | `{_format(parareal_wall)}` s | measured |",
        f"| speedup | `{_format(speedup)}x` | >= 2x |",
        f"| corrections used | `{corrections_used}` | <= 2 |",
        f"| outer residual | `{_format(correction_residuals[-1])}` | <= 0.005 |",
        (
            "| sequential-chain difference | "
            f"`{_format(scientific_difference)}` | <= 0.005 |"
        ),
        f"| branch histories identical | `{_format(branch_match)}` | true |",
        f"| ledger closures unchanged | `{_format(ledger_match)}` | true |",
        "",
    ]
    if verdict != "PASS":
        mechanisms = []
        if speedup < 2.0:
            mechanisms.append(
                "the required serialized execution makes each fine wave the sum "
                "of eight window calls, so Parareal adds fine work instead of "
                "overlapping it"
            )
        if not outer_converged:
            mechanisms.append(
                "the outer correction did not reach 0.005 in two corrections"
            )
        if scientific_difference > OUTER_TOLERANCE:
            mechanisms.append(
                "the corrected chain remained outside the fine-chain tolerance"
            )
        if not branch_match:
            mechanisms.append("branch history diverged")
        if not ledger_match:
            mechanisms.append("a nested fine-window ledger closure changed")
        lines.extend(
            [
                "The miss is retained as a negative result. Mechanism: "
                + "; ".join(mechanisms)
                + ". No tolerance, physics, device count or surrogate was changed.",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-window wall and nested convergence receipts",
            "",
            (
                "| lane | correction | window | wall (s) | iterations | "
                "contraction | exit gate | flux closure | current closure | branch |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for window, result in enumerate(serial_results):
        receipt = result.receipt
        lines.append(
            f"| serial | - | {window} | `{_format(result.wall_seconds)}` | "
            f"{receipt.convergence.iterations_used} | "
            f"`{_format(receipt.convergence.contraction_estimate)}` | "
            f"`{_format(receipt.convergence.gating_norm)}` | "
            f"`{_format(receipt.conservation.flux_closure_residual)}` | "
            f"`{_format(receipt.conservation.current_continuity_residual)}` | "
            f"`{result.boundary.history.selected_class.name.lower()}` |"
        )
    for correction, wave in enumerate(correction_results, start=1):
        for window, result in enumerate(wave):
            receipt = result.receipt
            lines.append(
                f"| Parareal | {correction} | {window} | "
                f"`{_format(result.wall_seconds)}` | "
                f"{receipt.convergence.iterations_used} | "
                f"`{_format(receipt.convergence.contraction_estimate)}` | "
                f"`{_format(receipt.convergence.gating_norm)}` | "
                f"`{_format(receipt.conservation.flux_closure_residual)}` | "
                f"`{_format(receipt.conservation.current_continuity_residual)}` | "
                f"`{result.boundary.history.selected_class.name.lower()}` |"
            )
    lines.extend(
        [
            "",
            "## Outer correction",
            "",
            "| correction | outer residual |",
            "|---:|---:|",
        ]
    )
    lines.extend(
        f"| {index} | `{_format(value)}` |"
        for index, value in enumerate(correction_residuals, start=1)
    )
    lines.extend(
        [
            "",
            (
                "`results.tsv` retains every fine-window residual trace and exit "
                "field, extraction-validity count, branch receipt, boundary hash, "
                "wall time and conservation closure, plus every coarse call and "
                "outer-correction residual."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run() -> int:
    tree_sha = _tree_sha()
    rows: list[dict[str, str]] = []
    print(f"tree={tree_sha}", flush=True)
    fixture = _prepare_fixture()
    print(f"fixture prepared in {fixture.preparation_seconds:.3f}s", flush=True)

    serial_states = [fixture.initial]
    serial_results: list[FineResult] = []
    for window in range(WINDOW_COUNT):
        print(f"serial window {window + 1}/{WINDOW_COUNT}", flush=True)
        result = _fine_propagate(fixture, serial_states[-1])
        serial_results.append(result)
        serial_states.append(result.boundary)
        _record_fine(rows, tree_sha, "serial", "", window, result)

    coarse_states = [fixture.initial]
    coarse_prediction: list[CoarseResult] = []
    coarse_wall = 0.0
    for window in range(WINDOW_COUNT):
        result = _coarse_propagate(fixture, coarse_states[-1])
        coarse_prediction.append(result)
        coarse_states.append(result.boundary)
        coarse_wall += result.wall_seconds
        _append(
            rows,
            tree_sha,
            "parareal",
            "coarse_wall",
            "wall_time",
            result.wall_seconds,
            "s",
            correction=0,
            window=window,
        )

    iterates = coarse_states
    correction_results: list[list[FineResult]] = []
    correction_residuals: list[float] = []
    correction_wall = 0.0
    for correction in range(1, MAXIMUM_CORRECTIONS + 1):
        print(f"Parareal correction {correction}/{MAXIMUM_CORRECTIONS}", flush=True)
        fine_wave: list[FineResult] = []
        old_coarse: list[CoarseResult] = []
        for window in range(WINDOW_COUNT):
            print(
                f"correction {correction} fine window {window + 1}/{WINDOW_COUNT}",
                flush=True,
            )
            fine = _fine_propagate(fixture, iterates[window])
            coarse = _coarse_propagate(fixture, iterates[window])
            fine_wave.append(fine)
            old_coarse.append(coarse)
            correction_wall += fine.wall_seconds + coarse.wall_seconds
            _record_fine(rows, tree_sha, "parareal", correction, window, fine)

        corrected = [fixture.initial]
        for window in range(WINDOW_COUNT):
            causal = _coarse_propagate(fixture, corrected[-1])
            correction_wall += causal.wall_seconds
            corrected.append(
                _correct_boundary(
                    causal.boundary, fine_wave[window], old_coarse[window]
                )
            )
            _append(
                rows,
                tree_sha,
                "parareal",
                "coarse_wall",
                "wall_time",
                causal.wall_seconds,
                "s",
                correction=correction,
                window=window,
                provenance="causal correction sweep",
            )
        residuals = _boundary_residuals(iterates, corrected)
        maximum = max(residuals.values())
        correction_residuals.append(maximum)
        _append(
            rows,
            tree_sha,
            "parareal",
            "outer_residual",
            "maximum",
            maximum,
            "relative",
            correction=correction,
        )
        for field, value in residuals.items():
            _append(
                rows,
                tree_sha,
                "parareal",
                "outer_residual_field",
                field,
                value,
                "relative",
                correction=correction,
            )
        correction_results.append(fine_wave)
        iterates = corrected
        if maximum <= OUTER_TOLERANCE:
            break

    scientific_residuals = _trajectory_difference(serial_states, iterates)
    scientific_difference = max(scientific_residuals.values())
    for field, value in scientific_residuals.items():
        _append(
            rows,
            tree_sha,
            "comparison",
            "serial_difference",
            field,
            value,
            "relative",
        )
    serial_classes = tuple(state.history.selected_class for state in serial_states)
    parareal_classes = tuple(state.history.selected_class for state in iterates)
    branch_match = serial_classes == parareal_classes
    serial_closures = tuple(
        (
            result.receipt.conservation.flux_closure_residual,
            result.receipt.conservation.current_continuity_residual,
        )
        for result in serial_results
    )
    final_fine_closures = tuple(
        (
            result.receipt.conservation.flux_closure_residual,
            result.receipt.conservation.current_continuity_residual,
        )
        for result in correction_results[-1]
    )
    ledger_match = all(
        np.allclose(left, right, rtol=0.0, atol=OUTER_TOLERANCE)
        for left, right in zip(serial_closures, final_fine_closures, strict=True)
    )
    outer_converged = correction_residuals[-1] <= OUTER_TOLERANCE

    _append(
        rows, tree_sha, "summary", "verdict", "branch_match", branch_match, "boolean"
    )
    _append(
        rows, tree_sha, "summary", "verdict", "ledger_match", ledger_match, "boolean"
    )
    _append(
        rows,
        tree_sha,
        "summary",
        "verdict",
        "scientific_difference",
        scientific_difference,
        "relative",
    )
    _write_results(rows)
    REPORT_PATH.write_text(
        _report(
            tree_sha,
            fixture,
            serial_results,
            correction_results,
            correction_residuals,
            coarse_wall,
            correction_wall,
            scientific_difference,
            branch_match,
            ledger_match,
            outer_converged,
        ),
        encoding="utf-8",
    )
    print(f"report={REPORT_PATH}", flush=True)
    print(f"results={RESULTS_PATH}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
