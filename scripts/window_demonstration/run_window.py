"""Run and report one free-boundary equilibrium--TORAX exchange window."""

from __future__ import annotations

import csv
import dataclasses
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import electron_volt

from nova.equilibrium import (
    FluxLattice,
    GreenSourceRepresentation,
    evaluate_forward_equilibrium,
    extract_flux_surface_geometry,
)
from nova.equilibrium.flux_surface_extraction import _axis_connected_core
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.jax.config import configure_dtypes
from nova.transport.coupled_window import (
    ConvergedNonConfinedError,
    ExchangeSweepResult,
    EquilibriumBranchReceipt,
    TransportSweepReceipt,
    Waveform,
    WindowConfig,
    WindowConservationError,
    WindowConvergenceError,
    equilibrium_sweep,
    solve_window,
    transport_sweep,
)
from nova.transport.evolved_state import (
    EvolvedFluxFunction,
    forward_source_from_receipt,
)
from nova.transport.forward import (
    TransportGeometry,
    TransportModel,
    TransportRung,
    TransportState,
)
from tests import test_equilibrium_forward_solve as forward_fixture

EVALUATIONS = forward_fixture.EVALUATIONS
free_boundary_machine_fixture = forward_fixture.machine

WINDOW_LENGTH_SECONDS = 0.01
EQUILIBRIUM_TIMES = np.array([0.0, WINDOW_LENGTH_SECONDS])
TRANSPORT_TIMES = np.array([0.0, WINDOW_LENGTH_SECONDS])
ITERATION_CAP = 10
CONTRACTION_THRESHOLD = 0.8
HARD_ITERATION_CEILING = 20
CONVERGENCE_TOLERANCE = 5.0e-3
DAMPING = 0.5
EQUILIBRIUM_SOLVE_TOLERANCE = 1.0e-6
RADIAL_CELLS = 8
SURFACE_BINS = 14
SOURCE_SAMPLES = 25
EXTRACTION_POINTS = 49
ION_DENSITY_PER_ELECTRON = 1.0
MINIMUM_TEMPERATURE_KEV = 0.2

OUTPUT_DIRECTORY = Path(__file__).resolve().parent
REPORT_PATH = OUTPUT_DIRECTORY / "report.md"
RECEIPTS_PATH = OUTPUT_DIRECTORY / "receipts.tsv"
TSV_FIELDS = (
    "regime",
    "candidate",
    "kind",
    "iteration",
    "sample",
    "side",
    "field",
    "value",
    "unit",
)


@dataclasses.dataclass(frozen=True)
class RegimeConfig:
    """One declared physical window and returned-source drive scale."""

    name: str
    window_length: float
    auxiliary_source_multiplier: float
    candidate: int | None = None

    @property
    def time_grid(self) -> np.ndarray:
        return np.asarray((0.0, self.window_length), dtype=np.float64)


@dataclasses.dataclass
class RegimeResult:
    """All receipts and measurements surrendered by one window attempt."""

    config: RegimeConfig
    outcome_type: str
    outcome: str
    convergence: Any = None
    conservation_receipt: Any = None
    transport_receipt: TransportSweepReceipt | None = None
    timings: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    extractions: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    branches: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    terminal_branch: dict[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.outcome_type == "WindowReceipt"


STRONG = RegimeConfig("strong", WINDOW_LENGTH_SECONDS, 1.0)
GENTLE_CANDIDATES = (
    RegimeConfig("gentle", 0.0025, 0.5, 1),
    RegimeConfig("gentle", 0.001, 0.5, 2),
    RegimeConfig("gentle", 0.0005, 0.5, 3),
    RegimeConfig("gentle", 0.001, 0.25, 4),
    RegimeConfig("gentle", 0.0005, 0.25, 5),
    RegimeConfig("gentle", 0.0005, 0.0, 6),
)


def _format(value: Any) -> str:
    """Return a stable full-precision receipt value."""
    if value is None:
        return "none"
    if isinstance(value, bool | np.bool_):
        return str(bool(value)).lower()
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, str):
        return value
    return f"{float(np.asarray(value)):.17g}"


def _block_and_copy(tree: Mapping[str, Any]) -> dict[str, Any]:
    """Materialise one service result after all device work completes."""
    ready = jax.tree.map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        tree,
    )
    return {name: np.asarray(value) for name, value in ready.items()}


def _fixture_machine():
    """Instantiate the repository's existing smallest free-boundary fixture."""
    factory = getattr(free_boundary_machine_fixture, "__wrapped__", None)
    if factory is None:
        raise RuntimeError("the free-boundary machine fixture has no callable factory")
    return factory()


def _field_function(
    source: ForwardSource, flux_span: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the source's physical field-function primitive for extraction."""
    psi_n = jnp.linspace(0.0, 1.0, 101, dtype=jnp.float64)
    squared = source.core.field_function_squared(
        psi_n,
        source.boundary_field_function,
        jnp.asarray(flux_span, dtype=jnp.float64),
    )
    return np.asarray(psi_n), np.sqrt(np.asarray(squared))


def _rectangle(radius: float, height: float, size: float = 0.05) -> np.ndarray:
    """Return one rectangular source section from the fixture definition."""
    half = 0.5 * size
    return np.asarray(
        (
            (radius - half, height - half),
            (radius + half, height - half),
            (radius + half, height + half),
            (radius - half, height + half),
        )
    )


def _fixture_sources(profile) -> GreenSourceRepresentation:
    """Retain the fixture source sections needed for exact flux evaluation."""
    angle = (
        2.0 * np.pi * np.arange(forward_fixture.CONDUCTORS) / forward_fixture.CONDUCTORS
    )
    conductor = np.c_[1.0 + 0.62 * np.cos(angle), 0.62 * np.sin(angle)]
    return GreenSourceRepresentation(
        external_sections=tuple(
            _rectangle(radius, height) for radius, height in conductor
        ),
        external_current=np.asarray(profile.operator.external_current),
        plasma_sections=tuple(
            _rectangle(radius, height) for radius, height in profile.lattice.coordinate
        ),
        external_kernel="hybrid_rectangle",
        plasma_kernel="hybrid_rectangle",
    )


def _extraction_lattice(profile) -> FluxLattice:
    """Build the denser target lattice required by the extraction service."""
    return FluxLattice(
        np.linspace(
            profile.lattice.radius[0],
            profile.lattice.radius[-1],
            EXTRACTION_POINTS,
        ),
        np.linspace(
            profile.lattice.height[0],
            profile.lattice.height[-1],
            EXTRACTION_POINTS,
        ),
    )


def _geometry_from_equilibrium(
    equilibrium,
    source: ForwardSource,
    lattice: FluxLattice,
    sources: GreenSourceRepresentation,
) -> tuple[TransportGeometry, dict[str, Any]]:
    """Evaluate and extract one solved fixture equilibrium on a dense lattice."""
    evaluation_start = time.perf_counter()
    psi_height_radius = evaluate_forward_equilibrium(
        equilibrium, lattice, sources
    ).block_until_ready()
    evaluation_seconds = time.perf_counter() - evaluation_start
    mesh_r, mesh_z = np.meshgrid(lattice.radius, lattice.height, indexing="xy")
    _wall, wall_flux = forward_fixture._wall_loop()
    inside_limiter = jnp.asarray(forward_fixture._solovev(mesh_r, mesh_z) >= wall_flux)
    axis_psi = float(equilibrium.topology.axis_flux)
    boundary_psi = float(equilibrium.topology.boundary_flux)
    flux_span = boundary_psi - axis_psi
    core_count = int(
        np.asarray(
            _axis_connected_core(
                (psi_height_radius - axis_psi) / flux_span,
                inside_limiter,
            )
        ).sum()
    )
    field_psi_n, field_function = _field_function(source, flux_span)
    major_radius = float(equilibrium.topology.axis[0])
    record = extract_flux_surface_geometry(
        psi_height_radius,
        jnp.asarray(lattice.radius),
        jnp.asarray(lattice.height),
        inside_limiter,
        axis_psi=jnp.asarray(axis_psi),
        boundary_psi=jnp.asarray(boundary_psi),
        profile_coefficients=jnp.zeros(2, dtype=jnp.float64),
        coefficient_scale=jnp.ones(2, dtype=jnp.float64),
        ip_amperes=jnp.asarray(equilibrium.moments.plasma_current),
        major_radius=jnp.asarray(major_radius),
        boundary_toroidal_field=jnp.asarray(
            source.boundary_field_function / major_radius
        ),
        field_function_psi_n=jnp.asarray(field_psi_n),
        field_function=jnp.asarray(field_function),
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=RADIAL_CELLS,
        n_surface_bins=SURFACE_BINS,
    )
    materialised = _block_and_copy(record)
    measurement = {
        "map_height": psi_height_radius.shape[0],
        "map_radius": psi_height_radius.shape[1],
        "core_count": core_count,
        "record_valid": bool(materialised["valid"]),
        "surface_arc_valid": bool(materialised["surface_arc_valid"]),
        "surface_arc_invalid_count": int(materialised["surface_arc_invalid_count"]),
        "surface_arc_first_invalid_cell": int(
            materialised["surface_arc_first_invalid_cell"]
        ),
        "surface_arc_first_invalid_level": float(
            materialised["surface_arc_first_invalid_level"]
        ),
        "exact_evaluation_seconds": evaluation_seconds,
    }
    if not measurement["record_valid"]:
        raise RuntimeError(
            "exact extraction returned an invalid geometry: "
            f"lattice={psi_height_radius.shape}, core={core_count}, "
            f"arcs_valid={measurement['surface_arc_valid']}, "
            f"invalid_arcs={measurement['surface_arc_invalid_count']}, "
            f"first_invalid_cell={measurement['surface_arc_first_invalid_cell']}, "
            f"first_invalid_level={measurement['surface_arc_first_invalid_level']}"
        )
    return TransportGeometry(materialised), measurement


def _source_from_sample(sample) -> ForwardSource:
    """Restore one sampled source waveform as the equilibrium callable seam."""
    return ForwardSource(
        core=DomainProfile(
            p_prime=EvolvedFluxFunction(sample.radial_grid, sample.values["p_prime"]),
            ff_prime=EvolvedFluxFunction(sample.radial_grid, sample.values["ff_prime"]),
        ),
        boundary_pressure=float(sample.values["boundary_pressure"]),
        boundary_field_function=float(sample.values["boundary_field_function"]),
    )


def _source_waveform(
    time_grid: np.ndarray,
    sources: Sequence[ForwardSource],
    coordinates: Sequence[Any],
) -> Waveform:
    """Sample callable equilibrium sources onto one coordinate-aware waveform."""
    psi_n = np.linspace(0.0, 1.0, SOURCE_SAMPLES)
    radial_grid = np.broadcast_to(psi_n, (len(sources), psi_n.size))
    return Waveform(
        time=time_grid,
        radial_grid=radial_grid,
        phi_boundary=np.asarray([sample.phi_boundary for sample in coordinates]),
        axis_reference=np.asarray([sample.axis_reference for sample in coordinates]),
        boundary_reference=np.asarray(
            [sample.boundary_reference for sample in coordinates]
        ),
        values={
            "p_prime": np.stack(
                [np.asarray(source.core.p_prime(psi_n)) for source in sources]
            ),
            "ff_prime": np.stack(
                [np.asarray(source.core.ff_prime(psi_n)) for source in sources]
            ),
            "boundary_pressure": np.asarray(
                [source.boundary_pressure for source in sources]
            ),
            "boundary_field_function": np.asarray(
                [source.boundary_field_function for source in sources]
            ),
        },
    )


def _initial_state(
    geometry: TransportGeometry, source: ForwardSource
) -> TransportState:
    """Form a positive multi-channel state consistent with the fixture pressure."""
    record = geometry.record
    rho = np.asarray(record["rho_face"])
    psi = np.asarray(record["flux_sign"]) * np.asarray(record["psi_face"])
    flux_span = float(record["boundary_psi"] - record["axis_psi"])
    pressure = np.asarray(
        source.core.pressure(
            jnp.asarray(record["r0"]),
            jnp.asarray(record["psi_n_face"]),
            source.boundary_pressure,
            flux_span,
        )
    )
    density = np.full_like(rho, 1.0e20)
    total_temperature = pressure / (density * 1.0e3 * electron_volt)
    temperature = np.maximum(0.5 * total_temperature, MINIMUM_TEMPERATURE_KEV)
    return TransportState(
        rho=rho,
        psi=psi,
        ion_temperature=temperature,
        electron_temperature=temperature,
        electron_density=density,
    )


def _torax_model(window_length: float = WINDOW_LENGTH_SECONDS) -> TransportModel:
    """Return the ordinary fixed-step TORAX multi-channel configuration."""
    from torax._src.test_utils.default_configs import get_default_config_dict

    config = get_default_config_dict()
    config["numerics"].update(
        {
            "fixed_dt": window_length,
            "max_dt": window_length,
            "min_dt": 1.0e-8,
            "adaptive_dt": False,
        }
    )
    config["time_step_calculator"] = {"calculator_type": "fixed"}
    return TransportModel(
        TransportRung.TORAX_MULTI_CHANNEL,
        torax_config=config,
    )


def _scaled_source_waveform(
    geometry_waveform: Waveform,
    time_grid: np.ndarray,
    baseline_source: ForwardSource,
    evolved_sources: Sequence[ForwardSource],
    multiplier: float,
) -> Waveform:
    """Scale the transport-returned source change about the fixture source."""
    if not 0.0 <= multiplier <= 1.0:
        raise ValueError("auxiliary source multiplier must lie in [0, 1]")
    psi_n = np.linspace(0.0, 1.0, SOURCE_SAMPLES)
    baseline_p = np.asarray(baseline_source.core.p_prime(psi_n))
    baseline_ff = np.asarray(baseline_source.core.ff_prime(psi_n))
    p_prime = []
    ff_prime = []
    boundary_pressure = []
    boundary_field_function = []
    coordinates = []
    for sample_time, evolved in zip(time_grid, evolved_sources, strict=True):
        evolved_p = np.asarray(evolved.core.p_prime(psi_n))
        evolved_ff = np.asarray(evolved.core.ff_prime(psi_n))
        p_prime.append(baseline_p + multiplier * (evolved_p - baseline_p))
        ff_prime.append(baseline_ff + multiplier * (evolved_ff - baseline_ff))
        boundary_pressure.append(
            baseline_source.boundary_pressure
            + multiplier
            * (evolved.boundary_pressure - baseline_source.boundary_pressure)
        )
        boundary_field_function.append(
            baseline_source.boundary_field_function
            + multiplier
            * (
                evolved.boundary_field_function
                - baseline_source.boundary_field_function
            )
        )
        coordinates.append(geometry_waveform.sample(float(sample_time)))
    radial_grid = np.broadcast_to(psi_n, (time_grid.size, psi_n.size))
    return Waveform(
        time=time_grid,
        radial_grid=radial_grid,
        phi_boundary=np.asarray([sample.phi_boundary for sample in coordinates]),
        axis_reference=np.asarray([sample.axis_reference for sample in coordinates]),
        boundary_reference=np.asarray(
            [sample.boundary_reference for sample in coordinates]
        ),
        values={
            "p_prime": np.stack(p_prime),
            "ff_prime": np.stack(ff_prime),
            "boundary_pressure": np.asarray(boundary_pressure),
            "boundary_field_function": np.asarray(boundary_field_function),
        },
    )


def _branch_measurement(
    branch: EquilibriumBranchReceipt, exchange: int
) -> dict[str, Any]:
    """Flatten one typed branch-selection receipt without interpreting it."""
    selection = branch.selection.as_dict()
    return {
        "exchange": exchange,
        "sample": branch.sample_index,
        "sample_time": branch.sample_time,
        "limited_core_cells": branch.core_cell_counts[0],
        "diverted_core_cells": branch.core_cell_counts[1],
        "selected_class": selection["selected_class"],
        "previous_class": selection["previous_class"],
        "switched": selection["switched"],
        "reason": selection["reason"],
        "limited_available": selection["availability"]["limited"],
        "diverted_available": selection["availability"]["diverted"],
        "limited_admissible": selection["admissibility"]["limited"],
        "diverted_admissible": selection["admissibility"]["diverted"],
        "limited_residual": selection["residuals"]["limited"],
        "diverted_residual": selection["residuals"]["diverted"],
    }


def _transport_conservation(receipt: TransportSweepReceipt) -> dict[str, float]:
    """Read absolute and relative closure directly from interval ledgers."""
    flux_ledgers = [item.flux_consumption for item in receipt.receipts]
    boundary = sum(float(item.boundary) for item in flux_ledgers)
    resistive = sum(float(item.resistive) for item in flux_ledgers)
    internal = sum(float(item.internal) for item in flux_ledgers)
    flux_error = abs(boundary - resistive - internal)
    flux_scale = max(abs(boundary), abs(resistive), abs(internal), 1.0e-30)

    current_ledgers = [item.plasma_current for item in receipt.receipts]
    errors = [
        abs(
            float(
                current_ledgers[0].requested_initial
                - current_ledgers[0].achieved_initial
            )
        ),
        abs(
            float(
                current_ledgers[-1].requested_final - current_ledgers[-1].achieved_final
            )
        ),
    ]
    for left, right in zip(current_ledgers[:-1], current_ledgers[1:], strict=True):
        errors.extend(
            [
                abs(float(left.achieved_final - right.achieved_initial)),
                abs(float(left.requested_final - right.requested_initial)),
            ]
        )
    current_error = max(errors)
    current_scale = max(
        *(
            abs(float(value))
            for item in current_ledgers
            for value in dataclasses.astuple(item)
        ),
        1.0,
    )
    return {
        "flux_boundary": boundary,
        "flux_resistive": resistive,
        "flux_internal": internal,
        "flux_closure_error": flux_error,
        "flux_closure_residual": flux_error / flux_scale,
        "current_requested_initial": float(current_ledgers[0].requested_initial),
        "current_requested_final": float(current_ledgers[-1].requested_final),
        "current_achieved_initial": float(current_ledgers[0].achieved_initial),
        "current_achieved_final": float(current_ledgers[-1].achieved_final),
        "current_continuity_error": current_error,
        "current_continuity_residual": current_error / current_scale,
    }


def _write_tsv(
    rows: Sequence[dict[str, Any]],
) -> None:
    """Write the machine-readable receipt ledger."""
    with RECEIPTS_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=TSV_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _append_row(
    rows: list[dict[str, Any]],
    regime: str,
    kind: str,
    field: str,
    value: Any,
    unit: str,
    *,
    candidate: int | str = "",
    iteration: int | str = "",
    sample: int | str = "",
    side: str = "",
) -> None:
    rows.append(
        {
            "regime": regime,
            "candidate": candidate,
            "kind": kind,
            "iteration": iteration,
            "sample": sample,
            "side": side,
            "field": field,
            "value": _format(value),
            "unit": unit,
        }
    )


def _run_regime(
    config: RegimeConfig,
    *,
    profile,
    baseline_equilibrium,
    baseline_geometry: TransportGeometry,
    baseline_extraction: Mapping[str, Any],
    extraction_lattice: FluxLattice,
    fixture_sources: GreenSourceRepresentation,
) -> RegimeResult:
    """Execute exactly one declared window and retain every available receipt."""
    time_grid = config.time_grid
    baseline_source = profile.source
    initial_geometry = Waveform.from_geometries(
        time_grid, (baseline_geometry, baseline_geometry)
    )
    coordinates = tuple(
        initial_geometry.sample(float(sample_time)) for sample_time in time_grid
    )
    initial_source = _source_waveform(
        time_grid, (baseline_source, baseline_source), coordinates
    )
    initial_transport_state = _initial_state(baseline_geometry, baseline_source)
    plasma_current = np.full(
        time_grid.shape, float(baseline_equilibrium.moments.plasma_current)
    )
    model = _torax_model(config.window_length)
    window = WindowConfig(
        length=config.window_length,
        equilibrium_grid=time_grid,
        transport_grid=time_grid,
        iteration_cap=ITERATION_CAP,
        tolerance=CONVERGENCE_TOLERANCE,
        contraction_threshold=CONTRACTION_THRESHOLD,
        hard_iteration_ceiling=HARD_ITERATION_CEILING,
    )
    result = RegimeResult(
        config=config,
        outcome_type="unknown",
        outcome="unknown",
        extractions=[dict(baseline_extraction)],
    )
    counters = {"transport": 0, "equilibrium": 0}
    latest: dict[str, Any] = {}

    def transport_update(geometry_waveform, sample_grid):
        counters["transport"] += 1
        started = time.perf_counter()
        receipt = transport_sweep(
            geometry_waveform,
            initial_transport_state,
            sample_grid,
            plasma_current,
            model,
        )
        evolved_sources = [baseline_source]
        for interval, item in enumerate(receipt.receipts):
            geometry_time = float(receipt.geometry_time[interval])
            evolved_sources.append(
                forward_source_from_receipt(
                    item,
                    geometry_waveform.sample(geometry_time).geometry(),
                    ion_density_per_electron=ION_DENSITY_PER_ELECTRON,
                )
            )
        waveform = _scaled_source_waveform(
            geometry_waveform,
            sample_grid,
            baseline_source,
            evolved_sources,
            config.auxiliary_source_multiplier,
        )
        result.timings.append(
            {
                "iteration": counters["transport"],
                "side": "transport",
                "seconds": time.perf_counter() - started,
            }
        )
        latest["transport"] = receipt
        return ExchangeSweepResult(waveform=waveform, receipt=receipt)

    def equilibrium_update(source_waveform, sample_grid):
        counters["equilibrium"] += 1
        started = time.perf_counter()
        try:
            receipt = equilibrium_sweep(
                profile,
                baseline_equilibrium.flux,
                source_waveform,
                sample_grid,
                _source_from_sample,
                route="anderson",
                solve_options={
                    "evaluations": EVALUATIONS,
                    "tolerance": EQUILIBRIUM_SOLVE_TOLERANCE,
                },
            )
        except ConvergedNonConfinedError:
            result.timings.append(
                {
                    "iteration": counters["equilibrium"],
                    "side": "equilibrium",
                    "seconds": time.perf_counter() - started,
                }
            )
            raise
        for branch in receipt.branch_receipts:
            result.branches.append(_branch_measurement(branch, counters["equilibrium"]))
        geometries = []
        for sample_index, (sample, equilibrium) in enumerate(
            zip(receipt.source_samples, receipt.equilibria, strict=True)
        ):
            geometry, extraction = _geometry_from_equilibrium(
                equilibrium,
                _source_from_sample(sample),
                extraction_lattice,
                fixture_sources,
            )
            extraction.update(iteration=counters["equilibrium"], sample=sample_index)
            result.extractions.append(extraction)
            geometries.append(geometry)
        waveform = Waveform.from_geometries(sample_grid, geometries)
        result.timings.append(
            {
                "iteration": counters["equilibrium"],
                "side": "equilibrium_plus_fsa",
                "seconds": time.perf_counter() - started,
            }
        )
        latest["equilibrium"] = receipt
        return ExchangeSweepResult(waveform=waveform, receipt=receipt)

    try:
        receipt = solve_window(
            initial_geometry,
            initial_source,
            window,
            equilibrium_update,
            transport_update,
            damping=DAMPING,
        )
        result.outcome_type = type(receipt).__name__
        result.outcome = "converged"
        result.convergence = receipt.convergence
        result.conservation_receipt = receipt.conservation
        result.transport_receipt = receipt.transport_receipt
    except ConvergedNonConfinedError as error:
        result.outcome_type = type(error).__name__
        result.outcome = str(error)
        result.terminal_branch = _branch_measurement(
            error.branch_receipt, int(error.exchange_index or counters["equilibrium"])
        )
        result.branches.append(result.terminal_branch)
        result.transport_receipt = latest.get("transport")
    except WindowConvergenceError as error:
        result.outcome_type = type(error).__name__
        result.outcome = str(error)
        result.convergence = error.convergence
        result.transport_receipt = error.transport_receipt
    except WindowConservationError as error:
        result.outcome_type = type(error).__name__
        result.outcome = str(error)
        result.conservation_receipt = error.conservation
        result.transport_receipt = latest.get("transport")
    return result


def _terminal_core(result: RegimeResult) -> str:
    """Return the final limited/diverted core pair from one attempt."""
    if not result.branches:
        return "unavailable"
    branch = result.branches[-1]
    return f"{branch['limited_core_cells']}/{branch['diverted_core_cells']}"


def _two_regime_report(
    strong: RegimeResult,
    gentle_attempts: Sequence[RegimeResult],
    preparation_seconds: float,
) -> str:
    """Render both regimes, preserving typed outcomes and numeric receipts."""
    gentle = next((result for result in gentle_attempts if result.converged), None)
    lines = [
        "# Coupled-window receipts across two regimes",
        "",
        (
            "This is one locked strong-window run and a bounded gentle search. "
            "Every row is from `solve_window`; typed refusals are reported and "
            "were not retried with altered tolerances."
        ),
        "",
        "## Experiment contract",
        "",
        (
            "The equilibrium is the repository's 25 x 25 free-boundary fixture. "
            f"Every accepted sample was evaluated exactly on a {EXTRACTION_POINTS} "
            f"x {EXTRACTION_POINTS} lattice and extracted into {RADIAL_CELLS} "
            f"TORAX radial cells with {SURFACE_BINS} surface bins. TORAX advances "
            "all four transport channels in one fixed step per window."
        ),
        f"Fixture-scale execution backend: `{jax.default_backend()}`.",
        (
            "The auxiliary source multiplier scales the transport-returned "
            "equilibrium-source change about the fixture source: 0 keeps the "
            "fixture source and 1 applies the full returned source. Coordinate "
            "maps still come from the evolving geometry waveform."
        ),
        (
            f"Common knobs: iteration cap `{ITERATION_CAP}`, contraction "
            f"threshold `{_format(CONTRACTION_THRESHOLD)}`, hard iteration "
            f"ceiling `{HARD_ITERATION_CEILING}`, convergence and conservation "
            f"tolerance `{_format(CONVERGENCE_TOLERANCE)}`, damping "
            f"`{_format(DAMPING)}`, equilibrium portfolio tolerance "
            f"`{_format(EQUILIBRIUM_SOLVE_TOLERANCE)}`. One-time fixture "
            f"preparation: `{_format(preparation_seconds)}` s."
        ),
        "",
        (
            "| regime | candidate | window (s) | auxiliary multiplier | "
            "outcome type | terminal limited/diverted core |"
        ),
        "|---|---:|---:|---:|---|---:|",
        (
            f"| strong | - | `{_format(strong.config.window_length)}` | "
            f"`{_format(strong.config.auxiliary_source_multiplier)}` | "
            f"`{strong.outcome_type}` | `{_terminal_core(strong)}` |"
        ),
    ]
    lines.extend(
        (
            f"| gentle | {result.config.candidate} | "
            f"`{_format(result.config.window_length)}` | "
            f"`{_format(result.config.auxiliary_source_multiplier)}` | "
            f"`{result.outcome_type}` | `{_terminal_core(result)}` |"
        )
        for result in gentle_attempts
    )
    lines.extend(
        [
            "",
            "## Strong regime: typed boundary outcome",
            "",
            f"`{strong.outcome}`",
            "",
            "The selector receipts are reproduced for every completed coarse sample:",
            "",
            (
                "| exchange | sample | limited core | diverted core | selected | "
                "verdict | limited/diverted available | limited/diverted residual |"
            ),
            "|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    lines.extend(
        (
            f"| {row['exchange']} | {row['sample']} | "
            f"{row['limited_core_cells']} | {row['diverted_core_cells']} | "
            f"`{row['selected_class']}` | `{row['reason']}` | "
            f"`{_format(row['limited_available'])}/"
            f"{_format(row['diverted_available'])}` | "
            f"`{_format(row['limited_residual'])}/"
            f"{_format(row['diverted_residual'])}` |"
        )
        for row in strong.branches
    )
    lines.extend(
        [
            "",
            "Exact dense-lattice core counts for the strong trajectory:",
            "",
            (
                "| exchange | sample | core cells | extraction valid | "
                "exact evaluation (s) |"
            ),
            "|---:|---:|---:|---|---:|",
        ]
    )
    lines.extend(
        (
            f"| {row['iteration']} | {row['sample']} | {row['core_count']} | "
            f"`{_format(row['record_valid'])}` | "
            f"`{_format(row['exact_evaluation_seconds'])}` |"
        )
        for row in strong.extractions
    )
    lines.extend(["", "## Gentle regime", ""])
    if gentle is None:
        lines.append(
            "None of the bounded candidates returned a converged window receipt. "
            "Their typed terminal outcomes and core counts are the result; no "
            "seventh candidate was attempted."
        )
        for result in gentle_attempts:
            lines.extend(
                ["", f"Candidate {result.config.candidate}: `{result.outcome}`"]
            )
    else:
        convergence = gentle.convergence
        conservation = _transport_conservation(gentle.transport_receipt)
        lines.extend(
            [
                (
                    f"Candidate {gentle.config.candidate} is the first converging "
                    f"candidate: window `{_format(gentle.config.window_length)}` s, "
                    "auxiliary multiplier "
                    f"`{_format(gentle.config.auxiliary_source_multiplier)}`."
                ),
                "",
                f"- Iterations used: `{convergence.iterations_used}`",
                (
                    "- Iterations past ordinary cap: "
                    f"`{convergence.iterations_past_cap}`"
                ),
                (
                    "- Measured contraction estimate: "
                    f"`{_format(convergence.contraction_estimate)}`"
                ),
                f"- Maximum exit residual: `{_format(convergence.maximum_residual)}`",
                f"- Damping applied: `{_format(convergence.damping_applied)}`",
                "",
                "| licensed iteration | licensing contraction |",
                "|---:|---:|",
            ]
        )
        lines.extend(
            (f"| {ITERATION_CAP + offset} | `{_format(contraction)}` |")
            for offset, contraction in enumerate(
                convergence.continuation_contractions, start=1
            )
        )
        lines.extend(
            [
                "",
                "| exchanged field | exit relative residual |",
                "|---|---:|",
            ]
        )
        lines.extend(
            f"| `{field}` | `{_format(value)}` |"
            for field, value in convergence.exit_residual.items()
        )
        lines.extend(
            [
                "",
                "### Conservation ledgers",
                "",
                (
                    "- Flux consumption boundary/resistive/internal: "
                    f"`{_format(conservation['flux_boundary'])}` / "
                    f"`{_format(conservation['flux_resistive'])}` / "
                    f"`{_format(conservation['flux_internal'])}` Wb."
                ),
                (
                    "- Flux closure absolute/relative: "
                    f"`{_format(conservation['flux_closure_error'])}` Wb / "
                    f"`{_format(conservation['flux_closure_residual'])}`."
                ),
                (
                    "- Plasma current requested initial/final: "
                    f"`{_format(conservation['current_requested_initial'])}` / "
                    f"`{_format(conservation['current_requested_final'])}` A; "
                    "achieved initial/final: "
                    f"`{_format(conservation['current_achieved_initial'])}` / "
                    f"`{_format(conservation['current_achieved_final'])}` A."
                ),
                (
                    "- Boundary current continuity absolute/relative: "
                    f"`{_format(conservation['current_continuity_error'])}` A / "
                    f"`{_format(conservation['current_continuity_residual'])}`."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Wall time per exchange sweep",
            "",
            "| regime | candidate | exchange | side | wall time (s) |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for result in (strong, *gentle_attempts):
        candidate = "-" if result.config.candidate is None else result.config.candidate
        lines.extend(
            (
                f"| {result.config.name} | {candidate} | {row['iteration']} | "
                f"{row['side']} | `{_format(row['seconds'])}` |"
            )
            for row in result.timings
        )
    lines.extend(
        [
            "",
            (
                "The TSV is the machine-readable record of every declared knob, "
                "attempt, selector receipt, exit residual, timing, exact extraction "
                "diagnostic and available transport ledger."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _result_rows(
    result: RegimeResult, preparation_seconds: float
) -> list[dict[str, Any]]:
    """Flatten one result into the stable tabular evidence schema."""
    rows: list[dict[str, Any]] = []
    regime = result.config.name
    candidate = result.config.candidate or ""

    def append(kind, field, value, unit, *, iteration="", sample="", side=""):
        _append_row(
            rows,
            regime,
            kind,
            field,
            value,
            unit,
            candidate=candidate,
            iteration=iteration,
            sample=sample,
            side=side,
        )

    for field, value, unit in (
        ("window_length", result.config.window_length, "s"),
        (
            "auxiliary_source_multiplier",
            result.config.auxiliary_source_multiplier,
            "fraction",
        ),
        ("iteration_cap", ITERATION_CAP, "count"),
        ("contraction_threshold", CONTRACTION_THRESHOLD, "ratio"),
        ("hard_iteration_ceiling", HARD_ITERATION_CEILING, "count"),
        ("tolerance", CONVERGENCE_TOLERANCE, "relative"),
        ("damping", DAMPING, "fraction"),
        ("equilibrium_solve_tolerance", EQUILIBRIUM_SOLVE_TOLERANCE, "relative"),
        ("radial_cells", RADIAL_CELLS, "count"),
        ("surface_bins", SURFACE_BINS, "count"),
        ("extraction_points_per_axis", EXTRACTION_POINTS, "count"),
        ("preparation_wall_time", preparation_seconds, "s"),
        ("outcome_type", result.outcome_type, "text"),
        ("outcome", result.outcome, "text"),
    ):
        append("configuration", field, value, unit)
    if result.convergence is not None:
        convergence = result.convergence
        for field, value, unit in (
            ("iterations_used", convergence.iterations_used, "count"),
            ("iterations_past_cap", convergence.iterations_past_cap, "count"),
            ("contraction_estimate", convergence.contraction_estimate, "ratio"),
            ("maximum_residual", convergence.maximum_residual, "relative"),
            ("damping_applied", convergence.damping_applied, "fraction"),
        ):
            append("convergence", field, value, unit)
        for field, value in convergence.exit_residual.items():
            append("exit_residual", field, value, "relative")
        for licensed_iteration, contraction in enumerate(
            convergence.continuation_contractions,
            start=ITERATION_CAP + 1,
        ):
            append(
                "continuation_license",
                "contraction_estimate",
                contraction,
                "ratio",
                iteration=licensed_iteration,
            )
        for iteration, residuals in enumerate(convergence.residual_trace, start=1):
            for field, value in residuals.items():
                append(
                    "residual_trace",
                    field,
                    value,
                    "relative",
                    iteration=iteration,
                )
    for row in result.branches:
        for field in (
            "sample_time",
            "limited_core_cells",
            "diverted_core_cells",
            "selected_class",
            "previous_class",
            "switched",
            "reason",
            "limited_available",
            "diverted_available",
            "limited_admissible",
            "diverted_admissible",
            "limited_residual",
            "diverted_residual",
        ):
            unit = "text"
            if field.endswith("core_cells"):
                unit = "count"
            elif field.endswith("residual"):
                unit = "relative"
            elif field.endswith("available") or field.endswith("admissible"):
                unit = "boolean"
            elif field == "sample_time":
                unit = "s"
            append(
                "branch_selection",
                field,
                row[field],
                unit,
                iteration=row["exchange"],
                sample=row["sample"],
                side="equilibrium",
            )
    for row in result.timings:
        append(
            "exchange_timing",
            "wall_time",
            row["seconds"],
            "s",
            iteration=row["iteration"],
            side=row["side"],
        )
    for row in result.extractions:
        for field, unit in (
            ("map_height", "count"),
            ("map_radius", "count"),
            ("core_count", "count"),
            ("record_valid", "boolean"),
            ("surface_arc_valid", "boolean"),
            ("surface_arc_invalid_count", "count"),
            ("surface_arc_first_invalid_cell", "index"),
            ("surface_arc_first_invalid_level", "normalized_flux"),
            ("exact_evaluation_seconds", "s"),
        ):
            append(
                "extraction_diagnostic",
                field,
                row[field],
                unit,
                iteration=row["iteration"],
                sample=row["sample"],
                side="equilibrium",
            )
    transport = result.transport_receipt
    if transport is not None:
        conservation = _transport_conservation(transport)
        if result.conservation_receipt is not None:
            receipt = result.conservation_receipt
            expected = {
                "flux_closure_error": receipt.flux_closure_error,
                "flux_closure_residual": receipt.flux_closure_residual,
                "current_continuity_error": receipt.current_continuity_error,
                "current_continuity_residual": receipt.current_continuity_residual,
            }
            for field, value in expected.items():
                np.testing.assert_allclose(
                    conservation[field], value, rtol=0.0, atol=0.0
                )
        for field, value in conservation.items():
            unit = "A"
            if field.startswith("flux_") and not field.endswith("residual"):
                unit = "Wb"
            if field.endswith("residual"):
                unit = "relative"
            append("conservation", field, value, unit)
        for interval, item in enumerate(transport.receipts, start=1):
            for field, value in zip(
                dataclasses.fields(item.flux_consumption),
                dataclasses.astuple(item.flux_consumption),
                strict=True,
            ):
                append(
                    "interval_flux_ledger",
                    field.name,
                    value,
                    "V" if "voltage" in field.name else "Wb",
                    iteration=interval,
                    side="transport",
                )
            for field, value in zip(
                dataclasses.fields(item.plasma_current),
                dataclasses.astuple(item.plasma_current),
                strict=True,
            ):
                append(
                    "interval_current_ledger",
                    field.name,
                    value,
                    "A",
                    iteration=interval,
                    side="transport",
                )
    return rows


def main() -> int:
    """Run the strong window and the bounded gentle candidate sequence."""
    configure_dtypes()
    preparation_start = time.perf_counter()
    profile, seed, _vacuum = _fixture_machine()
    extraction_lattice = _extraction_lattice(profile)
    fixture_sources = _fixture_sources(profile)
    baseline_equilibrium = profile.solve(
        seed,
        route="anderson",
        evaluations=EVALUATIONS,
    )
    baseline_geometry, baseline_extraction = _geometry_from_equilibrium(
        baseline_equilibrium,
        profile.source,
        extraction_lattice,
        fixture_sources,
    )
    baseline_extraction.update(iteration=0, sample=0)
    preparation_seconds = time.perf_counter() - preparation_start
    shared = {
        "profile": profile,
        "baseline_equilibrium": baseline_equilibrium,
        "baseline_geometry": baseline_geometry,
        "baseline_extraction": baseline_extraction,
        "extraction_lattice": extraction_lattice,
        "fixture_sources": fixture_sources,
    }

    print("running strong regime", flush=True)
    strong = _run_regime(STRONG, **shared)
    print(f"strong outcome={strong.outcome_type}", flush=True)
    gentle_attempts = []
    for candidate_config in GENTLE_CANDIDATES:
        print(
            f"running gentle candidate {candidate_config.candidate}: "
            f"length={candidate_config.window_length}, "
            f"multiplier={candidate_config.auxiliary_source_multiplier}",
            flush=True,
        )
        result = _run_regime(candidate_config, **shared)
        gentle_attempts.append(result)
        print(
            f"gentle candidate {candidate_config.candidate} "
            f"outcome={result.outcome_type}",
            flush=True,
        )
        if result.converged:
            break

    rows = []
    for result in (strong, *gentle_attempts):
        rows.extend(_result_rows(result, preparation_seconds))
    _write_tsv(rows)
    REPORT_PATH.write_text(
        _two_regime_report(strong, gentle_attempts, preparation_seconds),
        encoding="utf-8",
    )
    print(f"report={REPORT_PATH}")
    print(f"receipts={RECEIPTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
