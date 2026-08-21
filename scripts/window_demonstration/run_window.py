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

from nova.equilibrium.flux_surface_extraction import (
    extract_flux_surface_geometry,
)
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.jax.config import configure_dtypes
from nova.transport.coupled_window import (
    ExchangeSweepResult,
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
from tests.test_equilibrium_forward_solve import (
    EVALUATIONS,
    machine as free_boundary_machine_fixture,
)

WINDOW_LENGTH_SECONDS = 0.01
EQUILIBRIUM_TIMES = np.array([0.0, WINDOW_LENGTH_SECONDS])
TRANSPORT_TIMES = np.array([0.0, WINDOW_LENGTH_SECONDS])
ITERATION_CAP = 10
CONVERGENCE_TOLERANCE = 5.0e-3
DAMPING = 0.5
RADIAL_CELLS = 8
SURFACE_BINS = 14
SOURCE_SAMPLES = 25
ION_DENSITY_PER_ELECTRON = 1.0
MINIMUM_TEMPERATURE_KEV = 0.2

OUTPUT_DIRECTORY = Path(__file__).resolve().parent
REPORT_PATH = OUTPUT_DIRECTORY / "report.md"
RECEIPTS_PATH = OUTPUT_DIRECTORY / "receipts.tsv"
TSV_FIELDS = ("kind", "iteration", "side", "field", "value", "unit")


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


def _geometry_from_equilibrium(
    profile, equilibrium, source: ForwardSource
) -> TransportGeometry:
    """Pass one solved fixture map through the equilibrium extraction service."""
    lattice = profile.lattice
    grid_flux = jnp.asarray(equilibrium.flux[: lattice.node_count]).reshape(
        lattice.shape
    )
    psi_height_radius = grid_flux.T
    inside_limiter = (
        jnp.asarray(profile.operator.inside_material).reshape(lattice.shape).T
    )
    axis_psi = float(equilibrium.topology.axis_flux)
    boundary_psi = float(equilibrium.topology.boundary_flux)
    flux_span = boundary_psi - axis_psi
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
    return TransportGeometry(_block_and_copy(record))


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


def _torax_model() -> TransportModel:
    """Return the ordinary fixed-step TORAX multi-channel configuration."""
    from torax._src.test_utils.default_configs import get_default_config_dict

    config = get_default_config_dict()
    config["numerics"].update(
        {
            "fixed_dt": WINDOW_LENGTH_SECONDS,
            "max_dt": WINDOW_LENGTH_SECONDS,
            "min_dt": 1.0e-8,
            "adaptive_dt": False,
        }
    )
    config["time_step_calculator"] = {"calculator_type": "fixed"}
    return TransportModel(
        TransportRung.TORAX_MULTI_CHANNEL,
        torax_config=config,
    )


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
    kind: str,
    field: str,
    value: Any,
    unit: str,
    *,
    iteration: int | str = "",
    side: str = "",
) -> None:
    rows.append(
        {
            "kind": kind,
            "iteration": iteration,
            "side": side,
            "field": field,
            "value": _format(value),
            "unit": unit,
        }
    )


def _report(
    *,
    outcome: str,
    convergence,
    conservation: Mapping[str, float],
    timings: Sequence[dict[str, Any]],
    preparation_seconds: float,
    transport_receipt: TransportSweepReceipt,
) -> str:
    """Render the human-facing record without interpreting the receipts away."""
    contraction = (
        "none" if convergence is None else _format(convergence.contraction_estimate)
    )
    iterations = (
        "unavailable" if convergence is None else str(convergence.iterations_used)
    )
    maximum = (
        "unavailable" if convergence is None else _format(convergence.maximum_residual)
    )
    lines = [
        "# Real coupled-window demonstration",
        "",
        (
            f"Outcome: **{outcome}**. The window was attempted exactly once; "
            "no retry or tolerance adjustment was made."
        ),
        "",
        "## Run contract",
        "",
        (
            "- Equilibrium: the repository's existing 25 x 25 free-boundary "
            "machine fixture, solved with its declared Anderson budget."
        ),
        (
            "- Geometry: `nova.equilibrium.extract_flux_surface_geometry` at "
            "both equilibrium sample times, with 8 transport cells and 14 "
            "surface bins."
        ),
        (
            "- Transport: TORAX multi-channel, one fixed 10 ms step, with all "
            "four channels evolved."
        ),
        f"- Backend: `{jax.default_backend()}` (explicitly pinned before JAX import).",
        (
            f"- Window length: `{_format(WINDOW_LENGTH_SECONDS)}` s; "
            f"equilibrium grid `{EQUILIBRIUM_TIMES.tolist()}` s; transport grid "
            f"`{TRANSPORT_TIMES.tolist()}` s."
        ),
        (
            f"- Iteration cap: `{ITERATION_CAP}`; convergence and conservation "
            f"tolerance: `{_format(CONVERGENCE_TOLERANCE)}`; damping: "
            f"`{_format(DAMPING)}`."
        ),
        (
            "- One-time fixture solve plus first service assembly: "
            f"`{_format(preparation_seconds)}` s."
        ),
        "",
        "## Window receipt",
        "",
        f"- Iterations used: `{iterations}`",
        f"- Measured contraction estimate: `{contraction}`",
        f"- Maximum exit residual: `{maximum}`",
        f"- Damping applied: `{_format(DAMPING)}`",
        "",
        "Exit residuals are reproduced at full stored precision:",
        "",
        "| exchanged field | relative residual |",
        "|---|---:|",
    ]
    if convergence is None:
        lines.append("| unavailable from raised receipt | unavailable |")
    else:
        lines.extend(
            f"| `{field}` | `{_format(value)}` |"
            for field, value in convergence.exit_residual.items()
        )
    lines.extend(
        [
            "",
            "## Conservation ledgers",
            "",
            (
                "- Flux consumption: boundary "
                f"`{_format(conservation['flux_boundary'])}` Wb; resistive "
                f"`{_format(conservation['flux_resistive'])}` Wb; internal "
                f"`{_format(conservation['flux_internal'])}` Wb."
            ),
            (
                "- Flux closure: absolute "
                f"`{_format(conservation['flux_closure_error'])}` Wb; relative "
                f"`{_format(conservation['flux_closure_residual'])}`."
            ),
            (
                "- Plasma current: requested "
                f"`{_format(conservation['current_requested_initial'])}` -> "
                f"`{_format(conservation['current_requested_final'])}` A; achieved "
                f"`{_format(conservation['current_achieved_initial'])}` -> "
                f"`{_format(conservation['current_achieved_final'])}` A."
            ),
            (
                "- Boundary current continuity: absolute "
                f"`{_format(conservation['current_continuity_error'])}` A; relative "
                f"`{_format(conservation['current_continuity_residual'])}`."
            ),
            "",
            (
                "These closure figures are direct aggregations of the returned "
                "per-interval ledgers. They use the same absolute and "
                "scale-normalised definitions as the window receipt, including "
                "endpoint requested-versus-achieved current continuity."
            ),
            "",
            "## Exchange cost",
            "",
            "| iteration | side | wall time (s) |",
            "|---:|---|---:|",
        ]
    )
    lines.extend(
        f"| {row['iteration']} | {row['side']} | `{_format(row['seconds'])}` |"
        for row in timings
    )
    lines.extend(
        [
            "",
            (
                "The final transport sweep contains "
                f"`{len(transport_receipt.receipts)}` interval receipt(s). The TSV "
                "is the full machine-readable record, including every residual, "
                "timing, configuration value and ledger field."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Run one fixed window attempt and write both evidence artifacts."""
    configure_dtypes()
    preparation_start = time.perf_counter()
    profile, seed, _vacuum = _fixture_machine()
    baseline_equilibrium = profile.solve(
        seed,
        route="anderson",
        evaluations=EVALUATIONS,
    )
    baseline_source = profile.source
    baseline_geometry = _geometry_from_equilibrium(
        profile, baseline_equilibrium, baseline_source
    )
    preparation_seconds = time.perf_counter() - preparation_start

    initial_geometry = Waveform.from_geometries(
        EQUILIBRIUM_TIMES,
        (baseline_geometry, baseline_geometry),
    )
    baseline_coordinates = tuple(
        initial_geometry.sample(float(sample_time)) for sample_time in TRANSPORT_TIMES
    )
    initial_source = _source_waveform(
        TRANSPORT_TIMES,
        (baseline_source, baseline_source),
        baseline_coordinates,
    )
    initial_transport_state = _initial_state(baseline_geometry, baseline_source)
    plasma_current = np.full(
        TRANSPORT_TIMES.shape,
        float(baseline_equilibrium.moments.plasma_current),
    )
    model = _torax_model()
    config = WindowConfig(
        length=WINDOW_LENGTH_SECONDS,
        equilibrium_grid=EQUILIBRIUM_TIMES,
        transport_grid=TRANSPORT_TIMES,
        iteration_cap=ITERATION_CAP,
        tolerance=CONVERGENCE_TOLERANCE,
    )

    timings: list[dict[str, Any]] = []
    iteration_count = {"transport": 0, "equilibrium": 0}
    latest: dict[str, Any] = {}

    def transport_update(geometry_waveform, sample_grid):
        iteration_count["transport"] += 1
        started = time.perf_counter()
        receipt = transport_sweep(
            geometry_waveform,
            initial_transport_state,
            sample_grid,
            plasma_current,
            model,
        )
        coordinates = [geometry_waveform.sample(float(sample_grid[0]))]
        sources = [baseline_source]
        for interval, item in enumerate(receipt.receipts):
            geometry_time = float(receipt.geometry_time[interval])
            geometry = geometry_waveform.sample(geometry_time).geometry()
            sources.append(
                forward_source_from_receipt(
                    item,
                    geometry,
                    ion_density_per_electron=ION_DENSITY_PER_ELECTRON,
                )
            )
            coordinates.append(geometry_waveform.sample(geometry_time))
        waveform = _source_waveform(sample_grid, sources, coordinates)
        timings.append(
            {
                "iteration": iteration_count["transport"],
                "side": "transport",
                "seconds": time.perf_counter() - started,
            }
        )
        latest["transport"] = receipt
        return ExchangeSweepResult(waveform=waveform, receipt=receipt)

    def equilibrium_update(source_waveform, sample_grid):
        iteration_count["equilibrium"] += 1
        started = time.perf_counter()
        receipt = equilibrium_sweep(
            profile,
            baseline_equilibrium.flux,
            source_waveform,
            sample_grid,
            _source_from_sample,
            route="anderson",
            solve_options={"evaluations": EVALUATIONS},
        )
        geometries = []
        for sample, equilibrium in zip(
            receipt.source_samples, receipt.equilibria, strict=True
        ):
            geometries.append(
                _geometry_from_equilibrium(
                    profile,
                    equilibrium,
                    _source_from_sample(sample),
                )
            )
        waveform = Waveform.from_geometries(sample_grid, geometries)
        timings.append(
            {
                "iteration": iteration_count["equilibrium"],
                "side": "equilibrium_plus_fsa",
                "seconds": time.perf_counter() - started,
            }
        )
        latest["equilibrium"] = receipt
        return ExchangeSweepResult(waveform=waveform, receipt=receipt)

    convergence = None
    conservation_receipt = None
    outcome = "unknown"
    try:
        receipt = solve_window(
            initial_geometry,
            initial_source,
            config,
            equilibrium_update,
            transport_update,
            damping=DAMPING,
        )
        convergence = receipt.convergence
        conservation_receipt = receipt.conservation
        transport_receipt = receipt.transport_receipt
        outcome = "converged"
    except WindowConvergenceError as error:
        convergence = error.convergence
        transport_receipt = error.transport_receipt
        outcome = "iteration cap exhausted without convergence"
    except WindowConservationError as error:
        conservation_receipt = error.conservation
        transport_receipt = latest["transport"]
        outcome = "exchange converged but conservation tolerance was not met"

    conservation = _transport_conservation(transport_receipt)
    if conservation_receipt is not None:
        expected = {
            "flux_closure_error": conservation_receipt.flux_closure_error,
            "flux_closure_residual": conservation_receipt.flux_closure_residual,
            "current_continuity_error": conservation_receipt.current_continuity_error,
            "current_continuity_residual": (
                conservation_receipt.current_continuity_residual
            ),
        }
        for field, value in expected.items():
            np.testing.assert_allclose(conservation[field], value, rtol=0.0, atol=0.0)

    rows: list[dict[str, Any]] = []
    for field, value, unit in (
        ("window_length", WINDOW_LENGTH_SECONDS, "s"),
        ("iteration_cap", ITERATION_CAP, "count"),
        ("tolerance", CONVERGENCE_TOLERANCE, "relative"),
        ("damping", DAMPING, "fraction"),
        ("radial_cells", RADIAL_CELLS, "count"),
        ("surface_bins", SURFACE_BINS, "count"),
        ("preparation_wall_time", preparation_seconds, "s"),
        ("outcome", outcome, "text"),
    ):
        _append_row(rows, "configuration", field, value, unit)
    if convergence is not None:
        _append_row(
            rows,
            "convergence",
            "iterations_used",
            convergence.iterations_used,
            "count",
        )
        _append_row(
            rows,
            "convergence",
            "contraction_estimate",
            convergence.contraction_estimate,
            "ratio",
        )
        _append_row(
            rows,
            "convergence",
            "damping_applied",
            convergence.damping_applied,
            "fraction",
        )
        for field, value in convergence.exit_residual.items():
            _append_row(rows, "exit_residual", field, value, "relative")
        for iteration, residuals in enumerate(convergence.residual_trace, start=1):
            for field, value in residuals.items():
                _append_row(
                    rows,
                    "residual_trace",
                    field,
                    value,
                    "relative",
                    iteration=iteration,
                )
    for row in timings:
        _append_row(
            rows,
            "exchange_timing",
            "wall_time",
            row["seconds"],
            "s",
            iteration=row["iteration"],
            side=row["side"],
        )
    for field, value in conservation.items():
        unit = "Wb" if "flux_" in field and "residual" not in field else "A"
        if field.endswith("residual"):
            unit = "relative"
        _append_row(rows, "conservation", field, value, unit)
    for interval, item in enumerate(transport_receipt.receipts, start=1):
        for field, value in zip(
            dataclasses.fields(item.flux_consumption),
            dataclasses.astuple(item.flux_consumption),
            strict=True,
        ):
            _append_row(
                rows,
                "interval_flux_ledger",
                field.name,
                value,
                "Wb" if "voltage" not in field.name else "V",
                iteration=interval,
                side="transport",
            )
        for field, value in zip(
            dataclasses.fields(item.plasma_current),
            dataclasses.astuple(item.plasma_current),
            strict=True,
        ):
            _append_row(
                rows,
                "interval_current_ledger",
                field.name,
                value,
                "A",
                iteration=interval,
                side="transport",
            )

    _write_tsv(rows)
    REPORT_PATH.write_text(
        _report(
            outcome=outcome,
            convergence=convergence,
            conservation=conservation,
            timings=timings,
            preparation_seconds=preparation_seconds,
            transport_receipt=transport_receipt,
        ),
        encoding="utf-8",
    )
    print(f"outcome={outcome}")
    print(f"report={REPORT_PATH}")
    print(f"receipts={RECEIPTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
