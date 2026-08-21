"""Measure why a transported source removes the fixture's confined core."""

from __future__ import annotations

import dataclasses
import json
import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
from scipy.constants import electron_volt

from nova.equilibrium.topology import topology_solve_receipt
from nova.jax.config import configure_dtypes
from nova.transport import coupled_window
from nova.transport.coupled_window import Waveform, transport_sweep
from nova.transport.evolved_state import forward_source_from_receipt
from nova.transport.forward import TransportState
from scripts.window_demonstration import run_window as demonstration


MATCHED_FLUX = np.linspace(0.0, 1.0, 17)
KEV_JOULES = 1.0e3 * electron_volt
SOLVE_TOLERANCE = 1.0e-6


def _pressure(state: TransportState) -> np.ndarray:
    """Recover thermal pressure in Pa from one transport state."""
    return (
        np.asarray(state.electron_density)
        * KEV_JOULES
        * (np.asarray(state.ion_temperature) + np.asarray(state.electron_temperature))
    )


def _pressure_matched_state(
    state: TransportState, fixture_pressure: np.ndarray
) -> tuple[TransportState, dict[str, float]]:
    """Scale density and temperatures equally to match the fixture peak pressure."""
    initial_pressure = _pressure(state)
    fixture_scale = float(np.max(np.abs(fixture_pressure)))
    initial_scale = float(np.max(np.abs(initial_pressure)))
    pressure_factor = fixture_scale / initial_scale
    channel_factor = float(np.sqrt(pressure_factor))
    matched = dataclasses.replace(
        state,
        ion_temperature=np.asarray(state.ion_temperature) * channel_factor,
        electron_temperature=np.asarray(state.electron_temperature) * channel_factor,
        electron_density=np.asarray(state.electron_density) * channel_factor,
    )
    matched_pressure = _pressure(matched)
    return matched, {
        "fixture_peak_pressure_pa": fixture_scale,
        "initial_peak_pressure_pa": initial_scale,
        "pressure_factor": pressure_factor,
        "temperature_factor": channel_factor,
        "density_factor": channel_factor,
        "matched_peak_pressure_pa": float(np.max(np.abs(matched_pressure))),
        "initial_profile_relative_max_error": float(
            np.max(np.abs(initial_pressure - fixture_pressure)) / fixture_scale
        ),
        "matched_profile_relative_max_error": float(
            np.max(np.abs(matched_pressure - fixture_pressure)) / fixture_scale
        ),
    }


def _advance(geometry_waveform, baseline_source, initial_state, plasma_current, model):
    """Run exactly one transport exchange on the fixed demonstration interval."""
    sweep = transport_sweep(
        geometry_waveform,
        initial_state,
        demonstration.TRANSPORT_TIMES,
        plasma_current,
        model,
    )
    receipt = sweep.receipts[-1]
    geometry_time = float(sweep.geometry_time[-1])
    geometry = geometry_waveform.sample(geometry_time).geometry()
    source = forward_source_from_receipt(
        receipt,
        geometry,
        ion_density_per_electron=demonstration.ION_DENSITY_PER_ELECTRON,
    )
    sampled = _sampled_exchange_source(geometry_waveform, baseline_source, source)
    return receipt, source, sampled


def _sampled_exchange_source(geometry_waveform, baseline_source, evolved_source):
    """Cross the same sampled waveform seam used by the coupled callback."""
    coordinates = (
        geometry_waveform.sample(float(demonstration.TRANSPORT_TIMES[0])),
        geometry_waveform.sample(float(np.mean(demonstration.TRANSPORT_TIMES))),
    )
    waveform = demonstration._source_waveform(
        demonstration.TRANSPORT_TIMES,
        (baseline_source, evolved_source),
        coordinates,
    )
    return demonstration._source_from_sample(
        waveform.sample(float(demonstration.TRANSPORT_TIMES[-1]))
    )


def _drive_measurements(baseline, exchanged) -> dict[str, object]:
    """Compare both source families on one matched normalised-flux grid."""
    result: dict[str, object] = {"normalised_flux": MATCHED_FLUX.tolist()}
    for family in ("p_prime", "ff_prime"):
        baseline_value = np.asarray(getattr(baseline.core, family)(MATCHED_FLUX))
        exchanged_value = np.asarray(getattr(exchanged.core, family)(MATCHED_FLUX))
        baseline_magnitude = np.abs(baseline_value)
        ratio = np.divide(
            np.abs(exchanged_value),
            baseline_magnitude,
            out=np.full_like(exchanged_value, np.inf, dtype=np.float64),
            where=baseline_magnitude > 0.0,
        )
        result[family] = {
            "baseline": baseline_value.tolist(),
            "exchanged": exchanged_value.tolist(),
            "pointwise_absolute_ratio": ratio.tolist(),
            "rms_magnitude_ratio": float(
                np.linalg.norm(exchanged_value) / np.linalg.norm(baseline_value)
            ),
            "peak_magnitude_ratio": float(
                np.max(np.abs(exchanged_value)) / np.max(baseline_magnitude)
            ),
        }
    return result


def _exact_core_count(equilibrium, lattice, sources) -> int:
    """Read the connected core from the solve's exact Green evaluation."""
    flux = np.asarray(
        demonstration.evaluate_forward_equilibrium(
            equilibrium, lattice, sources
        ).block_until_ready()
    )
    mesh_radius, mesh_height = np.meshgrid(
        lattice.radius, lattice.height, indexing="xy"
    )
    _wall, wall_flux = demonstration.forward_fixture._wall_loop()
    inside_limiter = jnp.asarray(
        demonstration.forward_fixture._solovev(mesh_radius, mesh_height) >= wall_flux
    )
    axis_flux = float(equilibrium.topology.axis_flux)
    boundary_flux = float(equilibrium.topology.boundary_flux)
    return int(
        np.asarray(
            demonstration._axis_connected_core(
                (flux - axis_flux) / (boundary_flux - axis_flux),
                inside_limiter,
            )
        ).sum()
    )


def _solve(
    profile, seed_equilibrium, source, extraction_lattice, fixture_sources
) -> tuple[object, dict[str, object]]:
    """Solve one exchanged source and publish its topology qualification."""
    sampled_operator = dataclasses.replace(profile.operator, source=source)
    sampled_profile = dataclasses.replace(profile, operator=sampled_operator)
    entry_topology = sampled_operator.read(seed_equilibrium.flux)[1]
    started = time.perf_counter()
    equilibrium = sampled_profile.solve(
        seed_equilibrium.flux,
        route="anderson",
        evaluations=demonstration.EVALUATIONS,
    )
    elapsed = time.perf_counter() - started
    residual = float(equilibrium.fixed_point.residual)
    finite = bool(equilibrium.finite.passed)
    succeeded = finite and residual < SOLVE_TOLERANCE
    receipt = topology_solve_receipt(
        (entry_topology, equilibrium.topology), solver_succeeded=succeeded
    )
    measurement = {
        "fixed_point_residual": residual,
        "finite": finite,
        "core_cells": int(np.asarray(equilibrium.domains.core).sum()),
        "exact_core_cells": _exact_core_count(
            equilibrium, extraction_lattice, fixture_sources
        ),
        "plasma_current_a": float(equilibrium.moments.plasma_current),
        "solve_seconds": elapsed,
        "topology_receipt": receipt.as_dict(),
    }
    return equilibrium, measurement


def _trace_to_collapse(
    profile,
    baseline_equilibrium,
    baseline_geometry,
    extraction_lattice,
    fixture_sources,
    initial_state,
    plasma_current,
    model,
):
    """Follow ordinary geometry exchanges until the exact core disappears."""
    geometry_waveform = Waveform.from_geometries(
        demonstration.EQUILIBRIUM_TIMES,
        (baseline_geometry, baseline_geometry),
    )
    trace = []
    for iteration in range(1, demonstration.ITERATION_CAP + 1):
        started = time.perf_counter()
        transport, mapped_source, sampled_source = _advance(
            geometry_waveform,
            profile.source,
            initial_state,
            plasma_current,
            model,
        )
        equilibrium, solve = _solve(
            profile,
            baseline_equilibrium,
            sampled_source,
            extraction_lattice,
            fixture_sources,
        )
        trace.append(
            {
                "iteration": iteration,
                "exchange_seconds": time.perf_counter() - started,
                "transport_engine_status": transport.diagnostics.engine_status,
                "transport_steps": transport.diagnostics.steps,
                "boundary_pressure_pa": float(sampled_source.boundary_pressure),
                "boundary_field_function_tm": float(
                    sampled_source.boundary_field_function
                ),
                "mapped_drives": _drive_measurements(profile.source, mapped_source),
                "sampled_drives": _drive_measurements(profile.source, sampled_source),
                "solve": solve,
            }
        )
        if solve["exact_core_cells"] == 0:
            return geometry_waveform, trace
        evolved_geometry, _extraction = demonstration._geometry_from_equilibrium(
            equilibrium,
            sampled_source,
            extraction_lattice,
            fixture_sources,
        )
        candidate = Waveform.from_geometries(
            demonstration.EQUILIBRIUM_TIMES,
            (baseline_geometry, evolved_geometry),
        )
        geometry_waveform = coupled_window._blend_waveform(
            geometry_waveform, candidate, demonstration.DAMPING
        )
    raise RuntimeError("the declared iteration budget did not reproduce the collapse")


def main() -> int:
    """Run the baseline, ordinary exchange, and one pressure-matched exchange."""
    configure_dtypes()
    overall_start = time.perf_counter()
    profile, seed, _vacuum = demonstration._fixture_machine()
    baseline_start = time.perf_counter()
    baseline_equilibrium = profile.solve(
        seed,
        route="anderson",
        evaluations=demonstration.EVALUATIONS,
    )
    baseline_seconds = time.perf_counter() - baseline_start
    extraction_lattice = demonstration._extraction_lattice(profile)
    fixture_sources = demonstration._fixture_sources(profile)
    geometry, extraction = demonstration._geometry_from_equilibrium(
        baseline_equilibrium,
        profile.source,
        extraction_lattice,
        fixture_sources,
    )

    initial_state = demonstration._initial_state(geometry, profile.source)
    fixture_pressure = np.asarray(
        profile.source.core.pressure(
            jnp.asarray(geometry.record["r0"]),
            jnp.asarray(geometry.record["psi_n_face"]),
            profile.source.boundary_pressure,
            jnp.asarray(geometry.record["boundary_psi"] - geometry.record["axis_psi"]),
        )
    )
    pressure_matched_state, pressure_match = _pressure_matched_state(
        initial_state, fixture_pressure
    )
    plasma_current = np.full(
        demonstration.TRANSPORT_TIMES.shape,
        float(baseline_equilibrium.moments.plasma_current),
    )
    model = demonstration._torax_model()

    collapse_geometry_waveform, ordinary_trace = _trace_to_collapse(
        profile,
        baseline_equilibrium,
        geometry,
        extraction_lattice,
        fixture_sources,
        initial_state,
        plasma_current,
        model,
    )

    matched_start = time.perf_counter()
    matched_transport, matched_source, matched_sampled_source = _advance(
        collapse_geometry_waveform,
        profile.source,
        pressure_matched_state,
        plasma_current,
        model,
    )
    _matched_equilibrium, matched_solve = _solve(
        profile,
        baseline_equilibrium,
        matched_sampled_source,
        extraction_lattice,
        fixture_sources,
    )
    matched_seconds = time.perf_counter() - matched_start

    summary = {
        "baseline": {
            "fixed_point_residual": float(baseline_equilibrium.fixed_point.residual),
            "finite": bool(baseline_equilibrium.finite.passed),
            "core_cells_solve_lattice": int(
                np.asarray(baseline_equilibrium.domains.core).sum()
            ),
            "core_cells_extraction_lattice": extraction["core_count"],
            "plasma_current_a": float(baseline_equilibrium.moments.plasma_current),
            "solve_seconds": baseline_seconds,
            "exact_evaluation_seconds": extraction["exact_evaluation_seconds"],
        },
        "pressure_match": pressure_match,
        "ordinary_trace": ordinary_trace,
        "pressure_matched": {
            "exchange_seconds": matched_seconds,
            "transport_engine_status": matched_transport.diagnostics.engine_status,
            "transport_steps": matched_transport.diagnostics.steps,
            "boundary_pressure_pa": float(matched_sampled_source.boundary_pressure),
            "boundary_field_function_tm": float(
                matched_sampled_source.boundary_field_function
            ),
            "mapped_drives": _drive_measurements(profile.source, matched_source),
            "sampled_drives": _drive_measurements(
                profile.source, matched_sampled_source
            ),
            "solve": matched_solve,
        },
        "total_seconds": time.perf_counter() - overall_start,
    }
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
