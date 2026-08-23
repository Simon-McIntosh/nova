"""Trace passive-current amplification on one fixed dense carrier.

The diagnostic keeps geometry, profiles and the initial flux byte-identical
between two arms.  It first follows paired applications of the coupled map,
then applies the exact map tangent repeatedly to the passive external field.
Iteration one therefore tests closure arithmetic before feedback; later terms
test whether the measured expanding map supplies the terminal gain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np

from benchmarks.measurement_provenance import measurement_stamp
from nova.equilibrium.diagnostics import (
    DECAY_INDEX_WINDOW,
    vertical_conditioning_receipt,
)
from nova.jax.config import configure_dtypes
from tests import test_equilibrium_forward_solve as forward_solve_suite
from tests import test_equilibrium_forward_reference as reference


DEFAULT_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/passive-closure-trace.json"
)
DEFAULT_FIGURE = Path(
    "docs/figures/forward-operator-refinement/passive-closure-trace.png"
)
CONTROL_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/passive-closure-stability-control.json"
)
CONTROL_FIGURE = Path(
    "docs/figures/forward-operator-refinement/passive-closure-stability-control.png"
)
DENSE_CELL_REQUEST = -2100
PAIRED_MAP_ITERATIONS = 8
LINEAR_RESPONSE_ITERATIONS = 12
MEASURED_UNPINNED_SPECTRAL_RADIUS = 1.2577
MEASURED_PINNED_SPECTRAL_RADIUS = 1.1456
REGISTERED_CEILING_POINTS = 0.15
DOCUMENTED_DIRECT_RESPONSE_POINTS = 0.098
IDENTITY_RELATIVE_TOLERANCE = 2.0e-12
SPECTRAL_RATIO_RELATIVE_TOLERANCE = 0.25
ELONGATED_DIRECT_PEAK_POINTS = 0.09723915202112363
ELONGATED_REPRODUCTION_MOVE_POINTS = 0.6377635705066984
ELONGATED_FIELD_GAIN = 6.5998576353591725
ELONGATED_BOUNDARY_ELONGATION = 1.8176001064943421
ELONGATED_WALL_ELONGATION = 2.087719298245614
ANALYTIC_VACUUM_SHAPING = 0.3
PASSIVE_SHELL_EXPANSION = 1.08
CONTROL_NEWTON_STEPS = 10
CONTROL_GMRES_ITERATIONS = 30
CONTROL_NEWTON_WARMUP = 8
CONTROL_MINIMUM_CORE_FRACTION = 0.1
CONTROL_MAXIMUM_AXIS_DISPLACEMENT_FRACTION = 0.25
RUNTIME_MITIGATION_DEFAULTS = {
    "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


def _strict_json(value):
    """Return ordinary finite JSON values for the banked receipt."""
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("the trace receipt contains a non-finite scalar")
    return value


def _diagnostic_json(value):
    """Return printable values even when a rejected solve is non-finite."""
    if isinstance(value, dict):
        return {key: _diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_diagnostic_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _diagnostic_json(value.tolist())
    if isinstance(value, np.generic):
        return _diagnostic_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _elongation(points: np.ndarray) -> float:
    """Return total vertical span divided by total radial span."""
    coordinate = np.asarray(points, dtype=float)
    return float(np.ptp(coordinate[:, 1]) / np.ptp(coordinate[:, 0]))


@dataclass(frozen=True)
class AnalyticControl:
    """Free-boundary profiles and immutable reference data for the control."""

    with_passive: forward_solve_suite.ForwardProfile
    without_passive: forward_solve_suite.ForwardProfile
    seed_flux: jax.Array
    reference_grid_flux: np.ndarray
    material_boundary: np.ndarray
    reference_axis: np.ndarray
    reference_flux_span_wb: float
    construction_receipt: dict[str, object]


def _analytic_solovev(radius, height):
    """Return a near-circular Solov'ev seed with vacuum shaping."""
    alpha, offset, beta = forward_solve_suite._terms()
    axis_radius = forward_solve_suite.AXIS_RADIUS
    radial = np.asarray(radius)
    vertical = np.asarray(height)
    homogeneous = (
        radial**4 - 4.0 * radial**2 * vertical**2 - 2.0 * axis_radius**2 * radial**2
    )
    return (
        alpha * radial**4
        + offset * radial**2
        + beta * vertical**2
        + ANALYTIC_VACUUM_SHAPING * homogeneous
    )


def _analytic_material_loop(points: int) -> tuple[np.ndarray, float]:
    """Return a closed material flux surface of the analytic seed."""
    alpha, offset, beta = forward_solve_suite._terms()
    axis_radius = forward_solve_suite.AXIS_RADIUS
    axis_flux = float(_analytic_solovev(axis_radius, 0.0))
    wall_flux = axis_flux - forward_solve_suite.SEED_SPAN
    radial_quartic = alpha + ANALYTIC_VACUUM_SHAPING
    radial_square = offset - 2.0 * ANALYTIC_VACUUM_SHAPING * axis_radius**2
    inner, outer = np.sqrt(
        np.sort(np.roots([radial_quartic, radial_square, -wall_flux]))
    )
    centre = 0.5 * (inner + outer)
    half_width = 0.5 * (outer - inner)
    angle = 2.0 * np.pi * np.arange(points) / points
    radius = centre + half_width * np.cos(angle)
    vertical_coefficient = beta - 4.0 * ANALYTIC_VACUUM_SHAPING * radius**2
    radial_flux = radial_quartic * radius**4 + radial_square * radius**2
    height_squared = np.clip(
        (wall_flux - radial_flux) / vertical_coefficient,
        0.0,
        None,
    )
    height = np.sign(np.sin(angle)) * np.sqrt(height_squared)
    return np.c_[radius, height], wall_flux


def _vertical_translation_mode(radius, height) -> np.ndarray:
    """Return minus the vertical derivative of the analytic seed."""
    _alpha, _offset, beta = forward_solve_suite._terms()
    radial = np.asarray(radius)
    vertical = np.asarray(height)
    return -(
        2.0 * beta * vertical - 8.0 * ANALYTIC_VACUUM_SHAPING * radial**2 * vertical
    )


def _build_analytic_control() -> AnalyticControl:
    """Bootstrap the stable control from the forward-solve suite machinery."""
    lattice = forward_solve_suite.FluxLattice(
        np.linspace(0.6, 1.42, 25),
        np.linspace(-0.42, 0.42, 25),
    )
    coordinate = lattice.coordinate
    wall, wall_flux = _analytic_material_loop(61)
    seed_grid = _analytic_solovev(coordinate[:, 0], coordinate[:, 1])
    seed_wall = _analytic_solovev(wall[:, 0], wall[:, 1])
    seed = jnp.asarray(np.r_[seed_grid, seed_wall])
    inside = seed_grid >= wall_flux

    conductor_count = forward_solve_suite.CONDUCTORS
    angle = 2.0 * np.pi * np.arange(conductor_count) / conductor_count
    axis_radius = forward_solve_suite.AXIS_RADIUS
    active_coordinate = np.c_[
        axis_radius + 0.62 * np.cos(angle),
        0.62 * np.sin(angle),
    ]
    passive_boundary, _passive_flux = _analytic_material_loop(64)
    axis = np.asarray([axis_radius, 0.0])
    passive_coordinate = axis + PASSIVE_SHELL_EXPANSION * (passive_boundary[::4] - axis)

    plasma_to_grid = forward_solve_suite._green_block(coordinate, coordinate)
    plasma_to_wall = forward_solve_suite._green_block(wall, coordinate)
    active_to_grid = forward_solve_suite._green_block(coordinate, active_coordinate)
    active_to_wall = forward_solve_suite._green_block(wall, active_coordinate)
    passive_to_grid = forward_solve_suite._green_block(coordinate, passive_coordinate)
    passive_to_wall = forward_solve_suite._green_block(wall, passive_coordinate)

    def profile(current, source_to_grid, source_to_wall, source):
        """Build one profile on the shared analytic carrier."""
        return forward_solve_suite.ForwardProfile.from_lattice(
            lattice,
            source,
            external_current=current,
            source_to_grid=source_to_grid,
            plasma_to_grid=plasma_to_grid,
            source_to_wall=source_to_wall,
            plasma_to_wall=plasma_to_wall,
            wall_coordinate=wall,
            polarity=1,
            inside_material=inside,
            newton_steps=CONTROL_NEWTON_STEPS,
        )

    flat = profile(
        np.zeros(conductor_count),
        active_to_grid,
        active_to_wall,
        forward_solve_suite.ForwardSource(
            core=forward_solve_suite.DomainProfile(
                p_prime=forward_solve_suite._flat_profile(forward_solve_suite.P_PRIME),
                ff_prime=forward_solve_suite._flat_profile(
                    forward_solve_suite.FF_PRIME
                ),
            ),
            boundary_field_function=forward_solve_suite.BOUNDARY_FIELD_FUNCTION,
        ),
    )
    cell_current = np.asarray(flat.operator.cell_current(seed))
    active_target = np.r_[
        seed_grid - plasma_to_grid @ cell_current,
        seed_wall - plasma_to_wall @ cell_current,
    ]
    weight = np.r_[inside.astype(float), np.ones(len(wall))]
    active_matrix = np.r_[active_to_grid, active_to_wall]
    active_current = np.linalg.lstsq(
        active_matrix * weight[:, None],
        active_target * weight,
        rcond=None,
    )[0]
    active_fit = active_matrix @ active_current

    translation_target = np.r_[
        _vertical_translation_mode(coordinate[:, 0], coordinate[:, 1]),
        _vertical_translation_mode(wall[:, 0], wall[:, 1]),
    ]
    passive_matrix = np.r_[passive_to_grid, passive_to_wall]
    passive_current = np.linalg.lstsq(
        passive_matrix * weight[:, None],
        translation_target * weight,
        rcond=None,
    )[0]
    raw_passive_fit = passive_matrix @ passive_current
    passive_scale = (
        ELONGATED_DIRECT_PEAK_POINTS
        / 100.0
        * forward_solve_suite.SEED_SPAN
        / np.max(np.abs(passive_to_grid @ passive_current))
    )
    passive_current *= passive_scale

    source_to_grid = np.c_[active_to_grid, passive_to_grid]
    source_to_wall = np.c_[active_to_wall, passive_to_wall]
    with_current = np.r_[active_current, passive_current]
    without_current = np.r_[active_current, np.zeros_like(passive_current)]
    source = forward_solve_suite.ForwardSource(
        core=forward_solve_suite.DomainProfile(
            p_prime=forward_solve_suite._edge_vanishing_profile(
                2.0 * forward_solve_suite.DRIVE * forward_solve_suite.P_PRIME
            ),
            ff_prime=forward_solve_suite._edge_vanishing_profile(
                2.0 * forward_solve_suite.DRIVE * forward_solve_suite.FF_PRIME
            ),
        ),
        boundary_field_function=forward_solve_suite.BOUNDARY_FIELD_FUNCTION,
    )
    base = profile(with_current, source_to_grid, source_to_wall, source)
    without = replace(
        base,
        operator=replace(base.operator, external_current=jnp.asarray(without_current)),
    )
    passive_image = passive_to_grid @ passive_current
    control_direct_peak = (
        100.0 * float(np.max(np.abs(passive_image))) / forward_solve_suite.SEED_SPAN
    )
    return AnalyticControl(
        with_passive=base,
        without_passive=without,
        seed_flux=seed,
        reference_grid_flux=np.asarray(seed_grid),
        material_boundary=wall,
        reference_axis=axis,
        reference_flux_span_wb=forward_solve_suite.SEED_SPAN,
        construction_receipt={
            "construction": (
                "The forward-solve suite's real free-boundary lattice, material "
                "wall, Green couplings, tapered absolute source and fitted ring "
                "are retained. A Grad-Shafranov-homogeneous vacuum term makes "
                "the analytic seed near-circular and decay-index stable. A "
                "separate wall-following material shell is fitted to the seed's "
                "vertical-translation mode and scaled to the elongated passive "
                "field magnitude."
            ),
            "prior_art": "tests/test_equilibrium_forward_solve.py",
            "analytic_seed": ("suite Solovev seed + c*(R^4 - 4*R^2*Z^2 - 2*R0^2*R^2)"),
            "vacuum_shaping_coefficient": ANALYTIC_VACUUM_SHAPING,
            "reference_axis_m": axis,
            "reference_flux_span_wb": forward_solve_suite.SEED_SPAN,
            "elongated_boundary_elongation": ELONGATED_BOUNDARY_ELONGATION,
            "control_boundary_elongation": _elongation(wall),
            "elongated_wall_elongation": ELONGATED_WALL_ELONGATION,
            "control_wall_elongation": _elongation(wall),
            "lattice_shape": list(lattice.shape),
            "lattice_cells": int(lattice.node_count),
            "analytic_core_candidate_cells": int(inside.sum()),
            "active_fit": {
                "conductor_count": int(active_current.size),
                "ring_radius_m": 0.62,
                "current_l1_a": float(np.linalg.norm(active_current, ord=1)),
                "current_l2_a": float(np.linalg.norm(active_current)),
                "weighted_relative_l2_error": float(
                    np.linalg.norm((active_fit - active_target) * weight)
                    / np.linalg.norm(active_target * weight)
                ),
            },
            "passive_material_coupling": {
                "conductor_count": int(passive_current.size),
                "boundary_expansion_factor": PASSIVE_SHELL_EXPANSION,
                "green_section_width_m": 0.05,
                "green_section_height_m": 0.05,
                "source_to_grid_shape": list(passive_to_grid.shape),
                "source_to_wall_shape": list(passive_to_wall.shape),
                "source_to_grid_l2_wb_per_a": float(np.linalg.norm(passive_to_grid)),
                "source_to_wall_l2_wb_per_a": float(np.linalg.norm(passive_to_wall)),
                "translation_mode_weighted_relative_l2_error": float(
                    np.linalg.norm((raw_passive_fit - translation_target) * weight)
                    / np.linalg.norm(translation_target * weight)
                ),
                "current_l1_a": float(np.linalg.norm(passive_current, ord=1)),
                "current_l2_a": float(np.linalg.norm(passive_current)),
                "maximum_absolute_current_a": float(np.max(np.abs(passive_current))),
                "control_direct_external_peak_points": control_direct_peak,
                "elongated_direct_external_peak_points": (ELONGATED_DIRECT_PEAK_POINTS),
                "control_over_elongated_direct_peak": (
                    control_direct_peak / ELONGATED_DIRECT_PEAK_POINTS
                ),
            },
        },
    )


def _grid(values, cell_count: int) -> np.ndarray:
    """Return the plasma-cell prefix of one full operator vector."""
    return np.asarray(values, dtype=float)[:cell_count]


def _peak_points(values, cell_count: int, span: float) -> float:
    """Return a grid-vector peak in percentage points of reference span."""
    return 100.0 * float(np.max(np.abs(_grid(values, cell_count)))) / span


def _score(values, reference_flux, core, cell_count: int, span: float) -> float:
    """Return the reference reproduction sup-norm on one explicit support."""
    error = np.abs(_grid(values, cell_count) - reference_flux)
    return 100.0 * float(np.max(error[np.asarray(core, dtype=bool)])) / span


def _topology(operator, state) -> dict[str, object]:
    """Return the topology and centroid support read from one state."""
    masks, topology = operator.read(state)
    core = np.asarray(masks.core, dtype=bool)
    return {
        "core": core,
        "core_cells": int(core.sum()),
        "axis_m": np.asarray(topology.axis, dtype=float),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_flux_wb": float(topology.boundary_flux),
        "flux_span_wb": float(topology.flux_span),
        "diverted": bool(topology.diverted),
    }


def _support_comparison(left: dict[str, object], right: dict[str, object]):
    """Compare two domain reads without serialising their full masks."""
    left_core = left["core"]
    right_core = right["core"]
    return {
        "left_core_cells": int(left_core.sum()),
        "right_core_cells": int(right_core.sum()),
        "core_symmetric_difference_cells": int(
            np.logical_xor(left_core, right_core).sum()
        ),
        "both_diverted": bool(left["diverted"] and right["diverted"]),
        "axis_separation_mm": 1.0e3
        * float(np.linalg.norm(left["axis_m"] - right["axis_m"])),
        "flux_span_difference_wb": float(left["flux_span_wb"] - right["flux_span_wb"]),
    }


def _solve(case, machine, label: str, *, include_deviations: bool = True):
    """Run the production root and retain its convergence receipt."""
    started = perf_counter()
    solved = reference.solve(case, machine)
    residual = float(solved.fixed_point.residual)
    elapsed_seconds = perf_counter() - started
    trace = np.asarray(solved.fixed_point.trace, dtype=float)
    measured = trace[np.isfinite(trace)]
    core = np.asarray(solved.masks.core, dtype=bool)
    axis = np.asarray(solved.topology.axis, dtype=float)
    flux = np.asarray(solved.flux, dtype=float)
    flux_span = float(solved.topology.flux_span)
    if core.any():
        branch = (
            "diverted_plasma" if bool(solved.topology.diverted) else "limited_plasma"
        )
    else:
        branch = "vacuum_or_empty_core"
    receipt = {
        "arm": label,
        "elapsed_seconds": elapsed_seconds,
        "terminal_relative_fixed_point_residual": residual,
        "trace_slots": int(trace.size),
        "measured_residuals": int(measured.size),
        "first_measured_residual": float(measured[0]),
        "minimum_measured_residual": float(measured.min()),
        "direct_branch_receipt": {
            "state_finite": bool(np.isfinite(flux).all()),
            "core_cells": int(core.sum()),
            "core_fraction_of_carrier": float(core.mean()),
            "topology_branch": branch,
            "diverted": bool(solved.topology.diverted),
            "solved_axis_m": axis,
            "axis_finite": bool(np.isfinite(axis).all()),
            "flux_span_wb": flux_span,
            "flux_span_valid": bool(np.isfinite(flux_span) and abs(flux_span) > 0.0),
        },
    }
    if include_deviations:
        receipt["deviations"] = solved.deviations()
    return solved, receipt


def _equilibrium_core(solved) -> np.ndarray:
    """Return the labelled core from either forward-equilibrium interface."""
    domains = solved.masks if hasattr(solved, "masks") else solved.domains
    return np.asarray(domains.core, dtype=bool)


def _vertical_conditioning(profile, axis: np.ndarray) -> dict[str, object]:
    """Measure the vacuum decay index at one solved magnetic axis."""
    source_flux = (
        np.asarray(profile.operator.grid.source_target)
        @ np.asarray(profile.operator.external_current)
    ).reshape(profile.lattice.shape)
    radius = np.asarray(profile.lattice.radius, dtype=float)
    height = np.asarray(profile.lattice.height, dtype=float)
    vertical_index = int(np.argmin(np.abs(height - axis[1])))
    vertical_field = np.gradient(source_flux[:, vertical_index], radius) / (
        2.0 * np.pi * radius
    )
    receipt = asdict(
        vertical_conditioning_receipt(radius, vertical_field, float(axis[0]))
    )
    return {
        "definition": "n = -(R/Bz)*(dBz/dR)",
        "open_stability_window": list(DECAY_INDEX_WINDOW),
        "radial_sample_m": radius,
        "vertical_field_sample_t": vertical_field,
        **receipt,
    }


def _solve_analytic_control(profile, seed, label: str):
    """Solve and qualify one analytic-control conductor state."""
    started = perf_counter()
    solved = profile.solve(
        seed,
        route="newton_krylov",
        newton_steps=CONTROL_NEWTON_STEPS,
        gmres_iterations=CONTROL_GMRES_ITERATIONS,
        warmup=CONTROL_NEWTON_WARMUP,
    )
    elapsed_seconds = perf_counter() - started
    residual = float(solved.fixed_point.residual)
    trace = np.asarray(solved.fixed_point.trace, dtype=float)
    measured = trace[np.isfinite(trace)]
    core = _equilibrium_core(solved)
    axis = np.asarray(solved.topology.axis, dtype=float)
    flux = np.asarray(solved.flux, dtype=float)
    flux_span = float(solved.topology.flux_span)
    if core.any():
        branch = (
            "diverted_plasma" if bool(solved.topology.diverted) else "limited_plasma"
        )
    else:
        branch = "vacuum_or_empty_core"
    receipt = {
        "arm": label,
        "elapsed_seconds": elapsed_seconds,
        "solver": {
            "route": "newton_krylov",
            "warmup": CONTROL_NEWTON_WARMUP,
            "newton_steps": CONTROL_NEWTON_STEPS,
            "gmres_iterations": CONTROL_GMRES_ITERATIONS,
        },
        "terminal_relative_fixed_point_residual": residual,
        "trace_slots": int(trace.size),
        "measured_residuals": int(measured.size),
        "first_measured_residual": float(measured[0]),
        "minimum_measured_residual": float(measured.min()),
        "direct_branch_receipt": {
            "state_finite": bool(
                np.isfinite(flux).all() and bool(solved.finite.passed)
            ),
            "core_cells": int(core.sum()),
            "core_fraction_of_carrier": float(core.mean()),
            "topology_branch": branch,
            "diverted": bool(solved.topology.diverted),
            "solved_axis_m": axis,
            "axis_finite": bool(np.isfinite(axis).all()),
            "flux_span_wb": flux_span,
            "flux_span_valid": bool(np.isfinite(flux_span) and abs(flux_span) > 0.0),
        },
        "vertical_conditioning": _vertical_conditioning(profile, axis),
    }
    return solved, receipt


def _qualify_control_branch(
    reference_axis: np.ndarray,
    boundary: np.ndarray,
    solve_receipt: dict[str, object],
) -> dict[str, object]:
    """Qualify one analytic control root without optional shape reporting."""
    branch = solve_receipt["direct_branch_receipt"]
    axis = np.asarray(branch["solved_axis_m"], dtype=float)
    displacement = axis - np.asarray(reference_axis, dtype=float)
    displacement_m = float(np.linalg.norm(displacement))
    vertical_half_span = 0.5 * float(np.ptp(np.asarray(boundary)[:, 1]))
    displacement_limit = CONTROL_MAXIMUM_AXIS_DISPLACEMENT_FRACTION * vertical_half_span
    axis_inside = bool(PolygonPath(np.asarray(boundary)).contains_point(axis))
    residual = float(solve_receipt["terminal_relative_fixed_point_residual"])
    conditioning = solve_receipt["vertical_conditioning"]
    checks = {
        "root_residual_finite_and_below_reference_bound": bool(
            np.isfinite(residual) and residual < forward_solve_suite.RESIDUAL_TOLERANCE
        ),
        "state_finite": bool(branch["state_finite"]),
        "core_nonempty": int(branch["core_cells"]) > 0,
        "core_fraction_material": (
            float(branch["core_fraction_of_carrier"]) >= CONTROL_MINIMUM_CORE_FRACTION
        ),
        "axis_finite": bool(branch["axis_finite"]),
        "axis_inside_control_boundary": axis_inside,
        "axis_displacement_within_control_scale": (
            displacement_m <= displacement_limit
        ),
        "flux_span_valid": bool(branch["flux_span_valid"]),
        "vacuum_decay_index_inside_open_stability_window": bool(conditioning["stable"]),
    }
    return {
        **branch,
        "terminal_relative_fixed_point_residual": residual,
        "reference_axis_m": np.asarray(reference_axis, dtype=float),
        "axis_displacement_vector_m": displacement,
        "axis_displacement_m": displacement_m,
        "axis_displacement_limit_m": displacement_limit,
        "minimum_core_fraction": CONTROL_MINIMUM_CORE_FRACTION,
        "vertical_conditioning": conditioning,
        "checks": checks,
        "qualified": all(checks.values()),
    }


def _require_control_branch(
    reference_axis: np.ndarray,
    boundary: np.ndarray,
    solve_receipt: dict[str, object],
) -> None:
    """Stop before paired tracing when the analytic root is nonphysical."""
    qualification = _qualify_control_branch(
        reference_axis,
        boundary,
        solve_receipt,
    )
    solve_receipt["control_branch_qualification"] = qualification
    print(
        "CONTROL_BRANCH_QUALIFICATION="
        + json.dumps(_diagnostic_json(qualification), sort_keys=True)
    )
    if not qualification["qualified"]:
        failed = [
            name for name, passed in qualification["checks"].items() if not passed
        ]
        raise RuntimeError(
            "analytic near-circular control branch is not physical; "
            f"failed checks={failed}"
        )


def _paired_map_trace(
    with_operator,
    without_operator,
    seed,
    reference_flux,
    fixed_core,
    cell_count: int,
    span: float,
) -> dict[str, object]:
    """Follow the two coupled maps from their byte-identical seed."""
    with_map = with_operator.flux_map()
    without_map = without_operator.flux_map()
    with_external = with_operator.external()
    without_external = without_operator.external()
    external_delta = with_external - without_external
    direct_peak = _peak_points(external_delta, cell_count, span)

    with_state = jnp.asarray(seed)
    without_state = jnp.asarray(seed)
    previous_response = jnp.zeros_like(with_state)
    previous_peak = 0.0
    rows = []
    for iteration in range(1, PAIRED_MAP_ITERATIONS + 1):
        internal_with = with_operator.internal(with_state)
        internal_without = without_operator.internal(without_state)
        feedback = internal_with - internal_without
        next_with = with_map(with_state)
        next_without = without_map(without_state)
        response = next_with - next_without
        identity_error = response - (external_delta + feedback)
        increment = response - previous_response
        response_peak = _peak_points(response, cell_count, span)
        with_topology = _topology(with_operator, next_with)
        without_topology = _topology(without_operator, next_without)
        rows.append(
            {
                "iteration": iteration,
                "reproduction_move_points": _score(
                    next_without, reference_flux, fixed_core, cell_count, span
                )
                - _score(next_with, reference_flux, fixed_core, cell_count, span),
                "response_peak_points": response_peak,
                "response_growth_ratio": (
                    None if previous_peak == 0.0 else response_peak / previous_peak
                ),
                "response_increment_peak_points": _peak_points(
                    increment, cell_count, span
                ),
                "direct_external_peak_points": direct_peak,
                "plasma_feedback_peak_points": _peak_points(feedback, cell_count, span),
                "plasma_current_difference_a": float(
                    np.sum(
                        np.abs(
                            np.asarray(with_operator.cell_current(with_state))
                            - np.asarray(without_operator.cell_current(without_state))
                        )
                    )
                ),
                "map_decomposition_identity_error_wb": float(
                    np.max(np.abs(np.asarray(identity_error)))
                ),
                "support": _support_comparison(with_topology, without_topology),
            }
        )
        with_state = next_with
        without_state = next_without
        previous_response = response
        previous_peak = response_peak

    return {
        "iterations": rows,
        "direct_external_peak_points": direct_peak,
        "iteration_one_internal_difference_points": rows[0][
            "plasma_feedback_peak_points"
        ],
        "iteration_one_current_difference_a": rows[0]["plasma_current_difference_a"],
        "maximum_map_decomposition_identity_error_wb": max(
            row["map_decomposition_identity_error_wb"] for row in rows
        ),
    }


def _linear_response_trace(
    without_operator,
    without_root,
    external_delta,
    reference_flux,
    fixed_core,
    cell_count: int,
    span: float,
) -> dict[str, object]:
    """Apply the exact map tangent to each passive-response term."""
    without_map = without_operator.flux_map()
    mapped, tangent = jax.linearize(without_map, without_root)
    root_residual = _peak_points(mapped - without_root, cell_count, span)
    term = jnp.asarray(external_delta)
    cumulative = jnp.zeros_like(term)
    previous_term_peak = None
    rows = []
    for iteration in range(1, LINEAR_RESPONSE_ITERATIONS + 1):
        if iteration > 1:
            term = tangent(term)
        cumulative = cumulative + term
        term_peak = _peak_points(term, cell_count, span)
        counterfactual = without_root + cumulative
        rows.append(
            {
                "iteration": iteration,
                "increment_peak_points": term_peak,
                "increment_growth_ratio": (
                    None
                    if previous_term_peak is None
                    else term_peak / previous_term_peak
                ),
                "cumulative_response_peak_points": _peak_points(
                    cumulative, cell_count, span
                ),
                "counterfactual_reproduction_move_points": _score(
                    without_root, reference_flux, fixed_core, cell_count, span
                )
                - _score(counterfactual, reference_flux, fixed_core, cell_count, span),
            }
        )
        previous_term_peak = term_peak

    late_ratios = np.asarray(
        [
            row["increment_growth_ratio"]
            for row in rows[-5:]
            if row["increment_growth_ratio"] is not None
        ]
    )
    median_ratio = float(np.median(late_ratios))
    return {
        "linearisation_root_residual_points": root_residual,
        "iterations": rows,
        "late_increment_growth_ratio_median": median_ratio,
        "relative_difference_from_measured_spectral_radius": abs(
            median_ratio - MEASURED_UNPINNED_SPECTRAL_RADIUS
        )
        / MEASURED_UNPINNED_SPECTRAL_RADIUS,
    }


def _terminal_decomposition(
    with_solved,
    without_solved,
    external_delta,
    reference_flux,
    cell_count: int,
    span: float,
) -> dict[str, object]:
    """Decompose the terminal score into direct, feedback and support terms."""
    with_core = _equilibrium_core(with_solved)
    without_core = _equilibrium_core(without_solved)
    with_root = with_solved.flux
    without_root = without_solved.flux
    direct_counterfactual = without_root + external_delta

    without_score = _score(without_root, reference_flux, without_core, cell_count, span)
    direct_score = _score(
        direct_counterfactual, reference_flux, without_core, cell_count, span
    )
    with_common_score = _score(
        with_root, reference_flux, without_core, cell_count, span
    )
    with_own_score = _score(with_root, reference_flux, with_core, cell_count, span)
    direct_contribution = without_score - direct_score
    feedback_contribution = direct_score - with_common_score
    support_contribution = with_common_score - with_own_score
    total = without_score - with_own_score
    return {
        "without_passive_score_points": without_score,
        "direct_external_counterfactual_score_points": direct_score,
        "passive_inclusive_score_on_without_support_points": with_common_score,
        "passive_inclusive_score_on_own_support_points": with_own_score,
        "direct_external_contribution_points": direct_contribution,
        "coupled_plasma_feedback_contribution_points": feedback_contribution,
        "support_mask_contribution_points": support_contribution,
        "total_reproduction_move_points": total,
        "sum_of_contributions_points": (
            direct_contribution + feedback_contribution + support_contribution
        ),
        "sum_identity_error_points": abs(
            total - (direct_contribution + feedback_contribution + support_contribution)
        ),
        "root_response_peak_points": _peak_points(
            with_root - without_root, cell_count, span
        ),
        "direct_external_peak_points": _peak_points(external_delta, cell_count, span),
        "root_response_over_direct_peak": _peak_points(
            with_root - without_root, cell_count, span
        )
        / _peak_points(external_delta, cell_count, span),
        "support": {
            "with_core_cells": int(with_core.sum()),
            "without_core_cells": int(without_core.sum()),
            "core_symmetric_difference_cells": int(
                np.logical_xor(with_core, without_core).sum()
            ),
            "both_diverted": bool(
                with_solved.topology.diverted and without_solved.topology.diverted
            ),
            "axis_separation_mm": 1.0e3
            * float(
                np.linalg.norm(
                    np.asarray(with_solved.topology.axis, dtype=float)
                    - np.asarray(without_solved.topology.axis, dtype=float)
                )
            ),
        },
    }


def _verdict(
    seed_identity_error_wb: float,
    paired: dict[str, object],
    linear: dict[str, object],
    terminal: dict[str, object],
) -> dict[str, object]:
    """Choose between an expanding-map gain and an iteration-one path error."""
    scale = max(1.0, terminal["direct_external_peak_points"])
    iteration_one_correct = (
        seed_identity_error_wb == 0.0
        and paired["iteration_one_internal_difference_points"]
        <= IDENTITY_RELATIVE_TOLERANCE * scale
        and paired["iteration_one_current_difference_a"] == 0.0
        and paired["maximum_map_decomposition_identity_error_wb"]
        <= IDENTITY_RELATIVE_TOLERANCE
    )
    support_nonamplifying = terminal["support_mask_contribution_points"] == 0.0
    spectral_consistent = (
        linear["relative_difference_from_measured_spectral_radius"]
        <= SPECTRAL_RATIO_RELATIVE_TOLERANCE
    )
    feedback_dominates = abs(
        terminal["coupled_plasma_feedback_contribution_points"]
    ) > abs(terminal["direct_external_contribution_points"])

    if (
        iteration_one_correct
        and support_nonamplifying
        and spectral_consistent
        and feedback_dominates
    ):
        classification = "EXPANDING_MAP_AMPLIFICATION"
        mechanism = "noncontractive_free_boundary_map_resolvent_gain"
        repair_scope = [
            "nova/equilibrium/forward_operator.py",
            "nova/equilibrium/fixed_point.py",
        ]
        ruling = (
            "Iteration one is exactly the passive external field with no source "
            "or seed discrepancy. Its one-cell physical support move is removed "
            "by fixed-support scoring; later exact-tangent terms grow at the "
            "measured expanding-map rate and dominate the terminal score move."
        )
    else:
        classification = "CLOSURE_PATH_ERROR"
        mechanism = "iteration_one_closure_arithmetic_or_support_error"
        repair_scope = [
            "nova/equilibrium/source.py",
            "nova/equilibrium/domain.py",
            "nova/equilibrium/forward_operator.py",
        ]
        ruling = (
            "The passive response is already inconsistent before expanding-map "
            "feedback, or its support differs; repair the named iteration-one path."
        )

    return {
        "classification": classification,
        "mechanism": mechanism,
        "ruling": ruling,
        "iteration_one_arithmetic_correct": iteration_one_correct,
        "terminal_support_contribution_zero": support_nonamplifying,
        "growth_consistent_with_unpinned_spectral_radius": spectral_consistent,
        "plasma_feedback_dominates_direct_contribution": feedback_dominates,
        "measured_unpinned_spectral_radius": MEASURED_UNPINNED_SPECTRAL_RADIUS,
        "measured_pinned_spectral_radius": MEASURED_PINNED_SPECTRAL_RADIUS,
        "late_increment_growth_ratio_median": linear[
            "late_increment_growth_ratio_median"
        ],
        "repair_file_scope": repair_scope,
        "repair_scope_occupancy": (
            "The concurrent solver lane holds forward_operator.py and "
            "fixed_point.py; coordinate there rather than editing either here."
        ),
        "code_path": [
            "nova/equilibrium/forward_operator.py:569 captures the external field once",
            (
                "nova/equilibrium/forward_operator.py:571-573 adds the "
                "state-dependent plasma image"
            ),
            (
                "nova/equilibrium/fixed_point.py:244-255 forms and solves "
                "the exact (I-J) action"
            ),
            (
                "tests/test_equilibrium_forward_reference.py:1635-1640 "
                "supplies the same stored-map seed"
            ),
            (
                "tests/test_equilibrium_forward_reference.py:1552-1557 "
                "scores the topology core support"
            ),
        ],
        "excluded_closure_path_candidates": {
            "source_normalisation_amplification": (
                "excluded at iteration one: target_current is absent and the paired "
                "internal image plus plasma current are identical"
            ),
            "seed_or_basin_move": (
                "excluded: seed bytes are identical and both terminal roots remain "
                "on the same diverted branch"
            ),
            "support_or_mask_difference": (
                "excluded as amplification: paired iterations use fixed support, "
                "and the four-cell terminal mask difference contributes zero points"
            ),
        },
    }


def _stability_verdict(terminal: dict[str, object]) -> dict[str, object]:
    """Classify whether low elongation removes the elongated response gain."""
    gain = float(terminal["root_response_over_direct_peak"])
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("the stability-control field gain must be positive and finite")
    log_distance_from_direct = abs(math.log(gain))
    log_distance_from_elongated = abs(math.log(gain / ELONGATED_FIELD_GAIN))
    collapsed = log_distance_from_direct < log_distance_from_elongated
    if collapsed:
        classification = "PHYSICS"
        ruling = (
            "The near-circular control gain is logarithmically closer to unity "
            "than to the elongated 6.60x gain. The passive response therefore "
            "tracks vertical-mode conditioning rather than an anomalous map term."
        )
        next_action = (
            "Re-derive the reference ceiling for converged coupled response from "
            "stability conditioning; do not refuse healthy elongated equilibria "
            "solely because their passive response is amplified."
        )
    else:
        classification = "ANOMALOUS_GAIN"
        ruling = (
            "The near-circular control gain remains logarithmically closer to the "
            "elongated 6.60x gain than to unity, excluding the benign vertical-mode "
            "explanation."
        )
        next_action = (
            "Reopen the coupled-map gain investigation with the control receipt as "
            "evidence that low elongation does not remove the amplification."
        )
    return {
        "classification": classification,
        "ruling": ruling,
        "next_action": next_action,
        "control_field_gain": gain,
        "elongated_field_gain": ELONGATED_FIELD_GAIN,
        "log_distance_from_direct_gain": log_distance_from_direct,
        "log_distance_from_elongated_gain": log_distance_from_elongated,
        "decision_rule": (
            "PHYSICS when the control gain is closer to unity than to the "
            "elongated gain on a logarithmic scale; otherwise ANOMALOUS_GAIN."
        ),
    }


def _plot(receipt: dict[str, object], path: Path) -> None:
    """Plot iteration-resolved response and exact-tangent growth."""
    paired = receipt["paired_coupled_map"]["iterations"]
    linear = receipt["exact_tangent_response"]["iterations"]
    terminal = receipt["terminal_decomposition"]["total_reproduction_move_points"]

    figure, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), constrained_layout=True)
    upper, lower = axes
    upper.plot(
        [row["iteration"] for row in paired],
        [row["reproduction_move_points"] for row in paired],
        marker="o",
        color="#2a6099",
        label="paired coupled map",
    )
    upper.plot(
        [row["iteration"] for row in linear],
        [row["counterfactual_reproduction_move_points"] for row in linear],
        marker=".",
        color="0.4",
        label="exact-tangent response",
    )
    upper.axhline(terminal, color="#b24c3d", linestyle="--", linewidth=1.2)
    upper.text(
        LINEAR_RESPONSE_ITERATIONS,
        terminal,
        f" terminal {terminal:.4f} points ",
        color="#b24c3d",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    upper.axhline(
        DOCUMENTED_DIRECT_RESPONSE_POINTS,
        color="0.55",
        linestyle=":",
        linewidth=1.1,
    )
    upper.set_ylabel("reproduction move [points]")
    upper.legend(frameon=False, fontsize=9)
    upper.spines[["top", "right"]].set_visible(False)

    ratio_rows = [row for row in linear if row["increment_growth_ratio"] is not None]
    lower.plot(
        [row["iteration"] for row in ratio_rows],
        [row["increment_growth_ratio"] for row in ratio_rows],
        marker="o",
        color="#2a6099",
    )
    lower.axhline(
        MEASURED_UNPINNED_SPECTRAL_RADIUS,
        color="#b24c3d",
        linestyle="--",
        linewidth=1.2,
    )
    lower.text(
        LINEAR_RESPONSE_ITERATIONS,
        MEASURED_UNPINNED_SPECTRAL_RADIUS,
        f" measured spectral radius {MEASURED_UNPINNED_SPECTRAL_RADIUS:.4f} ",
        color="#b24c3d",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    lower.set_xlabel("coupled-map iteration")
    lower.set_ylabel("increment growth ratio")
    lower.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, transparent=True)
    plt.close(figure)


def _plot_stability_control(receipt: dict[str, object], path: Path) -> None:
    """Plot the control response by iteration beside the elongated baseline."""
    paired = receipt["paired_coupled_map"]["iterations"]
    linear = receipt["exact_tangent_response"]["iterations"]
    terminal = receipt["terminal_decomposition"]
    shape = receipt["shape_control"]

    figure, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), constrained_layout=True)
    trace, comparison = axes
    trace.plot(
        [row["iteration"] for row in paired],
        [row["response_peak_points"] for row in paired],
        marker="o",
        color="#2a6099",
        label="paired nonlinear response",
    )
    trace.plot(
        [row["iteration"] for row in linear],
        [row["cumulative_response_peak_points"] for row in linear],
        marker=".",
        color="0.4",
        label="exact-tangent cumulative response",
    )
    trace.axhline(
        terminal["direct_external_peak_points"],
        color="#b24c3d",
        linestyle="--",
        linewidth=1.2,
        label="control direct passive field",
    )
    trace.set_ylabel("response peak [points]")
    trace.set_xlabel("coupled-map iteration")
    trace.legend(frameon=False, fontsize=9)
    trace.spines[["top", "right"]].set_visible(False)

    labels = ["direct field", "root response", "score move"]
    elongated = [
        ELONGATED_DIRECT_PEAK_POINTS,
        ELONGATED_DIRECT_PEAK_POINTS * ELONGATED_FIELD_GAIN,
        ELONGATED_REPRODUCTION_MOVE_POINTS,
    ]
    control = [
        terminal["direct_external_peak_points"],
        terminal["root_response_peak_points"],
        abs(terminal["total_reproduction_move_points"]),
    ]
    position = np.arange(len(labels))
    width = 0.36
    comparison.bar(
        position - width / 2,
        elongated,
        width,
        color="0.65",
        label=(
            f"elongated κ={shape['elongated_boundary_elongation']:.3f}, "
            f"gain={ELONGATED_FIELD_GAIN:.2f}x"
        ),
    )
    comparison.bar(
        position + width / 2,
        control,
        width,
        color="#2a6099",
        label=(
            f"control κ={shape['control_boundary_elongation']:.3f}, "
            f"gain={terminal['root_response_over_direct_peak']:.2f}x"
        ),
    )
    comparison.set_xticks(position, labels)
    comparison.set_ylabel("flux-span percentage points")
    comparison.legend(frameon=False, fontsize=9)
    comparison.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, transparent=True)
    plt.close(figure)


def run() -> dict[str, object]:
    """Run the fixed-carrier mechanism trace."""
    source_commit = measurement_stamp(Path.cwd())
    configure_dtypes()
    case = reference.require_reference()
    machine_started = perf_counter()
    machine = reference.cached_machine(case, DENSE_CELL_REQUEST, passive=True)
    machine_seconds = perf_counter() - machine_started
    cache = machine.cache_receipt
    if cache is None:
        raise RuntimeError("the dense carrier has no persistent-cache receipt")
    without_current = machine.source_current.copy()
    without_current[-machine.passive_columns :] = 0.0
    without_machine = replace(
        machine,
        source_current=without_current,
        passive_columns=0,
        cache_receipt=None,
    )

    with_operator = reference.forward_operator(case, machine)
    without_operator = reference.forward_operator(case, without_machine)
    seed_with = reference.seed_flux(case, machine)
    seed_without = reference.seed_flux(case, without_machine)
    seed_identity_error = float(
        np.max(np.abs(np.asarray(seed_with) - np.asarray(seed_without)))
    )

    with_solved, with_receipt = _solve(case, machine, "declared_passive_currents")
    without_solved, without_receipt = _solve(
        case, without_machine, "passive_currents_zeroed"
    )
    cell_count = len(machine.node)
    span = abs(float(case.flux_span))
    reference_flux = case.flux(machine.radius, machine.node[:, 1])
    without_core = np.asarray(without_solved.masks.core, dtype=bool)
    external_delta = with_operator.external() - without_operator.external()

    paired = _paired_map_trace(
        with_operator,
        without_operator,
        seed_with,
        reference_flux,
        without_core,
        cell_count,
        span,
    )
    linear = _linear_response_trace(
        without_operator,
        without_solved.flux,
        external_delta,
        reference_flux,
        without_core,
        cell_count,
        span,
    )
    terminal = _terminal_decomposition(
        with_solved,
        without_solved,
        external_delta,
        reference_flux,
        cell_count,
        span,
    )
    verdict = _verdict(seed_identity_error, paired, linear, terminal)
    return {
        "receipt": {
            "kind": "passive_closure_iteration_trace",
            "status": "complete",
            "source_commit": source_commit,
            "checkout_porcelain_empty_before_measurement": True,
            "device_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "production_solve_count": 2,
            "runtime_mitigations": {
                "defaults": RUNTIME_MITIGATION_DEFAULTS,
                "effective": {
                    name: os.environ.get(name) for name in RUNTIME_MITIGATION_DEFAULTS
                },
                "purpose": (
                    "Disable CUDA command-buffer graphs and whole-device "
                    "preallocation so the dense control can coexist with other "
                    "accelerator work."
                ),
            },
        },
        "carrier": {
            "requested_cells": DENSE_CELL_REQUEST,
            "realised_plasma_cells": cell_count,
            "machine_request_seconds": machine_seconds,
            "cache": asdict(cache),
            "passive_columns": machine.passive_columns,
            "reference_flux_span_wb": span,
        },
        "comparators": {
            "documented_direct_response_points": DOCUMENTED_DIRECT_RESPONSE_POINTS,
            "registered_ceiling_points": REGISTERED_CEILING_POINTS,
            "measured_unpinned_spectral_radius": MEASURED_UNPINNED_SPECTRAL_RADIUS,
            "measured_pinned_spectral_radius": MEASURED_PINNED_SPECTRAL_RADIUS,
        },
        "controlled_inputs": {
            "seed_max_absolute_difference_wb": seed_identity_error,
            "geometry_shared": True,
            "source_profiles_shared": True,
            "only_changed_input": "declared passive-current columns retained or zeroed",
        },
        "production_solves": [with_receipt, without_receipt],
        "paired_coupled_map": paired,
        "exact_tangent_response": linear,
        "terminal_decomposition": terminal,
        "verdict": verdict,
    }


def run_stability_control() -> dict[str, object]:
    """Repeat the passive trace on an analytic stable control."""
    source_commit = measurement_stamp(Path.cwd())
    configure_dtypes()
    construction_started = perf_counter()
    control = _build_analytic_control()
    construction_seconds = perf_counter() - construction_started
    with_operator = control.with_passive.operator
    without_operator = control.without_passive.operator
    seed_with = control.seed_flux
    seed_without = control.seed_flux
    seed_identity_error = float(
        np.max(np.abs(np.asarray(seed_with) - np.asarray(seed_without)))
    )
    seed_plasma_current = float(
        np.sum(np.asarray(with_operator.cell_current(seed_with)))
    )

    with_solved, with_receipt = _solve_analytic_control(
        control.with_passive,
        seed_with,
        "declared_passive_currents",
    )
    _require_control_branch(
        control.reference_axis,
        control.material_boundary,
        with_receipt,
    )
    without_solved, without_receipt = _solve_analytic_control(
        control.without_passive,
        seed_without,
        "passive_currents_zeroed",
    )
    _require_control_branch(
        control.reference_axis,
        control.material_boundary,
        without_receipt,
    )
    cell_count = control.with_passive.operator.grid.node_number
    span = control.reference_flux_span_wb
    reference_flux = control.reference_grid_flux
    without_core = _equilibrium_core(without_solved)
    external_delta = with_operator.external() - without_operator.external()
    paired = _paired_map_trace(
        with_operator,
        without_operator,
        seed_with,
        reference_flux,
        without_core,
        cell_count,
        span,
    )
    linear = _linear_response_trace(
        without_operator,
        without_solved.flux,
        external_delta,
        reference_flux,
        without_core,
        cell_count,
        span,
    )
    terminal = _terminal_decomposition(
        with_solved,
        without_solved,
        external_delta,
        reference_flux,
        cell_count,
        span,
    )
    stability = {
        "definition": "n = -(R/Bz)*(dBz/dR)",
        "open_stability_window": list(DECAY_INDEX_WINDOW),
        "passive_retained": with_receipt["vertical_conditioning"],
        "passive_zeroed": without_receipt["vertical_conditioning"],
        "both_arms_stable": bool(
            with_receipt["vertical_conditioning"]["stable"]
            and without_receipt["vertical_conditioning"]["stable"]
        ),
        "elongated_boundary_elongation": ELONGATED_BOUNDARY_ELONGATION,
        "control_boundary_elongation": control.construction_receipt[
            "control_boundary_elongation"
        ],
    }
    return {
        "receipt": {
            "kind": "passive_closure_vertical_stability_control",
            "status": "complete",
            "source_commit": source_commit,
            "checkout_porcelain_empty_before_measurement": True,
            "device_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "production_solve_count": 2,
            "runtime_mitigations": {
                "defaults": RUNTIME_MITIGATION_DEFAULTS,
                "effective": {
                    name: os.environ.get(name) for name in RUNTIME_MITIGATION_DEFAULTS
                },
                "purpose": (
                    "Disable CUDA command-buffer graphs and whole-device "
                    "preallocation so the dense control can coexist with other "
                    "accelerator work."
                ),
            },
        },
        "carrier": {
            "construction": "analytic structured free-boundary carrier",
            "lattice_shape": control.construction_receipt["lattice_shape"],
            "realised_plasma_cells": cell_count,
            "construction_seconds": construction_seconds,
            "passive_columns": control.construction_receipt[
                "passive_material_coupling"
            ]["conductor_count"],
            "reference_flux_span_wb": span,
        },
        "shape_control": control.construction_receipt,
        "vertical_stability": stability,
        "comparators": {
            "elongated_direct_external_peak_points": ELONGATED_DIRECT_PEAK_POINTS,
            "elongated_coupled_reproduction_move_points": (
                ELONGATED_REPRODUCTION_MOVE_POINTS
            ),
            "elongated_field_gain": ELONGATED_FIELD_GAIN,
            "elongated_boundary_elongation": ELONGATED_BOUNDARY_ELONGATION,
            "elongated_wall_elongation": ELONGATED_WALL_ELONGATION,
            "documented_direct_response_points": DOCUMENTED_DIRECT_RESPONSE_POINTS,
            "registered_direct_response_ceiling_points": REGISTERED_CEILING_POINTS,
        },
        "controlled_inputs": {
            "seed_max_absolute_difference_wb": seed_identity_error,
            "geometry_shared_between_passive_arms": True,
            "source_profiles_shared_between_passive_arms": True,
            "only_changed_input_between_passive_arms": (
                "declared passive-current columns retained or zeroed"
            ),
            "seed_plasma_current_a": seed_plasma_current,
            "passive_drive_matches_elongated_peak": bool(
                abs(
                    _peak_points(external_delta, cell_count, span)
                    - ELONGATED_DIRECT_PEAK_POINTS
                )
                <= 1.0e-12
            ),
        },
        "production_solves": [with_receipt, without_receipt],
        "paired_coupled_map": paired,
        "exact_tangent_response": linear,
        "terminal_decomposition": terminal,
        "stability_verdict": _stability_verdict(terminal),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-control", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--figure", type=Path)
    arguments = parser.parse_args()
    if arguments.stability_control:
        receipt_path = arguments.receipt or CONTROL_RECEIPT
        figure_path = arguments.figure or CONTROL_FIGURE
        receipt = _strict_json(run_stability_control())
        summary = {
            "classification": receipt["stability_verdict"]["classification"],
            "control_field_gain": receipt["stability_verdict"]["control_field_gain"],
            "direct_external_peak_points": receipt["terminal_decomposition"][
                "direct_external_peak_points"
            ],
            "root_response_peak_points": receipt["terminal_decomposition"][
                "root_response_peak_points"
            ],
            "terminal_move_points": receipt["terminal_decomposition"][
                "total_reproduction_move_points"
            ],
        }
    else:
        receipt_path = arguments.receipt or DEFAULT_RECEIPT
        figure_path = arguments.figure or DEFAULT_FIGURE
        receipt = _strict_json(run())
        summary = {
            "classification": receipt["verdict"]["classification"],
            "mechanism": receipt["verdict"]["mechanism"],
            "terminal_move_points": receipt["terminal_decomposition"][
                "total_reproduction_move_points"
            ],
            "direct_contribution_points": receipt["terminal_decomposition"][
                "direct_external_contribution_points"
            ],
            "feedback_contribution_points": receipt["terminal_decomposition"][
                "coupled_plasma_feedback_contribution_points"
            ],
            "late_increment_growth_ratio": receipt["exact_tangent_response"][
                "late_increment_growth_ratio_median"
            ],
        }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if arguments.stability_control:
        _plot_stability_control(receipt, figure_path)
    else:
        _plot(receipt, figure_path)
    summary.update({"receipt": str(receipt_path), "figure": str(figure_path)})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
