"""Measure fixed two-level triangle masking on an analytic plasma boundary.

Regular hexagons are decomposed into six centre-to-edge triangles, then every
triangle into four fixed sub-triangles.  The hierarchy is geometric and static:
an iteration changes only a scalar separatrix label and the resulting Boolean
masks before contracting precomputed point-filament coupling columns.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, stats

from nova.biot.greens import greens_psi
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import RotatingEquilibrium, reference_cases

ANALYTIC_CASE = "moderate-rotation-conventional"
HEX_RADII_M = (0.12, 0.08, 0.06, 0.04)
TARGET_COUNT = 48
REFERENCE_ORDERS = (80, 120)
PRODUCTION_CELL_COUNT = 1587
TIMED_ITERATIONS = 20
REPAIRED_CLIP_MAX_RELATIVE_RESIDUAL = 6.56e-16


def _fit_order(cell_size: np.ndarray, error: np.ndarray) -> dict[str, float]:
    """Fit a zero-limit power law to a positive error series."""

    fit = stats.linregress(np.log(cell_size), np.log(error))
    return {
        "observed_order": float(fit.slope),
        "order_standard_error": float(fit.stderr),
        "coefficient": float(np.exp(fit.intercept)),
        "r_squared": float(fit.rvalue**2),
    }


def _analytic_area(case: RotatingEquilibrium) -> tuple[float, float]:
    """Return the exact smooth-boundary poloidal area and quadrature receipt."""

    inboard, outboard = case.boundary_midplane_radii()

    def full_height(radius: float) -> float:
        midplane_flux = max(float(case.flux(radius, 0.0)), 0.0)
        return 2.0 * np.sqrt(midplane_flux / case.field_coefficient)

    area, error = integrate.quad(
        full_height,
        inboard,
        outboard,
        epsabs=2.0e-14,
        epsrel=2.0e-13,
        points=[case.major_radius],
        limit=300,
    )
    return float(area), float(error)


def _hex_centres(case: RotatingEquilibrium, radius: float) -> np.ndarray:
    """Return a flat-top regular-hexagonal tiling covering the plasma."""

    inboard, outboard = case.boundary_midplane_radii()
    half_height = float(np.sqrt(case.axis_flux / case.field_coefficient))
    radial_step = 1.5 * radius
    vertical_step = np.sqrt(3.0) * radius
    q_limit = (
        int(
            np.ceil(
                max(outboard - case.major_radius, case.major_radius - inboard)
                / radial_step
            )
        )
        + 3
    )
    r_limit = int(np.ceil(half_height / vertical_step)) + q_limit // 2 + 4
    rows = []
    for q_index in range(-q_limit, q_limit + 1):
        source_r = case.major_radius + radial_step * q_index
        for r_index in range(-r_limit, r_limit + 1):
            source_z = vertical_step * (r_index + 0.5 * q_index)
            if (
                inboard - 2.0 * radius <= source_r <= outboard + 2.0 * radius
                and -half_height - 2.0 * radius
                <= source_z
                <= half_height + 2.0 * radius
            ):
                rows.append((source_r, source_z))
    return np.asarray(rows, dtype=float)


def _triangle_geometry(
    case: RotatingEquilibrium, radius: float
) -> dict[str, np.ndarray | float]:
    """Precompute both triangle levels and fixed sub-triangle centroids."""

    centres = _hex_centres(case, radius)
    angles = np.arange(6, dtype=float) * np.pi / 3.0
    offsets = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    vertices = centres[:, None, :] + offsets[None, :, :]
    next_vertices = np.roll(vertices, -1, axis=1)
    repeated_centres = np.broadcast_to(centres[:, None, :], vertices.shape)
    parents = np.stack((repeated_centres, vertices, next_vertices), axis=2)

    first = parents[:, :, 0]
    second = parents[:, :, 1]
    third = parents[:, :, 2]
    first_second = 0.5 * (first + second)
    second_third = 0.5 * (second + third)
    third_first = 0.5 * (third + first)
    children = np.stack(
        (
            np.stack((first, first_second, third_first), axis=2),
            np.stack((first_second, second, second_third), axis=2),
            np.stack((third_first, second_third, third), axis=2),
            np.stack((first_second, second_third, third_first), axis=2),
        ),
        axis=2,
    )
    parent_area = 3.0 * np.sqrt(3.0) * radius**2 / 12.0
    child_area = parent_area / 4.0
    centroids = np.mean(children, axis=3)
    return {
        "hex_centres": centres,
        "parent_nodes": parents,
        "child_nodes": children,
        "child_centroids": centroids,
        "hex_area_m2": 3.0 * np.sqrt(3.0) * radius**2 / 2.0,
        "parent_area_m2": parent_area,
        "child_area_m2": child_area,
    }


def _hierarchical_mask(
    parent_flux: np.ndarray, child_flux: np.ndarray, cutoff: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return parent, crossing and retained child masks for one separatrix."""

    parent_inside_nodes = parent_flux > cutoff
    parent_inside = np.all(parent_inside_nodes, axis=-1)
    parent_crossing = np.any(parent_inside_nodes, axis=-1) & ~parent_inside
    child_inside = np.all(child_flux > cutoff, axis=-1)
    retained = parent_inside[:, :, None] | (parent_crossing[:, :, None] & child_inside)
    return parent_inside, parent_crossing, retained


def _targets(case: RotatingEquilibrium) -> np.ndarray:
    """Return fixed off-plasma targets, avoiding filament self singularities."""

    inboard, outboard = case.boundary_midplane_radii()
    centre = 0.5 * (inboard + outboard)
    half_width = 0.5 * (outboard - inboard)
    half_height = float(np.sqrt(case.axis_flux / case.field_coefficient))
    angle = 2.0 * np.pi * np.arange(TARGET_COUNT) / TARGET_COUNT
    return np.column_stack(
        (centre + 1.45 * half_width * np.cos(angle), 1.45 * half_height * np.sin(angle))
    )


def _coupling_columns(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Precompute exact point-filament columns at fixed source centroids."""

    flat = np.asarray(sources, dtype=float).reshape(-1, 2)
    columns = greens_psi(
        targets[:, 0, None],
        targets[:, 1, None],
        flat[None, :, 0],
        flat[None, :, 1],
    )
    return np.asarray(columns, dtype=float)


def _reference_flux(
    case: RotatingEquilibrium, targets: np.ndarray, order: int
) -> np.ndarray:
    """Integrate the analytic current over its exact smooth plasma domain."""

    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(order)
    vertical_nodes, vertical_weights = np.polynomial.legendre.leggauss(order)
    inboard, outboard = case.boundary_midplane_radii()
    radius = 0.5 * (outboard - inboard) * radial_nodes + 0.5 * (inboard + outboard)
    radius_weight = 0.5 * (outboard - inboard) * radial_weights
    midplane_flux = np.maximum(case.flux(radius, 0.0), 0.0)
    half_height = np.sqrt(midplane_flux / case.field_coefficient)
    source_r = np.repeat(radius, order)
    source_z = (half_height[:, None] * vertical_nodes[None, :]).reshape(-1)
    area_weight = (
        radius_weight[:, None] * half_height[:, None] * vertical_weights[None, :]
    ).reshape(-1)
    current = case.toroidal_current_density(source_r, source_z) * area_weight
    columns = _coupling_columns(targets, np.column_stack((source_r, source_z)))
    return columns @ current


def _memory_receipt(
    cell_count: int, boundary_count: int, target_count: int
) -> dict[str, int | float]:
    """Price boundary-band sub-triangle columns against one column per cell."""

    scalar_bytes = np.dtype(np.float64).itemsize
    baseline = cell_count * target_count * scalar_bytes
    refined_columns = cell_count + 23 * boundary_count
    refined = refined_columns * target_count * scalar_bytes
    boundary_fraction = boundary_count / cell_count
    production_boundary = int(np.ceil(PRODUCTION_CELL_COUNT * boundary_fraction))
    production_baseline = PRODUCTION_CELL_COUNT * PRODUCTION_CELL_COUNT * scalar_bytes
    production_refined_columns = PRODUCTION_CELL_COUNT + 23 * production_boundary
    production_refined = (
        production_refined_columns * PRODUCTION_CELL_COUNT * scalar_bytes
    )
    return {
        "float_bytes": scalar_bytes,
        "target_count": target_count,
        "baseline_columns": cell_count,
        "boundary_band_cells": boundary_count,
        "boundary_band_fraction": boundary_fraction,
        "refined_columns": refined_columns,
        "baseline_bytes": baseline,
        "boundary_band_refined_bytes": refined,
        "growth_bytes": refined - baseline,
        "growth_factor": refined / baseline,
        "production_cell_count": PRODUCTION_CELL_COUNT,
        "production_target_count": PRODUCTION_CELL_COUNT,
        "production_extrapolated_boundary_cells": production_boundary,
        "production_baseline_bytes": production_baseline,
        "production_boundary_band_refined_bytes": production_refined,
        "production_growth_bytes": production_refined - production_baseline,
        "production_growth_factor": production_refined / production_baseline,
    }


def _timing_receipt(
    parent_flux: np.ndarray,
    child_flux: np.ndarray,
    child_current: np.ndarray,
    child_area: float,
    columns: np.ndarray,
) -> dict[str, Any]:
    """Trace and time the fixed-shape mask-and-contraction iteration."""

    trace_count = 0
    parent = jnp.asarray(parent_flux)
    child = jnp.asarray(child_flux)
    current = jnp.asarray(child_current)
    coupling = jnp.asarray(columns)

    def iteration(cutoff: jax.Array) -> jax.Array:
        nonlocal trace_count
        trace_count += 1
        parent_nodes_inside = parent > cutoff
        parent_inside = jnp.all(parent_nodes_inside, axis=-1)
        crossing = jnp.any(parent_nodes_inside, axis=-1) & ~parent_inside
        child_inside = jnp.all(child > cutoff, axis=-1)
        retained = parent_inside[:, :, None] | (crossing[:, :, None] & child_inside)
        drive = jnp.where(retained, current * child_area, 0.0).reshape(-1)
        return coupling @ drive

    traced = jax.jit(iteration)
    cutoffs = np.linspace(-0.025, 0.025, TIMED_ITERATIONS) * float(np.max(parent_flux))
    traced(jnp.asarray(cutoffs[0])).block_until_ready()
    elapsed = []
    for cutoff in cutoffs:
        started = perf_counter()
        traced(jnp.asarray(cutoff)).block_until_ready()
        elapsed.append(perf_counter() - started)
    device = jax.devices()[0]
    return {
        "jax_version": jax.__version__,
        "platform": device.platform,
        "device": str(device),
        "device_kind": getattr(device, "device_kind", str(device)),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "iterations": TIMED_ITERATIONS,
        "moving_separatrix_cutoff_min": float(np.min(cutoffs)),
        "moving_separatrix_cutoff_max": float(np.max(cutoffs)),
        "trace_count": trace_count,
        "fixed_shapes": True,
        "per_iteration_work": (
            "node comparisons, Boolean masks, gather by where, and one fixed "
            "matrix-vector contraction"
        ),
        "mean_seconds": float(np.mean(elapsed)),
        "median_seconds": float(np.median(elapsed)),
        "minimum_seconds": float(np.min(elapsed)),
        "maximum_seconds": float(np.max(elapsed)),
    }


def _resolution_receipt(
    case: RotatingEquilibrium,
    hex_radius: float,
    exact_area: float,
    targets: np.ndarray,
    reference_flux: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray | float]]:
    """Measure area, flux and memory for one fixed tiling."""

    geometry = _triangle_geometry(case, hex_radius)
    parents = np.asarray(geometry["parent_nodes"])
    children = np.asarray(geometry["child_nodes"])
    centroids = np.asarray(geometry["child_centroids"])
    parent_flux = case.flux(parents[..., 0], parents[..., 1])
    child_flux = case.flux(children[..., 0], children[..., 1])
    parent_inside, parent_crossing, retained = _hierarchical_mask(
        parent_flux, child_flux
    )
    child_area = float(geometry["child_area_m2"])
    approximate_area = float(np.sum(retained) * child_area)
    signed_area_difference = (approximate_area - exact_area) / exact_area
    if signed_area_difference > 5.0e-14:
        raise AssertionError("all-node-inside triangle masking must not overcount")

    current_density = case.toroidal_current_density(
        centroids[..., 0], centroids[..., 1]
    )
    columns = _coupling_columns(targets, centroids)
    approximate_flux = columns @ (retained * current_density * child_area).reshape(-1)
    difference = approximate_flux - reference_flux
    span = float(np.ptp(reference_flux))
    boundary_cells = np.any(parent_crossing, axis=1)
    receipt = {
        "hex_circumradius_m": hex_radius,
        "hex_centre_pitch_radial_m": 1.5 * hex_radius,
        "hex_centre_pitch_vertical_m": float(np.sqrt(3.0) * hex_radius),
        "cell_count": len(np.asarray(geometry["hex_centres"])),
        "level_one_triangle_count": int(np.prod(parent_inside.shape)),
        "level_two_subtriangle_count": int(np.prod(retained.shape)),
        "kept_whole_level_one_triangles": int(np.count_nonzero(parent_inside)),
        "crossing_level_one_triangles": int(np.count_nonzero(parent_crossing)),
        "retained_level_two_equivalent_count": int(np.count_nonzero(retained)),
        "exact_analytic_clipped_area_m2": exact_area,
        "hierarchical_inside_area_m2": approximate_area,
        "signed_relative_area_difference": signed_area_difference,
        "one_sided_undercount": bool(signed_area_difference <= 0.0),
        "flux_error_against_converged_exact_reference": {
            "reference_span_wb": span,
            "sup_wb": float(np.max(np.abs(difference))),
            "rms_wb": float(np.sqrt(np.mean(difference**2))),
            "sup_fraction_of_span": float(np.max(np.abs(difference)) / span),
            "rms_fraction_of_span": float(np.sqrt(np.mean(difference**2)) / span),
        },
        "memory": _memory_receipt(
            len(np.asarray(geometry["hex_centres"])),
            int(np.count_nonzero(boundary_cells)),
            len(targets),
        ),
    }
    timing_inputs = {
        "parent_flux": parent_flux,
        "child_flux": child_flux,
        "child_current": current_density,
        "child_area": child_area,
        "columns": columns,
    }
    return receipt, timing_inputs


def measure(*, timing_only: bool) -> dict[str, Any]:
    """Return the accuracy series or a timing-only receipt on the finest tiling."""

    configure_dtypes()
    case = reference_cases()[ANALYTIC_CASE]
    targets = _targets(case)
    if timing_only:
        geometry = _triangle_geometry(case, HEX_RADII_M[-1])
        parents = np.asarray(geometry["parent_nodes"])
        children = np.asarray(geometry["child_nodes"])
        centroids = np.asarray(geometry["child_centroids"])
        return {
            "mode": "timing_only",
            "hex_circumradius_m": HEX_RADII_M[-1],
            "timing": _timing_receipt(
                case.flux(parents[..., 0], parents[..., 1]),
                case.flux(children[..., 0], children[..., 1]),
                case.toroidal_current_density(centroids[..., 0], centroids[..., 1]),
                float(geometry["child_area_m2"]),
                _coupling_columns(targets, centroids),
            ),
        }

    exact_area, exact_area_error = _analytic_area(case)
    reference_rows = [
        _reference_flux(case, targets, order) for order in REFERENCE_ORDERS
    ]
    reference_flux = reference_rows[-1]
    reference_span = float(np.ptp(reference_flux))
    reference_self_difference = float(
        np.max(np.abs(reference_rows[-1] - reference_rows[-2])) / reference_span
    )
    rows = []
    finest_timing_inputs: dict[str, np.ndarray | float] | None = None
    for radius in HEX_RADII_M:
        row, finest_timing_inputs = _resolution_receipt(
            case, radius, exact_area, targets, reference_flux
        )
        rows.append(row)
    assert finest_timing_inputs is not None
    cell_size = np.asarray([row["hex_circumradius_m"] for row in rows])
    area_error = -np.asarray([row["signed_relative_area_difference"] for row in rows])
    flux_sup = np.asarray(
        [
            row["flux_error_against_converged_exact_reference"]["sup_fraction_of_span"]
            for row in rows
        ]
    )
    return {
        "mode": "full_measurement",
        "analytic_source": ANALYTIC_CASE,
        "python_version": ".".join(map(str, __import__("sys").version_info[:3])),
        "geometry": {
            "tiling": "regular flat-top hexagons",
            "hexagon_split": "six centre-to-edge triangles",
            "triangle_split": "four midpoint sub-triangles",
            "precomputed_once": True,
            "hex_radius_sequence_m": list(HEX_RADII_M),
            "linear_pitch_factor": HEX_RADII_M[0] / HEX_RADII_M[-1],
        },
        "hierarchical_rule": (
            "keep all four children when all parent nodes are inside; only a "
            "parent with mixed node states descends, and then retain a child "
            "only when all three child nodes are inside"
        ),
        "exact_reference": {
            "analytic_area_m2": exact_area,
            "analytic_area_quadrature_error_estimate_m2": exact_area_error,
            "flux_quadrature_orders": list(REFERENCE_ORDERS),
            "flux_target_count": len(targets),
            "consecutive_flux_reference_sup_difference_fraction_of_span": (
                reference_self_difference
            ),
            "coupling_columns": (
                "exact point-filament Green values at every fixed sub-triangle centroid"
            ),
        },
        "resolutions": rows,
        "area_undercount_convergence": _fit_order(cell_size, area_error),
        "flux_sup_convergence": _fit_order(cell_size, flux_sup),
        "comparison_to_repaired_clip": {
            "repaired_shared_atomic_edge_clip_max_relative_conservation_residual": (
                REPAIRED_CLIP_MAX_RELATIVE_RESIDUAL
            ),
            "triangle_rule": (
                "strict all-node-inside masking intentionally discards partial "
                "sub-triangles and therefore undercounts instead of conserving "
                "the clipped area"
            ),
        },
        "timing": _timing_receipt(**finest_timing_inputs),
        "sol_current_extension": (
            "for nonzero J(psi_N > 1), use the complementary child mask for "
            "discarded sub-triangles and contract a separately evaluated "
            "SOL-current vector through the same fixed coupling columns; "
            "geometry and trace shapes remain unchanged"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-only", action="store_true")
    return parser


def main() -> None:
    """Print a stable JSON measurement receipt."""

    arguments = _parser().parse_args()
    print(
        json.dumps(measure(timing_only=arguments.timing_only), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
