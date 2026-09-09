"""Discriminate cut-cell moment and curvature errors on exact Solovev states.

One invocation measures one static analytic case and carrier resolution.  The
production path is left unchanged: its own-cell quadratic and fixed Duffy rule
produce the current moments consumed by the polygon coupling blocks.  A
separate adaptive integral supplies the exact current-density moments over the
true analytic plasma intersection of every cell.

The aggregate receipt compares four direct images of the exact state.  The
first is production, the second substitutes exact physical moments only in
boundary-cut cells, and the remaining two suppress the radial and vertical
coupling coefficients either everywhere or only in boundary-cut cells.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/uniform-cell-clip-and-coupling/cut-cell-moments"
RECEIPT = OUTPUT_ROOT / "receipt.json"
ERROR_FIGURE = OUTPUT_ROOT / "error-maps.svg"
CASE_REQUESTS = (
    ("weak-rotation-reactor-static", -110),
    ("moderate-rotation-conventional-static", -110),
    ("weak-rotation-reactor-static", -300),
)
VARIANTS = (
    "production",
    "exact_cut_cell_moments",
    "second_order_zeroed_everywhere",
    "second_order_zeroed_in_cut_cells",
)
OUTBOARD_WINDOW = (7.25, 7.75, 0.0, 0.5)
ADAPTIVE_RELATIVE_TOLERANCE = 5.0e-13
AGREEMENT_BOUND = 1.0e-12


@dataclass(frozen=True)
class CellIntegral:
    """Area and density moments about one cell's coupling expansion point."""

    area: float
    moment: np.ndarray
    error_estimate: np.ndarray


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _lane_receipt() -> dict[str, Any]:
    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _array_digest(values: np.ndarray) -> str:
    packed = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _case_key(case_name: str, requested_cells: int) -> str:
    return f"{case_name}-{abs(requested_cells)}"


def _part_path(case_name: str, requested_cells: int) -> Path:
    return OUTPUT_ROOT / "parts" / f"{_case_key(case_name, requested_cells)}.json"


def _vertical_bounds(vertices: np.ndarray, radius: float) -> tuple[float, float] | None:
    """Return a convex polygon's vertical interval at one radius."""

    scale = max(float(np.max(np.abs(vertices))), 1.0)
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    heights: list[float] = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        radial_delta = second[0] - first[0]
        if abs(radial_delta) <= tolerance:
            if abs(radius - first[0]) <= tolerance:
                heights.extend((float(first[1]), float(second[1])))
            continue
        fraction = (radius - first[0]) / radial_delta
        if -tolerance <= fraction <= 1.0 + tolerance:
            heights.append(float(first[1] + fraction * (second[1] - first[1])))
    if len(heights) < 2:
        return None
    return min(heights), max(heights)


def _plasma_half_height(case: Any, radius: float) -> float:
    remaining = float(case.axis_flux - case._flux_offset(case._flux_label(radius)))
    if remaining <= 0.0:
        return 0.0
    return math.sqrt(remaining / float(case.field_coefficient))


def _radial_breaks(
    case: Any, vertices: np.ndarray, lower: float, upper: float
) -> list[float]:
    points = [lower, upper]
    points.extend(float(value) for value in vertices[:, 0] if lower < value < upper)
    axis = float(case.major_radius)
    if lower < axis < upper:
        points.append(axis)
    return sorted(set(points))


def _integrate_component(
    case: Any,
    vertices: np.ndarray,
    centre: np.ndarray,
    radial_power: int,
    vertical_power: int,
    *,
    density: bool,
    relative_tolerance: float,
) -> tuple[float, float]:
    plasma_lower, plasma_upper = case.boundary_midplane_radii()
    lower = max(float(np.min(vertices[:, 0])), float(plasma_lower))
    upper = min(float(np.max(vertices[:, 0])), float(plasma_upper))
    if upper <= lower:
        return 0.0, 0.0

    def integrand(radius: float) -> float:
        cell_bounds = _vertical_bounds(vertices, radius)
        if cell_bounds is None:
            return 0.0
        half_height = _plasma_half_height(case, radius)
        vertical_lower = max(cell_bounds[0], -half_height)
        vertical_upper = min(cell_bounds[1], half_height)
        if vertical_upper <= vertical_lower:
            return 0.0
        vertical_integral = (
            (vertical_upper - centre[1]) ** (vertical_power + 1)
            - (vertical_lower - centre[1]) ** (vertical_power + 1)
        ) / (vertical_power + 1)
        radial_factor = (radius - centre[0]) ** radial_power
        density_value = (
            float(case.toroidal_current_density(radius, 0.0)) if density else 1.0
        )
        return density_value * radial_factor * vertical_integral

    total = 0.0
    error = 0.0
    breaks = _radial_breaks(case, vertices, lower, upper)
    for first, second in zip(breaks, breaks[1:]):
        if second <= first:
            continue
        value, estimate = quad(
            integrand,
            first,
            second,
            epsabs=1.0e-12,
            epsrel=relative_tolerance,
            limit=300,
        )
        total += value
        error += estimate
    return float(total), float(error)


def _exact_cell_integral(
    case: Any, vertices: np.ndarray, centre: np.ndarray, relative_tolerance: float
) -> CellIntegral:
    area, _area_error = _integrate_component(
        case,
        vertices,
        centre,
        0,
        0,
        density=False,
        relative_tolerance=relative_tolerance,
    )
    powers = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    values = []
    estimates = []
    for radial_power, vertical_power in powers:
        value, estimate = _integrate_component(
            case,
            vertices,
            centre,
            radial_power,
            vertical_power,
            density=True,
            relative_tolerance=relative_tolerance,
        )
        values.append(value)
        estimates.append(estimate)
    return CellIntegral(area, np.asarray(values), np.asarray(estimates))


def _production_flux_coefficients(
    operator: Any, centroid_flux: np.ndarray, sample_flux: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the exact coefficient arrays used by the production route."""

    cell_count = len(operator.grid.coordinate)
    coefficient = np.full((cell_count, 6), np.nan, dtype=np.float64)
    centre = np.full((cell_count, 2), np.nan, dtype=np.float64)
    scale = np.full((cell_count, 2), np.nan, dtype=np.float64)
    value_pool = np.concatenate((centroid_flux, sample_flux))
    for stencil in operator._support_moment_stencils:
        ring = np.asarray(stencil.ring_centre, dtype=np.intp)
        gathered = value_pool[np.asarray(stencil.ring_gather_index, dtype=np.intp)]
        coefficient[ring] = np.einsum(
            "rps,rs->rp", np.asarray(stencil.ring_flux_weight), gathered
        )
        centre[ring] = np.asarray(stencil.ring_sampling_centre)
        scale[ring] = np.asarray(stencil.ring_coordinate_scale)
    if np.any(~np.isfinite(coefficient)):
        missing = np.flatnonzero(~np.all(np.isfinite(coefficient), axis=1))
        raise RuntimeError(f"production flux coefficients missing for cells {missing}")
    return coefficient, centre, scale


def _duffy_rule(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed degree-fifteen product rule used in production."""

    nodes, weights = np.polynomial.legendre.leggauss(8)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    points: list[np.ndarray] = []
    area_weights: list[float] = []
    for index in range(1, len(vertices) - 1):
        first, second, third = vertices[[0, index, index + 1]]
        edge_first = second - first
        edge_second = third - first
        cross = abs(edge_first[0] * edge_second[1] - edge_first[1] * edge_second[0])
        for radial, radial_weight in zip(unit_nodes, unit_weights, strict=True):
            for vertical, vertical_weight in zip(unit_nodes, unit_weights, strict=True):
                points.append(
                    first
                    + radial * edge_first
                    + (1.0 - radial) * vertical * edge_second
                )
                area_weights.append(
                    cross * (1.0 - radial) * radial_weight * vertical_weight
                )
    return np.asarray(points), np.asarray(area_weights)


def _quadratic_values(
    points: np.ndarray,
    coefficient: np.ndarray,
    centre: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    local = (points - centre) / scale
    radial, vertical = local[:, 0], local[:, 1]
    design = np.column_stack(
        (
            np.ones(len(points)),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        )
    )
    return design @ coefficient


def _production_cell_integrals(
    operator: Any,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Evaluate production density moments through its support and ring fit."""

    masks, topology, sample_flux, support = operator._support_partition(
        jnp.asarray(state)
    )
    centroid_flux = np.asarray(masks.psi_norm, dtype=np.float64)
    sample_flux_array = np.asarray(sample_flux, dtype=np.float64)
    coefficient, sampling_centre, coordinate_scale = _production_flux_coefficients(
        operator, centroid_flux, sample_flux_array
    )
    counts = np.asarray(support.vertex_count, dtype=np.intp)
    support_vertices = np.asarray(support.support_vertices, dtype=np.float64)
    moment_centre = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    values = np.zeros((len(counts), 6), dtype=np.float64)
    for cell, count in enumerate(counts):
        if count < 3:
            continue
        points, weights = _duffy_rule(support_vertices[cell, :count])
        fitted_flux = _quadratic_values(
            points,
            coefficient[cell],
            sampling_centre[cell],
            coordinate_scale[cell],
        )
        density = np.asarray(
            operator.source.core.current_density(
                jnp.asarray(points[:, 0]), jnp.asarray(fitted_flux)
            ),
            dtype=np.float64,
        )
        offset = points - moment_centre[cell]
        weighted = weights * density
        values[cell] = (
            np.sum(weighted),
            np.sum(weighted * offset[:, 0]),
            np.sum(weighted * offset[:, 1]),
            np.sum(weighted * offset[:, 0] ** 2),
            np.sum(weighted * offset[:, 0] * offset[:, 1]),
            np.sum(weighted * offset[:, 1] ** 2),
        )
    return (
        values,
        np.asarray(masks.profile_participation, dtype=bool),
        {
            "coefficient": coefficient,
            "sampling_centre": sampling_centre,
            "coordinate_scale": coordinate_scale,
            "axis_flux_wb": np.asarray(topology.axis_flux),
            "flux_span_wb": np.asarray(topology.flux_span),
        },
    )


def _ring_hessian(
    operator: Any, grid_flux: np.ndarray, sample_flux: np.ndarray
) -> np.ndarray:
    coefficient, _centre, scale = _production_flux_coefficients(
        operator, grid_flux, sample_flux
    )
    hessian = np.zeros((len(coefficient), 2, 2), dtype=np.float64)
    hessian[:, 0, 0] = 2.0 * coefficient[:, 3] / scale[:, 0] ** 2
    hessian[:, 0, 1] = coefficient[:, 4] / (scale[:, 0] * scale[:, 1])
    hessian[:, 1, 0] = hessian[:, 0, 1]
    hessian[:, 1, 1] = 2.0 * coefficient[:, 5] / scale[:, 1] ** 2
    return hessian


def _relative_error(actual: np.ndarray, exact: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(exact))
    if denominator == 0.0:
        return None
    return float(np.linalg.norm(actual - exact) / denominator)


def _order_record(actual: np.ndarray, exact: np.ndarray) -> dict[str, Any]:
    return {
        "production": actual,
        "exact": exact,
        "relative_error": _relative_error(actual, exact),
        "absolute_error_norm": float(np.linalg.norm(actual - exact)),
        "exact_norm": float(np.linalg.norm(exact)),
    }


def _moment_record(actual: np.ndarray, exact: np.ndarray) -> dict[str, Any]:
    return {
        "zeroth": _order_record(actual[:1], exact[:1]),
        "first": _order_record(actual[1:3], exact[1:3]),
        "second": _order_record(actual[3:6], exact[3:6]),
    }


def _summary(values: list[float | None]) -> dict[str, Any]:
    finite = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "maximum": None, "median": None, "p95": None}
    return {
        "count": int(finite.size),
        "maximum": float(np.max(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _outboard(coordinate: np.ndarray) -> bool:
    r_min, r_max, z_min, z_max = OUTBOARD_WINDOW
    return bool(r_min <= coordinate[0] <= r_max and z_min <= coordinate[1] <= z_max)


def _flux_error_metrics(
    coordinates: np.ndarray, image: np.ndarray, exact: np.ndarray
) -> dict[str, Any]:
    error = np.asarray(image) - np.asarray(exact)
    r_min, r_max, z_min, z_max = OUTBOARD_WINDOW
    selected = (
        (coordinates[:, 0] >= r_min)
        & (coordinates[:, 0] <= r_max)
        & (coordinates[:, 1] >= z_min)
        & (coordinates[:, 1] <= z_max)
    )
    squared = error**2
    total = float(np.sum(squared))
    fraction = float(np.sum(squared[selected]) / max(total, np.finfo(float).tiny))
    return {
        "squared_error_wb2": total,
        "rms_error_wb": float(np.sqrt(np.mean(squared))),
        "sup_error_wb": float(np.max(np.abs(error))),
        "outboard_window_squared_error_fraction": fraction,
        "outboard_window_node_count": int(np.count_nonzero(selected)),
        "outboard_window_domain_fraction": float(np.mean(selected)),
    }


def _topology_axis(
    operator: Any, state: np.ndarray, exact_axis: np.ndarray
) -> dict[str, Any]:
    read = certificate._topology(operator, state)
    axis = read["axis_rz_m"]
    return {
        "read": read,
        "axis_error_mm": (
            None
            if axis is None
            else float(1.0e3 * np.linalg.norm(np.asarray(axis) - exact_axis))
        ),
    }


def _physical_moments(values: np.ndarray) -> CellCurrentMoments:
    return CellCurrentMoments(
        jnp.asarray(values[:, 0]),
        jnp.asarray(values[:, 1]),
        jnp.asarray(values[:, 2]),
    )


def _replace_cut_moments(
    production: np.ndarray, exact: np.ndarray, cut: np.ndarray
) -> np.ndarray:
    result = np.array(production, copy=True)
    result[cut, :3] = exact[cut, :3]
    return result


def _zero_coefficients(
    coefficients: CellCurrentMoments, selection: np.ndarray | None
) -> CellCurrentMoments:
    radial = np.asarray(coefficients.radial_moment, dtype=np.float64)
    vertical = np.asarray(coefficients.vertical_moment, dtype=np.float64)
    if selection is None:
        radial = np.zeros_like(radial)
        vertical = np.zeros_like(vertical)
    else:
        radial = np.where(selection, 0.0, radial)
        vertical = np.where(selection, 0.0, vertical)
    return CellCurrentMoments(
        coefficients.cell_current,
        jnp.asarray(radial),
        jnp.asarray(vertical),
    )


def _variant_images(
    operator: Any,
    production: np.ndarray,
    exact: np.ndarray,
    cut: np.ndarray,
) -> dict[str, np.ndarray]:
    production_coefficients = operator.coupling_current_moments(
        _physical_moments(production)
    )
    substituted_coefficients = operator.coupling_current_moments(
        _physical_moments(_replace_cut_moments(production, exact, cut))
    )
    variants = {
        "production": production_coefficients,
        "exact_cut_cell_moments": substituted_coefficients,
        "second_order_zeroed_everywhere": _zero_coefficients(
            production_coefficients, None
        ),
        "second_order_zeroed_in_cut_cells": _zero_coefficients(
            production_coefficients, cut
        ),
    }
    external = np.asarray(operator.external(), dtype=np.float64)
    return {
        name: external
        + np.asarray(operator.current_moment_image(coefficients), dtype=np.float64)
        for name, coefficients in variants.items()
    }


def measure_rung(case_name: str, requested_cells: int) -> dict[str, Any]:
    """Measure one case-resolution rung and retain arrays needed for plotting."""

    started = perf_counter()
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the cut-cell discriminator requires JAX extended precision")
    if jax.default_backend() != "cpu":
        raise RuntimeError("the cut-cell discriminator requires the CPU backend")
    carrier_case, source_case, exact_case = certificate._case(case_name)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        requested_cells,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    exact_state = certificate._exact_state(case_name, exact_case, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    production, participation, _fit = _production_cell_integrals(
        empty_operator, exact_state
    )
    centres = np.asarray(empty_operator.moment_geometry.atomic_mesh.centroids)
    polygons = tuple(
        np.asarray(cell, dtype=np.float64) for cell in machine.cell_polygons
    )

    exact_integrals = [
        _exact_cell_integral(exact_case, polygon, centre, ADAPTIVE_RELATIVE_TOLERANCE)
        for polygon, centre in zip(polygons, centres, strict=True)
    ]
    exact_values = np.stack([item.moment for item in exact_integrals])
    exact_areas = np.asarray([item.area for item in exact_integrals])
    cell_areas = np.asarray(machine.area, dtype=np.float64)
    area_tolerance = 2.0e-11 * np.maximum(cell_areas, 1.0)
    cut = (exact_areas > area_tolerance) & (exact_areas < cell_areas - area_tolerance)
    interior = exact_areas >= cell_areas - area_tolerance

    agreement = []
    for cell in np.flatnonzero(cut):
        repeated = _exact_cell_integral(
            exact_case, polygons[cell], centres[cell], 1.0e-12
        )
        scale = np.maximum(np.abs(exact_values[cell]), 1.0)
        agreement.append(
            float(np.max(np.abs(repeated.moment - exact_values[cell]) / scale))
        )
    maximum_agreement = max(agreement, default=0.0)
    if maximum_agreement > AGREEMENT_BOUND:
        raise RuntimeError(
            "adaptive exact-moment agreement exceeded the declared bound: "
            f"{maximum_agreement:.17g}"
        )

    grid_count = len(machine.node)
    sample_offset = grid_count + len(machine.wall_node)
    production_hessian = _ring_hessian(
        empty_operator,
        exact_state[:grid_count],
        exact_state[sample_offset:],
    )
    _gradient, exact_hessian = certificate._exact_derivatives(
        case_name, exact_case, np.asarray(machine.node)
    )
    curvature_relative = np.asarray(
        [
            _relative_error(actual, reference)
            for actual, reference in zip(production_hessian, exact_hessian, strict=True)
        ],
        dtype=np.float64,
    )

    cut_records = []
    for cell in np.flatnonzero(cut):
        cut_records.append(
            {
                "cell": int(cell),
                "centre_rz_m": centres[cell],
                "outboard_window_member": _outboard(centres[cell]),
                "production_participation": bool(participation[cell]),
                "exact_area_fraction": float(exact_areas[cell] / cell_areas[cell]),
                "moments": _moment_record(production[cell], exact_values[cell]),
                "adaptive_error_estimate": exact_integrals[cell].error_estimate,
                "curvature": _order_record(
                    production_hessian[cell], exact_hessian[cell]
                ),
            }
        )

    interior_records = [
        {
            "cell": int(cell),
            "centre_rz_m": centres[cell],
            "outboard_window_member": _outboard(centres[cell]),
            "production_participation": bool(participation[cell]),
            "curvature": _order_record(production_hessian[cell], exact_hessian[cell]),
        }
        for cell in np.flatnonzero(interior)
    ]

    exact_coefficients = empty_operator.coupling_current_moments(
        _physical_moments(exact_values)
    )
    exact_internal = np.asarray(
        empty_operator.current_moment_image(exact_coefficients), dtype=np.float64
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, exact_state - exact_internal
    )
    images = _variant_images(operator, production, exact_values, cut)
    exact_axis = np.asarray(exact_case.magnetic_axis, dtype=np.float64)
    image_records = {}
    for name, image in images.items():
        grid_image = image[:grid_count]
        image_records[name] = {
            "flux_error": _flux_error_metrics(
                np.asarray(machine.node), grid_image, exact_state[:grid_count]
            ),
            "axis": _topology_axis(operator, image, exact_axis),
            "state_sha256_binary64": _array_digest(image),
        }

    moment_errors = {
        order: _summary(
            [record["moments"][order]["relative_error"] for record in cut_records]
        )
        for order in ("zeroth", "first", "second")
    }
    row = {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": grid_count,
        "source_revision": _source_revision(),
        "lane": _lane_receipt(),
        "contract": {
            "exact_density": "R p_prime + FF_prime / (mu0 R)",
            "exact_region": "cell polygon intersected with the analytic Solovev plasma",
            "adaptive_relative_tolerance": ADAPTIVE_RELATIVE_TOLERANCE,
            "repeat_agreement_bound": AGREEMENT_BOUND,
            "outboard_window_rz_m": list(OUTBOARD_WINDOW),
            "production_rule": (
                "own-cell quadratic with fixed degree-fifteen Duffy product rule"
            ),
        },
        "cell_census": {
            "boundary_cut": int(np.count_nonzero(cut)),
            "interior": int(np.count_nonzero(interior)),
            "exterior": int(len(cut) - np.count_nonzero(cut | interior)),
            "production_participating_boundary_cut": int(
                np.count_nonzero(cut & participation)
            ),
            "outboard_boundary_cut": int(
                sum(record["outboard_window_member"] for record in cut_records)
            ),
        },
        "adaptive_quadrature": {
            "maximum_repeat_relative_disagreement": maximum_agreement,
            "passed": maximum_agreement <= AGREEMENT_BOUND,
        },
        "cut_cells": cut_records,
        "interior_cells": interior_records,
        "summaries": {
            "cut_cell_moment_relative_error": moment_errors,
            "cut_cell_curvature_relative_error": _summary(
                curvature_relative[cut].tolist()
            ),
            "interior_cell_curvature_relative_error": _summary(
                curvature_relative[interior].tolist()
            ),
        },
        "images": image_records,
        "plot_data": {
            "node_rz_m": np.asarray(machine.node),
            "wall_rz_m": np.asarray(machine.wall_node),
            "exact_state_wb": exact_state[:grid_count],
            "exact_axis_rz_m": exact_axis,
            "variant_state_wb": {
                name: image[:grid_count] for name, image in images.items()
            },
        },
        "elapsed_seconds": perf_counter() - started,
    }
    print(
        f"CUT_CELL_RESULT case={case_name} requested={requested_cells} "
        f"cut={len(cut_records)} agreement={maximum_agreement:.3e} "
        f"production_axis_mm={image_records['production']['axis']['axis_error_mm']}",
        flush=True,
    )
    return row


def _shared_error_levels(rows: list[dict[str, Any]]) -> np.ndarray:
    absolute = []
    for row in rows:
        exact = np.asarray(row["plot_data"]["exact_state_wb"], dtype=np.float64)
        for state in row["plot_data"]["variant_state_wb"].values():
            absolute.append(np.abs(np.asarray(state, dtype=np.float64) - exact))
    values = np.concatenate(absolute)
    nonzero = values[values > 0.0]
    if nonzero.size == 0:
        return np.asarray([np.finfo(float).tiny])
    lower = max(float(np.percentile(nonzero, 5.0)), np.finfo(float).tiny)
    upper = float(np.max(nonzero))
    return np.geomspace(lower, upper, 11) if upper > lower else np.asarray([upper])


def _shared_flux_levels(rows: list[dict[str, Any]]) -> np.ndarray:
    values = np.concatenate(
        [np.asarray(row["plot_data"]["exact_state_wb"]) for row in rows]
    )
    return np.linspace(float(np.min(values)), float(np.max(values)), 12)[1:-1]


def _draw_figure(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    error_levels = _shared_error_levels(rows)
    flux_levels = _shared_flux_levels(rows)
    figure, axes = plt.subplots(
        len(rows), len(VARIANTS), figsize=(15.0, 9.2), constrained_layout=True
    )
    for row_index, row in enumerate(rows):
        plot = row["plot_data"]
        node = np.asarray(plot["node_rz_m"], dtype=np.float64)
        wall = np.asarray(plot["wall_rz_m"], dtype=np.float64)
        exact = np.asarray(plot["exact_state_wb"], dtype=np.float64)
        exact_axis = np.asarray(plot["exact_axis_rz_m"], dtype=np.float64)
        for column, name in enumerate(VARIANTS):
            axes_cell = poloidal_axes(axes[row_index, column])
            image = np.asarray(plot["variant_state_wb"][name], dtype=np.float64)
            error = np.abs(image - exact)
            axes_cell.tricontour(
                node[:, 0],
                node[:, 1],
                np.maximum(error, error_levels[0]),
                levels=error_levels,
                colors="firebrick",
                linewidths=0.8,
            )
            axes_cell.tricontour(
                node[:, 0],
                node[:, 1],
                exact,
                levels=flux_levels,
                colors="dimgray",
                linewidths=0.42,
            )
            axes_cell.tricontour(
                node[:, 0],
                node[:, 1],
                image,
                levels=flux_levels,
                colors="royalblue",
                linewidths=0.42,
                linestyles="dashed",
            )
            poloidal.draw_wall(axes_cell, wall[:, 0], wall[:, 1])
            image_axis = row["images"][name]["axis"]["read"]["axis_rz_m"]
            reference_style = DEFAULT_INK.variant(axis_marker="^", axis_color="dimgray")
            image_style = DEFAULT_INK.variant(axis_marker="^", axis_color="royalblue")
            poloidal.draw_nulls(
                axes_cell,
                magnetic_axis=exact_axis,
                style=reference_style,
                contain=wall,
            )
            poloidal.draw_nulls(
                axes_cell,
                magnetic_axis=image_axis,
                style=image_style,
                contain=wall,
            )
            axis_error = row["images"][name]["axis"]["axis_error_mm"]
            outboard = row["images"][name]["flux_error"][
                "outboard_window_squared_error_fraction"
            ]
            axis_text = "unavailable" if axis_error is None else f"{axis_error:.3g} mm"
            axes_cell.set_title(
                f"{name.replace('_', ' ')}\naxis {axis_text}; outboard {outboard:.3g}",
                fontsize=7,
            )
        axes[row_index, 0].text(
            -0.04,
            0.5,
            f"{row['case']}\n{row['requested_cells']} requested",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=8,
        )
    figure.suptitle(
        "Solovev exact-state flux images\n"
        "red: |image - exact| shared levels; grey: exact flux; "
        "blue dashed: image flux; "
        "grey/blue triangles: exact/imaged axes",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return {
        "path": str(output.relative_to(ROOT)),
        "project_src": "/nova/figures/uniform-cell-clip-and-coupling/cut-cell-moments/"
        + output.name,
        "shared_absolute_error_levels_wb": error_levels,
        "shared_flux_levels_wb": flux_levels,
    }


def _fraction_removed(production: float, candidate: float) -> float:
    return float(1.0 - candidate / max(production, np.finfo(float).tiny))


def aggregate(output: Path = RECEIPT) -> dict[str, Any]:
    """Combine the three independently measured rungs and render shared maps."""

    rows = [
        json.loads(_part_path(case_name, cells).read_text(encoding="utf-8"))
        for case_name, cells in CASE_REQUESTS
    ]
    revisions = {row["source_revision"] for row in rows}
    if len(revisions) != 1:
        raise RuntimeError(f"rungs do not share one source revision: {revisions}")
    job_ids = set()
    for row in rows:
        lane = row["lane"]
        if (
            lane["execution"] != "slurm"
            or lane["partition"] != "all_debug"
            or lane["jax_platform"] != "cpu"
            or not lane["jax_enable_x64"]
        ):
            raise RuntimeError("every rung must be an all_debug CPU-x64 measurement")
        job_ids.add(lane["job_id"])
        if not row["adaptive_quadrature"]["passed"]:
            raise RuntimeError("an adaptive exact-moment agreement gate failed")
    if len(job_ids) != 1:
        raise RuntimeError("the three rungs must share one scheduler allocation")

    figure = _draw_figure(rows, ERROR_FIGURE)
    compact_rows = []
    for row in rows:
        compact = dict(row)
        compact.pop("plot_data")
        compact_rows.append(compact)

    headline_rows = {}
    for row in compact_rows:
        production_error = row["images"]["production"]["flux_error"][
            "squared_error_wb2"
        ]
        variant_summary = {}
        for name in VARIANTS:
            metrics = row["images"][name]
            variant_error = metrics["flux_error"]["squared_error_wb2"]
            variant_summary[name] = {
                "squared_error_wb2": variant_error,
                "fraction_of_production_squared_error_removed": _fraction_removed(
                    production_error, variant_error
                ),
                "outboard_window_squared_error_fraction": metrics["flux_error"][
                    "outboard_window_squared_error_fraction"
                ],
                "axis_error_mm": metrics["axis"]["axis_error_mm"],
            }
        headline_rows[_case_key(row["case"], row["requested_cells"])] = {
            "boundary_cut_cells": row["cell_census"]["boundary_cut"],
            "outboard_boundary_cut_cells": row["cell_census"]["outboard_boundary_cut"],
            "cut_cell_moment_relative_error": row["summaries"][
                "cut_cell_moment_relative_error"
            ],
            "cut_cell_curvature_relative_error": row["summaries"][
                "cut_cell_curvature_relative_error"
            ],
            "interior_cell_curvature_relative_error": row["summaries"][
                "interior_cell_curvature_relative_error"
            ],
            "images": variant_summary,
        }

    receipt = {
        "schema": "nova.solovev-cut-cell-moments",
        "completed": True,
        "measurement_driver_revision": revisions.pop(),
        "contract": {
            "requests": [
                {"case": case_name, "requested_cells": cells}
                for case_name, cells in CASE_REQUESTS
            ],
            "one_rung_per_fresh_process": True,
            "one_scheduler_job": True,
            "lane": "all_debug CPU float64",
            "solver_source_changed": False,
            "variants": list(VARIANTS),
            "outboard_window_rz_m": list(OUTBOARD_WINDOW),
            "adaptive_relative_tolerance": ADAPTIVE_RELATIVE_TOLERANCE,
            "adaptive_repeat_agreement_bound": AGREEMENT_BOUND,
        },
        "rows": compact_rows,
        "figure": figure,
        "headline": headline_rows,
    }
    _write_json(output, receipt)
    for case_name, cells in CASE_REQUESTS:
        _part_path(case_name, cells).unlink()
    parts = OUTPUT_ROOT / "parts"
    parts.rmdir()
    return receipt


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=tuple(sorted({name for name, _cells in CASE_REQUESTS}))
    )
    parser.add_argument("--requested-cells", type=int, choices=(-110, -300))
    parser.add_argument("--part", type=Path)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--output", type=Path, default=RECEIPT)
    return parser.parse_args()


def main() -> None:
    configure_dtypes()
    arguments = _parse()
    if arguments.aggregate:
        receipt = aggregate(arguments.output)
        print(json.dumps(receipt["headline"], sort_keys=True), flush=True)
        print("CUT_CELL_AGGREGATE_EXIT=0", flush=True)
        return
    if arguments.case is None or arguments.requested_cells is None:
        raise SystemExit("one --case and --requested-cells pair is required")
    if (arguments.case, arguments.requested_cells) not in CASE_REQUESTS:
        raise SystemExit("the requested case-resolution pair is outside the contract")
    part = arguments.part or _part_path(arguments.case, arguments.requested_cells)
    row = measure_rung(arguments.case, arguments.requested_cells)
    _write_json(part, row)
    print(f"CUT_CELL_ROW_EXIT=0 part={part}", flush=True)


if __name__ == "__main__":
    main()
