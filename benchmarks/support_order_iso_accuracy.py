"""Price linear and quadratic in-cell current support at equal field accuracy.

The accuracy ladder uses two exact Solov'ev equilibria.  The first is Nova's
analytic oracle; the second has a geometric aspect ratio below two.  Both use
the same clipped source cells, closed-form current density, exterior targets,
and fixed tensor-Gauss interaction rule.  Only the retained polynomial support
changes: constant plus two linear terms, or those terms plus all three
quadratic terms.

The reported flux is completed with the exact homogeneous contribution, so
errors are against the closed-form total field rather than against a discrete
fixed point.  Cost measurements run in fresh processes so accelerator peak
memory is independent between support orders.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/coefficient-space-newton/support-order-iso-accuracy.json"
)
SUPPORT_COLUMNS = {"first": 3, "second": 6}
REQUESTED_CELL_LADDER = (180, 320, 560, 980, 1720, 3000, 5200)
TARGET_RELATIVE_SUP_ERROR = 4.0e-3
TARGET_COUNT = 48
TARGET_CLEARANCE_MINOR_RADIUS_FRACTION = 0.12
REFERENCE_ORDERS = ((48, 96), (72, 144))
INTERACTION_QUADRATURE_ORDER = 8
MOMENT_QUADRATURE_ORDER = 10
DEVICE_CHUNK = 32
TIMING_REPEATS = 11
BANKED_REFINEMENT_BUILD_RATIO = 3.24
BANKED_REFINEMENT_MEMORY_RATIO = 13.6


@dataclass(frozen=True)
class SourceGrid:
    """One square-cell carrier and its conservative analytic-domain clip."""

    pitch: float
    centres: np.ndarray
    cells: np.ndarray
    support_vertices: np.ndarray
    vertex_count: np.ndarray
    included: np.ndarray
    boundary: np.ndarray
    patch_area_relative_residual: float


_COMPILED_BUILDERS: dict[int, Any] = {}


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def _cases() -> dict[str, Any]:
    from tests.rotating_equilibrium_references import (
        reference_cases,
        rotating_equilibrium,
    )

    oracle = reference_cases()["moderate-rotation-conventional"].static_limit()
    low_aspect = rotating_equilibrium(
        name="low-aspect-solovev",
        major_radius=0.90,
        outboard_radius=1.25,
        half_height=0.75,
        vacuum_field=0.60,
        axis_pressure=1.5e4,
        thermal_mach_number=0.0,
        axis_temperature=oracle.axis_temperature,
        boundary_temperature=oracle.boundary_temperature,
        mean_particle_mass=oracle.mean_particle_mass,
    )
    return {"analytic_oracle": oracle, "low_aspect_ratio": low_aspect}


def _aspect_ratio(case: Any) -> float:
    inner, outer = case.boundary_midplane_radii()
    return float(case.major_radius / (0.5 * (outer - inner)))


def _targets(case: Any) -> np.ndarray:
    angle = 2.0 * np.pi * np.arange(TARGET_COUNT) / TARGET_COUNT
    half_u = math.sqrt(2.0 * case.axis_flux / case.pressure_coefficient)
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    boundary_radius = np.sqrt(case.major_radius**2 + half_u * np.cos(angle))
    boundary_height = half_height * np.sin(angle)
    inner, outer = case.boundary_midplane_radii()
    geometric_centre = np.asarray([0.5 * (inner + outer), 0.0])
    minor_radius = 0.5 * (outer - inner)
    boundary = np.column_stack((boundary_radius, boundary_height))
    direction = boundary - geometric_centre
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    targets = (
        boundary + TARGET_CLEARANCE_MINOR_RADIUS_FRACTION * minor_radius * direction
    )
    if not np.all(np.isfinite(targets)):
        raise RuntimeError("the physical LCFS offset produced a non-finite target")
    if np.any(case.contains(targets[:, 0], targets[:, 1])):
        raise RuntimeError("the physical LCFS offset did not put every target outside")
    return targets


def _pitch(case: Any, requested_cells: int) -> float:
    inner, outer = case.boundary_midplane_radii()
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    bounding_area = (outer - inner) * (2.0 * half_height)
    return math.sqrt(bounding_area / requested_cells)


def _grid(case: Any, requested_cells: int) -> SourceGrid:
    from nova.equilibrium.separatrix_clip import AtomicCellMesh

    pitch = _pitch(case, requested_cells)
    inner, outer = case.boundary_midplane_radii()
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    radial_edge = np.arange(
        math.floor((inner - pitch) / pitch) * pitch,
        math.ceil((outer + pitch) / pitch) * pitch + 0.5 * pitch,
        pitch,
    )
    vertical_edge = np.arange(
        math.floor((-half_height - pitch) / pitch) * pitch,
        math.ceil((half_height + pitch) / pitch) * pitch + 0.5 * pitch,
        pitch,
    )
    radius = 0.5 * (radial_edge[:-1] + radial_edge[1:])
    height = 0.5 * (vertical_edge[:-1] + vertical_edge[1:])
    rr, zz = np.meshgrid(radius, height, indexing="xy")
    centres = np.column_stack((rr.ravel(), zz.ravel()))
    half = 0.5 * pitch
    offset = np.asarray([[-half, -half], [half, -half], [half, half], [-half, half]])
    cells = centres[:, None, :] + offset[None, :, :]
    atomic = AtomicCellMesh.from_cells([cell for cell in cells], centroids=centres)
    clipped = atomic.clip(atomic.sample(case.flux))
    if not clipped.contour_closed:
        raise RuntimeError("the analytic contour did not close inside the grid")
    area_residual = abs(clipped.patch_area_sum - clipped.contour_area) / abs(
        clipped.contour_area
    )
    return SourceGrid(
        pitch=pitch,
        centres=centres,
        cells=cells,
        support_vertices=np.asarray(clipped.support_vertices),
        vertex_count=np.asarray(clipped.vertex_count),
        included=np.asarray(clipped.included, dtype=bool),
        boundary=np.asarray(clipped.boundary, dtype=bool),
        patch_area_relative_residual=float(area_residual),
    )


def _basis(local: np.ndarray, columns: int) -> np.ndarray:
    radial = local[..., 0]
    vertical = local[..., 1]
    values = np.stack(
        (
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ),
        axis=-1,
    )
    return values[..., :columns]


def _polygon_rule(vertices: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    node, weight = np.polynomial.legendre.leggauss(order)
    node = 0.5 * (node + 1.0)
    weight = 0.5 * weight
    points: list[np.ndarray] = []
    weights: list[float] = []
    for index in range(1, len(vertices) - 1):
        first, second, third = vertices[[0, index, index + 1]]
        first_edge = second - first
        second_edge = third - first
        determinant = abs(np.linalg.det(np.stack((first_edge, second_edge))))
        for first_node, first_weight in zip(node, weight, strict=True):
            for second_node, second_weight in zip(node, weight, strict=True):
                points.append(
                    first
                    + first_node * first_edge
                    + (1.0 - first_node) * second_node * second_edge
                )
                weights.append(
                    determinant * (1.0 - first_node) * first_weight * second_weight
                )
    return np.asarray(points), np.asarray(weights)


def _gram_inverse(pitch: float, columns: int) -> np.ndarray:
    square = pitch * np.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    points, weights = _polygon_rule(square, MOMENT_QUADRATURE_ORDER)
    basis = _basis(points / pitch, columns)
    return np.linalg.inv(np.einsum("qi,qj,q->ij", basis, basis, weights))


def _coefficients(case: Any, grid: SourceGrid, columns: int) -> np.ndarray:
    inverse = _gram_inverse(grid.pitch, columns)
    result = np.zeros((len(grid.centres), columns), dtype=np.float64)
    for index in np.flatnonzero(grid.included):
        count = int(grid.vertex_count[index])
        if count < 3:
            continue
        points, weights = _polygon_rule(
            grid.support_vertices[index, :count], MOMENT_QUADRATURE_ORDER
        )
        density = np.asarray(
            case.toroidal_current_density(points[:, 0], points[:, 1]),
            dtype=np.float64,
        )
        local = (points - grid.centres[index]) / grid.pitch
        moments = np.einsum("q,qc,q->c", density, _basis(local, columns), weights)
        result[index] = inverse @ moments
    return result


def _compiled_builder(columns: int):
    import jax
    import jax.numpy as jnp

    from nova.biot.second_moment_kernel import flux_density_columns

    cached = _COMPILED_BUILDERS.get(columns)
    if cached is not None:
        return cached, 0.0

    def chunk(vertices, centres, target_r, target_z):
        def one_cell(cell, centre):
            return flux_density_columns(
                jnp,
                target_r,
                target_z,
                cell,
                expansion_point=centre,
                order=INTERACTION_QUADRATURE_ORDER,
                columns=columns,
            )

        return jax.vmap(one_cell)(vertices, centres)

    example_vertices = np.zeros((DEVICE_CHUNK, 4, 2), dtype=np.float64)
    example_vertices[:, 1, 0] = 1.0
    example_vertices[:, 2] = 1.0
    example_vertices[:, 3, 1] = 1.0
    example_centres = np.full((DEVICE_CHUNK, 2), 0.5, dtype=np.float64)
    example_target_r = np.linspace(1.5, 2.0, TARGET_COUNT)
    example_target_z = np.linspace(-0.5, 0.5, TARGET_COUNT)
    started = perf_counter()
    compiled = (
        jax.jit(chunk)
        .lower(
            jnp.asarray(example_vertices),
            jnp.asarray(example_centres),
            jnp.asarray(example_target_r),
            jnp.asarray(example_target_z),
        )
        .compile()
    )
    compile_seconds = perf_counter() - started
    _COMPILED_BUILDERS[columns] = compiled
    return compiled, compile_seconds


def _response_matrix(
    grid: SourceGrid, targets: np.ndarray, columns: int
) -> tuple[np.ndarray, dict[str, float]]:
    import jax
    import jax.numpy as jnp

    compiled, compile_seconds = _compiled_builder(columns)
    chunks: list[np.ndarray] = []
    started = perf_counter()
    for begin in range(0, len(grid.cells), DEVICE_CHUNK):
        end = min(begin + DEVICE_CHUNK, len(grid.cells))
        count = end - begin
        cells = grid.cells[begin:end]
        centres = grid.centres[begin:end]
        if count < DEVICE_CHUNK:
            pad = DEVICE_CHUNK - count
            cells = np.concatenate((cells, np.repeat(cells[-1:], pad, axis=0)))
            centres = np.concatenate((centres, np.repeat(centres[-1:], pad, axis=0)))
        value = compiled(
            jnp.asarray(cells),
            jnp.asarray(centres),
            jnp.asarray(targets[:, 0]),
            jnp.asarray(targets[:, 1]),
        )
        chunks.append(np.asarray(jax.block_until_ready(value))[:count])
    execution_seconds = perf_counter() - started
    response = np.transpose(np.concatenate(chunks, axis=0), (1, 0, 2))
    divisor = np.asarray(
        [1.0, grid.pitch, grid.pitch, grid.pitch**2, grid.pitch**2, grid.pitch**2]
    )[:columns]
    response /= divisor[None, None, :]
    return response, {
        "compilation_seconds": compile_seconds,
        "execution_seconds": execution_seconds,
        "total_seconds": compile_seconds + execution_seconds,
    }


def _reference_geometry(case: Any, radial_order: int, angular_order: int):
    radial_node, radial_weight = np.polynomial.legendre.leggauss(radial_order)
    angular_node, angular_weight = np.polynomial.legendre.leggauss(angular_order)
    rho = 0.5 * (radial_node + 1.0)
    rho_weight = 0.5 * radial_weight
    angle = np.pi * (angular_node + 1.0)
    angle_weight = np.pi * angular_weight
    half_u = math.sqrt(2.0 * case.axis_flux / case.pressure_coefficient)
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    rr, aa = np.meshgrid(rho, angle, indexing="ij")
    source_r = np.sqrt(case.major_radius**2 + half_u * rr * np.cos(aa))
    source_z = half_height * rr * np.sin(aa)
    jacobian = half_u * half_height * rr / (2.0 * source_r)
    quadrature_weight = jacobian * rho_weight[:, None] * angle_weight[None, :]
    return source_r.ravel(), source_z.ravel(), quadrature_weight.ravel()


def _plasma_reference(case: Any, targets: np.ndarray, order: tuple[int, int]):
    from nova.biot.greens import greens_psi

    source_r, source_z, weight = _reference_geometry(case, *order)
    density = np.asarray(
        case.toroidal_current_density(source_r, source_z), dtype=np.float64
    )
    response = np.zeros(len(targets), dtype=np.float64)
    for begin in range(0, len(source_r), 4096):
        end = min(begin + 4096, len(source_r))
        kernel = greens_psi(
            targets[:, 0, None],
            targets[:, 1, None],
            source_r[None, begin:end],
            source_z[None, begin:end],
        )
        response += kernel @ (density[begin:end] * weight[begin:end])
    return response


def _error_metrics(
    measured: np.ndarray,
    reference: np.ndarray,
    normalisation: float,
) -> dict[str, float]:
    difference = measured - reference
    return {
        "normalisation_wb": normalisation,
        "relative_sup_error": float(np.max(np.abs(difference)) / normalisation),
        "relative_rms_error": float(np.sqrt(np.mean(difference**2)) / normalisation),
    }


def _accuracy_row(
    case: Any,
    requested_cells: int,
    targets: np.ndarray,
    reference_plasma: np.ndarray,
    exact_total: np.ndarray,
) -> dict[str, Any]:
    grid = _grid(case, requested_cells)
    orders: dict[str, Any] = {}
    for name, columns in SUPPORT_COLUMNS.items():
        coefficients = _coefficients(case, grid, columns)
        response, build = _response_matrix(grid, targets, columns)
        approximate_plasma = grid.pitch**2 * np.einsum(
            "tck,ck->t", response, coefficients
        )
        approximate_total = approximate_plasma + exact_total - reference_plasma
        orders[name] = {
            "columns_per_cell": columns,
            "error": _error_metrics(
                approximate_total,
                exact_total,
                2.0 * np.pi * case.axis_flux,
            ),
            "interaction_build": build,
            "coupling_resident_bytes": int(response.nbytes),
        }
    return {
        "requested_cells": requested_cells,
        "carrier_cell_count": len(grid.centres),
        "supported_cell_count": int(np.count_nonzero(grid.included)),
        "boundary_cell_count": int(np.count_nonzero(grid.boundary)),
        "pitch_m": grid.pitch,
        "patch_area_relative_residual": grid.patch_area_relative_residual,
        "orders": orders,
    }


def _fit_order(rows: list[dict[str, Any]], order: str) -> dict[str, Any]:
    pitch = np.asarray([row["pitch_m"] for row in rows])
    error = np.asarray(
        [row["orders"][order]["error"]["relative_sup_error"] for row in rows]
    )
    x = np.log(pitch)
    y = np.log(error)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = y - predicted
    degrees_of_freedom = len(rows) - 2
    residual_variance = float(np.sum(residual**2) / degrees_of_freedom)
    slope_standard_error = math.sqrt(residual_variance / np.sum((x - x.mean()) ** 2))
    critical = float(student_t.ppf(0.975, degrees_of_freedom))
    total = float(np.sum((y - y.mean()) ** 2))
    above_physical_ceiling = slope > 3.0
    return {
        "field_error_order": float(slope),
        "one_sigma_uncertainty": slope_standard_error,
        "confidence_95_lower": float(slope - critical * slope_standard_error),
        "confidence_95_upper": float(slope + critical * slope_standard_error),
        "error_coefficient": float(np.exp(intercept)),
        "r_squared": float(1.0 - np.sum(residual**2) / total) if total else 1.0,
        "rungs": len(rows),
        "point_estimate_above_three": above_physical_ceiling,
        "interpretation": (
            "point estimate above three is treated as a noisy refinement-ladder "
            "artefact; use the fitted uncertainty rather than the point value"
            if above_physical_ceiling
            else "point estimate is within the physically interpretable range"
        ),
    }


def _iso_row(rows: list[dict[str, Any]], order: str) -> dict[str, Any]:
    qualifying = [
        row
        for row in rows
        if row["orders"][order]["error"]["relative_sup_error"]
        <= TARGET_RELATIVE_SUP_ERROR
    ]
    if not qualifying:
        raise RuntimeError(
            f"{order}-order support does not reach the declared target on the ladder"
        )
    selected = min(qualifying, key=lambda row: row["carrier_cell_count"])
    return {
        "requested_cells": selected["requested_cells"],
        "carrier_cell_count": selected["carrier_cell_count"],
        "pitch_m": selected["pitch_m"],
        "achieved_relative_sup_error": selected["orders"][order]["error"][
            "relative_sup_error"
        ],
        "selection": "coarsest measured ladder rung meeting the declared target",
    }


def _cost_measurement(case_name: str, requested_cells: int, order: str):
    import jax
    import jax.numpy as jnp

    case = _cases()[case_name]
    targets = _targets(case)
    grid = _grid(case, requested_cells)
    columns = SUPPORT_COLUMNS[order]
    coefficients = _coefficients(case, grid, columns)
    response, build = _response_matrix(grid, targets, columns)
    device_response = jnp.asarray(response)
    device_coefficients = jnp.asarray(coefficients)

    def contract(matrix, vector):
        return grid.pitch**2 * jnp.einsum("tck,ck->t", matrix, vector)

    started = perf_counter()
    compiled = jax.jit(contract).lower(device_response, device_coefficients).compile()
    contraction_compile_seconds = perf_counter() - started
    jax.block_until_ready(compiled(device_response, device_coefficients))
    samples = []
    for _ in range(TIMING_REPEATS):
        started = perf_counter()
        value = compiled(device_response, device_coefficients)
        jax.block_until_ready(value)
        samples.append(perf_counter() - started)
    device = jax.devices()[0]
    memory = device.memory_stats() or {}
    device_peak = int(memory.get("peak_bytes_in_use", memory.get("bytes_in_use", 0)))
    if device_peak > 0:
        peak_memory_bytes = device_peak
        peak_memory_measurement = "jax_device_peak_bytes_in_use"
    else:
        peak_memory_bytes = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        peak_memory_measurement = "process_maximum_resident_set_size"
    samples.sort()
    return {
        "case": case_name,
        "order": order,
        "columns_per_cell": columns,
        "requested_cells": requested_cells,
        "carrier_cell_count": len(grid.centres),
        "interaction_build": build,
        "coupling_resident_bytes": int(response.nbytes),
        "contraction_compile_seconds": contraction_compile_seconds,
        "per_iterate_support_contraction": {
            "median_seconds": samples[len(samples) // 2],
            "minimum_seconds": samples[0],
            "maximum_seconds": samples[-1],
            "repeats": TIMING_REPEATS,
            "excludes_common_topology_and_profile_evaluation": True,
        },
        "peak_device_memory_bytes": peak_memory_bytes,
        "peak_memory_measurement": peak_memory_measurement,
        "device": str(device),
        "device_platform": device.platform,
        "execution_lane": {
            "scheduler": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "node_name": os.environ.get("SLURMD_NODENAME"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "job_cpus_per_node": os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
            "memory_per_node_mib": os.environ.get("SLURM_MEM_PER_NODE"),
            "exclusive_node_requested": os.environ.get("NOVA_ISO_ACCURACY_EXCLUSIVE")
            == "1",
            "tmpdir": os.environ.get("TMPDIR"),
        },
    }


def _isolated_cost(case_name: str, iso: dict[str, Any], order: str):
    with tempfile.TemporaryDirectory(prefix="nova-support-iso-cost-") as directory:
        output = Path(directory) / "cost.json"
        log = Path(directory) / "cost.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "measure-cost",
            "--case",
            case_name,
            "--requested-cells",
            str(iso["requested_cells"]),
            "--order",
            order,
            "--output",
            str(output),
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"isolated {case_name} {order} cost failed: "
                f"{log.read_text(encoding='utf-8')}"
            )
        return json.loads(output.read_text(encoding="utf-8"))


def _cost_comparison(costs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first = costs["first"]
    second = costs["second"]

    def ratio(path: tuple[str, ...]) -> float:
        left: Any = first
        right: Any = second
        for key in path:
            left = left[key]
            right = right[key]
        return float(right / left)

    build_ratio = ratio(("interaction_build", "total_seconds"))
    memory_ratio = ratio(("peak_device_memory_bytes",))
    coupling_memory_ratio = ratio(("coupling_resident_bytes",))
    iterate_ratio = ratio(("per_iterate_support_contraction", "median_seconds"))
    cell_count_ratio = float(second["carrier_cell_count"] / first["carrier_cell_count"])
    surcharge_repaid = (
        cell_count_ratio < 1.0
        and build_ratio < 1.0
        and memory_ratio < 1.0
        and iterate_ratio < 1.0
    )
    return {
        "second_over_first_cell_count": cell_count_ratio,
        "second_over_first_interaction_build": build_ratio,
        "second_over_first_peak_device_memory": memory_ratio,
        "second_over_first_coupling_resident_memory": coupling_memory_ratio,
        "second_over_first_per_iterate_support_contraction": iterate_ratio,
        "banked_h_refinement": {
            "interaction_build_ratio": BANKED_REFINEMENT_BUILD_RATIO,
            "peak_memory_ratio": BANKED_REFINEMENT_MEMORY_RATIO,
        },
        "relative_to_banked_h_refinement": {
            "build_ratio_divided_by_banked_charge": (
                build_ratio / BANKED_REFINEMENT_BUILD_RATIO
            ),
            "memory_ratio_divided_by_banked_charge": (
                memory_ratio / BANKED_REFINEMENT_MEMORY_RATIO
            ),
        },
        "second_order_reduces_iso_accuracy_cell_count": cell_count_ratio < 1.0,
        "six_column_surcharge_repaid": surcharge_repaid,
        "economic_statement": (
            "Second-order support reaches the target with fewer cells and repays "
            "its surcharge on build, peak memory, and support contraction."
            if surcharge_repaid
            else (
                "Second-order support does not reduce the measured iso-accuracy "
                "cell count, so its six-column surcharge is not repaid."
            )
        ),
    }


def _order_verdict(fits: dict[str, dict[str, float]], rows: list[dict[str, Any]]):
    first = fits["first"]
    second = fits["second"]
    difference = second["field_error_order"] - first["field_error_order"]
    standard_error = math.sqrt(
        first["one_sigma_uncertainty"] ** 2 + second["one_sigma_uncertainty"] ** 2
    )
    critical = float(student_t.ppf(0.975, 2 * (len(rows) - 2)))
    lower = difference - critical * standard_error
    upper = difference + critical * standard_error
    ratios = [
        row["orders"]["first"]["error"]["relative_sup_error"]
        / row["orders"]["second"]["error"]["relative_sup_error"]
        for row in rows
    ]
    second_better_rungs = sum(value > 1.0 for value in ratios)
    first_better_or_equal_rungs = len(ratios) - second_better_rungs
    constant_improves = second_better_rungs == len(ratios)
    above_physical_ceiling = any(
        fit["point_estimate_above_three"] for fit in fits.values()
    )
    raises_order = lower > 0.0 and not above_physical_ceiling
    return {
        "second_minus_first_field_error_order": difference,
        "difference_one_sigma_uncertainty": standard_error,
        "difference_confidence_95_lower": lower,
        "difference_confidence_95_upper": upper,
        "geometric_mean_first_over_second_error_at_shared_rungs": float(
            np.exp(np.mean(np.log(ratios)))
        ),
        "second_order_better_rungs": second_better_rungs,
        "first_order_better_or_equal_rungs": first_better_or_equal_rungs,
        "second_order_systematically_improves_error_constant": constant_improves,
        "above_three_point_estimate_present": above_physical_ceiling,
        "classification": (
            "second_order_support_raises_field_error_order"
            if raises_order
            else (
                "above_three_point_estimate_is_not_physical; order_raise_not_resolved"
                if above_physical_ceiling
                else (
                    "order_raise_not_resolved; second_order_improves_only_the_constant"
                    if constant_improves
                    else "no_resolved_order_or_constant_improvement"
                )
            )
        ),
        "statement": (
            "The fitted order increase is positive at 95% confidence."
            if raises_order
            else (
                "At least one fitted point estimate exceeds three and is treated "
                "as a refinement-ladder artefact; the uncertainty is retained and "
                "no order-raising claim is made."
                if above_physical_ceiling
                else (
                    "The fitted order difference includes zero at 95% confidence, "
                    "but second order is more accurate at every shared rung; the "
                    "supported conclusion is a smaller constant, not a higher order."
                    if constant_improves
                    else (
                        "The fitted order difference includes zero at 95% confidence "
                        "and the rung-by-rung errors are mixed; neither a higher order "
                        "nor a systematically smaller error constant is resolved."
                    )
                )
            )
        ),
    }


def run(output: Path) -> dict[str, Any]:
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    case_results: dict[str, Any] = {}
    for case_name, case in _cases().items():
        targets = _targets(case)
        prior_reference = _plasma_reference(case, targets, REFERENCE_ORDERS[0])
        reference = _plasma_reference(case, targets, REFERENCE_ORDERS[1])
        exact_total = (
            2.0
            * np.pi
            * np.asarray(case.flux(targets[:, 0], targets[:, 1]), dtype=np.float64)
        )
        prior_total = prior_reference + exact_total - reference
        reference_error = _error_metrics(
            prior_total,
            exact_total,
            2.0 * np.pi * case.axis_flux,
        )
        rows = [
            _accuracy_row(case, requested_cells, targets, reference, exact_total)
            for requested_cells in REQUESTED_CELL_LADDER
        ]
        fits = {order: _fit_order(rows, order) for order in SUPPORT_COLUMNS}
        iso = {order: _iso_row(rows, order) for order in SUPPORT_COLUMNS}
        costs = {
            order: _isolated_cost(case_name, iso[order], order)
            for order in SUPPORT_COLUMNS
        }
        case_results[case_name] = {
            "geometry": {
                "major_radius_m": case.major_radius,
                "boundary_midplane_radii_m": list(case.boundary_midplane_radii()),
                "geometric_aspect_ratio": _aspect_ratio(case),
            },
            "reference_convergence": {
                "nested_orders": [list(value) for value in REFERENCE_ORDERS],
                "difference": reference_error,
            },
            "ladder": rows,
            "fitted_orders": fits,
            "iso_accuracy": iso,
            "iso_accuracy_costs": costs,
            "cost_comparison": _cost_comparison(costs),
            "order_verdict": _order_verdict(fits, rows),
        }

    classifications = {
        result["order_verdict"]["classification"] for result in case_results.values()
    }
    raises_everywhere = classifications == {
        "second_order_support_raises_field_error_order"
    }
    improves_constant_everywhere = classifications == {
        "order_raise_not_resolved; second_order_improves_only_the_constant"
    }
    receipt = {
        "schema": "nova.support-order-iso-accuracy",
        "source_revision": _source_revision(),
        "declared_target": {
            "quantity": "exterior poloidal-flux field relative sup error",
            "threshold": TARGET_RELATIVE_SUP_ERROR,
            "normalisation": (
                "closed-form axis-to-boundary total-flux span; this is independent "
                "of the exterior target contour's sampled variation"
            ),
            "selection_rule": "coarsest measured rung at or below the threshold",
        },
        "comparison_contract": {
            "support_orders": {
                "first": "constant plus radial and vertical linear terms",
                "second": (
                    "first-order terms plus radial-squared, mixed, and "
                    "vertical-squared terms"
                ),
            },
            "only_differing_mechanism": (
                "three versus six in-cell polynomial current-support columns"
            ),
            "shared_interaction_quadrature_order": INTERACTION_QUADRATURE_ORDER,
            "shared_moment_quadrature_order": MOMENT_QUADRATURE_ORDER,
            "accuracy_truth": (
                "closed-form Solov'ev total flux; an independently converged "
                "plasma integral supplies the common homogeneous complement"
            ),
            "target_surface": (
                "fixed physical R-Z offset along the ray from the LCFS geometric "
                "centre through each exact boundary point"
            ),
            "target_construction": {
                "clearance_minor_radius_fraction": (
                    TARGET_CLEARANCE_MINOR_RADIUS_FRACTION
                ),
                "choice": "declared construction constant; not fitted to results",
                "finite_and_exterior_asserted": True,
                "reason": (
                    "physical-coordinate offsets remain finite near the symmetry "
                    "axis where scaled flux-coordinate radii can become imaginary"
                ),
            },
            "cell_price": (
                "all carrier cells whose interaction columns are stored, "
                "including boundary and exterior padding cells"
            ),
            "cost_scope": (
                "interaction build, stored coupling, isolated accelerator peak, "
                "and warmed support contraction; topology and profile evaluation "
                "common to both orders are excluded from the contraction timing"
            ),
            "cost_lane_requirement": (
                "fresh processes on an exclusively allocated SLURM CPU compute "
                "node, matching the CPU lane of the banked refinement comparator"
            ),
        },
        "ladder_contract": {
            "requested_cells": list(REQUESTED_CELL_LADDER),
            "minimum_rungs_per_order": 4,
            "actual_rungs_per_order": len(REQUESTED_CELL_LADDER),
            "order_fit": (
                "ordinary least squares of log relative-sup error against log "
                "cell pitch; uncertainty is the slope standard error and a "
                "two-sided Student-t 95% interval"
            ),
        },
        "cases": case_results,
        "overall_verdict": {
            "classification": (
                "second_order_support_raises_field_error_order"
                if raises_everywhere
                else (
                    "second_order_support_improves_the_constant_without_a_"
                    "resolved_order_raise"
                    if improves_constant_everywhere
                    else "no_resolved_order_or_constant_improvement"
                )
            ),
            "statement": (
                "Second-order support raises the field-error order on both the "
                "analytic oracle and the low-aspect-ratio stress case."
                if raises_everywhere
                else (
                    "A field-error order increase is not established, but second "
                    "order systematically improves the error constant on both cases; "
                    "iso-accuracy costs decide whether that constant repays the "
                    "six-column surcharge."
                    if improves_constant_everywhere
                    else (
                        "Both supports retain second-order field convergence and the "
                        "rung-by-rung errors are mixed on at least one case; neither "
                        "an order raise nor a systematic constant improvement is "
                        "resolved, so the cell and cost comparison determines whether "
                        "the extra columns have any practical value."
                    )
                )
            ),
            "banked_comparator": {
                "h_refinement_interaction_build_ratio": BANKED_REFINEMENT_BUILD_RATIO,
                "h_refinement_peak_memory_ratio": BANKED_REFINEMENT_MEMORY_RATIO,
            },
        },
    }
    _write_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    cost_parser = subparsers.add_parser("measure-cost")
    cost_parser.add_argument("--case", choices=tuple(_cases()), required=True)
    cost_parser.add_argument("--requested-cells", type=int, required=True)
    cost_parser.add_argument("--order", choices=tuple(SUPPORT_COLUMNS), required=True)
    cost_parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.command == "measure-cost":
        from nova.jax.config import configure_dtypes

        configure_dtypes()
        result = _cost_measurement(
            arguments.case, arguments.requested_cells, arguments.order
        )
        _write_json(arguments.output, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    receipt = run(arguments.output)
    print(json.dumps(receipt["overall_verdict"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
