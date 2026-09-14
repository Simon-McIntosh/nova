"""Measure production Newton contraction from perturbed analytic flux.

The committed fixture exterior controls the admitted whole-cell single-null
rows.  A probe-only exterior posed from the exact booking at the analytic state
isolates the same iteration on the high-resolution limited and diverted rows.
Each row compiles one fixed-shape Newton program and reuses it for all four
perturbation magnitudes while persisting every completed arm.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np

from benchmarks import analytic_operator_ladder
from benchmarks import solovev_certificate as certificate
from nova.equilibrium import fixed_point
from nova.equilibrium import ForwardProfile
from nova.equilibrium.forward_operator import (
    ForwardFluxOperator,
    set_support_clip_mode,
    support_clip_mode,
)
from nova.equilibrium.source import CurrentNormalisationError
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/figures/cut-cell-current-attribution/oracle-start"
    / "map-floor-jacobian.json"
)
DEFAULT_REPORT_DIRECTORY = DEFAULT_OUTPUT.parent
DEFAULT_NEWTON_OUTPUT = DEFAULT_OUTPUT.parent / "newton-contraction.json"
PART_DIRECTORY_NAME = "parts"
NEWTON_PART_DIRECTORY_NAME = "newton-parts"
PERTURBATION_FRACTIONS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
NEWTON_STEPS = 12
ACTIVE_SET_STEPS = 12
FIXED_POINT_TOLERANCE = 1.0e-12
FINITE_DIFFERENCE_STEPS = (1.0e-5, 1.0e-7)
RANDOM_DIRECTION_COUNT = 4
RANDOM_SEED = 271828
MODES = ("exact", "chord")
ROWS = (
    ("weak-rotation-reactor-static", -1000),
    ("moderate-rotation-conventional-static", -1000),
    (certificate.DIVERTED_CASE_NAME, -1000),
    (certificate.DIVERTED_CASE_NAME, -500),
)


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("the measurement requires one scheduler allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    reservation = os.environ.get("SLURM_JOB_RESERVATION", "")
    platforms = os.environ.get("JAX_PLATFORMS", "")
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    if cpus != 8:
        raise RuntimeError(f"expected eight CPUs, received {cpus}")
    if partition != "all_debug":
        raise RuntimeError(f"expected all_debug, received {partition!r}")
    if platforms != "cpu":
        raise RuntimeError(f"expected JAX_PLATFORMS=cpu, received {platforms!r}")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": partition,
        "reservation": reservation,
        "allocated_cpus": cpus,
        "allocated_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", "0")),
        "memory_mb": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "gpu": None,
        "tmpdir": os.environ.get("TMPDIR"),
        "jax_platforms": platforms.split(","),
        "jax_cuda_devices": [],
        "jax_cpu_devices": [str(device) for device in jax.devices("cpu")],
    }


def _row_slug(case_name: str, requested_cells: int) -> str:
    return f"{case_name}-cells-{abs(requested_cells)}"


def _part_path(output: Path, case_name: str, requested_cells: int, mode: str) -> Path:
    name = f"{_row_slug(case_name, requested_cells)}-map-jacobian-{mode}.json"
    return output.parent / PART_DIRECTORY_NAME / name


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _number_or_token(value: float) -> float | str:
    number = float(value)
    if np.isnan(number):
        return "nan"
    if np.isposinf(number):
        return "inf"
    if np.isneginf(number):
        return "-inf"
    return number


def _pointer(
    function: Callable[..., Any], markers: tuple[str, ...] = ()
) -> dict[str, Any]:
    lines, start = inspect.getsourcelines(function)
    path = Path(inspect.getsourcefile(function) or "")
    try:
        rendered_path = str(path.relative_to(ROOT))
    except ValueError:
        rendered_path = str(path)
    located = []
    for marker in markers:
        matches = [index for index, line in enumerate(lines) if marker in line]
        if not matches:
            raise RuntimeError(
                f"source marker {marker!r} is absent from {rendered_path}"
            )
        located.append({"text": marker, "line": start + matches[0]})
    return {
        "path": rendered_path,
        "line_start": start,
        "line_end": start + len(lines) - 1,
        "markers": located,
    }


def _norms(delta: np.ndarray, span: float, grid_count: int) -> dict[str, float]:
    grid = np.asarray(delta, dtype=np.float64)[:grid_count]
    absolute_rms = float(np.sqrt(np.mean(grid**2)))
    absolute_sup = float(np.max(np.abs(grid)))
    return {
        "absolute_rms_wb": absolute_rms,
        "absolute_sup_wb": absolute_sup,
        "relative_rms_of_span": absolute_rms / span,
        "relative_sup_of_span": absolute_sup / span,
    }


def _topology(operator: Any, state: np.ndarray) -> dict[str, Any]:
    try:
        _masks, topology = operator.read(jnp.asarray(state))
    except NoQualifiedAxisError as error:
        return {
            "read_status": "no_qualified_axis",
            "class": None,
            "boundary_rz_m": None,
            "axis_flux_wb": None,
            "boundary_flux_wb": None,
            "x_point_rz_m": None,
            "x_point_flux_wb": None,
            "o_candidate_count": None,
            "x_candidate_count": None,
            "o_second_best_flux_margin_wb": None,
            "x_second_best_flux_margin_wb": None,
            "exception_text": str(error),
        }
    axis = np.asarray(topology.axis, dtype=np.float64)
    boundary = np.asarray(topology.boundary, dtype=np.float64)
    x_point = np.asarray(topology.x_point, dtype=np.float64)
    determinate = bool(topology.class_determinate)
    axis_flux = float(topology.axis_flux)
    boundary_flux = float(topology.boundary_flux)
    polarity = 1.0 if boundary_flux <= axis_flux else -1.0
    candidates = certificate.candidate_flux_margins(
        operator, jnp.asarray(state), polarity=polarity
    )
    topology_class = (
        "indeterminate"
        if not determinate
        else "diverted"
        if bool(topology.diverted)
        else "limited"
    )
    return {
        "read_status": "qualified_axis",
        "class": topology_class,
        "axis_rz_m": axis.tolist() if np.all(np.isfinite(axis)) else None,
        "boundary_rz_m": (boundary.tolist() if np.all(np.isfinite(boundary)) else None),
        "wall_contact_rz_m": (
            boundary.tolist()
            if topology_class == "limited" and np.all(np.isfinite(boundary))
            else None
        ),
        "axis_flux_wb": axis_flux,
        "boundary_flux_wb": boundary_flux,
        "analytic_authored_boundary_flux_wb": 0.0,
        "boundary_flux_offset_from_analytic_zero_wb": boundary_flux,
        "x_point_rz_m": (x_point.tolist() if np.all(np.isfinite(x_point)) else None),
        "x_point_flux_wb": (
            float(topology.x_point_flux)
            if np.isfinite(float(topology.x_point_flux))
            else None
        ),
        **candidates,
        "exception_text": None,
    }


def _point_cell_booking(
    cell_polygons: tuple[np.ndarray, ...],
    point: np.ndarray | None,
    booked_moments: Any,
    analytic_moments: Any,
    amplitude: float | None,
) -> dict[str, Any] | None:
    """Describe the allocation in the atomic cell containing one point."""
    if point is None:
        return None
    location = np.asarray(point, dtype=np.float64)
    booked = np.asarray(booked_moments.cell_current, dtype=np.float64)
    analytic = np.asarray(analytic_moments.cell_current, dtype=np.float64)
    if len(booked) != len(cell_polygons) or len(analytic) != len(cell_polygons):
        raise RuntimeError("cell polygons and current-moment vectors differ in length")
    containing = [
        index
        for index, polygon in enumerate(cell_polygons)
        if PolygonPath(
            np.vstack((np.asarray(polygon), np.asarray(polygon)[0]))
        ).contains_point(location, radius=1.0e-12)
    ]
    if containing:
        centres = np.asarray(
            [np.mean(np.asarray(cell_polygons[index]), axis=0) for index in containing]
        )
        selected = containing[
            int(np.argmin(np.linalg.norm(centres - location, axis=1)))
        ]
        selection = "containing polygon, nearest cell centre breaks an edge tie"
    else:
        centres = np.asarray(
            [np.mean(np.asarray(polygon), axis=0) for polygon in cell_polygons]
        )
        selected = int(np.argmin(np.linalg.norm(centres - location, axis=1)))
        selection = "nearest cell centre because no polygon contained the point"
    booked_cell = float(booked[selected])
    analytic_cell = float(analytic[selected])
    normalized_cell = booked_cell * amplitude if amplitude is not None else None
    return {
        "point_rz_m": location.tolist(),
        "containing_cell_indices": containing,
        "selected_cell_index": selected,
        "selection_rule": selection,
        "selected_cell_centre_rz_m": centres[
            containing.index(selected) if containing else selected
        ].tolist(),
        "booked_before_target_normalisation_a": booked_cell,
        "booked_after_target_normalisation_a": normalized_cell,
        "analytic_integrated_a": analytic_cell,
        "booked_before_over_analytic": (
            booked_cell / analytic_cell if analytic_cell != 0.0 else None
        ),
        "booked_after_over_analytic": (
            normalized_cell / analytic_cell
            if normalized_cell is not None and analytic_cell != 0.0
            else None
        ),
    }


def _per_cell_moment_attribution(
    operator: Any,
    state: jax.Array,
    requested_class: int | None,
    booked_moments: Any,
    analytic_moments: Any,
    amplitude: float,
    residual_shadow: np.ndarray,
    span: float,
    grid_count: int,
) -> dict[str, Any]:
    """Attribute a booked-moment image difference by support-cell class."""
    _masks, _topology_state, _sample, support = operator._support_partition(
        state, requested_class
    )
    area = np.asarray(support.area, dtype=np.float64)
    full_area = np.asarray(support.full_area, dtype=np.float64)
    included = np.asarray(support.included, dtype=bool)
    scale = max(float(np.max(np.abs(full_area))), np.finfo(np.float64).tiny)
    tolerance = 4096.0 * np.finfo(np.float64).eps * scale
    active = included & (area > tolerance)
    whole = active & (np.abs(area - full_area) <= tolerance)
    cut = active & ~whole
    inactive = ~active

    scaled = operator.scaled_current_moments(booked_moments, amplitude)
    booked_values = [np.asarray(value, dtype=np.float64) for value in scaled]
    analytic_values = [
        np.asarray(value, dtype=np.float64) for value in analytic_moments
    ]
    difference = [
        booked - analytic
        for booked, analytic in zip(booked_values, analytic_values, strict=True)
    ]
    moment_type = type(booked_moments)
    active_carriers = ~np.asarray(residual_shadow, dtype=bool)
    category_masks = {
        "cut": cut,
        "whole": whole,
        "inactive": inactive,
        "all": np.ones(len(area), dtype=bool),
    }
    image_attribution = {}
    absolute_current_total = float(np.sum(np.abs(difference[0])))
    for name, mask in category_masks.items():
        selected = moment_type(*(value * mask for value in difference))
        image = np.asarray(operator.current_moment_image(selected), dtype=np.float64)
        image_attribution[name] = {
            "cell_count": int(np.count_nonzero(mask)),
            "absolute_current_difference_a": float(np.sum(np.abs(difference[0][mask]))),
            "absolute_current_difference_fraction": (
                float(np.sum(np.abs(difference[0][mask]))) / absolute_current_total
                if absolute_current_total > 0.0
                else 0.0
            ),
            "image_on_residual_carriers": _norms(
                np.where(active_carriers, image, 0.0), span, grid_count
            ),
        }

    category = np.full(len(area), "inactive", dtype=object)
    category[whole] = "whole"
    category[cut] = "cut"
    return {
        "definition": (
            "target-normalised production coupling moments minus fixture analytic "
            "coupling moments, per atomic cell, imaged through the frozen blocks"
        ),
        "support_classification": (
            "cut means included with support area strictly between zero and full "
            "atomic area; whole means included at full area"
        ),
        "support_area_tolerance_m2": tolerance,
        "cell_index": np.arange(len(area), dtype=np.int64),
        "cell_centre_rz_m": np.asarray(
            operator.moment_geometry.atomic_mesh.centroids, dtype=np.float64
        ),
        "cell_class": category.tolist(),
        "support_area_fraction": np.divide(
            area,
            full_area,
            out=np.zeros_like(area),
            where=full_area != 0.0,
        ),
        "booked_target_normalised": {
            "cell_current_a": booked_values[0],
            "radial_coupling_coefficient": booked_values[1],
            "vertical_coupling_coefficient": booked_values[2],
        },
        "fixture_analytic": {
            "cell_current_a": analytic_values[0],
            "radial_coupling_coefficient": analytic_values[1],
            "vertical_coupling_coefficient": analytic_values[2],
        },
        "booked_minus_analytic": {
            "cell_current_a": difference[0],
            "radial_coupling_coefficient": difference[1],
            "vertical_coupling_coefficient": difference[2],
        },
        "image_attribution": image_attribution,
    }


def _current_booking(
    operator: Any,
    analytic: np.ndarray,
    requested_class: int | None,
    target_current: float,
) -> tuple[dict[str, Any], Any, float | None]:
    moments = operator.cell_current_moments(jnp.asarray(analytic), requested_class)
    booked = float(jnp.sum(moments.cell_current))
    error_record = None
    try:
        amplitude = float(
            operator.current_normalisation_amplitude(target_current, booked)
        )
        status = "finite"
    except CurrentNormalisationError as error:
        amplitude = None
        status = "current_normalisation_error"
        error_record = {
            "type": type(error).__name__,
            "message": str(error),
            "attempted_amplitude": _number_or_token(error.amplitude),
        }
    record = {
        "booked_plasma_current_a": booked,
        "analytic_plasma_current_a": target_current,
        "booked_over_analytic": booked / target_current,
        "normalisation_amplitude": amplitude,
        "status": status,
        "normalisation_error": error_record,
        "cell_current_sha256_binary64": _array_digest(moments.cell_current),
    }
    return record, moments, amplitude


def _smooth_directions(
    coordinates: np.ndarray, count: int, seed: int
) -> list[np.ndarray]:
    points = np.asarray(coordinates, dtype=np.float64)
    centre = np.mean(points, axis=0)
    scale = np.maximum(np.ptp(points, axis=0), 1.0e-12)
    normalized = (points - centre) / scale
    radius = normalized[:, 0]
    height = normalized[:, 1]
    basis = np.column_stack(
        (
            np.ones(len(points)),
            radius,
            height,
            radius * height,
            radius**2 - np.mean(radius**2),
            height**2 - np.mean(height**2),
            np.sin(np.pi * radius),
            np.cos(np.pi * height),
            np.sin(np.pi * (radius + height)),
            np.exp(-5.0 * (radius**2 + height**2)),
        )
    )
    generator = np.random.default_rng(seed)
    directions: list[np.ndarray] = []
    for _ in range(count):
        direction = basis @ generator.normal(size=basis.shape[1])
        direction -= np.mean(direction)
        norm = float(np.max(np.abs(direction)))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError("the smooth-direction instrument produced no signal")
        directions.append(direction / norm)
    return directions


def _certificate_residual(candidate, shadowed_map, frozen_shadow):
    """Return the I-minus-J residual action used by the Newton inner solve."""
    return candidate - shadowed_map(candidate, frozen_shadow)


def _relative_discrepancy(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = float(np.linalg.norm(candidate - reference))
    denominator = max(
        float(np.linalg.norm(reference)),
        float(np.linalg.norm(candidate)),
        np.finfo(np.float64).tiny,
    )
    return numerator / denominator


def _affine_decomposition(
    residual: np.ndarray,
    carrier_coordinates: np.ndarray,
    span: float,
) -> dict[str, Any]:
    """Split one carrier residual into an affine fit and its remainder."""
    values = np.asarray(residual, dtype=np.float64)[: len(carrier_coordinates)]
    coordinates = np.asarray(carrier_coordinates, dtype=np.float64)
    design = np.column_stack(
        (np.ones(len(coordinates)), coordinates[:, 0], coordinates[:, 1])
    )
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design, values, rcond=None
    )
    fitted = design @ coefficients
    remainder = values - fitted
    total_energy = float(np.sum(values**2))
    remainder_energy = float(np.sum(remainder**2))
    return {
        "basis": "constant plus raw R and Z in metres",
        "coefficient_constant_wb": float(coefficients[0]),
        "coefficient_r_wb_per_m": float(coefficients[1]),
        "coefficient_z_wb_per_m": float(coefficients[2]),
        "rank": int(rank),
        "singular_values": singular_values.tolist(),
        "affine_part": _norms(fitted, span, len(fitted)),
        "remainder": _norms(remainder, span, len(remainder)),
        "rms_energy_fraction_explained": (
            1.0 - remainder_energy / total_energy if total_energy > 0.0 else 1.0
        ),
    }


def _decompose_image(
    analytic: np.ndarray,
    mapped: np.ndarray,
    internal_image: np.ndarray,
    analytic_internal_image: np.ndarray,
    certificate_external: np.ndarray,
    carrier_coordinates: np.ndarray,
    residual_shadow: np.ndarray,
    span: float,
) -> dict[str, Any]:
    """Attribute a one-application residual to exterior and internal images."""
    grid_count = len(carrier_coordinates)
    actual_residual = np.asarray(mapped) - analytic
    implied_exterior = analytic - internal_image
    external_difference = certificate_external - implied_exterior
    internal_difference = internal_image - analytic_internal_image
    analytic_anchor_residual = certificate_external + analytic_internal_image - analytic
    raw_reconstruction = analytic_anchor_residual + internal_difference
    active = ~np.asarray(residual_shadow, dtype=bool)
    masked_reconstruction = np.where(active, raw_reconstruction, 0.0)
    closure = actual_residual - masked_reconstruction
    return {
        "a_affine_residual": _affine_decomposition(
            actual_residual, carrier_coordinates, span
        ),
        "b_external_term": {
            "certificate_external_definition": (
                "the external array captured by ForwardFluxOperator.flux_map"
            ),
            "implied_exterior_definition": (
                "analytic total flux minus the evaluated booked internal image"
            ),
            "certificate_minus_implied_exterior": _norms(
                external_difference, span, grid_count
            ),
            "certificate_minus_implied_exterior_on_residual_carriers": _norms(
                np.where(active, external_difference, 0.0), span, grid_count
            ),
            "certificate_external_constructed_for_evaluated_booking": bool(
                np.max(np.abs(external_difference[:grid_count]))
                <= 4096.0
                * np.finfo(np.float64).eps
                * max(float(np.max(np.abs(analytic[:grid_count]))), 1.0)
            ),
            "source_pointers": {
                "implied_exterior": _pointer(
                    _decompose_image,
                    ("implied_exterior = analytic - internal_image",),
                ),
                "certificate_external_capture": _pointer(
                    ForwardFluxOperator.flux_map,
                    ("external = self.external",),
                ),
            },
        },
        "c_internal_image": {
            "definition": (
                "evaluated booked internal image minus analytically integrated "
                "moments imaged through the same frozen blocks"
            ),
            "booked_minus_analytic_moments": _norms(
                internal_difference, span, grid_count
            ),
            "booked_minus_analytic_moments_on_residual_carriers": _norms(
                np.where(active, internal_difference, 0.0), span, grid_count
            ),
        },
        "analytic_moment_anchor": _norms(analytic_anchor_residual, span, grid_count),
        "actual_map_residual": _norms(actual_residual, span, grid_count),
        "decomposition_closure": _norms(closure, span, grid_count),
        "residual_carrier_count": int(np.count_nonzero(active[:grid_count])),
        "carrier_count": grid_count,
    }


def _unavailable_decomposition(
    analytic: np.ndarray,
    unscaled_internal: np.ndarray,
    analytic_internal: np.ndarray,
    external: np.ndarray,
    carrier_coordinates: np.ndarray,
    span: float,
) -> dict[str, Any]:
    """Retain the finite unscaled terms when normalisation is undefined."""
    grid_count = len(carrier_coordinates)
    implied_exterior = analytic - unscaled_internal
    return {
        "status": "certificate_map_unavailable_due_to_current_normalisation",
        "a_affine_residual": None,
        "b_external_term": {
            "unscaled_certificate_minus_implied_exterior": _norms(
                external - implied_exterior, span, grid_count
            ),
            "certificate_external_constructed_for_evaluated_booking": False,
        },
        "c_internal_image": {
            "unscaled_booked_minus_analytic_moments": _norms(
                unscaled_internal - analytic_internal, span, grid_count
            )
        },
    }


def _jacobian_probe(
    operator: Any,
    analytic: np.ndarray,
    coordinates: np.ndarray,
    span: float,
    requested_class: int,
    target_current: float,
) -> dict[str, Any]:
    shadowed_map = operator.flux_map_with_shadow(
        requested_class=requested_class,
        target_current=target_current,
    )
    state = jnp.asarray(analytic)
    frozen_shadow = operator.residual_shadow_mask(state, requested_class)

    def residual(candidate):
        return _certificate_residual(candidate, shadowed_map, frozen_shadow)

    residual_at_analytic, tangent = jax.linearize(residual, state)
    directions = _smooth_directions(coordinates, RANDOM_DIRECTION_COUNT, RANDOM_SEED)
    records = []
    for index, direction in enumerate(directions):
        exact_jvp = np.asarray(tangent(jnp.asarray(direction)), dtype=np.float64)
        step_records = []
        for relative_step in FINITE_DIFFERENCE_STEPS:
            absolute_step = relative_step * span
            plus = np.asarray(
                residual(state + absolute_step * jnp.asarray(direction)),
                dtype=np.float64,
            )
            minus = np.asarray(
                residual(state - absolute_step * jnp.asarray(direction)),
                dtype=np.float64,
            )
            finite_difference = (plus - minus) / (2.0 * absolute_step)
            step_records.append(
                {
                    "relative_step_of_flux_span": relative_step,
                    "absolute_step_wb": absolute_step,
                    "relative_jvp_discrepancy": _relative_discrepancy(
                        exact_jvp, finite_difference
                    ),
                    "exact_jvp_rms": float(np.sqrt(np.mean(exact_jvp**2))),
                    "finite_difference_rms": float(
                        np.sqrt(np.mean(finite_difference**2))
                    ),
                    "finite_difference_detected_nonzero_action": bool(
                        np.any(finite_difference != 0.0)
                    ),
                }
            )
        records.append(
            {
                "direction": f"smooth_random_{index + 1}",
                "unit_sup_direction_sha256": _array_digest(direction),
                "exact_jvp_detected_nonzero_action": bool(np.any(exact_jvp != 0.0)),
                "relative_steps": step_records,
            }
        )
    jax.block_until_ready(residual_at_analytic)
    return {
        "residual_definition": (
            "state minus the certificate's shadow-frozen target-normalised map; "
            "fixed_point.newton_krylov forms the same I-minus-J linear action "
            "after linearizing its frozen map"
        ),
        "nonlinear_solve_entered": False,
        "linear_solve_entered": False,
        "random_seed": RANDOM_SEED,
        "residual_at_analytic_rms_wb": float(
            np.sqrt(np.mean(np.asarray(residual_at_analytic, dtype=np.float64) ** 2))
        ),
        "directions": records,
        "source_pointers": {
            "benchmark_residual": _pointer(
                _certificate_residual,
                ("return candidate - shadowed_map(candidate, frozen_shadow)",),
            ),
            "production_inner_newton": _pointer(
                fixed_point._newton_krylov_inner,
                (
                    "mapped, tangent = jax.linearize(frozen_map, state)",
                    "residual_vector = mapped - state",
                    "return vector - tangent(vector)",
                ),
            ),
        },
    }


def _mode_measure(
    output: Path,
    case_name: str,
    requested_cells: int,
    mode: str,
    operator: Any,
    analytic: np.ndarray,
    coordinates: np.ndarray,
    grid_count: int,
    span: float,
    requested_class: int,
    target_current: float,
    exact_internal: np.ndarray,
) -> dict[str, Any]:
    started = perf_counter()
    set_support_clip_mode(mode)
    if support_clip_mode() != mode:
        raise RuntimeError(f"clip-mode setter did not select {mode}")

    external = np.asarray(operator.external(), dtype=np.float64)
    analytic_moment_map = external + exact_internal
    unscaled_map = operator.flux_map(requested_class=requested_class)
    certificate_map = operator.flux_map(
        requested_class=requested_class,
        target_current=target_current,
    )
    unscaled_mapped = np.asarray(
        jax.block_until_ready(unscaled_map(jnp.asarray(analytic))), dtype=np.float64
    )
    certificate_mapped = np.asarray(
        jax.block_until_ready(certificate_map(jnp.asarray(analytic))),
        dtype=np.float64,
    )
    part = _part_path(output, case_name, requested_cells, mode)
    measured = {
        "schema": "nova.oracle-start-map-jacobian-part",
        "version": 1,
        "source_revision": _source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": grid_count,
        "mode": mode,
        "mode_semantics": (
            "signed-flux spline-chain exact clip"
            if mode == "exact"
            else "production whole-cell booking control"
        ),
        "exterior_term": {
            "definition": (
                "analytic total flux minus the analytically integrated exact "
                "plasma-current moment image; it is constructed before selecting "
                "a clip mode and is identical for exact and control"
            ),
            "sha256_binary64": _array_digest(external),
            "source_pointers": {
                "construction": _pointer(
                    run,
                    (
                        "analytic - exact_internal",
                        "operator = oracle_fixture.forward_operator",
                    ),
                ),
                "fixture_image": _pointer(
                    oracle_fixture._internal_flux_image,
                    ("return np.asarray(",),
                ),
            },
        },
        "residual_definitions": {
            "map_floor": (
                "mapped analytic flux minus analytic flux, normalized only by "
                "the analytic grid-flux span"
            ),
            "certificate_relative_residual": (
                "max(abs(mapped-state)) / max(abs(mapped)); recorded as a pointer "
                "but not substituted for the requested span-normalized map floor"
            ),
            "jacobian": (
                "state minus the target-normalised map on the residual shadow "
                "frozen at the analytic state"
            ),
            "source_pointers": {
                "production_map": _pointer(
                    ForwardFluxOperator.flux_map,
                    (
                        "external = self.external",
                        "return self._exclude_shadow_residual",
                    ),
                ),
                "production_relative_residual": _pointer(
                    fixed_point._relative_residual,
                    ("return jnp.max(jnp.abs(mapped - state))",),
                ),
            },
        },
        "one_application": {
            "analytic_moments_anchor": {
                "definition": "external plus analytically integrated exact moments",
                **_norms(analytic_moment_map - analytic, span, grid_count),
            },
            "unscaled_production_map": {
                "definition": (
                    "production allocation and moment conversion without "
                    "target-current normalisation, requested topology class fixed"
                ),
                **_norms(unscaled_mapped - analytic, span, grid_count),
            },
            "certificate_target_normalised_map": {
                "definition": (
                    "the production certificate map with requested topology class and "
                    "analytic total-current target"
                ),
                **_norms(certificate_mapped - analytic, span, grid_count),
            },
            "booked_current": _current_booking(
                operator, analytic, requested_class, target_current
            )[0],
        },
        "jacobian": None,
        "completed": False,
        "wall_seconds": None,
    }
    _write_json(part, measured)
    certificate_floor = measured["one_application"]["certificate_target_normalised_map"]
    print(
        f"MAP_FLOOR case={case_name} cells={abs(requested_cells)} mode={mode} "
        f"rms={certificate_floor['relative_rms_of_span']:.8e} "
        f"sup={certificate_floor['relative_sup_of_span']:.8e}",
        flush=True,
    )
    measured["jacobian"] = _jacobian_probe(
        operator,
        analytic,
        coordinates,
        span,
        requested_class,
        target_current,
    )
    measured["wall_seconds"] = perf_counter() - started
    measured["completed"] = True
    _write_json(part, measured)
    print(
        f"JACOBIAN_DONE case={case_name} cells={abs(requested_cells)} mode={mode}",
        flush=True,
    )
    return measured


def _mode_measure_decomposed(
    output: Path,
    case_name: str,
    requested_cells: int,
    mode: str,
    operator: Any,
    analytic: np.ndarray,
    coordinates: np.ndarray,
    grid_count: int,
    span: float,
    requested_class: int,
    target_current: float,
    exact_internal: np.ndarray,
    exact_physical: Any,
    exact_coefficients: Any,
    cell_polygons: tuple[np.ndarray, ...],
    analytic_x_point: np.ndarray | None,
) -> dict[str, Any]:
    """Measure and persist one mode, retaining normalisation refusals."""
    started = perf_counter()
    set_support_clip_mode(mode)
    if support_clip_mode() != mode:
        raise RuntimeError(f"clip-mode setter did not select {mode}")

    state = jnp.asarray(analytic)
    topology = _topology(operator, analytic)
    external = np.asarray(operator.external(), dtype=np.float64)
    analytic_moment_map = external + exact_internal
    booking, booked_moments, amplitude = _current_booking(
        operator, analytic, requested_class, target_current
    )
    point_cell_booking = _point_cell_booking(
        cell_polygons,
        analytic_x_point,
        booked_moments,
        exact_physical,
        amplitude,
    )
    unscaled_internal = np.asarray(
        operator.current_moment_image(booked_moments), dtype=np.float64
    )
    certificate_map = operator.flux_map(
        requested_class=requested_class,
        target_current=target_current,
    )
    residual_shadow = np.asarray(
        operator.residual_shadow_mask(state, requested_class), dtype=bool
    )
    unscaled_mapped = np.where(
        residual_shadow,
        analytic,
        external + unscaled_internal,
    )

    normalisation_finding = None
    if amplitude is None:
        try:
            certificate_map(state)
        except CurrentNormalisationError as error:
            map_error = {
                "type": type(error).__name__,
                "message": str(error),
                "attempted_amplitude": _number_or_token(error.amplitude),
            }
        else:
            raise RuntimeError(
                "the certificate map did not reproduce its recorded "
                "current-normalisation refusal"
            )
        certificate_floor = {
            "status": "unavailable_due_to_current_normalisation",
            "definition": (
                "the production certificate map with requested topology class "
                "and analytic total-current target"
            ),
            "absolute_rms_wb": None,
            "absolute_sup_wb": None,
            "relative_rms_of_span": None,
            "relative_sup_of_span": None,
            "error": map_error,
        }
        decomposition = _unavailable_decomposition(
            analytic,
            unscaled_internal,
            exact_internal,
            external,
            coordinates[:grid_count],
            span,
        )
        normalisation_finding = {
            "classification": "analytic_flux_books_no_admissible_current",
            "statement": (
                "The production certificate map is undefined at the analytic "
                "flux because its selected allocation books no admissible current."
            ),
            "normalisation_error": map_error,
            "production_topology_state": topology,
            "booked_current": booking,
        }
    else:
        scaled_moments = operator.scaled_current_moments(booked_moments, amplitude)
        certificate_internal = np.asarray(
            operator.current_moment_image(scaled_moments), dtype=np.float64
        )
        certificate_mapped = np.asarray(
            jax.block_until_ready(certificate_map(state)), dtype=np.float64
        )
        certificate_floor = {
            "status": "finite",
            "definition": (
                "the production certificate map with requested topology class "
                "and analytic total-current target"
            ),
            **_norms(certificate_mapped - analytic, span, grid_count),
            "error": None,
        }
        decomposition = _decompose_image(
            analytic,
            certificate_mapped,
            certificate_internal,
            exact_internal,
            external,
            coordinates[:grid_count],
            residual_shadow,
            span,
        )

    per_cell_attribution = None
    if (
        case_name == "weak-rotation-reactor-static"
        and requested_cells == -1000
        and mode == "exact"
        and amplitude is not None
    ):
        per_cell_attribution = _per_cell_moment_attribution(
            operator,
            state,
            requested_class,
            booked_moments,
            exact_coefficients,
            amplitude,
            residual_shadow,
            span,
            grid_count,
        )

    part = _part_path(output, case_name, requested_cells, mode)
    measured = {
        "schema": "nova.oracle-start-map-jacobian-part",
        "version": 3,
        "source_revision": _source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": grid_count,
        "mode": mode,
        "mode_semantics": (
            "signed-flux spline-chain exact clip"
            if mode == "exact"
            else "production whole-cell booking control"
        ),
        "exterior_term": {
            "definition": (
                "analytic total flux minus the analytically integrated exact "
                "plasma-current moment image; constructed before mode selection"
            ),
            "boundary_basis": (
                "the authored analytic total field with its zero-level boundary; "
                "the production topology read and wall-contact level are not inputs"
            ),
            "analytic_authored_boundary_flux_wb": 0.0,
            "sha256_binary64": _array_digest(external),
            "source_pointers": {
                "construction": _pointer(
                    run,
                    (
                        "analytic - fixture_exact_internal",
                        "            operator = oracle_fixture.forward_operator(",
                    ),
                ),
                "fixture_image": _pointer(
                    oracle_fixture._internal_flux_image,
                    ("return np.asarray(",),
                ),
            },
        },
        "residual_definitions": {
            "map_floor": (
                "mapped analytic flux minus analytic flux, normalized only by "
                "the analytic grid-flux span"
            ),
            "certificate_relative_residual": (
                "max(abs(mapped-state)) / max(abs(mapped)); recorded as a source "
                "pointer but not substituted for the span-normalized map floor"
            ),
            "jacobian": (
                "state minus the target-normalised map on the residual shadow "
                "frozen at the analytic state"
            ),
            "source_pointers": {
                "production_map": _pointer(
                    ForwardFluxOperator.flux_map,
                    (
                        "external = self.external",
                        "return self._exclude_shadow_residual",
                    ),
                ),
                "production_relative_residual": _pointer(
                    fixed_point._relative_residual,
                    ("return jnp.max(jnp.abs(mapped - state))",),
                ),
            },
        },
        "one_application": {
            "analytic_moments_anchor": {
                "definition": "external plus analytically integrated exact moments",
                **_norms(analytic_moment_map - analytic, span, grid_count),
            },
            "unscaled_production_map": {
                "definition": (
                    "production allocation and moment conversion without "
                    "target-current normalisation, requested topology class fixed"
                ),
                **_norms(unscaled_mapped - analytic, span, grid_count),
            },
            "certificate_target_normalised_map": certificate_floor,
            "booked_current": booking,
            "decomposition": decomposition,
        },
        "production_topology_state_at_analytic_flux": topology,
        "analytic_x_point_cell_booking": point_cell_booking,
        "per_cell_moment_attribution": per_cell_attribution,
        "finding": normalisation_finding,
        "jacobian": None,
        "completed": False,
        "wall_seconds": None,
    }
    _write_json(part, measured)

    if amplitude is None:
        print(
            f"MAP_FLOOR_REFUSED case={case_name} cells={abs(requested_cells)} "
            f"mode={mode} booked_current={booking['booked_plasma_current_a']:.8e} "
            f"topology_class={topology['class']} error={map_error['message']}",
            flush=True,
        )
        jacobian_status = "unavailable_due_to_current_normalisation"
        jacobian_reason = map_error
    else:
        print(
            f"MAP_FLOOR case={case_name} cells={abs(requested_cells)} mode={mode} "
            f"rms={certificate_floor['relative_rms_of_span']:.8e} "
            f"sup={certificate_floor['relative_sup_of_span']:.8e}",
            flush=True,
        )
        jacobian_status = "not_requested_for_one_application_decomposition"
        jacobian_reason = None
    measured["jacobian"] = {
        "status": jacobian_status,
        "reason": jacobian_reason,
        "nonlinear_solve_entered": False,
        "linear_solve_entered": False,
        "directions": [],
    }
    measured["wall_seconds"] = perf_counter() - started
    measured["completed"] = True
    _write_json(part, measured)
    print(
        f"ROW_MODE_DONE case={case_name} cells={abs(requested_cells)} mode={mode} "
        f"jacobian_status={jacobian_status}",
        flush=True,
    )
    return measured


def _row_explanation(modes: dict[str, Any]) -> dict[str, Any]:
    exact = modes["exact"]["one_application"]
    control = modes["chord"]["one_application"]
    anchor = exact["analytic_moments_anchor"]["relative_sup_of_span"]
    exact_floor = exact["certificate_target_normalised_map"]["relative_sup_of_span"]
    control_floor = control["certificate_target_normalised_map"]["relative_sup_of_span"]
    same_exterior = (
        modes["exact"]["exterior_term"]["sha256_binary64"]
        == modes["chord"]["exterior_term"]["sha256_binary64"]
    )
    if anchor > 1.0e-10:
        classification = "exterior_completion_does_not_close_exact_moments"
        sentence = (
            "The analytic-moment anchor itself misses the analytic flux, so the "
            "exterior completion is the first inconsistent term."
        )
    elif exact_floor > 5.0 * max(control_floor, 1.0e-12):
        classification = "exact_allocation_or_moment_path"
        sentence = (
            "The same exterior and residual close analytic moments to roundoff, "
            "while the exact production allocation has a substantially larger floor "
            "than whole-cell booking; the floor enters through the exact allocation "
            "or its production moment conversion, not the exterior or residual sign."
        )
    elif max(exact_floor, control_floor) > 1.0e-8:
        classification = "shared_production_map_path"
        sentence = (
            "Both allocation modes miss despite a roundoff analytic-moment anchor, "
            "so a production-map term shared by both modes is responsible."
        )
    else:
        classification = "analytic_fixed_point_admitted"
        sentence = (
            "Both production allocation modes admit the analytic fixed point at the "
            "measured precision."
        )
    return {
        "classification": classification,
        "same_exterior_sha256": same_exterior,
        "analytic_moment_anchor_relative_sup": anchor,
        "exact_certificate_map_relative_sup": exact_floor,
        "whole_cell_certificate_map_relative_sup": control_floor,
        "sentence": sentence,
    }


def _mode_attribution(mode: str, measured: dict[str, Any]) -> dict[str, Any]:
    application = measured["one_application"]
    floor = application["certificate_target_normalised_map"]
    decomposition = application["decomposition"]
    if floor["status"] != "finite":
        return {
            "classification": "certificate_map_undefined",
            "certificate_exterior_constructed_for_different_booking": True,
            "sentence": (
                f"{mode} has no finite target-normalised map: the analytic-flux "
                "topology read led the allocation to book no admissible current, "
                "so neither a map-floor Jacobian nor Newton trajectory exists."
            ),
        }
    affine = decomposition["a_affine_residual"]
    external = decomposition["b_external_term"][
        "certificate_minus_implied_exterior_on_residual_carriers"
    ]
    internal = decomposition["c_internal_image"][
        "booked_minus_analytic_moments_on_residual_carriers"
    ]
    floor_rms = floor["relative_rms_of_span"]
    affine_fraction = affine["rms_energy_fraction_explained"]
    different_booking = not decomposition["b_external_term"][
        "certificate_external_constructed_for_evaluated_booking"
    ]
    return {
        "classification": "booked_internal_image_mismatch",
        "certificate_map_relative_rms": floor_rms,
        "affine_relative_rms": affine["affine_part"]["relative_rms_of_span"],
        "affine_remainder_relative_rms": affine["remainder"]["relative_rms_of_span"],
        "affine_rms_energy_fraction_explained": affine_fraction,
        "certificate_minus_implied_exterior_relative_rms": external[
            "relative_rms_of_span"
        ],
        "booked_minus_analytic_internal_relative_rms": internal["relative_rms_of_span"],
        "decomposition_closure_relative_sup": decomposition["decomposition_closure"][
            "relative_sup_of_span"
        ],
        "certificate_exterior_constructed_for_different_booking": different_booking,
        "sentence": (
            f"{mode} floor {floor_rms:.6g} rms of span is carried by the booked "
            f"internal-image difference ({internal['relative_rms_of_span']:.6g}); "
            "the external-minus-implied-exterior value is the same incompatibility "
            "viewed from the boundary supply "
            f"({external['relative_rms_of_span']:.6g}). "
            f"The affine fit explains {affine_fraction:.1%} of residual rms energy "
            f"and leaves {affine['remainder']['relative_rms_of_span']:.6g} rms of "
            "span. The certificate exterior closes analytically integrated moments, "
            "not the production booking evaluated by this map."
        ),
    }


def _row_explanation_decomposed(modes: dict[str, Any]) -> dict[str, Any]:
    attributions = {
        mode: _mode_attribution(mode, measured) for mode, measured in modes.items()
    }
    same_exterior = (
        modes["exact"]["exterior_term"]["sha256_binary64"]
        == modes["chord"]["exterior_term"]["sha256_binary64"]
    )
    return {
        "same_exterior_sha256": same_exterior,
        "by_mode": attributions,
        "sentence": " ".join(attributions[mode]["sentence"] for mode in MODES),
    }


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    lines = [
        "# Analytic map floor and residual tangent",
        "",
        "No Newton or linear solve was entered. Every row-mode part records the "
        "exterior construction and the exact source lines for the map, relative "
        "residual, benchmark residual, and production I-minus-J action.",
        "",
        "Comparison anchors: the operator refinement ladder closes the analytic "
        "field to 3e-15 relative sup when analytically integrated moments are "
        "imaged with their implied exterior; the committed weak-110 whole-cell "
        "certificate row is self-consistent at residual 0.0074.",
        "",
        "| Row | Mode | Map rms / sup | Affine rms / remainder (energy) | "
        "External mismatch rms | Internal mismatch rms | Booked / analytic | "
        "Worst JVP discrepancy at 1e-5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in receipt["rows"]:
        for mode in MODES:
            measured = row["modes"][mode]
            application = measured["one_application"]
            mapped = application["certificate_target_normalised_map"]
            current = application["booked_current"]
            if mapped["status"] == "finite":
                decomposition = application["decomposition"]
                affine = decomposition["a_affine_residual"]
                external = decomposition["b_external_term"][
                    "certificate_minus_implied_exterior_on_residual_carriers"
                ]
                internal = decomposition["c_internal_image"][
                    "booked_minus_analytic_moments_on_residual_carriers"
                ]
                discrepancies = [
                    step["relative_jvp_discrepancy"]
                    for direction in measured["jacobian"]["directions"]
                    for step in direction["relative_steps"]
                    if step["relative_step_of_flux_span"] == 1.0e-5
                ]
                mapped_text = (
                    f"{mapped['relative_rms_of_span']:.3e} / "
                    f"{mapped['relative_sup_of_span']:.3e}"
                )
                affine_text = (
                    f"{affine['affine_part']['relative_rms_of_span']:.3e} / "
                    f"{affine['remainder']['relative_rms_of_span']:.3e} "
                    f"({affine['rms_energy_fraction_explained']:.1%})"
                )
                external_text = f"{external['relative_rms_of_span']:.3e}"
                internal_text = f"{internal['relative_rms_of_span']:.3e}"
                jvp_text = (
                    f"{max(discrepancies):.3e}"
                    if discrepancies
                    else measured["jacobian"]["status"]
                )
            else:
                mapped_text = "normalisation refused"
                affine_text = "unavailable"
                external_text = "unavailable"
                internal_text = "unavailable"
                jvp_text = "unavailable"
            lines.append(
                f"| {row['case']} {abs(row['requested_cells'])} | {mode} | "
                f"{mapped_text} | {affine_text} | {external_text} | "
                f"{internal_text} | {current['booked_over_analytic']:.8f} | "
                f"{jvp_text} |"
            )
        lines.extend(("", row["explanation"]["sentence"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path, report_directory: Path) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the measurement requires JAX double precision")
    started = perf_counter()
    lane = _allocation()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    original_mode = support_clip_mode()
    rows = []
    try:
        for case_name, requested_cells in ROWS:
            carrier_case, source_case, exact = certificate._case(case_name)
            machine = certificate._case_machine(
                case_name, carrier_case, exact, requested_cells
            )
            coordinates = np.vstack(
                (machine.node, machine.wall_node, machine.sample_coordinates)
            )
            analytic = certificate._exact_state(case_name, exact, coordinates)
            empty_operator = oracle_fixture.forward_operator(source_case, machine)
            exact_physical = oracle_fixture.exact_current_moments(
                source_case, empty_operator, analytic
            )
            exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
            fixture_exact_internal = np.asarray(
                oracle_fixture._internal_flux_image(empty_operator, exact_coefficients),
                dtype=np.float64,
            )
            operator = oracle_fixture.forward_operator(
                source_case,
                machine,
                analytic - fixture_exact_internal,
            )
            exact_internal = np.asarray(
                operator.current_moment_image(exact_coefficients), dtype=np.float64
            )
            target_current, _centroid, target_receipt = (
                certificate._closed_form_current_target(
                    case_name, source_case, operator, exact_physical
                )
            )
            requested_class = int(
                TopologyClass.DIVERTED
                if certificate._is_diverted_case(case_name)
                else TopologyClass.LIMITED
            )
            topology = _topology(operator, analytic)
            if topology["axis_flux_wb"] is None:
                raise RuntimeError(
                    f"the analytic topology instrument could not read {case_name}"
                )
            span = abs(
                float(topology["axis_flux_wb"]) - float(topology["boundary_flux_wb"])
            )
            if not np.isfinite(span) or span <= 0.0:
                raise RuntimeError(f"the analytic flux span is invalid for {case_name}")
            modes = {}
            for mode in MODES:
                modes[mode] = _mode_measure_decomposed(
                    output,
                    case_name,
                    requested_cells,
                    mode,
                    operator,
                    analytic,
                    coordinates,
                    len(machine.node),
                    span,
                    requested_class,
                    target_current,
                    exact_internal,
                    exact_physical,
                    exact_coefficients,
                    machine.cell_polygons,
                    (
                        np.asarray(certificate.X_POINT_M, dtype=np.float64)
                        if certificate._is_diverted_case(case_name)
                        else None
                    ),
                )
            row = {
                "case": case_name,
                "requested_cells": requested_cells,
                "realised_cells": len(machine.node),
                "state_dimension": len(analytic),
                "analytic_flux_span_wb": span,
                "analytic_current_target_a": target_current,
                "analytic_current_target_receipt": target_receipt,
                "interaction_matrix_cache": machine.cache,
                "interaction_matrix_construction_count": 1,
                "fixture_and_same_operator_exact_image": {
                    "fixture_sha256": _array_digest(fixture_exact_internal),
                    "same_operator_sha256": _array_digest(exact_internal),
                    "absolute_sup_delta_wb": float(
                        np.max(np.abs(fixture_exact_internal - exact_internal))
                    ),
                },
                "modes": modes,
                "explanation": _row_explanation_decomposed(modes),
            }
            rows.append(row)
            _write_json(
                output.parent
                / PART_DIRECTORY_NAME
                / f"{_row_slug(case_name, requested_cells)}.json",
                row,
            )
    finally:
        set_support_clip_mode(original_mode)
    receipt = {
        "schema": "nova.oracle-start-map-decomposition",
        "version": 2,
        "source_revision": _source_revision(),
        "production_code_modified": False,
        "nonlinear_solve_entered": False,
        "linear_solve_entered": False,
        "lane": {
            **lane,
            "persistent_compilation_cache": cache.receipt(),
            "wall_seconds": perf_counter() - started,
            "exit_marker": "ORACLE_START_MAP_DECOMPOSITION_EXIT=0",
        },
        "design": {
            "rows": [
                {"case": case_name, "requested_cells": requested_cells}
                for case_name, requested_cells in ROWS
            ],
            "modes": {
                "exact": "signed-flux spline-chain exact clip",
                "chord": "production whole-cell booking control",
            },
            "jacobian_measured": False,
            "map_applications_per_row_mode": 1,
            "interaction_matrix_policy": (
                "one cached machine and operator per row, reused across both modes"
            ),
            "comparison_anchors": {
                "analytic_operator_ladder_relative_sup": 3.0e-15,
                "committed_weak_110_whole_cell_self_consistency": 0.0074,
                "operator_ladder_source": _pointer(
                    analytic_operator_ladder._measure,
                    (
                        "prescribed_exterior = analytic - exact_internal",
                        "mapped_analytic = np.asarray(",
                    ),
                ),
            },
        },
        "rows": rows,
    }
    _write_json(output, receipt)
    _write_report(report_directory / "map-jacobian-report.md", receipt)
    return receipt


NEWTON_ROWS = (
    (
        "reposed_exact_booking_iteration_probe",
        "weak-rotation-reactor-static",
        -300,
        "exact",
    ),
    (
        "reposed_exact_booking_iteration_probe",
        "moderate-rotation-conventional-static",
        -300,
        "exact",
    ),
    ("fixture_exterior_control", certificate.DIVERTED_CASE_NAME, -1000, "chord"),
)


def _gpu_allocation() -> dict[str, Any]:
    """Return the guarded accelerator allocation used by the Newton probe."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the Newton measurement requires one scheduler allocation")
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    reservation = os.environ.get("SLURM_JOB_RESERVATION", "")
    platforms = os.environ.get("JAX_PLATFORMS", "")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower()
    if partition != "betelgeuse":
        raise RuntimeError(f"expected betelgeuse, received {partition!r}")
    if reservation != "gpu_0003_grpA":
        raise RuntimeError(f"unexpected reservation {reservation!r}")
    if platforms != "cuda,cpu":
        raise RuntimeError(f"expected JAX_PLATFORMS=cuda,cpu, received {platforms!r}")
    if cpus != 8:
        raise RuntimeError(f"expected eight CPUs, received {cpus}")
    if preallocate != "false":
        raise RuntimeError("XLA_PYTHON_CLIENT_PREALLOCATE must be false")
    return {
        "job_id": int(os.environ["SLURM_JOB_ID"]),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": partition,
        "reservation": reservation,
        "allocated_cpus": cpus,
        "allocated_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", "1")),
        "memory_mb": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "tmpdir": os.environ.get("TMPDIR"),
        "jax_platforms": platforms.split(","),
        "xla_python_client_preallocate": preallocate,
        "visible_gpu_memory_used_mib": int(
            os.environ.get("NOVA_VISIBLE_GPU_MEMORY_USED_MIB", "-1")
        ),
        "visible_gpu_memory_total_mib": int(
            os.environ.get("NOVA_VISIBLE_GPU_MEMORY_TOTAL_MIB", "-1")
        ),
        "visible_gpu_memory_free_mib": int(
            os.environ.get("NOVA_VISIBLE_GPU_MEMORY_FREE_MIB", "-1")
        ),
        "jax_cuda_devices": [str(device) for device in jax.devices("gpu")],
        "jax_cpu_devices": [str(device) for device in jax.devices("cpu")],
    }


def _newton_part_path(
    output: Path,
    exterior_kind: str,
    case_name: str,
    requested_cells: int,
) -> Path:
    return (
        output.parent
        / NEWTON_PART_DIRECTORY_NAME
        / f"{_row_slug(case_name, requested_cells)}-{exterior_kind}.json"
    )


def _newton_arm_path(
    output: Path,
    exterior_kind: str,
    case_name: str,
    requested_cells: int,
    fraction: float,
) -> Path:
    exponent = int(round(-np.log10(fraction)))
    return (
        output.parent
        / NEWTON_PART_DIRECTORY_NAME
        / (
            f"{_row_slug(case_name, requested_cells)}-{exterior_kind}"
            f"-perturb-1e-{exponent}.json"
        )
    )


def _solver_functions(operator: Any, requested_class: int, target_current: float):
    mapped = operator.flux_map(
        requested_class=requested_class, target_current=target_current
    )
    shadowed = operator.flux_map_with_shadow(
        requested_class=requested_class, target_current=target_current
    )

    def shadow_mask(state):
        return operator.residual_shadow_mask(state, requested_class)

    def promoted_shadow_mask(state, previous):
        return operator.residual_shadow_mask(
            state, requested_class, previous_shadow=previous
        )

    return mapped, shadowed, shadow_mask, promoted_shadow_mask


def _device_value(value: Any) -> Any:
    array = np.asarray(jax.device_get(value))
    if array.shape == ():
        return array.item()
    return array


def _fixed_point_telemetry(history: fixed_point.FixedPointResult) -> dict[str, Any]:
    """Persist the production result fields without adding an observer."""
    telemetry = {
        name: _device_value(getattr(history, name))
        for name in history._fields
        if name not in {"state", "trajectory_state"}
    }
    state = np.asarray(history.state, dtype=np.float64)
    trajectory = np.asarray(history.trajectory_state)
    telemetry["state"] = {
        "shape": list(state.shape),
        "sha256_binary64": _array_digest(state),
    }
    telemetry["trajectory_state"] = {
        "shape": list(trajectory.shape),
        "sha256_binary64": _array_digest(trajectory),
    }
    telemetry["termination_name"] = fixed_point.FixedPointTerminationReason(
        int(history.termination_reason)
    ).name.lower()
    telemetry["krylov_action_qualification_name"] = (
        fixed_point.KrylovActionQualification(
            int(history.krylov_action_qualification)
        ).name.lower()
    )
    return telemetry


def _point_error(point: Any, reference: Any) -> float | None:
    if point is None or reference is None:
        return None
    value = np.asarray(point, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if value.shape != (2,) or target.shape != (2,):
        return None
    return float(np.linalg.norm(value - target))


def _terminal_measurement(
    operator: Any,
    state: np.ndarray,
    analytic: np.ndarray,
    analytic_read: dict[str, Any],
    closed_form_axis: np.ndarray,
    closed_form_x_point: np.ndarray | None,
    span: float,
    grid_count: int,
    pitch: float,
    requested_class: int | None,
    target_current: float,
) -> dict[str, Any]:
    topology = _topology(operator, state)
    axis_read_error = _point_error(
        topology.get("axis_rz_m"), analytic_read.get("axis_rz_m")
    )
    x_read_error = _point_error(
        topology.get("x_point_rz_m"), analytic_read.get("x_point_rz_m")
    )
    axis_closed_error = _point_error(topology.get("axis_rz_m"), closed_form_axis)
    x_closed_error = _point_error(topology.get("x_point_rz_m"), closed_form_x_point)
    topology_matches = topology.get("class") == analytic_read.get("class")
    positions_within_pitch = (
        axis_closed_error is not None
        and axis_closed_error <= pitch
        and (
            closed_form_x_point is None
            or (x_closed_error is not None and x_closed_error <= pitch)
        )
    )
    try:
        booked_current = _current_booking(
            operator, state, requested_class, target_current
        )[0]
    except Exception as error:
        booked_current = {
            "status": "unavailable",
            "exception_type": type(error).__name__,
            "exception_text": str(error),
        }
    return {
        "distance_to_analytic": _norms(state - analytic, span, grid_count),
        "topology": topology,
        "axis_error_to_analytic_read_m": axis_read_error,
        "x_point_error_to_analytic_read_m": x_read_error,
        "axis_error_to_closed_form_m": axis_closed_error,
        "x_point_error_to_closed_form_m": x_closed_error,
        "topology_class_matches_analytic_read": topology_matches,
        "analytic_nulls_within_one_pitch": positions_within_pitch,
        "booked_current": booked_current,
    }


def _render_newton_terminal(
    path: Path,
    case_name: str,
    exterior_kind: str,
    coordinates: np.ndarray,
    analytic: np.ndarray,
    terminal: np.ndarray,
    wall: np.ndarray,
    boundary: np.ndarray,
    analytic_read: dict[str, Any],
    terminal_measurement: dict[str, Any],
    terminal_residual: float,
    converged: bool,
) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(5.5, 5.2), constrained_layout=True)
    analytic_r, analytic_z, analytic_field = certificate._raster_field(
        coordinates, analytic, wall
    )
    terminal_r, terminal_z, terminal_field = certificate._raster_field(
        coordinates, terminal, wall
    )
    levels = poloidal.contour_levels(
        np.concatenate((analytic_field.ravel(), terminal_field.ravel())), count=12
    )
    wall_units = (wall,)
    poloidal.draw_flux_contours(
        axis, analytic_r, analytic_z, analytic_field, levels, color="#3366cc"
    )
    poloidal.draw_flux_contours(
        axis, terminal_r, terminal_z, terminal_field, levels, color="#cc7722"
    )
    poloidal.draw_boundary(axis, boundary[:, 0], boundary[:, 1], color="#3366cc")
    poloidal.draw_wall(axis, units=wall_units)
    poloidal.draw_nulls(
        axis,
        magnetic_axis=analytic_read["axis_rz_m"],
        x_points=analytic_read["x_point_rz_m"],
        style=DEFAULT_INK.variant(
            axis_marker="^", axis_color="#3366cc", xpoint_color="#3366cc"
        ),
        contain=wall_units,
    )
    terminal_topology = terminal_measurement["topology"]
    poloidal.draw_nulls(
        axis,
        magnetic_axis=terminal_topology["axis_rz_m"],
        x_points=terminal_topology["x_point_rz_m"],
        style=DEFAULT_INK.variant(
            axis_marker="^", axis_color="#cc7722", xpoint_color="#cc7722"
        ),
        contain=wall_units,
    )
    poloidal_axes(axis)
    axis.set_title(
        "analytic blue / terminal ochre; shared Wb levels\n"
        f"residual={terminal_residual:.3e}; converged={converged}",
        fontsize=8,
    )
    figure.suptitle(f"{case_name} · {exterior_kind} · 1e-2 perturbation", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _build_newton_operator(
    exterior_kind: str,
    case_name: str,
    requested_cells: int,
    mode: str,
) -> dict[str, Any]:
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)

    set_support_clip_mode("chord")
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    fixture_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, analytic
    )
    fixture_coefficients = empty_operator.coupling_current_moments(fixture_physical)
    fixture_internal = np.asarray(
        empty_operator.current_moment_image(fixture_coefficients), dtype=np.float64
    )
    fixture_external = analytic - fixture_internal
    fixture_operator = oracle_fixture.forward_operator(
        source_case, machine, fixture_external
    )
    target_current, _centroid, target_receipt = certificate._closed_form_current_target(
        case_name, source_case, fixture_operator, fixture_physical
    )
    requested_class = int(
        TopologyClass.DIVERTED
        if certificate._is_diverted_case(case_name)
        else TopologyClass.LIMITED
    )

    if exterior_kind == "fixture_exterior_control":
        external = fixture_external
        closure_internal = fixture_internal
        operator = fixture_operator
        exterior_receipt = {
            "kind": exterior_kind,
            "certificate_changed": False,
            "definition": (
                "committed fixture exterior: analytic total flux minus fixture "
                "whole-cell analytic moment image"
            ),
            "analytic_booking_amplitude": 1.0,
        }
    else:
        set_support_clip_mode("exact")
        allocation_topology = _topology(fixture_operator, analytic)
        booked = fixture_operator.cell_current_moments(
            jnp.asarray(analytic), requested_class
        )
        booked_total = float(jnp.sum(booked.cell_current))
        amplitude = float(
            fixture_operator.current_normalisation_amplitude(
                target_current, booked_total
            )
        )
        normalised = fixture_operator.scaled_current_moments(booked, amplitude)
        exact_booking_internal = np.asarray(
            fixture_operator.current_moment_image(normalised), dtype=np.float64
        )
        external = analytic - exact_booking_internal
        closure_internal = exact_booking_internal
        operator = oracle_fixture.forward_operator(source_case, machine, external)
        exterior_receipt = {
            "kind": exterior_kind,
            "certificate_changed": False,
            "probe_only": True,
            "definition": (
                "analytic total flux minus the target-normalised exact-clip "
                "internal image evaluated once at the analytic state"
            ),
            "analytic_booking_topology": allocation_topology,
            "analytic_booking_current_a": booked_total,
            "analytic_booking_amplitude": amplitude,
        }

    set_support_clip_mode(mode)
    analytic_read = _topology(operator, analytic)
    span = abs(
        float(analytic_read["axis_flux_wb"]) - float(analytic_read["boundary_flux_wb"])
    )
    construction_floor = _norms(
        external + closure_internal - analytic, span, len(machine.node)
    )
    if construction_floor["relative_sup_of_span"] > FIXED_POINT_TOLERANCE:
        raise RuntimeError(
            f"{exterior_kind} {case_name} does not provide the required analytic "
            f"closure: {construction_floor['relative_sup_of_span']:.17g}"
        )
    return {
        "carrier_case": carrier_case,
        "source_case": source_case,
        "exact": exact,
        "machine": machine,
        "coordinates": coordinates,
        "analytic": analytic,
        "operator": operator,
        "target_current": target_current,
        "target_receipt": target_receipt,
        "requested_class": requested_class,
        "analytic_read": analytic_read,
        "span": span,
        "construction_floor": construction_floor,
        "exterior": exterior_receipt,
        "external_sha256_binary64": _array_digest(external),
    }


def _measure_newton_row(
    output: Path,
    exterior_kind: str,
    case_name: str,
    requested_cells: int,
    mode: str,
) -> dict[str, Any]:
    started = perf_counter()
    built = _build_newton_operator(exterior_kind, case_name, requested_cells, mode)
    machine = built["machine"]
    analytic = built["analytic"]
    operator = built["operator"]
    coordinates = built["coordinates"]
    grid_count = len(machine.node)
    pitch = float(np.sqrt(np.median(np.asarray(machine.area))))
    closed_form_axis = np.asarray(built["exact"].magnetic_axis, dtype=np.float64)
    closed_form_x_point = (
        np.asarray(certificate.X_POINT_M, dtype=np.float64)
        if certificate._is_diverted_case(case_name)
        else None
    )
    direction = _smooth_directions(coordinates, 1, RANDOM_SEED)[0]
    analytic_shadow = np.asarray(
        operator.residual_shadow_mask(jnp.asarray(analytic)), dtype=bool
    )
    direction[analytic_shadow] = 0.0
    direction /= float(np.max(np.abs(direction[:grid_count])))
    initials = [
        analytic + fraction * built["span"] * direction
        for fraction in PERTURBATION_FRACTIONS
    ]
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=NEWTON_STEPS,
    )
    start_map = operator.flux_map(target_current=built["target_current"])

    row = {
        "schema": "nova.oracle-start-newton-row",
        "version": 1,
        "source_revision": _source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": grid_count,
        "mode": mode,
        "exterior": built["exterior"],
        "external_sha256_binary64": built["external_sha256_binary64"],
        "analytic_fixed_point_construction_floor": built["construction_floor"],
        "analytic_topology_read": built["analytic_read"],
        "analytic_flux_span_wb": built["span"],
        "characteristic_pitch_m": pitch,
        "analytic_current_target_a": built["target_current"],
        "analytic_current_target_receipt": built["target_receipt"],
        "smooth_direction_sha256_binary64": _array_digest(direction),
        "analytic_residual_shadow_count": int(np.count_nonzero(analytic_shadow)),
        "analytic_residual_carrier_count": int(np.count_nonzero(~analytic_shadow)),
        "program_reuse": (
            "one ForwardProfile and one public request policy per row; the first "
            "receipt compiles and the remaining same-shape requests reuse it"
        ),
        "perturbations": [],
        "completed": False,
        "wall_seconds": None,
    }
    part = _newton_part_path(output, exterior_kind, case_name, requested_cells)
    _write_json(part, row)

    figure_state = None
    figure_terminal = None
    for fraction, initial in zip(PERTURBATION_FRACTIONS, initials, strict=True):
        arm_started = perf_counter()
        mapped_initial = np.asarray(
            jax.block_until_ready(start_map(jnp.asarray(initial))), dtype=np.float64
        )
        direct_start_residual = float(
            fixed_point._relative_residual(
                jnp.asarray(mapped_initial), jnp.asarray(initial)
            )
        )
        if not np.isfinite(direct_start_residual) or direct_start_residual <= 0.0:
            raise RuntimeError(
                f"the perturbed start has no production-map residual: "
                f"{direct_start_residual!r}"
            )
        request = certificate._certificate_solve_request(
            profile,
            jnp.asarray(initial),
            built["target_current"],
            carrier_identity=(
                f"oracle-start:{exterior_kind}:{case_name}:{requested_cells}"
            ),
        )
        request = replace(
            request,
            policy=replace(
                request.policy,
                newton_steps=NEWTON_STEPS,
                active_set_steps=ACTIVE_SET_STEPS,
            ),
        )
        solve_receipt = profile.solve(request)
        history = solve_receipt.equilibrium.fixed_point
        jax.block_until_ready(history.state)
        terminal = np.asarray(solve_receipt.equilibrium.flux, dtype=np.float64)
        terminal_measurement = _terminal_measurement(
            operator,
            terminal,
            analytic,
            built["analytic_read"],
            closed_form_axis,
            closed_form_x_point,
            built["span"],
            grid_count,
            pitch,
            None,
            built["target_current"],
        )
        initial_distance = _norms(initial - analytic, built["span"], grid_count)
        terminal_distance = terminal_measurement["distance_to_analytic"]
        residual = float(history.residual)
        contracted = (
            terminal_distance["relative_sup_of_span"]
            < initial_distance["relative_sup_of_span"]
        )
        returned = (
            residual <= FIXED_POINT_TOLERANCE
            and terminal_distance["relative_sup_of_span"] <= FIXED_POINT_TOLERANCE
            and terminal_measurement["topology_class_matches_analytic_read"]
            and terminal_measurement["analytic_nulls_within_one_pitch"]
        )
        trace = np.asarray(history.trace, dtype=np.float64)
        finite_trace = trace[np.isfinite(trace)]
        arm = {
            "requested_relative_perturbation": fraction,
            "initial_distance_to_analytic": initial_distance,
            "initial_relative_fixed_point_residual_from_result_trace": (
                float(finite_trace[0]) if len(finite_trace) else None
            ),
            "initial_relative_fixed_point_residual_direct": direct_start_residual,
            "terminal": terminal_measurement,
            "contracted_toward_analytic": contracted,
            "left_analytic_neighbourhood": not contracted,
            "residual_at_or_below_1e_12": residual <= FIXED_POINT_TOLERANCE,
            "state_distance_at_or_below_1e_12": (
                terminal_distance["relative_sup_of_span"] <= FIXED_POINT_TOLERANCE
            ),
            "returned_to_analytic_fixed_point": returned,
            "fixed_point_result": _fixed_point_telemetry(history),
            "public_solve_receipt": {
                "compilation_cache_hit": solve_receipt.compilation_cache_hit,
                "qualified": bool(solve_receipt.qualified),
                "wall_seconds": solve_receipt.wall_seconds,
                "resolved_defaults": solve_receipt.resolved_defaults.to_dict(),
            },
            "solve_wall_seconds": perf_counter() - arm_started,
        }
        row["perturbations"].append(arm)
        if fraction == 1.0e-2:
            figure_state = terminal
            figure_terminal = arm
        _write_json(
            _newton_arm_path(
                output, exterior_kind, case_name, requested_cells, fraction
            ),
            arm,
        )
        _write_json(part, row)
        print(
            f"NEWTON_ARM_DONE case={case_name} cells={abs(requested_cells)} "
            f"exterior={exterior_kind} mode={mode} perturbation={fraction:.0e} "
            f"residual={residual:.8e} "
            f"distance={terminal_distance['relative_sup_of_span']:.8e} "
            f"returned={returned}",
            flush=True,
        )

    if figure_state is None or figure_terminal is None:
        raise RuntimeError("the 1e-2 terminal state was not retained")
    figure = (
        output.parent / f"{_row_slug(case_name, requested_cells)}-{exterior_kind}.png"
    )
    _render_newton_terminal(
        figure,
        case_name,
        exterior_kind,
        coordinates,
        analytic,
        figure_state,
        np.asarray(machine.wall_node, dtype=np.float64),
        certificate._boundary(case_name, built["exact"]),
        built["analytic_read"],
        figure_terminal["terminal"],
        float(figure_terminal["fixed_point_result"]["residual"]),
        bool(figure_terminal["fixed_point_result"]["converged"]),
    )
    row["figure"] = {
        "filesystem_path": str(figure.relative_to(ROOT)),
        "project_absolute_src": f"/nova/{figure.relative_to(ROOT / 'docs')}",
        "sha256": hashlib.sha256(figure.read_bytes()).hexdigest(),
        "caption": (
            "Terminal 1e-2 iterate in ochre against analytic blue on shared "
            "levels, both topology reads, and the authored wall."
        ),
    }
    row["returned_for_every_perturbation"] = all(
        arm["returned_to_analytic_fixed_point"] for arm in row["perturbations"]
    )
    row["wall_seconds"] = perf_counter() - started
    row["completed"] = True
    _write_json(part, row)
    return row


def _write_newton_report(path: Path, receipt: dict[str, Any]) -> None:
    lines = [
        "# Analytic-start Newton contraction",
        "",
        (
            "Variant `fixture_exterior_control` is the committed certificate "
            "fixture. Variant `reposed_exact_booking_iteration_probe` changes no "
            "certificate: it poses the exterior once so the analytic exact booking "
            "is a fixed point, solely to isolate the production iteration."
        ),
        "",
        (
            "| Exterior | Row | Mode | Perturbation | Start residual | Terminal "
            "residual | Distance / span | Trips | Accepted / attempted | Axis / X "
            "error to analytic read (m) | Axis / X error to closed form (m) | "
            "Returned |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in receipt["rows"]:
        for arm in row["perturbations"]:
            terminal = arm["terminal"]
            result = arm["fixed_point_result"]
            read_x_error = terminal["x_point_error_to_analytic_read_m"]
            closed_x_error = terminal["x_point_error_to_closed_form_m"]
            lines.append(
                f"| {row['exterior']['kind']} | {row['case']} "
                f"{abs(row['requested_cells'])} | "
                f"{row['mode']} | {arm['requested_relative_perturbation']:.0e} | "
                f"{arm['initial_relative_fixed_point_residual_direct']:.3e} | "
                f"{result['residual']:.3e} | "
                f"{terminal['distance_to_analytic']['relative_sup_of_span']:.3e} | "
                f"{result['active_set_iterations']} | "
                f"{result['accepted_newton_promotions']} / "
                f"{result['attempted_newton_promotions']} | "
                f"{terminal['axis_error_to_analytic_read_m']:.3e} / "
                f"{'n/a' if read_x_error is None else f'{read_x_error:.3e}'} | "
                f"{terminal['axis_error_to_closed_form_m']:.3e} / "
                f"{'n/a' if closed_x_error is None else f'{closed_x_error:.3e}'} | "
                f"{arm['returned_to_analytic_fixed_point']} |"
            )
    map_receipt = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    lines.extend(
        [
            "",
            "## Jacobian-vector products from the map-only allocation",
            "",
            (
                "Each value is the worst relative discrepancy across four fixed-seed "
                "smooth directions. The two columns are central finite differences at "
                "relative steps 1e-5 and 1e-7 of the analytic flux span."
            ),
            "",
            "| Row | Mode | Map floor rms / span | JVP discrepancy 1e-5 | "
            "JVP discrepancy 1e-7 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for map_row in map_receipt["rows"]:
        for mode, measured in map_row["modes"].items():
            mapped = measured["one_application"]["certificate_target_normalised_map"]
            directions = measured["jacobian"].get("directions", [])
            discrepancies = []
            for relative_step in FINITE_DIFFERENCE_STEPS:
                values = [
                    step["relative_jvp_discrepancy"]
                    for direction in directions
                    for step in direction["relative_steps"]
                    if step["relative_step_of_flux_span"] == relative_step
                ]
                discrepancies.append(max(values) if values else None)
            floor = mapped.get("relative_rms_of_span")
            lines.append(
                f"| {map_row['case']} {abs(map_row['requested_cells'])} | {mode} | "
                f"{'refused' if floor is None else f'{floor:.3e}'} | "
                f"{'n/a' if discrepancies[0] is None else f'{discrepancies[0]:.3e}'} | "
                f"{'n/a' if discrepancies[1] is None else f'{discrepancies[1]:.3e}'} |"
            )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "Where the reference is an actual fixed point, the production "
                "Newton-Krylov iteration contracts to it from every tested start, "
                "including one tenth of the analytic span away. The whole-cell "
                "single-null 500 control reaches residual 6.203e-15 and relative "
                "sup distance 1.428e-14 in one accepted promotion."
            ),
            "",
            (
                "The limited whole-cell rows are allocation-floor failures against "
                "the fixture, not Jacobian or globalisation failures: their JVP "
                "discrepancies are 8e-10 or smaller at the finer finite-difference "
                "step while their map floors are 0.047 to 0.202 of span."
            ),
            "",
            (
                "The limited exact rows carry both an allocation floor of 0.076 to "
                "0.285 and a 1.5 to 1.7 percent JVP discrepancy. The single-null 500 "
                "exact row has floor 2.063 and a 20.1 percent JVP discrepancy; its "
                "re-posed solve is currently blocked by the exact-clip solve-memory "
                "temporary rather than by measured globalisation."
            ),
            "",
            (
                "Single-null 300 is a topology-read refusal in both modes: the "
                "analytic flux is classified limited with zero retained X-point "
                "candidates, so neither a trustworthy Jacobian nor a contraction "
                "arm exists there."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _completed_fixture_control(output: Path) -> dict[str, Any]:
    part = _newton_part_path(
        output,
        "fixture_exterior_control",
        certificate.DIVERTED_CASE_NAME,
        -500,
    )
    row = json.loads(part.read_text(encoding="utf-8"))
    fractions = [
        arm["requested_relative_perturbation"] for arm in row.get("perturbations", [])
    ]
    if (
        not row.get("completed")
        or not row.get("returned_for_every_perturbation")
        or fractions != list(PERTURBATION_FRACTIONS)
    ):
        raise RuntimeError(f"the persisted fixture control is incomplete: {part}")
    row["measurement_origin"] = (
        "persisted completed control from the preceding allocation"
    )
    return row


def run_newton_probe(output: Path, report_directory: Path) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the measurement requires JAX double precision")
    lane = _gpu_allocation()
    started = perf_counter()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    original_mode = support_clip_mode()
    rows = [_completed_fixture_control(output)]
    try:
        for exterior_kind, case_name, requested_cells, mode in NEWTON_ROWS:
            row = _measure_newton_row(
                output, exterior_kind, case_name, requested_cells, mode
            )
            rows.append(row)
            _write_json(
                output,
                {
                    "schema": "nova.oracle-start-newton-contraction",
                    "version": 1,
                    "source_revision": _source_revision(),
                    "production_code_modified": False,
                    "rows": rows,
                    "completed": False,
                },
            )
    finally:
        set_support_clip_mode(original_mode)
    receipt = {
        "schema": "nova.oracle-start-newton-contraction",
        "version": 1,
        "source_revision": _source_revision(),
        "production_code_modified": False,
        "lane": {
            **lane,
            "persistent_compilation_cache": cache.receipt(),
            "wall_seconds": perf_counter() - started,
            "exit_marker": "ORACLE_START_NEWTON_CONTRACTION_EXIT=0",
        },
        "design": {
            "perturbation_relative_sup_fractions": PERTURBATION_FRACTIONS,
            "smooth_random_seed": RANDOM_SEED,
            "program_identity_per_row": 1,
            "program_reuse_evidence": (
                "ForwardSolveReceipt.compilation_cache_hit on each public solve"
            ),
            "newton_steps": NEWTON_STEPS,
            "active_set_steps": ACTIVE_SET_STEPS,
            "gmres_iterations": recovery.KRYLOV_ITERATIONS,
            "warmup": 0,
            "fixed_point_tolerance": FIXED_POINT_TOLERANCE,
            "telemetry_source": "FixedPointResult without per-step observers",
            "skipped_rows": {
                "diverted-single-null 300 exact": (
                    "the production read at the analytic flux is limited and "
                    "retains zero X-point candidates"
                )
            },
            "exact_clip_solve_memory_evidence_gib": {
                "weak 500": 131.47,
                "weak 1000": 278.08,
            },
        },
        "rows": rows,
        "completed": True,
    }
    _write_json(output, receipt)
    _write_newton_report(report_directory / "newton-report.md", receipt)
    return receipt


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_NEWTON_OUTPUT)
    parser.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    receipt = run_newton_probe(arguments.output, arguments.report_directory)
    print(
        json.dumps(
            {
                "completed_rows": len(receipt["rows"]),
                "completed": receipt["completed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("ORACLE_START_NEWTON_CONTRACTION_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
