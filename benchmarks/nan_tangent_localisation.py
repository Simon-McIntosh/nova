"""Localise the non-finite tangent in analytic current moments.

The measurement retains the production operator unchanged.  It follows the
reduced analytic carrier from its closed-form near-root state through the
fixed-support current-moment quadrature, then repeats the public current-moment
JVP on the persisted MAST response carrier.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
from time import perf_counter
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from benchmarks.solovev_certificate import (
    _case,
    _closed_form_current_target,
    _exact_state,
)
from benchmarks.stagnation_mechanism_probe import _prepare_reference
from nova.equilibrium import ForwardProfile
from nova.equilibrium.rotation import IsothermalRotation
from nova.equilibrium.stencil_mesh import (
    StencilMesh,
    _DENSITY_UNIT_NODE,
    _DENSITY_UNIT_WEIGHT,
    _quadratic_flux_design,
)
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/gs-absolute-accuracy/nan-tangent-localisation.json"
)
CASE_NAME = "strong-rotation-compact-static"
REQUESTED_CELLS = -110
MAST_REFERENCE = (22086, 43)
EXPECTED_CPUS = 16


@contextmanager
def _stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    """Print flushed boundaries around one harvestable measurement stage."""

    started = perf_counter()
    print(f"NAN_TANGENT_STAGE_BEGIN name={name}", flush=True)
    try:
        yield
    finally:
        elapsed = perf_counter() - started
        timings[name] = elapsed
        print(
            f"NAN_TANGENT_STAGE_END name={name} elapsed_seconds={elapsed:.6f}",
            flush=True,
        )


def _strict(value: Any) -> Any:
    """Convert arrays and non-finite scalars to strict JSON values."""

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
    """Write a strict, stable receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _census(value: Any) -> dict[str, Any]:
    """Summarise finite values and retain the first non-finite index."""

    array = np.asarray(value)
    flat = array.reshape(-1)
    nonfinite = np.flatnonzero(~np.isfinite(flat))
    first = None
    if len(nonfinite):
        index = int(nonfinite[0])
        first = {
            "flat_index": index,
            "index": list(np.unravel_index(index, array.shape)),
            "kind": (
                "nan"
                if np.isnan(flat[index])
                else "positive_infinity"
                if np.isposinf(flat[index])
                else "negative_infinity"
            ),
        }
    return {
        "shape": list(array.shape),
        "finite": not len(nonfinite),
        "finite_count": int(np.count_nonzero(np.isfinite(flat))),
        "nonfinite_count": int(len(nonfinite)),
        "nan_count": int(np.count_nonzero(np.isnan(flat))),
        "first_nonfinite": first,
    }


def _jvp_census(function, state: jax.Array, direction: jax.Array) -> dict[str, Any]:
    """Measure one primal and tangent without hiding terminal non-finites."""

    value, tangent = jax.jvp(function, (state,), (direction,))
    jax.block_until_ready(tangent)
    return {
        "value": _census(value),
        "jvp": _census(tangent),
    }


def _probe_direction(
    profile: ForwardProfile, state: jax.Array, target_current: float
) -> tuple[jax.Array, dict[str, Any]]:
    """Use the unit-sup production-map residual as the tangent direction."""

    operator = profile.operator
    shadow = operator.residual_shadow_mask(state)
    map_with_shadow = operator.flux_map_with_shadow(target_current=target_current)
    mapped = jax.jit(lambda candidate: map_with_shadow(candidate, shadow))(state)
    jax.block_until_ready(mapped)
    residual = np.asarray(mapped - state, dtype=np.float64)
    residual_sup = float(np.max(np.abs(residual)))
    if residual_sup > np.finfo(np.float64).tiny:
        direction = residual / residual_sup
        construction = "unit_sup_fixed_point_residual"
    else:
        direction = np.zeros_like(residual)
        direction[int(np.argmax(np.abs(np.asarray(state))))] = 1.0
        construction = "unit_coordinate_at_largest_state_component"
    return jnp.asarray(direction), {
        "construction": construction,
        "map_residual_absolute_sup_wb": residual_sup,
        "map_residual_relative_sup": residual_sup
        / max(float(np.max(np.abs(np.asarray(mapped)))), np.finfo(np.float64).tiny),
        "direction_sup_norm": float(np.max(np.abs(direction))),
        "direction_census": _census(direction),
        "frozen_shadow_cell_count": int(np.count_nonzero(np.asarray(shadow))),
    }


def _point_segment_distance(
    point: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    edge = second - first
    squared = float(edge @ edge)
    if squared == 0.0:
        return float(np.linalg.norm(point - first))
    fraction = float(np.clip((point - first) @ edge / squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (first + fraction * edge)))


def _polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the minimum vertex-to-edge separation of two polygons."""

    distances = []
    for point in first:
        distances.extend(
            _point_segment_distance(point, edge, following)
            for edge, following in zip(second, np.roll(second, -1, axis=0), strict=True)
        )
    for point in second:
        distances.extend(
            _point_segment_distance(point, edge, following)
            for edge, following in zip(first, np.roll(first, -1, axis=0), strict=True)
        )
    return min(distances)


def _polygon_area(vertices: np.ndarray) -> float:
    following = np.roll(vertices, -1, axis=0)
    return 0.5 * abs(
        float(
            np.sum(vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1])
        )
    )


def _cell_geometry(operator, state: jax.Array, cell: int = 0) -> dict[str, Any]:
    masks, _topology, _sample_flux, support = operator._support_partition(state)
    polygon = np.asarray(operator.moment_geometry.polygons[cell], dtype=np.float64)
    wall = np.asarray(operator.wall.coordinate, dtype=np.float64)
    support_count = int(np.asarray(support.vertex_count)[cell])
    support_vertices = np.asarray(support.support_vertices)[cell, :support_count]
    polygon_area = _polygon_area(polygon)
    support_area = float(np.asarray(support.area)[cell])
    tolerance = 512.0 * np.finfo(np.float64).eps * max(polygon_area, 1.0)
    return {
        "plasma_cell_index": cell,
        "centre_rz_m": np.asarray(operator.moment_geometry.atomic_mesh.centroids)[cell],
        "area_m2": polygon_area,
        "operator_area_m2": float(np.asarray(operator.area)[cell]),
        "support_area_m2": support_area,
        "clipped": bool(abs(support_area - polygon_area) > tolerance),
        "vertex_count": int(len(polygon)),
        "support_vertex_count": support_count,
        "distance_to_wall_m": _polygon_distance(polygon, wall),
        "profile_participation": bool(np.asarray(masks.profile_participation)[cell]),
        "normalised_flux_at_centre": float(np.asarray(masks.psi_norm)[cell]),
        "vertices_rz_m": polygon,
        "support_vertices_rz_m": support_vertices,
    }


def _cell_zero_quadrature(operator, candidate: jax.Array) -> tuple[jax.Array, ...]:
    """Reproduce the fixed Duffy inputs for plasma cell zero."""

    masks, _topology, sample_flux, support = operator._support_partition(candidate)
    matching = [
        (stencil, int(np.flatnonzero(stencil.ring_centre == 0)[0]))
        for stencil in operator._support_moment_stencils
        if stencil.ring_centre is not None and np.any(stencil.ring_centre == 0)
    ]
    if len(matching) != 1:
        raise RuntimeError(f"cell zero belongs to {len(matching)} moment stencils")
    stencil, ring_index = matching[0]
    value_pool = jnp.concatenate([masks.psi_norm, sample_flux])
    gathered = value_pool[stencil.ring_gather_index]
    coefficient = jnp.einsum(
        "rps,rs->rp",
        jnp.asarray(stencil.ring_flux_weight, dtype=value_pool.dtype),
        gathered,
    )[ring_index]

    vertices = support.support_vertices[0]
    count = support.vertex_count[0]
    capacity = vertices.shape[0]
    triangle_slot = jnp.arange(1, capacity - 1)
    first = jnp.broadcast_to(vertices[:1], (capacity - 2, 2))
    second = vertices[triangle_slot]
    third = vertices[triangle_slot + 1]
    node = jnp.asarray(_DENSITY_UNIT_NODE, dtype=vertices.dtype)
    node_weight = jnp.asarray(_DENSITY_UNIT_WEIGHT, dtype=vertices.dtype)
    radial, vertical = jnp.meshgrid(node, node, indexing="ij")
    radial_weight, vertical_weight = jnp.meshgrid(
        node_weight, node_weight, indexing="ij"
    )
    radial = radial.reshape(-1)
    vertical = vertical.reshape(-1)
    rule_weight = (radial_weight * vertical_weight).reshape(-1)
    edge_first = second - first
    edge_second = third - first
    points = (
        first[:, None, :]
        + radial[None, :, None] * edge_first[:, None, :]
        + (1.0 - radial)[None, :, None]
        * vertical[None, :, None]
        * edge_second[:, None, :]
    )
    cross = jnp.abs(
        edge_first[:, 0] * edge_second[:, 1] - edge_first[:, 1] * edge_second[:, 0]
    )
    included = count >= 3
    live = (triangle_slot + 1 < count) & included
    weights = cross[:, None] * (1.0 - radial)[None, :] * rule_weight[None, :]
    weights = jnp.where(live[:, None], weights, 0.0).reshape(-1)
    points = points.reshape(-1, 2)
    sampling_centre = jnp.asarray(stencil.ring_sampling_centre[ring_index])
    coordinate_scale = jnp.asarray(stencil.ring_coordinate_scale[ring_index])
    points = jnp.where((weights > 0.0)[:, None], points, sampling_centre[None, :])
    local = (points - sampling_centre[None, :]) / coordinate_scale[None, :]
    flux = jnp.einsum("qi,i->q", _quadratic_flux_design(local), coefficient)
    return points, weights, flux


def _stage_census(
    operator, profile, state: jax.Array, direction: jax.Array
) -> dict[str, Any]:
    """Follow finite primal values through the exact cell-zero tangent path."""

    source_profile = profile.operator.source.core

    def partition(candidate):
        masks, topology, sample_flux, support = operator._support_partition(candidate)
        return (
            masks.psi_norm,
            sample_flux,
            topology.axis_flux,
            topology.flux_span,
            support.area,
        )

    partition_value, partition_tangent = jax.jvp(partition, (state,), (direction,))
    quadrature_value, quadrature_tangent = jax.jvp(
        lambda candidate: _cell_zero_quadrature(operator, candidate),
        (state,),
        (direction,),
    )
    points, weights, flux = quadrature_value
    _point_tangent, _weight_tangent, flux_tangent = quadrature_tangent
    public = {
        "support_partition": {
            "value": [_census(value) for value in partition_value],
            "jvp": [_census(value) for value in partition_tangent],
            "fields": [
                "centroid_normalised_flux",
                "sample_normalised_flux",
                "axis_flux",
                "flux_span",
                "support_area",
            ],
        },
        "cell_current_moments": _jvp_census(
            lambda candidate: jnp.stack(operator.cell_current_moments(candidate)),
            state,
            direction,
        ),
    }

    radius = points[:, 0]
    stages = {
        "quadrature_points": {"value": _census(points), "jvp": _census(_point_tangent)},
        "quadrature_weights": {
            "value": _census(weights),
            "jvp": _census(_weight_tangent),
        },
        "quadrature_normalised_flux": {
            "value": _census(flux),
            "jvp": _census(flux_tangent),
        },
    }
    closure_functions = {
        "temperature": source_profile.rotation.temperature,
        "angular_frequency": source_profile.rotation.angular_frequency,
        "centrifugal_exponent": source_profile.rotation.centrifugal_exponent,
        "centrifugal_exponent_gradient": (
            source_profile.rotation.centrifugal_exponent_gradient
        ),
        "centrifugal_factor": (
            lambda argument: source_profile.rotation.centrifugal_factor(
                radius, argument
            )
        ),
        "pressure_gradient": lambda argument: source_profile.pressure_gradient(
            radius, argument
        ),
        "current_density": lambda argument: source_profile.current_density(
            radius, argument
        ),
    }
    for name, function in closure_functions.items():
        value, tangent = jax.jvp(function, (flux,), (flux_tangent,))
        stages[name] = {"value": _census(value), "jvp": _census(tangent)}

    density = source_profile.current_density(radius, flux)
    _, density_tangent = jax.jvp(
        lambda argument: source_profile.current_density(radius, argument),
        (flux,),
        (flux_tangent,),
    )
    stages["weighted_density"] = {
        "value": _census(density * weights),
        "jvp": _census(density_tangent * weights),
    }
    first_nonfinite = next(
        (
            {"stage": name, **entry["jvp"]["first_nonfinite"]}
            for name, entry in stages.items()
            if entry["jvp"]["first_nonfinite"] is not None
        ),
        None,
    )
    return {
        "public_intermediates": public,
        "cell_zero_quadrature_trace": stages,
        "first_nonfinite_stage": first_nonfinite,
    }


def _debug_nan_evidence(
    rotation: IsothermalRotation, flux: jax.Array, flux_tangent: jax.Array
) -> dict[str, Any]:
    """Capture the eager debug-nans report at the first source operation."""

    error = None
    try:
        with jax.debug_nans(True):
            jax.jvp(
                rotation.angular_frequency,
                (flux,),
                (flux_tangent,),
            )
    except FloatingPointError as exception:
        error = str(exception)
    function = rotation.angular_frequency
    source_lines, line = inspect.getsourcelines(function)
    path = inspect.getsourcefile(function)
    return {
        "raised": error is not None,
        "exception": error,
        "operation": "jnp.sqrt(2 * rotation_parameter * temperature / mass)",
        "autodiff_primitive_reported": (
            "mul" if error is not None and "mul" in error else None
        ),
        "file": str(Path(path).resolve().relative_to(ROOT)) if path else None,
        "line": line
        + next(index for index, text in enumerate(source_lines) if "jnp.sqrt" in text),
        "class": "sqrt at a zero argument",
        "zero_argument_cause": (
            "the static Solovev source carries rotation_parameter=0, so the sqrt "
            "primal is zero while its forward derivative evaluates the singular "
            "zero-argument sqrt rule"
        ),
    }


def _solovev_measurement() -> dict[str, Any]:
    carrier_case, source_case, exact = _case(CASE_NAME)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        REQUESTED_CELLS,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = jnp.asarray(_exact_state(CASE_NAME, exact, coordinates))
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, np.asarray(state)
    )
    coefficients = empty_operator.coupling_current_moments(exact_physical)
    internal = oracle_fixture._internal_flux_image(empty_operator, coefficients)
    operator = oracle_fixture.forward_operator(
        source_case, machine, np.asarray(state) - internal
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    target_current, _centroid, _current_receipt = _closed_form_current_target(
        CASE_NAME, source_case, operator, exact_physical
    )
    direction, direction_receipt = _probe_direction(profile, state, target_current)
    trace = _stage_census(operator, profile, state, direction)
    quadrature_value, quadrature_tangent = jax.jvp(
        lambda candidate: _cell_zero_quadrature(operator, candidate)[2],
        (state,),
        (direction,),
    )
    debug = _debug_nan_evidence(
        operator.source.core.rotation, quadrature_value, quadrature_tangent
    )
    return {
        "case": CASE_NAME,
        "requested_cells": REQUESTED_CELLS,
        "realised_cells": int(len(machine.node)),
        "state": {
            "kind": "closed_form_near_root",
            "sha256": _array_digest(state),
            "primal_census": _census(state),
        },
        "probe_direction": direction_receipt,
        "cell_zero_geometry": _cell_geometry(operator, state),
        "trace": trace,
        "jax_debug_nans": debug,
        "cell_current_moments_jvp_finite": trace["public_intermediates"][
            "cell_current_moments"
        ]["jvp"]["finite"],
        "first_nonfinite_operation": debug,
    }


def _mast_measurement() -> dict[str, Any]:
    response_cache, carrier = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    if MAST_REFERENCE not in selected:
        raise RuntimeError(f"frozen selection lacks MAST reference {MAST_REFERENCE}")
    profile, seed, target_current, reference = _prepare_reference(
        *selected[MAST_REFERENCE], response_cache
    )
    state = jnp.asarray(seed)
    direction, direction_receipt = _probe_direction(profile, state, target_current)
    moments = _jvp_census(
        lambda candidate: jnp.stack(profile.operator.cell_current_moments(candidate)),
        state,
        direction,
    )
    return {
        "machine": "MAST",
        "shot": MAST_REFERENCE[0],
        "slice_index": MAST_REFERENCE[1],
        "state": {
            "kind": "persisted frozen-six production seed",
            "sha256": _array_digest(state),
            "reference": reference,
        },
        "target_current_a": target_current,
        "probe_direction": direction_receipt,
        "cell_current_moments": moments,
        "cell_current_moments_jvp_finite": moments["jvp"]["finite"],
        "carrier": carrier,
    }


def _lane() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    platforms = os.environ.get("JAX_PLATFORMS", "")
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    if not job_id:
        raise RuntimeError("measurement requires a SLURM allocation")
    if cpus != EXPECTED_CPUS:
        raise RuntimeError(
            f"measurement requires {EXPECTED_CPUS} CPUs, received {cpus}"
        )
    if platforms != "cpu":
        raise RuntimeError(
            f"measurement requires JAX_PLATFORMS=cpu, received {platforms!r}"
        )
    if "rigel" not in partition:
        raise RuntimeError(
            f"measurement requires a Rigel partition, received {partition!r}"
        )
    return {
        "slurm_job_id": job_id,
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": partition,
        "cpu_count": cpus,
        "jax_platforms": platforms,
        "tmpdir": os.environ.get("TMPDIR"),
        "threaded_settings": {
            name: os.environ.get(name)
            for name in (
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def measure(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run both carriers on one allocation and write the localisation receipt."""

    started = perf_counter()
    lane = _lane()
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    timings: dict[str, float] = {}
    with _stage("solovev_near_root", timings):
        solovev = _solovev_measurement()
    with _stage("mast_persisted_carrier", timings):
        mast = _mast_measurement()

    oracle_specific = (
        not solovev["cell_current_moments_jvp_finite"]
        and mast["cell_current_moments_jvp_finite"]
    )
    proposed_fix = (
        "In the analytic fixture, return an identically zero angular-frequency "
        "function when rotation_parameter is zero, before evaluating jnp.sqrt."
    )
    receipt = {
        "schema": {
            "type": "object",
            "required": [
                "headline",
                "solovev",
                "mast_comparison",
                "classification",
                "proposed_fix",
                "lane",
            ],
        },
        "headline": {
            "first_nonfinite_operation": solovev["first_nonfinite_operation"],
            "sentence": (
                "The static analytic fixture differentiates jnp.sqrt at its zero "
                "rotation argument; the resulting NaN tangent enters current "
                "density before current normalisation."
            ),
        },
        "solovev": solovev,
        "mast_comparison": mast,
        "classification": {
            "oracle_machine_specific": oracle_specific,
            "general": not oracle_specific,
            "sentence": (
                "The defect is specific to the static analytic-oracle profile "
                "construction; the persisted MAST frozen-six current-moment JVP "
                "is finite."
                if oracle_specific
                else "The same non-finite current-moment tangent is present on MAST."
            ),
        },
        "proposed_fix": proposed_fix,
        "persistent_compilation_cache": {
            "directory": str(cache.directory),
            "version": cache.version_key,
        },
        "stage_wall_seconds": timings,
        "lane": {
            **lane,
            "elapsed_seconds": perf_counter() - started,
            "exit_marker": "NAN_TANGENT_LOCALISATION_EXIT=0",
        },
    }
    _write_json(output, receipt)
    print(f"NAN_TANGENT_RECEIPT path={output}", flush=True)
    print("NAN_TANGENT_LOCALISATION_EXIT=0", flush=True)
    return receipt


def validate(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Fail closed on the evidence fields required by the receipt contract."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    required = set(payload["schema"]["required"])
    if missing := required.difference(payload):
        raise ValueError(f"receipt misses required fields {sorted(missing)}")
    operation = payload["headline"]["first_nonfinite_operation"]
    if not operation["file"] or not operation["line"] or not operation["class"]:
        raise ValueError("first non-finite operation lacks source location or class")
    geometry = payload["solovev"]["cell_zero_geometry"]
    for field in ("area_m2", "clipped", "vertex_count", "distance_to_wall_m"):
        if field not in geometry:
            raise ValueError(f"cell-zero geometry lacks {field}")
    if payload["mast_comparison"]["cell_current_moments_jvp_finite"] is not True:
        raise ValueError("MAST comparison did not produce a finite JVP")
    if payload["lane"]["exit_marker"] != "NAN_TANGENT_LOCALISATION_EXIT=0":
        raise ValueError("receipt lacks the successful exit marker")
    print(f"NAN_TANGENT_VALIDATION_OK path={path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.command == "measure":
        measure(arguments.output)
    else:
        validate(arguments.input)


if __name__ == "__main__":
    main()
