"""Measure wall-target convergence on one fixed MAST plasma carrier.

The 37-row authored control is solved once.  Count rungs rebuild only wall
response rows and wall seed samples; the terminal plasma-grid field, source
policy, conductor currents, and residual trajectory stay fixed.  Topology is
then reread post-hoc through the production saddle-aware margin diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RectBivariateSpline

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    RELAXATION,
    STEP_CAP,
    WARMUP_SWEEPS,
    _circuit_drives,
    _mast_case_from_selection,
    _passive_inclusive_case,
    _plasma_response,
    _source_response,
    _stored_circuit_fields,
    _stored_map,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.biot.null import Null1D
from nova.biot.target import FluxTarget
from nova.equilibrium.connectivity_boundary import traced_margin_candidate_diagnostics
from nova.equilibrium.forward_operator import PrescribedCurrentField
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path(
    "docs/figures/efit-baseline-demonstration/wall-resolution-ladder/receipt.json"
)
REFERENCE_SHOT = 21985
REFERENCE_SLICE = 51
SAMPLES_PER_SEGMENT = (1, 2, 4, 8)
EXPECTED_COUNTS = (36, 72, 144, 288)
SCHEMA = "isolated-wall-resolution-ladder"
NO_PREDICTED_CLOSED_BOUNDARY = "no_predicted_closed_boundary"


def _array_digest(values: Any) -> str:
    """Return a dtype-and-shape-sensitive array digest."""

    packed = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(packed.dtype.str.encode())
    digest.update(b"\0")
    digest.update(np.asarray(packed.shape, dtype=np.int64).tobytes())
    digest.update(packed.tobytes())
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    """Return a canonical digest for a JSON-compatible policy record."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sample_authored_wall(authored: np.ndarray, samples: int) -> np.ndarray:
    """Sample every authored straight segment without duplicating closure."""

    wall = np.asarray(authored, dtype=np.float64)
    if wall.ndim != 2 or wall.shape[1] != 2:
        raise ValueError("wall coordinates must have shape (row, 2)")
    if len(wall) != 37 or not np.array_equal(wall[0], wall[-1]):
        raise ValueError("the MAST control must carry 37 rows with exact closure")
    if samples not in SAMPLES_PER_SEGMENT:
        raise ValueError(f"unsupported samples per segment: {samples}")
    start = wall[:-1]
    end = np.roll(start, -1, axis=0)
    fraction = np.arange(samples, dtype=np.float64) / samples
    points = start[:, None, :] + fraction[None, :, None] * (end - start)[:, None, :]
    return points.reshape(-1, 2)


def _spacing(coordinate: np.ndarray, closure_duplicate: bool) -> dict[str, float]:
    """Return cyclic chord spacing, retaining the control's zero closure edge."""

    coordinate = np.asarray(coordinate, dtype=np.float64)
    following = np.roll(coordinate, -1, axis=0)
    spacing = np.linalg.norm(following - coordinate, axis=1)
    positive = spacing[spacing > 0.0]
    return {
        "minimum_m": float(np.min(spacing)),
        "minimum_positive_m": float(np.min(positive)),
        "maximum_m": float(np.max(spacing)),
        "closure_duplicate": closure_duplicate,
    }


def _combine_digest(named: dict[str, Any]) -> str:
    """Combine named array identities without losing their boundaries."""

    digest = hashlib.sha256()
    for name, values in sorted(named.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_array_digest(values)))
    return digest.hexdigest()


def _plasma_carrier_digest(
    case: dict[str, Any], profile, terminal_plasma_flux: np.ndarray
) -> str:
    """Bind the solved terminal plasma field into the plasma carrier identity."""

    operator = profile.operator
    grid = operator.grid
    return _combine_digest(
        {
            "grid_coordinate": case["grid_coordinate"],
            "grid_source_target": grid.source_target,
            "grid_plasma_target": grid.plasma_target,
            "grid_plasma_target_r": grid.plasma_target_r,
            "grid_plasma_target_z": grid.plasma_target_z,
            "cell_area": operator.area,
            "inside_material": operator.inside_material,
            "declared_support": operator.declared_support,
            "terminal_plasma_flux": terminal_plasma_flux,
        }
    )


def _fixed_carrier_digests(
    case: dict[str, Any],
    profile,
    current: np.ndarray,
    context: dict[str, Any],
    terminal_plasma_flux: np.ndarray,
) -> dict[str, str]:
    """Return every held-constant carrier identity used by the ladder."""

    operator = profile.operator
    grid = operator.grid
    group = context["group"]
    row = context["row"]
    source_policy = {
        "pprime": _array_digest(group["pprime"][row]),
        "ffprime": _array_digest(group["ffprime"][row]),
        "boundary_pressure": float(group["ppsi_c"][row, -1]),
        "boundary_field_function": float(group["fpsi_c"][row, -1]),
        "declared_axis_flux": float(operator.declared_axis_flux),
        "declared_boundary_flux": float(operator.declared_boundary_flux),
        "target_current_a": abs(float(case["reference"]["plasma_current_a"])),
        "requested_class": "diverted",
        "fixed_point_criterion": FIXED_POINT_CRITERION,
        "newton_steps": NEWTON_STEPS,
        "gmres_iterations": GMRES_ITERATIONS,
        "warmup_sweeps": WARMUP_SWEEPS,
        "relaxation": RELAXATION,
        "step_cap": STEP_CAP,
    }
    full_r, full_z, reference = _stored_map(group, row)
    seed_field = {
        "radius": full_r,
        "height": full_z,
        "reference_flux": reference,
        "plasma_prefix": np.asarray(case["state"])[: grid.node_number],
    }
    return {
        "plasma_carrier_sha256": _plasma_carrier_digest(
            case, profile, terminal_plasma_flux
        ),
        "terminal_plasma_flux_sha256": _array_digest(terminal_plasma_flux),
        "source_policy_sha256": _json_digest(source_policy),
        "conductor_currents_sha256": _array_digest(current),
        "seed_field_sha256": _combine_digest(seed_field),
        "plasma_seed_prefix_sha256": _array_digest(seed_field["plasma_prefix"]),
    }


def _reference_case() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the frozen 21985/51 carrier selected by the parity bank."""

    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    selected_row, qualification = next(
        (row, gate)
        for row, gate in selected
        if int(row["shot"]) == REFERENCE_SHOT
        and int(row["slice_index"]) == REFERENCE_SLICE
    )
    return _mast_case_from_selection(SHOT_STORE, selected_row, qualification)


def _solve_control(case: dict[str, Any], profile) -> tuple[np.ndarray, list[Any], Any]:
    """Solve the authored control once and retain its fixed plasma field."""

    target_current = abs(float(case["reference"]["plasma_current_a"]))
    branch = profile.solve_branch(
        jnp.asarray(case["state"]),
        TopologyClass.DIVERTED,
        route="newton_krylov",
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        warmup=WARMUP_SWEEPS,
        relaxation=RELAXATION,
        step_cap=STEP_CAP,
    )
    branch.equilibrium.flux.block_until_ready()
    trace = np.asarray(branch.equilibrium.fixed_point.trace, dtype=np.float64)
    trajectory = [float(value) if np.isfinite(value) else None for value in trace]
    grid_number = profile.operator.grid.node_number
    plasma_flux = np.asarray(branch.equilibrium.flux, dtype=np.float64)[:grid_number]
    return plasma_flux, trajectory, branch


def _build_wall_responses(
    case: dict[str, Any], context: dict[str, Any], base_profile, coordinate: np.ndarray
) -> dict[str, Any]:
    """Evaluate every varying wall row once at the finest nested cloud."""

    group = context["group"]
    row = context["row"]
    geometry = (
        MachineGeometryRegistry.default()
        .select(int(case["reference"]["shot"]))
        .configuration.geometry
    )
    families, _drive, active_mapping = _circuit_drives(group, row, geometry, "fcoil_c")
    source = _source_response(geometry, coordinate, families)
    lattice = base_profile.lattice
    plasma = _plasma_response(
        coordinate,
        lattice.coordinate,
        float(lattice.radial_step),
        float(lattice.vertical_step),
    )
    circuits, audit = _stored_circuit_fields(
        group, row, coordinate, geometry, active_mapping
    )
    prescribed = np.column_stack([circuit["response_wb_per_a"] for circuit in circuits])
    current = np.asarray(
        [circuit["fitted_current_a"] for circuit in circuits], dtype=np.float64
    )
    return {
        "source": source,
        "plasma": plasma,
        "prescribed": prescribed,
        "current": current,
        "audit": audit,
    }


def _control_batch_coordinates(
    case: dict[str, Any], authored: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact target batches used to author the stored control."""

    source_targets = np.asarray(authored, dtype=np.float64)
    stored_wall = np.asarray(case["wall_coordinate"], dtype=np.float64)
    grid_targets = np.asarray(case["grid_coordinate"], dtype=np.float64)
    if source_targets.shape != (37, 2):
        raise ValueError("the direct source control requires exactly 37 by 2 targets")
    if not np.array_equal(source_targets, stored_wall):
        raise ValueError("the direct source control changed authored wall order")
    prescribed_targets = np.vstack((grid_targets, source_targets))
    if prescribed_targets.shape != (1126, 2):
        raise ValueError(
            "the direct prescribed control requires exactly 1126 by 2 targets"
        )
    grid_number = len(grid_targets)
    if not np.array_equal(prescribed_targets[:grid_number], grid_targets):
        raise ValueError("the direct prescribed control changed grid-prefix order")
    if not np.array_equal(prescribed_targets[grid_number:], source_targets):
        raise ValueError("the direct prescribed control changed wall-suffix order")
    return source_targets, prescribed_targets


def _build_control_responses(
    case: dict[str, Any], context: dict[str, Any], base_profile, authored: np.ndarray
) -> dict[str, Any]:
    """Reconstruct the control at its original source and prescribed batches."""

    source_targets, prescribed_targets = _control_batch_coordinates(case, authored)
    group = context["group"]
    row = context["row"]
    geometry = (
        MachineGeometryRegistry.default()
        .select(int(case["reference"]["shot"]))
        .configuration.geometry
    )
    families, _drive, active_mapping = _circuit_drives(group, row, geometry, "fcoil_c")
    source = _source_response(geometry, source_targets, families)
    lattice = base_profile.lattice
    plasma = _plasma_response(
        source_targets,
        lattice.coordinate,
        float(lattice.radial_step),
        float(lattice.vertical_step),
    )
    circuits, audit = _stored_circuit_fields(
        group, row, prescribed_targets, geometry, active_mapping
    )
    prescribed = np.column_stack([circuit["response_wb_per_a"] for circuit in circuits])
    current = np.asarray(
        [circuit["fitted_current_a"] for circuit in circuits], dtype=np.float64
    )
    return {
        "source": source,
        "plasma": plasma,
        "prescribed": prescribed,
        "current": current,
        "audit": audit,
        "source_targets": source_targets,
        "prescribed_targets": prescribed_targets,
    }


def _rung_profile(
    base_profile,
    coordinate: np.ndarray,
    response: dict[str, Any],
    indices: np.ndarray,
) -> Any:
    """Replace only the wall target and prescribed response wall suffix."""

    operator = base_profile.operator
    grid_number = operator.grid.node_number
    base_prescribed = np.asarray(operator.prescribed_current_field.response)
    prescribed = np.vstack(
        (base_prescribed[:grid_number], response["prescribed"][indices])
    )
    wall = FluxTarget(
        source_target=jnp.asarray(response["source"][indices]),
        plasma_target=jnp.asarray(response["plasma"][indices]),
        null=Null1D(jnp.asarray(coordinate, dtype=jnp.float64)),
    )
    policy = PrescribedCurrentField(
        response=jnp.asarray(prescribed), current=jnp.asarray(response["current"])
    )
    return replace(
        base_profile,
        operator=replace(
            operator,
            wall=wall,
            prescribed_current_field=policy,
        ),
    )


def _fixed_plasma_state(
    profile, plasma_flux: np.ndarray, target_current: float
) -> np.ndarray:
    """Complete the fixed grid field with its self-consistent wall image."""

    wall_seed = np.zeros(profile.operator.wall.node_number, dtype=np.float64)
    state = jnp.asarray(np.r_[plasma_flux, wall_seed])
    mapped = profile.flux_map(
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
    )(state)
    wall = np.asarray(mapped, dtype=np.float64)[profile.operator.grid.node_number :]
    return np.r_[plasma_flux, wall]


def _strict(value: float) -> float | None:
    """Return finite JSON numerics without inventing sentinels."""

    return float(value) if np.isfinite(value) else None


def _contact_fields(topology, operands: dict[str, Any]) -> dict[str, Any]:
    """Publish wall contact only when the terminal has a closed boundary."""

    boundary = np.asarray(topology.boundary, dtype=np.float64)
    if boundary.shape != (2,):
        raise ValueError("the predicted boundary coordinate must have shape (2,)")
    if not np.all(np.isfinite(boundary)):
        return {
            "contact_coordinate_m": None,
            "contact_coordinate_absence_reason": NO_PREDICTED_CLOSED_BOUNDARY,
            "contact_arc_m": None,
            "contact_arc_absence_reason": NO_PREDICTED_CLOSED_BOUNDARY,
        }
    contact = np.asarray(operands["limiter_coordinate"], dtype=np.float64)
    return {
        "contact_coordinate_m": contact.tolist(),
        "contact_arc_m": operands["limiter_arc"],
    }


def _posthoc_margin_read(profile, state: np.ndarray, topology) -> dict[str, Any]:
    """Return exact selected-saddle and reachable-wall limiter operands."""

    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    coordinate = np.asarray(operator.grid.coordinate, dtype=np.float64)
    radius = np.unique(coordinate[:, 0])
    height = np.unique(coordinate[:, 1])
    grid_flux, wall_flux = operator.topology.split_flux_map(physical)
    _vmap_o, typed_candidates = operator._fixed_design_topology.grid(grid_flux)
    classification_wall = jnp.concatenate(
        (topology.wall_point, topology.wall_point_flux[None])
    )
    _axis_seed, connectivity_material = operator.connectivity_axis_seed(topology.axis)
    reading = traced_margin_candidate_diagnostics(
        grid_flux.reshape((radius.size, height.size)).T,
        jnp.asarray(radius, dtype=jnp.float64),
        jnp.asarray(height, dtype=jnp.float64),
        connectivity_material.reshape((radius.size, height.size)).T,
        topology.axis[0],
        topology.axis[1],
        96,
        18,
        operator.wall.coordinate[:, 0],
        operator.wall.coordinate[:, 1],
        wall_flux,
        typed_candidates,
        classification_wall,
    )
    host = {name: np.asarray(value) for name, value in reading.items()}
    return {
        "class_margin": float(host["class_margin"]),
        "selected_x_coordinate": host["selected_typed_candidate"][:2],
        "selected_x_flux": float(host["selected_typed_candidate"][2]),
        "limiter_coordinate": host["limiter_coordinate"],
        "limiter_flux": float(host["limiter_flux"]),
        "limiter_arc": float(host["limiter_arc"]),
    }


def _read_rung(
    profile,
    state: np.ndarray,
    trajectory: list[Any],
    coordinate: np.ndarray,
    samples: int | None,
    fixed_digests: dict[str, str],
    seed_wall: np.ndarray,
    target_current: float,
) -> dict[str, Any]:
    """Read one wall design against the held terminal plasma field."""

    mapped = np.asarray(
        profile.flux_map(
            requested_class=TopologyClass.DIVERTED,
            target_current=target_current,
        )(jnp.asarray(state)),
        dtype=np.float64,
    )
    grid_number = profile.operator.grid.node_number
    grid_scale = max(float(np.max(np.abs(mapped[:grid_number]))), 1.0e-30)
    residual = float(
        np.max(np.abs(mapped[:grid_number] - state[:grid_number])) / grid_scale
    )
    _masks, topology = profile.operator.read(jnp.asarray(state))
    operands = _posthoc_margin_read(profile, state, topology)
    margin = operands["class_margin"]
    achieved = "diverted" if margin >= 0.0 else "limited"
    if achieved == "limited":
        boundary_flux = operands["limiter_flux"]
        boundary_source = "reachable_wall_limiter"
    else:
        boundary_flux = operands["selected_x_flux"]
        boundary_source = "selected_saddle"
    return {
        "kind": "authored_37_row_control" if samples is None else "count_rung",
        "wall_target_count": int(len(coordinate)),
        "samples_per_authored_segment": samples,
        "authored_segment_count": 36,
        "spacing": _spacing(coordinate, samples is None),
        "response_shapes": {
            "source_to_wall": list(profile.operator.wall.source_target.shape),
            "plasma_to_wall": list(profile.operator.wall.plasma_target.shape),
            "prescribed_full": list(
                profile.operator.prescribed_current_field.response.shape
            ),
        },
        "boundary_flux_wb": boundary_flux,
        "boundary_operand": boundary_source,
        **_contact_fields(topology, operands),
        "achieved_class": achieved,
        "class_margin": _strict(margin),
        "class_margin_nonfinite": (
            "positive_infinity"
            if np.isposinf(margin)
            else "negative_infinity"
            if np.isneginf(margin)
            else None
        ),
        "fixed_plasma_terminal_relative_residual": residual,
        "residual_trajectory": trajectory,
        "residual_trajectory_sha256": _json_digest(trajectory),
        "fixed_carrier_digests": fixed_digests,
        "wall_seed_sha256": _array_digest(seed_wall),
    }


def _convergence(rows: list[dict[str, Any]], plasma_pitch: float) -> dict[str, Any]:
    """Bank next-finer deltas and state whether motion actually stops."""

    counts = [row["wall_target_count"] for row in rows]
    for coarse, fine in zip(rows[:-1], rows[1:], strict=True):
        coarse_contact = coarse["contact_coordinate_m"]
        fine_contact = fine["contact_coordinate_m"]
        if coarse_contact is None or fine_contact is None:
            contact_change = {
                "contact_coordinate_distance_m": None,
                "contact_coordinate_distance_absence_reason": (
                    NO_PREDICTED_CLOSED_BOUNDARY
                ),
            }
        else:
            contact_change = {
                "contact_coordinate_distance_m": float(
                    np.linalg.norm(
                        np.asarray(coarse_contact) - np.asarray(fine_contact)
                    )
                )
            }
        coarse["change_to_next_finer"] = {
            "next_finer_wall_target_count": fine["wall_target_count"],
            "boundary_flux_absolute_wb": abs(
                coarse["boundary_flux_wb"] - fine["boundary_flux_wb"]
            ),
            **contact_change,
            "class_changed": coarse["achieved_class"] != fine["achieved_class"],
        }
    rows[-1]["change_to_next_finer"] = None
    fine_step = rows[-2]["change_to_next_finer"]
    for row in rows:
        row["maximum_spacing_in_plasma_pitches"] = (
            row["spacing"]["maximum_m"] / plasma_pitch
        )
    plasma_equivalent = next(
        (
            row["wall_target_count"]
            for row in rows
            if row["maximum_spacing_in_plasma_pitches"] <= 1.0
        ),
        None,
    )
    stopped = next(
        (
            row["wall_target_count"]
            for row in rows[:-1]
            if row["change_to_next_finer"]["boundary_flux_absolute_wb"] == 0.0
            and row["change_to_next_finer"]["contact_coordinate_distance_m"] == 0.0
            and not row["change_to_next_finer"]["class_changed"]
        ),
        None,
    )
    fine_contact_distance = fine_step["contact_coordinate_distance_m"]
    if fine_contact_distance is None:
        contact_statement = (
            "contact distance is unavailable because a terminal has no "
            "predicted closed boundary."
        )
    else:
        contact_statement = f"contact by {fine_contact_distance:.6g} m."
    return {
        "count_sequence": counts,
        "plasma_grid_pitch_m": plasma_pitch,
        "first_plasma_equivalent_wall_count": plasma_equivalent,
        "limiter_operand_stop_count": stopped,
        "finest_observed_step": fine_step,
        "statement": (
            "The limiter operand does not stop moving by 288 targets; the "
            f"144-to-288 step moves boundary flux by "
            f"{fine_step['boundary_flux_absolute_wb']:.6g} Wb and "
            f"{contact_statement} The finest "
            "maximum wall spacing is "
            f"{rows[-1]['maximum_spacing_in_plasma_pitches']:.6g} "
            "plasma pitches, so no prescribed rung is plasma-equivalent."
        ),
    }


def _control_reproduction_passes(reproduction: dict[str, Any]) -> bool:
    """Require exact identity for the control coordinates, seed, and response rows."""

    identity_fields = (
        "authored_coordinate_exact",
        "control_operator_wall_coordinate_exact",
        "control_seed_exact",
    )
    batch_contract = reproduction.get("original_batch_contract")
    batch_fields = (
        "source_shape_37_by_2",
        "source_authored_order_exact",
        "prescribed_shape_1126_by_2",
        "prescribed_grid_before_wall_order_exact",
    )
    audit = reproduction.get("direct_control_audit")
    response_fields = ("source_to_wall", "plasma_to_wall", "prescribed_full")
    return (
        all(reproduction.get(name) is True for name in identity_fields)
        and isinstance(batch_contract, dict)
        and all(batch_contract.get(name) is True for name in batch_fields)
        and isinstance(audit, dict)
        and all(
            isinstance(audit.get(name), dict) and audit[name].get("array_equal") is True
            for name in response_fields
        )
    )


def validate_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate schema, ladder counts, fixed-carrier isolation, and numerics."""

    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected wall-resolution receipt schema")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("receipt must carry one control and four count rungs")
    counts = tuple(row.get("wall_target_count") for row in rows)
    if counts != (37, *EXPECTED_COUNTS):
        raise ValueError(f"unexpected wall count sequence: {counts}")
    required = (
        "boundary_flux_wb",
        "contact_coordinate_m",
        "achieved_class",
        "class_margin",
        "residual_trajectory",
        "residual_trajectory_sha256",
        "fixed_carrier_digests",
    )
    if any(any(name not in row for name in required) for row in rows):
        raise ValueError("a rung omits a required wall convergence operand")
    for row in rows:
        contact = row["contact_coordinate_m"]
        absence_reason = row.get("contact_coordinate_absence_reason")
        if contact is None:
            if absence_reason != NO_PREDICTED_CLOSED_BOUNDARY:
                raise ValueError(
                    "an absent contact must name the missing closed boundary"
                )
        else:
            coordinate = np.asarray(contact, dtype=np.float64)
            if coordinate.shape != (2,) or not np.all(np.isfinite(coordinate)):
                raise ValueError("a present contact must be a finite coordinate pair")
            if absence_reason is not None:
                raise ValueError("a present contact cannot carry an absence reason")
        recomputed = _json_digest(row["residual_trajectory"])
        if row["residual_trajectory_sha256"] != recomputed:
            raise ValueError("a residual trajectory digest does not match its values")
    trajectory = rows[0]["residual_trajectory_sha256"]
    if any(row["residual_trajectory_sha256"] != trajectory for row in rows):
        raise ValueError("residual trajectory changed across the isolated ladder")
    isolation = payload.get("isolation")
    if (
        not isinstance(isolation, dict)
        or isolation.get("wall_not_reevaluated_in_fixed_terminal_ladder") is not True
    ):
        raise ValueError(
            "the receipt must state that the fixed-terminal ladder did not "
            "re-evaluate the wall"
        )
    fixed = (
        isolation.get("fixed_carrier_digests") if isinstance(isolation, dict) else None
    )
    if not isinstance(fixed, dict):
        raise ValueError("the top-level isolation carrier map is missing")
    if any(row["fixed_carrier_digests"] != fixed for row in rows):
        raise ValueError("a row carrier map differs from the top-level isolation map")
    reproduction = payload.get("control_reproduction")
    if not isinstance(reproduction, dict):
        raise ValueError("the control reproduction record is missing")
    exact_reproduction = _control_reproduction_passes(reproduction)
    if reproduction.get("passes") is not exact_reproduction:
        raise ValueError(
            "the control reproduction pass flag disagrees with exact audits"
        )
    if not exact_reproduction:
        raise ValueError("the 37-row control did not reproduce the current operator")
    return {
        "valid": True,
        "row_count": len(rows),
        "wall_target_counts": list(counts),
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Generate a ladder that re-reads wall diagnostics on one fixed terminal."""

    configure_dtypes()
    case, context = _reference_case()
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    passive_case, base_profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    if policy["section_kernel_evaluations_this_shot"] != 0:
        raise RuntimeError("the control entered a direct response builder")
    current = np.asarray(
        base_profile.operator.prescribed_current_field.current, dtype=np.float64
    )
    plasma_flux, trajectory, control_branch = _solve_control(passive_case, base_profile)
    fixed_digests = _fixed_carrier_digests(
        passive_case, base_profile, current, context, plasma_flux
    )
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    authored = np.asarray(base_profile.operator.wall.coordinate, dtype=np.float64)
    fine_coordinate = sample_authored_wall(authored, 8)
    response = _build_wall_responses(
        passive_case, context, base_profile, fine_coordinate
    )
    if not np.array_equal(response["current"], current):
        raise RuntimeError("direct wall rows changed the prescribed current vector")

    full_r, full_z, reference = _stored_map(context["group"], context["row"])
    seed_spline = RectBivariateSpline(full_r, full_z, reference, kx=3, ky=3, s=0.0)
    control_state = np.asarray(control_branch.equilibrium.flux, dtype=np.float64)
    control_seed_wall = np.asarray(passive_case["state"])[
        base_profile.operator.grid.node_number :
    ]
    rows = [
        _read_rung(
            base_profile,
            control_state,
            trajectory,
            authored,
            None,
            fixed_digests,
            control_seed_wall,
            target_current,
        )
    ]
    rung_profiles = []
    for samples in SAMPLES_PER_SEGMENT:
        stride = 8 // samples
        indices = np.arange(0, len(fine_coordinate), stride, dtype=int)
        coordinate = fine_coordinate[indices]
        profile = _rung_profile(base_profile, coordinate, response, indices)
        state = _fixed_plasma_state(profile, plasma_flux, target_current)
        seed_wall = seed_spline.ev(coordinate[:, 0], coordinate[:, 1])
        rows.append(
            _read_rung(
                profile,
                state,
                trajectory,
                coordinate,
                samples,
                fixed_digests,
                seed_wall,
                target_current,
            )
        )
        rung_profiles.append(profile)

    control_response = _build_control_responses(
        passive_case, context, base_profile, authored
    )
    if not np.array_equal(control_response["current"], current):
        raise RuntimeError("direct control rows changed the prescribed current vector")
    unique_profile = rung_profiles[0]
    direct_control = {
        "source_to_wall": control_response["source"],
        "plasma_to_wall": control_response["plasma"],
        "prescribed_full": control_response["prescribed"],
    }
    control_arrays = {
        "source_to_wall": np.asarray(base_profile.operator.wall.source_target),
        "plasma_to_wall": np.asarray(base_profile.operator.wall.plasma_target),
        "prescribed_full": np.asarray(
            base_profile.operator.prescribed_current_field.response
        ),
    }
    grid_number = base_profile.operator.grid.node_number
    reproduction = {
        "authored_coordinate_exact": bool(
            np.array_equal(authored[:-1], unique_profile.operator.wall.coordinate)
        ),
        "control_operator_wall_coordinate_exact": bool(
            np.array_equal(case["wall_coordinate"], authored)
        ),
        "control_seed_exact": bool(
            np.array_equal(case["state"], passive_case["state"])
        ),
        "original_batch_contract": {
            "source_shape_37_by_2": control_response["source_targets"].shape == (37, 2),
            "source_authored_order_exact": bool(
                np.array_equal(control_response["source_targets"], authored)
            ),
            "prescribed_shape_1126_by_2": control_response["prescribed_targets"].shape
            == (1126, 2),
            "prescribed_grid_before_wall_order_exact": bool(
                np.array_equal(
                    control_response["prescribed_targets"][:grid_number],
                    passive_case["grid_coordinate"],
                )
                and np.array_equal(
                    control_response["prescribed_targets"][grid_number:], authored
                )
            ),
        },
        "direct_control_section_kernel_evaluations": control_response["audit"][
            "section_kernel_evaluations"
        ],
        "direct_control_audit": {
            name: {
                "array_equal": bool(np.array_equal(control_arrays[name], rebuilt)),
                "maximum_absolute_difference": float(
                    np.max(np.abs(control_arrays[name] - rebuilt))
                ),
                "maximum_relative_difference": float(
                    np.max(np.abs(control_arrays[name] - rebuilt))
                    / max(float(np.max(np.abs(control_arrays[name]))), 1.0e-300)
                ),
            }
            for name, rebuilt in direct_control.items()
        },
    }
    reproduction["passes"] = _control_reproduction_passes(reproduction)
    plasma_pitch = max(
        float(base_profile.lattice.radial_step),
        float(base_profile.lattice.vertical_step),
    )
    payload = {
        "schema": SCHEMA,
        "artifact": "isolated MAST wall-target resolution convergence",
        "reference": {
            "machine": "MAST",
            "shot": REFERENCE_SHOT,
            "slice_index": REFERENCE_SLICE,
            "arm": "passive-inclusive reference-seeded diverted branch",
        },
        "isolation": {
            "plasma_terminal_held_fixed": True,
            "residual_trajectory_held_fixed": True,
            "wall_not_reevaluated_in_fixed_terminal_ladder": True,
            "varying_inputs": [
                "wall coordinate",
                "source-to-wall rows",
                "plasma-to-wall rows",
                "prescribed-response wall rows",
                "wall seed suffix",
            ],
            "fixed_carrier_digests": fixed_digests,
            "response_carrier": carrier_evidence,
            "direct_wall_section_kernel_evaluations": response["audit"][
                "section_kernel_evaluations"
            ],
        },
        "control_reproduction": reproduction,
        "rows": rows,
        "convergence": _convergence(rows[1:], plasma_pitch),
    }
    try:
        payload["validation"] = validate_receipt(payload)
    except ValueError as error:
        payload["validation"] = {"valid": False, "error": str(error)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


def main() -> None:
    """Generate the receipt or validate an existing bank."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate", type=Path)
    arguments = parser.parse_args()
    if arguments.validate is not None:
        payload = json.loads(arguments.validate.read_text())
        print(json.dumps(validate_receipt(payload), indent=2, sort_keys=True))
        return
    payload = run(arguments.output)
    print(json.dumps(payload["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
