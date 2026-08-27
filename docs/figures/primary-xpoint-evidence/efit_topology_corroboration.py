"""Corroborate Nova wall topology against the independent MAST EFIT geometry.

EFIT is a magnetics-fitted reconstruction, not ground truth.  This measurement
therefore compares geometry and topology labels only; it does not use a
magnetics-reproduction score, which would privilege the reconstruction fitted
to those measurements.
"""

from __future__ import annotations

import gc
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import time
from typing import Any

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.connectivity_boundary import traced_margin_candidate_diagnostics
from nova.equilibrium.boundary_comparison import (
    BoundaryMode,
    classify_boundary_mode,
    compare_closed_boundaries,
)
from nova.equilibrium.observation import ConstraintViolationError
from nova.equilibrium.separatrix_branches import assemble_separatrix_branches
from nova.equilibrium.topology import NoQualifiedAxisError
from nova.imas.mast_efit_referee import read_efit_referee
from nova.imas.mast_geometry import DD_VERSION
from nova.imas.mast_solve_inputs import (
    RECONSTRUCTION_GROUP,
    SOURCE_CONVENTION,
    TARGET_CONVENTION,
)
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parent
OUTPUT_PNG = HERE / "efit-topology-corroboration.png"
OUTPUT_JSON = HERE / "efit-topology-corroboration.json"
CACHE_PATH = HERE / ".efit-topology-corroboration-cache.npz"
REACHABILITY_SCRIPT = HERE / "real_equilibria_reachability.py"
SELECTION_COMMIT = "80706f89"
CACHE_SCHEMA_REVISION = 3
RESAMPLE_POINTS = 2000
CURVE_SAMPLES_PER_SEGMENT = 9


def _reachability_module():
    """Load the existing frozen-six reconstruction driver without duplicating it."""

    spec = spec_from_file_location("real_equilibria_reachability", REACHABILITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reconstruction driver {REACHABILITY_SCRIPT}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_polyline(points: np.ndarray, *, close: bool) -> np.ndarray:
    """Return finite points, optionally with one explicit closing point."""

    result = np.asarray(points, dtype=float)
    result = result[np.isfinite(result).all(axis=1)]
    if len(result) < 3:
        raise RuntimeError("a comparison boundary carries fewer than three points")
    if close and not np.allclose(result[0], result[-1], rtol=0.0, atol=1.0e-12):
        result = np.vstack((result, result[0]))
    return result


def _sample_cubic_controls(controls: np.ndarray) -> np.ndarray | None:
    """Sample ordered cubic controls without joining separate branches."""

    controls = np.asarray(controls, dtype=float)
    if controls.ndim != 3 or controls.shape[1:] != (4, 2) or not len(controls):
        return None
    coordinate = np.linspace(
        0.0, 1.0, CURVE_SAMPLES_PER_SEGMENT, endpoint=False, dtype=float
    )
    basis = np.column_stack(
        (
            (1.0 - coordinate) ** 3,
            3.0 * (1.0 - coordinate) ** 2 * coordinate,
            3.0 * (1.0 - coordinate) * coordinate**2,
            coordinate**3,
        )
    )
    points = np.einsum("tk,skd->std", basis, controls).reshape(-1, 2)
    return np.vstack((points, controls[-1, -1]))


def _assembled_branch_polylines(
    geometry: dict[str, Any],
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Return the valid closed branch and separately typed valid open legs."""

    required = ("radius", "height", "flux", "axis", "boundary_flux")
    if any(name not in geometry for name in required):
        return None, []
    try:
        radius = np.asarray(geometry["radius"], dtype=float)
        height = np.asarray(geometry["height"], dtype=float)
        flux = np.asarray(geometry["flux"], dtype=float)
        axis = np.asarray(geometry["axis"], dtype=float)
        level = float(geometry["boundary_flux"])
    except TypeError, ValueError:
        return None, []
    if (
        radius.ndim != 1
        or height.ndim != 1
        or flux.shape != (height.size, radius.size)
        or axis.shape != (2,)
        or not np.isfinite(radius).all()
        or not np.isfinite(height).all()
        or not np.isfinite(flux).all()
        or not np.isfinite(axis).all()
        or not np.isfinite(level)
    ):
        return None, []

    assembled = jax.device_get(
        assemble_separatrix_branches(
            jnp.asarray(flux),
            jnp.asarray(radius),
            jnp.asarray(height),
            jnp.asarray(level),
            jnp.asarray(axis),
        )
    )
    if not bool(assembled["well_formed"]):
        return None, []
    closed = _sample_cubic_controls(
        np.asarray(assembled["closed_controls_rz"])[
            np.asarray(assembled["closed_valid"], dtype=bool)
        ]
    )
    legs = []
    for index in np.flatnonzero(np.asarray(assembled["open_branch_valid"], dtype=bool)):
        leg = _sample_cubic_controls(
            np.asarray(assembled["open_controls_rz"])[index][
                np.asarray(assembled["open_valid"], dtype=bool)[index]
            ]
        )
        if leg is not None:
            legs.append(leg)
    return closed, legs


def _segment_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray | None:
    """Return the intersection of two closed line segments, if unique."""

    first_direction = first_end - first_start
    second_direction = second_end - second_start
    cross = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(float(cross)) <= 1.0e-14:
        return None
    offset = second_start - first_start
    first_fraction = (
        offset[0] * second_direction[1] - offset[1] * second_direction[0]
    ) / cross
    second_fraction = (
        offset[0] * first_direction[1] - offset[1] * first_direction[0]
    ) / cross
    if 0.0 <= first_fraction <= 1.0 and 0.0 <= second_fraction <= 1.0:
        return first_start + first_fraction * first_direction
    return None


def _boundary_wall_contacts(boundary: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Return geometrically identifiable EFIT LCFS and governed-wall crossings."""

    boundary = _finite_polyline(boundary, close=True)
    wall = _finite_polyline(wall, close=True)
    contacts: list[np.ndarray] = []
    for boundary_start, boundary_end in zip(boundary[:-1], boundary[1:], strict=True):
        for wall_start, wall_end in zip(wall[:-1], wall[1:], strict=True):
            contact = _segment_intersection(
                boundary_start, boundary_end, wall_start, wall_end
            )
            if contact is None:
                continue
            if not any(
                np.linalg.norm(contact - retained) <= 1.0e-8 for retained in contacts
            ):
                contacts.append(contact)
    return np.stack(contacts) if contacts else np.empty((0, 2), dtype=float)


def _strict_value(value: float) -> float | None:
    """Convert a finite value for strict JSON, retaining absence as null."""

    return float(value) if np.isfinite(value) else None


def _post_cutover_geometry(profile, state, topology) -> dict[str, Any]:
    """Read the exact saddle-aware class operands used in production."""

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
    host = jax.device_get(reading)
    selected = np.asarray(host["selected_typed_candidate"], dtype=float)
    margin = float(host["class_margin"])
    limiter = np.asarray(host["limiter_coordinate"], dtype=float)
    limiter_flux = float(host["limiter_flux"])
    try:
        achieved_mode = classify_boundary_mode(margin)
    except ValueError:
        achieved_mode = None
    achieved_class = achieved_mode.value if achieved_mode is not None else None
    binding_flux = (
        float(selected[2])
        if achieved_mode is BoundaryMode.DIVERTED and np.isfinite(selected[2])
        else limiter_flux
        if achieved_mode is BoundaryMode.LIMITED and np.isfinite(limiter_flux)
        else float("nan")
    )
    return {
        "achieved_class": achieved_class,
        "class_margin": margin,
        "binding_flux": binding_flux,
        "selected_saddle": selected[:2],
        "limiter_coordinate": limiter,
    }


def _carrier_semantic_identity(carrier_evidence: dict[str, Any]) -> str:
    """Return the persisted response carrier's semantic cache identity."""

    carrier = carrier_evidence.get("carrier", carrier_evidence)
    identity = carrier.get("semantic_response_identity")
    if not isinstance(identity, str) or not identity:
        raise RuntimeError("the persisted response carrier has no semantic identity")
    return identity


def _cache_authority(carrier_evidence: dict[str, Any]) -> dict[str, Any]:
    """Return the exact authority tuple that makes cached operands reusable."""

    return {
        "schema_revision": CACHE_SCHEMA_REVISION,
        "response_carrier_semantic_identity": _carrier_semantic_identity(
            carrier_evidence
        ),
        "selection_source_commit": SELECTION_COMMIT,
    }


def _write_operand_cache(
    rows: list[dict[str, Any]], carrier_evidence: dict[str, Any]
) -> None:
    """Persist exact solve operands before any plotting or distance reduction."""

    metadata_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        prefix = f"arm_{index:02d}"
        metadata_row = {
            key: row[key]
            for key in (
                "identity",
                "shot",
                "slice_index",
                "time_s",
                "arm",
                "efit_label",
                "nova_achieved_class",
                "converged",
                "terminal_residual",
                "tolerance",
                "termination_reason",
            )
        }
        metadata_row["failure_exception_class"] = row.get("failure_exception_class")
        metadata_rows.append(metadata_row)
        for name in (
            "radius",
            "height",
            "flux",
            "axis",
            "wall",
            "binding_flux",
            "selected_saddle",
            "limiter_coordinate",
            "class_margin",
            "efit_lcfs",
            "efit_x_points",
        ):
            arrays[f"{prefix}_{name}"] = np.asarray(row[name])
    metadata = _cache_authority(carrier_evidence) | {
        "purpose": (
            "unbanked exact solve operands for reproducible plot-only regeneration"
        ),
        "arm_count": len(rows),
        "rows": metadata_rows,
    }
    temporary = CACHE_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    temporary.replace(CACHE_PATH)


def _read_operand_cache(
    carrier_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load exact solve operands after fail-closed authority validation."""

    expected = _cache_authority(carrier_evidence)
    with np.load(CACHE_PATH, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        observed = {key: metadata.get(key) for key in expected}
        if observed != expected:
            raise RuntimeError(
                f"intermediate operand cache is stale: expected {expected}, "
                f"observed {observed}"
            )
        if int(metadata.get("arm_count", -1)) != 12:
            raise RuntimeError("intermediate operand cache does not carry twelve arms")
        rows: list[dict[str, Any]] = []
        for index, metadata_row in enumerate(metadata["rows"]):
            prefix = f"arm_{index:02d}"
            row = dict(metadata_row)
            for name in (
                "radius",
                "height",
                "flux",
                "axis",
                "wall",
                "binding_flux",
                "selected_saddle",
                "limiter_coordinate",
                "class_margin",
                "efit_lcfs",
                "efit_x_points",
            ):
                row[name] = np.array(stored[f"{prefix}_{name}"], copy=True)
            rows.append(row)
    return rows


def _build_operand_cache(
    response_cache, carrier_evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run each production arm once and persist its exact corroboration operands."""

    reachability = _reachability_module()
    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    rows: list[dict[str, Any]] = []
    for selected_row, qualification in selected:
        shot = int(selected_row["shot"])
        slice_index = int(selected_row["slice_index"])
        print(f"solving MAST {shot}/{slice_index}", flush=True)
        started = time.perf_counter()
        try:
            rows.extend(
                _build_identity_operands(
                    reachability,
                    response_cache,
                    selected_row,
                    qualification,
                )
            )
        finally:
            jax.clear_caches()
            gc.collect()
        elapsed = time.perf_counter() - started
        print(
            f"completed MAST {shot}/{slice_index} in {elapsed:.3f} s; "
            "released compilation caches",
            flush=True,
        )
    _write_operand_cache(rows, carrier_evidence)
    print(f"wrote exact operand cache {CACHE_PATH}", flush=True)
    return rows


def _build_identity_operands(
    reachability,
    response_cache,
    selected_row: dict[str, Any],
    qualification,
) -> list[dict[str, Any]]:
    """Build both arms while one identity's solve objects have bounded lifetime."""

    shot = int(selected_row["shot"])
    slice_index = int(selected_row["slice_index"])
    case, context = _mast_case_from_selection(SHOT_STORE, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    if policy["section_kernel_evaluations_this_shot"] != 0:
        raise RuntimeError("MAST reconstruction entered a direct response builder")
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    states = reachability._mast_states(
        profile, jnp.asarray(passive_case["state"]), target_current
    )
    referee = read_efit_referee(shot, store=SHOT_STORE)
    referee_usable = bool(referee.usable[slice_index])
    efit_lcfs = np.asarray(referee.lcfs_m[slice_index], dtype=float)
    efit_lcfs = efit_lcfs[np.isfinite(efit_lcfs).all(axis=1)]
    efit_x_points = referee.x_points_m[slice_index]
    efit_x_points = efit_x_points[np.isfinite(efit_x_points).all(axis=1)]
    efit_label = (
        ("diverted" if bool(referee.diverted[slice_index]) else "limited")
        if referee_usable
        else None
    )
    return [
        _build_arm_operand(
            reachability,
            profile,
            arm_result,
            identity=f"{shot}/{slice_index}",
            shot=shot,
            slice_index=slice_index,
            time_s=float(referee.time_s[slice_index]),
            arm=arm,
            efit_label=efit_label,
            efit_lcfs=efit_lcfs,
            efit_x_points=efit_x_points,
        )
        for arm, arm_result in states.items()
    ]


def _build_arm_operand(
    reachability,
    profile,
    arm_result,
    *,
    identity: str,
    shot: int,
    slice_index: int,
    time_s: float,
    arm: str,
    efit_label: str | None,
    efit_lcfs: np.ndarray,
    efit_x_points: np.ndarray,
) -> dict[str, Any]:
    """Build one arm operand or retain its named host-side qualification failure."""

    state = arm_result.state
    common = {
        "identity": identity,
        "shot": shot,
        "slice_index": slice_index,
        "time_s": time_s,
        "arm": arm,
        "tolerance": arm_result.tolerance,
        "efit_label": efit_label,
        "efit_lcfs": efit_lcfs,
        "efit_x_points": efit_x_points,
    }
    try:
        geometry = reachability._grid_geometry(profile, state)
        _masks, topology = profile.operator.read(state)
        post_cutover = _post_cutover_geometry(profile, state, topology)
    except (NoQualifiedAxisError, ConstraintViolationError) as error:
        exception_class = type(error).__name__
        return common | {
            "converged": False,
            "terminal_residual": None,
            "termination_reason": exception_class,
            "failure_exception_class": exception_class,
            "nova_achieved_class": None,
            "radius": np.empty(0),
            "height": np.empty(0),
            "flux": np.empty((0, 0)),
            "axis": np.full(2, np.nan),
            "wall": np.empty((0, 2)),
            "binding_flux": np.asarray(np.nan),
            "selected_saddle": np.full(2, np.nan),
            "limiter_coordinate": np.full(2, np.nan),
            "class_margin": np.asarray(np.nan),
        }
    return common | {
        "converged": arm_result.converged,
        "terminal_residual": arm_result.terminal_residual,
        "termination_reason": arm_result.termination_reason,
        "failure_exception_class": None,
        "nova_achieved_class": post_cutover["achieved_class"],
        "radius": geometry["radius"],
        "height": geometry["height"],
        "flux": geometry["flux"],
        "axis": geometry["axis"],
        "wall": geometry["wall"],
        "binding_flux": post_cutover["binding_flux"],
        "selected_saddle": post_cutover["selected_saddle"],
        "limiter_coordinate": post_cutover["limiter_coordinate"],
        "class_margin": post_cutover["class_margin"],
    }


def _draw_panel(axis, row: dict[str, Any]) -> None:
    """Draw one arm with both independent and Nova geometry."""

    if row.get("failure_exception_class") is not None:
        axis.text(
            0.5,
            0.5,
            f"Geometry unavailable\n{row['failure_exception_class']}",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title(
            f"({row['panel']}) {row['identity']} · {row['arm']} · non-converged"
        )
        axis.set_axis_off()
        return
    wall = np.asarray(row["wall_m"], dtype=float)
    efit_lcfs = np.asarray(row["efit_lcfs_m"], dtype=float)
    nova_boundary = np.asarray(row["nova_closed_boundary_m"], dtype=float)
    axis.plot(wall[:, 0], wall[:, 1], color="#8a8a8a", linewidth=1.0)
    if efit_lcfs.ndim == 2 and len(efit_lcfs):
        axis.plot(efit_lcfs[:, 0], efit_lcfs[:, 1], color="#087e8b", linewidth=1.8)
    if nova_boundary.ndim == 2 and len(nova_boundary):
        axis.plot(
            nova_boundary[:, 0], nova_boundary[:, 1], color="#d1495b", linewidth=1.35
        )
    for leg in row["nova_open_legs_m"]:
        leg = np.asarray(leg, dtype=float)
        axis.plot(leg[:, 0], leg[:, 1], color="#d1495b", linewidth=1.0, alpha=0.7)
    for point in row["efit_x_points_m"]:
        axis.scatter(
            *point, marker="+", s=48, color="#087e8b", linewidths=1.6, zorder=5
        )
    if row["nova_selected_saddle_m"] is not None:
        axis.scatter(
            *row["nova_selected_saddle_m"],
            marker="X",
            s=34,
            color="#d1495b",
            zorder=6,
        )
    if row["nova_limiter_point_m"] is not None:
        axis.scatter(
            *row["nova_limiter_point_m"], marker="D", s=24, color="#f2c14e", zorder=6
        )
    agreement = (
        "AGREE"
        if row["label_agreement"] is True
        else "DISAGREE"
        if row["label_agreement"] is False
        else "UNAVAILABLE"
    )
    axis.set_title(
        f"{row['panel']}  MAST {row['identity']} {row['arm']}\n"
        f"EFIT {row['efit_label']} · Nova {row['nova_achieved_class']} · {agreement}",
        loc="left",
        fontsize=8.3,
        fontweight="semibold",
    )
    metric_text = (
        f"LCFS sup {row['binding_to_efit_lcfs_sup_m']:.3f} m · "
        f"RMS {row['binding_to_efit_lcfs_rms_m']:.3f} m\n"
        f"X nearest {row['selected_saddle_to_efit_x_point_m']:.3f} m"
        if row["binding_to_efit_lcfs_sup_m"] is not None
        and row["binding_to_efit_lcfs_rms_m"] is not None
        and row["selected_saddle_to_efit_x_point_m"] is not None
        else "comparison unavailable: " + ", ".join(row["comparison_failures"])
    )
    axis.text(
        0.02,
        0.02,
        metric_text,
        transform=axis.transAxes,
        fontsize=6.8,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(0.08, 1.72)
    axis.set_ylim(-1.9, 1.9)
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.spines[["top", "right"]].set_visible(False)


def _finite_point_list(point: object) -> list[float] | None:
    """Serialize one finite physical point without emitting non-finite JSON."""

    try:
        result = np.asarray(point, dtype=float)
    except TypeError, ValueError:
        return None
    return (
        result.tolist() if result.shape == (2,) and np.isfinite(result).all() else None
    )


def _finite_points(points: object) -> np.ndarray:
    """Return finite physical points for strict serialization and plotting."""

    try:
        result = np.asarray(points, dtype=float)
    except TypeError, ValueError:
        return np.empty((0, 2), dtype=float)
    if result.ndim != 2 or result.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    return result[np.isfinite(result).all(axis=1)]


def _score_operand(operand: dict[str, Any]) -> dict[str, Any]:
    """Build one bank row even when one or more comparison inputs are absent."""

    converged_value = operand.get("converged")
    if not isinstance(converged_value, bool | np.bool_):
        raise RuntimeError("a corroboration operand lacks an explicit convergence flag")
    converged = bool(converged_value)
    terminal_value = operand.get("terminal_residual")
    terminal_residual = (
        None if terminal_value is None else _strict_value(float(terminal_value))
    )
    tolerance = _strict_value(float(operand.get("tolerance")))
    termination_reason = operand.get("termination_reason")
    if tolerance is None or tolerance <= 0.0:
        raise RuntimeError(
            "a corroboration operand lacks a positive residual tolerance"
        )
    if not isinstance(termination_reason, str) or not termination_reason:
        raise RuntimeError("a corroboration operand lacks a termination reason")
    qualified_terminal = bool(
        converged
        and terminal_residual is not None
        and terminal_residual <= tolerance
        and termination_reason == "converged"
    )
    failure_exception_class = operand.get("failure_exception_class")
    if failure_exception_class is not None:
        if not isinstance(failure_exception_class, str) or not failure_exception_class:
            raise RuntimeError("a retained arm failure lacks an exception class")
        return {
            "identity": operand.get("identity"),
            "shot": operand.get("shot"),
            "slice_index": operand.get("slice_index"),
            "time_s": operand.get("time_s"),
            "arm": operand.get("arm"),
            "converged": False,
            "terminal_residual": None,
            "tolerance": tolerance,
            "termination_reason": termination_reason,
            "failure_exception_class": failure_exception_class,
            "qualified_terminal": False,
            "efit_label": operand.get("efit_label"),
            "nova_achieved_class": None,
            "nova_post_cutover_class_margin": None,
            "nova_post_cutover_class_margin_nonfinite": None,
            "label_agreement": None,
            "rms_threshold_eligible": False,
            "binding_to_efit_lcfs_sup_m": None,
            "binding_to_efit_lcfs_rms_m": None,
            "nova_selected_saddle_m": None,
            "efit_x_points_m": None,
            "selected_saddle_to_efit_x_point_m": None,
            "comparison_failures": [
                f"arm_geometry_exception:{failure_exception_class}"
            ],
            "nova_limiter_point_m": None,
            "efit_boundary_wall_contacts_m": None,
            "limiter_to_efit_boundary_wall_contact_m": None,
            "limiter_contact_metric_unavailable_reason": termination_reason,
            "nova_closed_boundary_m": None,
            "nova_open_legs_m": None,
            "efit_lcfs_m": None,
            "wall_m": None,
        }

    geometry = {
        "radius": operand.get("radius"),
        "height": operand.get("height"),
        "flux": operand.get("flux"),
        "axis": operand.get("axis"),
        "boundary_flux": operand.get("binding_flux"),
    }
    closed_boundary, open_legs = _assembled_branch_polylines(geometry)
    comparison = compare_closed_boundaries(
        closed_boundary,
        operand.get("efit_lcfs"),
        class_margin=operand.get("class_margin"),
        reference_mode=operand.get("efit_label"),
        predicted_saddle_rz_m=operand.get("selected_saddle"),
        reference_x_points_rz_m=operand.get("efit_x_points"),
        sample_count=RESAMPLE_POINTS,
    )

    efit_lcfs = _finite_points(operand.get("efit_lcfs"))
    wall = _finite_points(operand.get("wall"))
    try:
        contacts = _boundary_wall_contacts(efit_lcfs, wall)
    except RuntimeError, TypeError, ValueError, IndexError:
        contacts = np.empty((0, 2), dtype=float)
    limiter = _finite_point_list(operand.get("limiter_coordinate"))
    contact_distance = (
        float(
            np.min(np.linalg.norm(contacts - np.asarray(limiter, dtype=float), axis=1))
        )
        if len(contacts) and limiter is not None
        else None
    )
    class_margin = operand.get("class_margin")
    try:
        margin = float(class_margin)
    except TypeError, ValueError:
        margin = float("nan")
    achieved_class = (
        comparison.achieved_mode.value if comparison.achieved_mode is not None else None
    )
    reference_class = (
        comparison.reference_mode.value
        if comparison.reference_mode is not None
        else operand.get("efit_label")
    )
    return {
        "identity": operand.get("identity"),
        "shot": operand.get("shot"),
        "slice_index": operand.get("slice_index"),
        "time_s": operand.get("time_s"),
        "arm": operand.get("arm"),
        "converged": converged,
        "terminal_residual": terminal_residual,
        "tolerance": tolerance,
        "termination_reason": termination_reason,
        "failure_exception_class": None,
        "qualified_terminal": qualified_terminal,
        "efit_label": reference_class,
        "nova_achieved_class": achieved_class,
        "nova_post_cutover_class_margin": _strict_value(margin),
        "nova_post_cutover_class_margin_nonfinite": (
            "positive_infinity"
            if np.isposinf(margin)
            else "negative_infinity"
            if np.isneginf(margin)
            else None
        ),
        "label_agreement": comparison.topology_class_agreement,
        "rms_threshold_eligible": bool(
            qualified_terminal
            and comparison.topology_class_agreement is True
            and comparison.symmetric_rms_distance_m is not None
        ),
        "binding_to_efit_lcfs_sup_m": comparison.symmetric_sup_distance_m,
        "binding_to_efit_lcfs_rms_m": comparison.symmetric_rms_distance_m,
        "nova_selected_saddle_m": _finite_point_list(operand.get("selected_saddle")),
        "efit_x_points_m": _finite_points(operand.get("efit_x_points")).tolist(),
        "selected_saddle_to_efit_x_point_m": comparison.x_point_distance_m,
        "comparison_failures": list(comparison.failures),
        "nova_limiter_point_m": limiter,
        "efit_boundary_wall_contacts_m": contacts.tolist(),
        "limiter_to_efit_boundary_wall_contact_m": contact_distance,
        "limiter_contact_metric_unavailable_reason": (
            None
            if contact_distance is not None
            else (
                "the stored EFIT LCFS has no geometric intersection "
                "with the governed wall"
                if limiter is not None
                else "the Nova limiter coordinate is unavailable"
            )
        ),
        "nova_closed_boundary_m": (
            closed_boundary.tolist() if closed_boundary is not None else None
        ),
        "nova_open_legs_m": [leg.tolist() for leg in open_legs],
        "efit_lcfs_m": efit_lcfs.tolist(),
        "wall_m": wall.tolist(),
    }


def _rms_threshold_eligibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Name RMS-eligible arms while retaining the fixed cohort denominator."""

    return {
        "eligible_count": sum(row["rms_threshold_eligible"] for row in rows),
        "declared_arm_denominator": len(rows),
        "eligible_arms": [
            f"{row['identity']} {row['arm']}"
            for row in rows
            if row["rms_threshold_eligible"]
        ],
        "excluded_nonconverged_arms": [
            f"{row['identity']} {row['arm']}" for row in rows if not row["converged"]
        ],
    }


def run() -> dict[str, Any]:
    """Regenerate the twelve-arm corroboration figure and strict-JSON receipt."""

    configure_dtypes()
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    cache_reused = False
    if CACHE_PATH.exists():
        try:
            operand_rows = _read_operand_cache(carrier_evidence)
            cache_reused = True
            print(f"reusing exact operand cache {CACHE_PATH}", flush=True)
        except RuntimeError as error:
            print(f"rejecting stale operand cache: {error}", flush=True)
            operand_rows = _build_operand_cache(response_cache, carrier_evidence)
    else:
        operand_rows = _build_operand_cache(response_cache, carrier_evidence)

    rows = [_score_operand(operand) for operand in operand_rows]

    for index, row in enumerate(rows):
        row["panel"] = chr(ord("A") + index)
    figure, axes = plt.subplots(6, 2, figsize=(10.2, 24.0), constrained_layout=True)
    for axis, row in zip(axes.ravel(), rows, strict=True):
        _draw_panel(axis, row)
    figure.legend(
        handles=[
            Line2D([0], [0], color="#087e8b", lw=2, label="EFIT efm LCFS"),
            Line2D([0], [0], color="#d1495b", lw=2, label="Nova binding contour"),
            Line2D(
                [0], [0], marker="+", color="#087e8b", lw=0, label="EFIT efm X-point"
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="#d1495b",
                lw=0,
                label="Nova selected saddle",
            ),
            Line2D(
                [0], [0], marker="D", color="#f2c14e", lw=0, label="Nova limiter point"
            ),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(figure)

    agreements = [row for row in rows if row["label_agreement"] is True]
    disagreements = [row for row in rows if row["label_agreement"] is False]
    unavailable = [row for row in rows if row["label_agreement"] is None]

    def metric_range(name: str) -> tuple[float | None, float | None]:
        values = [row[name] for row in rows if row[name] is not None]
        return (min(values), max(values)) if values else (None, None)

    boundary_sup_range = metric_range("binding_to_efit_lcfs_sup_m")
    boundary_rms_range = metric_range("binding_to_efit_lcfs_rms_m")
    x_point_range = metric_range("selected_saddle_to_efit_x_point_m")
    payload = {
        "artifact": "independent EFIT topology corroboration of Nova wall reachability",
        "headline": (
            f"All {len(rows)} declared MAST arms are retained: "
            f"{len(agreements)} agree with the EFIT label, "
            f"{len(disagreements)} disagree, and {len(unavailable)} carry "
            "explicitly unavailable comparisons."
        ),
        "qualification": (
            "EFIT efm is an independent magnetics-fitted reconstruction, not truth. "
            "No magnetics-reproduction metric is used because that would not be a "
            "neutral cross-source comparison."
        ),
        "project_absolute_src": (
            "/nova/figures/primary-xpoint-evidence/efit-topology-corroboration.png"
        ),
        "data_authority": {
            "efit_reader": "nova.imas.mast_efit_referee.read_efit_referee",
            "efit_group": RECONSTRUCTION_GROUP,
            "catalogue_store": str(SHOT_STORE),
            "imas_machine_description_dd_version_written_and_opened": DD_VERSION,
            "catalogue_format_qualification": (
                "efm is a raw read-only Zarr catalogue group rather than an IMAS "
                "IDS, so it has no DD version to infer or open; the governed "
                "machine-description seam it is compared within is authored and "
                "reopened at its written DD version"
            ),
            "source_cocos": SOURCE_CONVENTION,
            "target_cocos": TARGET_CONVENTION,
            "cocos_resolution": (
                "source COCOS 3 is the measured MAST convention declared by "
                "mast_solve_inputs; the solve seam converts typed quantities to DD "
                "COCOS 17. R and Z are ONE_LIKE geometric coordinates, so their "
                "conversion factor is exactly one and the EFIT geometry is overlaid "
                "without a sign or scale edit"
            ),
            "nova_selection_source_commit": SELECTION_COMMIT,
            "carrier": carrier_evidence,
            "intermediate_operand_cache": {
                "path": str(CACHE_PATH),
                "banked_artifact": False,
                "reused_for_this_render": cache_reused,
                "authority": _cache_authority(carrier_evidence),
                "exact_operands_per_arm": [
                    "binding_flux",
                    "contained_selected_saddle",
                    "refined_limiter_coordinate",
                    "governed_wall_polyline",
                    "full_flux_grid_with_tensor_axes",
                ],
            },
            "legacy_label_trap": (
                "ForwardTopologyState.diverted is the legacy geometric topology "
                "label and is not the post-cutover achieved class. This artifact "
                "derives achieved class only from the saddle-aware class margin: "
                "non-negative or positive infinity is diverted; negative is limited."
            ),
        },
        "distance_method": {
            "boundary_sup_m": (
                "shared symmetric sampled Hausdorff distance after 2000-point "
                "arc-length resampling of the valid spline-assembled closed branch "
                "and the closed EFIT polyline"
            ),
            "boundary_rms_m": (
                "root mean square of both directed nearest-polyline sample "
                "distances on the same per-branch resampling"
            ),
            "compound_path_rule": (
                "the spline assembler types one axis-enclosing closed branch and "
                "separate open legs; only the valid closed branch enters metrics"
            ),
            "x_point_m": "Nova selected saddle to the nearest finite EFIT efm X-point",
            "limiter_contact_m": (
                "Nova limiter point to the nearest exact EFIT LCFS and "
                "governed-wall segment intersection; null when none exists"
            ),
        },
        "label_comparison": {
            "cohort_description": (
                "six frozen MAST DIVERTED-LABEL EFIT references, two Nova arms each"
            ),
            "arm_count": len(rows),
            "agreement_count": len(agreements),
            "disagreement_count": len(disagreements),
            "unavailable_count": len(unavailable),
            "disagreements": [
                f"{row['identity']} {row['arm']}" for row in disagreements
            ],
            "rms_threshold_eligibility": _rms_threshold_eligibility(rows),
        },
        "summary": {
            "boundary_sup_m_min": boundary_sup_range[0],
            "boundary_sup_m_max": boundary_sup_range[1],
            "boundary_rms_m_min": boundary_rms_range[0],
            "boundary_rms_m_max": boundary_rms_range[1],
            "x_point_distance_m_min": x_point_range[0],
            "x_point_distance_m_max": x_point_range[1],
            "identifiable_efit_boundary_wall_contact_count": sum(
                bool(row["efit_boundary_wall_contacts_m"]) for row in rows
            ),
        },
        "rows": rows,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "label_comparison": result["label_comparison"],
            },
            indent=2,
        )
    )
