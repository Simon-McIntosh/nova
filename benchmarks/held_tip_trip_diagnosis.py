"""Record topology evidence at the start of every production active-set trip.

The measurement patches the in-memory active-set implementation only long
enough to add an ordered callback at each frozen-mask trip boundary.  It does
not replace the solve, alter any numerical value, or modify Nova source.  Each
arm runs in its own detached checkout so imports, source lines, and revision
identity remain attributable to exactly one candidate tree.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import efit_forward_parity_slice as parity
from benchmarks.receipt_raster_check import _profile_and_seed
from nova.equilibrium import fixed_point
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache


ROOT = Path(os.environ.get("NOVA_DIAG_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile")
TRIP_BUDGET = 24
REFERENCE_SHOT = 22086
REFERENCE_SLICE = 43
FLOAT_TOLERANCE = 5.0e-11


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_location(value: Any) -> dict[str, Any]:
    value = inspect.unwrap(value)
    lines, start = inspect.getsourcelines(value)
    path = Path(inspect.getsourcefile(value) or "")
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        relative = path
    return {
        "path": str(relative),
        "start_line": start,
        "end_line": start + len(lines) - 1,
    }


def _scheduler() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "node": os.environ.get("SLURMD_NODENAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "temporary_directory": os.environ.get("TMPDIR"),
    }


def _require_h200() -> None:
    device = jax.devices()[0]
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError("measurement requires one JAX-visible H200")
    scheduler = _scheduler()
    if scheduler["partition"] != "betelgeuse":
        raise RuntimeError("measurement requires partition betelgeuse")
    if scheduler["reservation"] != "gpu_0003_grpA":
        raise RuntimeError("measurement requires reservation gpu_0003_grpA")
    if scheduler["temporary_directory"] != "/tmp":
        raise RuntimeError("measurement requires TMPDIR=/tmp")


class _TripTrace:
    """Collect ordered start-state and end-of-trip callbacks from one solve."""

    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.ends: list[dict[str, Any]] = []

    def start(self, index, state, mask) -> None:
        trip = int(np.asarray(index)) + 1
        expected = len(self.starts) + 1
        if trip != expected:
            raise RuntimeError(f"trip-start callback order {trip} != {expected}")
        self.starts.append(
            {
                "trip": trip,
                "state": np.asarray(state, dtype=np.float64).copy(),
                "mask": np.asarray(mask, dtype=bool).copy(),
            }
        )

    def end(
        self, active, trip_index, mask_difference, live_residual, inner_iterations
    ) -> None:
        if not bool(np.asarray(active)):
            return
        self.ends.append(
            {
                "trip": int(np.asarray(trip_index)) + 1,
                "mask_difference": int(np.asarray(mask_difference)),
                "residual": float(np.asarray(live_residual)),
                "attempted_newton_promotions": int(np.asarray(inner_iterations)),
            }
        )


def _instrument_active_set(trace: _TripTrace):
    """Return the original functions after adding runtime-only trip callbacks."""
    original_active_set = fixed_point._active_set_newton_krylov
    original_trip_printer = fixed_point._print_active_set_trip
    source = inspect.getsource(original_active_set)
    first_marker = "    first_result, first_globalization = solve_frozen(\n"
    later_marker = "            inner_result, inner_globalization = solve_frozen(\n"
    if source.count(first_marker) != 1 or source.count(later_marker) != 1:
        raise RuntimeError("active-set source no longer has the two trip-start anchors")
    first_callback = (
        "    jax.debug.callback(\n"
        "        _held_tip_trip_start_callback,\n"
        "        jnp.asarray(0, dtype=jnp.int32),\n"
        "        initial,\n"
        "        initial_mask,\n"
        "        ordered=True,\n"
        "    )\n"
    )
    later_callback = (
        "            jax.debug.callback(\n"
        "                _held_tip_trip_start_callback,\n"
        "                jnp.asarray(index, dtype=jnp.int32),\n"
        "                solve_state,\n"
        "                carry.mask,\n"
        "                ordered=True,\n"
        "            )\n"
    )
    instrumented = source.replace(first_marker, first_callback + first_marker)
    instrumented = instrumented.replace(later_marker, later_callback + later_marker)
    fixed_point.__dict__["_held_tip_trip_start_callback"] = trace.start
    namespace: dict[str, Any] = {}
    exec(
        compile(instrumented, str(Path(fixed_point.__file__)), "exec"),
        fixed_point.__dict__,
        namespace,
    )
    fixed_point._active_set_newton_krylov = namespace["_active_set_newton_krylov"]
    fixed_point._print_active_set_trip = trace.end
    return original_active_set, original_trip_printer


def _restore_active_set(original_active_set, original_trip_printer) -> None:
    fixed_point._active_set_newton_krylov = original_active_set
    fixed_point._print_active_set_trip = original_trip_printer
    fixed_point.__dict__.pop("_held_tip_trip_start_callback", None)


def _point_inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return a host-side ray-cast verdict for every candidate point."""
    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if not len(points):
        return np.empty(0, dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    x0 = polygon[:, 0]
    y0 = polygon[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    crosses = (y0[None, :] > y[:, None]) != (y1[None, :] > y[:, None])
    denominator = np.where(y1 != y0, y1 - y0, 1.0)
    crossing_x = x0[None, :] + (
        (y[:, None] - y0[None, :]) * (x1 - x0)[None, :] / denominator[None, :]
    )
    return np.count_nonzero(crosses & (x[:, None] < crossing_x), axis=1) % 2 == 1


def _legacy_census(locator, grid_flux, wall: np.ndarray) -> dict[str, Any]:
    candidate, fitted_mask = locator._candidate_census(grid_flux)
    retained = locator(grid_flux)
    candidate = np.asarray(candidate, dtype=np.float64)
    fitted_mask = np.asarray(fitted_mask, dtype=bool)
    retained = np.asarray(retained, dtype=np.float64)
    stencil = np.asarray(locator.locator.stencil, dtype=np.int64)
    origin = np.asarray(locator.locator.physical_origin, dtype=np.float64)
    ring_values = np.asarray(grid_flux, dtype=np.float64)[stencil]
    crossing = np.asarray(locator.locator.crossing_count(ring_values), dtype=np.int64)
    polarity = int(getattr(locator, "extremum_polarity", 0) or 0)
    retained_rows = []
    for kind_index, kind_name in enumerate(("o", "x")):
        finite = np.all(np.isfinite(retained[kind_index]), axis=1)
        source_indices = np.flatnonzero(fitted_mask[kind_index])[: len(finite)]
        for slot in np.flatnonzero(finite):
            source = int(source_indices[slot])
            values = retained[kind_index, slot]
            containment = crossing[source] == (0 if kind_name == "o" else 4)
            polarity_consistent = (
                bool(values[3] == polarity) if kind_name == "o" and polarity else True
            )
            retained_rows.append(
                {
                    "type": kind_name,
                    "slot": int(slot),
                    "position_m": values[:2],
                    "value": float(values[2]),
                    "kind": float(values[3]),
                    "origin_index": int(stencil[source, 0]),
                    "origin_position_m": origin[source],
                    "ring_crossing_count": int(crossing[source]),
                    "containment_admitted": bool(containment),
                    "polarity_consistent": polarity_consistent,
                    "inside_wall": bool(
                        _point_inside_polygon(values[None, :2], wall)[0]
                    ),
                    "admitted": True,
                }
            )
    diagnostic_rows = []
    for index in np.flatnonzero(np.any(fitted_mask, axis=0)):
        values = candidate[index]
        diagnostic_rows.append(
            {
                "source_row": int(index),
                "origin_index": int(stencil[index, 0]),
                "origin_position_m": origin[index],
                "candidate": values,
                "ring_crossing_count": int(crossing[index]),
                "fit_extremum": bool(fitted_mask[0, index]),
                "fit_saddle": bool(fitted_mask[1, index]),
                "inside_wall": bool(_point_inside_polygon(values[None, :2], wall)[0]),
            }
        )
    return {
        "authority": "local_quadratic_fit_with_host_ring_audit",
        "containment_is_selection_rule": False,
        "precapacity_count": [
            int(np.count_nonzero(fitted_mask[0])),
            int(np.count_nonzero(fitted_mask[1])),
        ],
        "capacity": int(retained.shape[1]),
        "admitted_candidates": retained_rows,
        "diagnostic_rows": diagnostic_rows,
    }


def _structured_census(locator, grid_flux, wall: np.ndarray) -> dict[str, Any]:
    census = jax.tree.map(np.asarray, locator.candidate_census(grid_flux))
    retained = np.asarray(census["retained_candidate"], dtype=np.float64)
    valid = np.asarray(census["retained_valid"], dtype=bool)
    origin_index = np.asarray(
        census["retained_representative_origin_index"], dtype=np.int64
    )
    origin_position = np.asarray(
        census["retained_representative_origin_rz"], dtype=np.float64
    )
    multiplicity = np.asarray(census["retained_multiplicity"], dtype=np.int64)
    polarity = int(getattr(locator, "extremum_polarity", 0) or 0)
    retained_rows = []
    for kind_index, kind_name in enumerate(("o", "x")):
        for slot in np.flatnonzero(valid[kind_index]):
            values = retained[kind_index, slot]
            polarity_consistent = (
                bool(values[3] == polarity) if kind_name == "o" and polarity else True
            )
            retained_rows.append(
                {
                    "type": kind_name,
                    "slot": int(slot),
                    "position_m": values[:2],
                    "value": float(values[2]),
                    "kind": float(values[3]),
                    "origin_index": int(origin_index[kind_index, slot]),
                    "origin_position_m": origin_position[kind_index, slot],
                    "ring_crossing_count": 0 if kind_name == "o" else 4,
                    "containment_admitted": True,
                    "polarity_consistent": polarity_consistent,
                    "inside_wall": bool(
                        _point_inside_polygon(values[None, :2], wall)[0]
                    ),
                    "multiplicity": int(multiplicity[kind_index, slot]),
                    "admitted": bool(polarity_consistent),
                }
            )
    candidate = np.asarray(census["candidate"], dtype=np.float64)
    crossing = np.asarray(census["ring_crossing_count"], dtype=np.int64)
    source_origin = np.asarray(census["source_origin_index"], dtype=np.int64)
    locator_origin = np.asarray(locator.locator.physical_origin, dtype=np.float64)
    diagnostic_rows = []
    for index in range(candidate.shape[0]):
        row = {
            "source_row": index,
            "origin_index": int(source_origin[index]),
            "origin_position_m": locator_origin[index],
            "candidate": candidate[index],
            "ring_crossing_count": int(crossing[index]),
            "ring_admitted_o": bool(census["ring_admitted_mask"][0, index]),
            "ring_admitted_x": bool(census["ring_admitted_mask"][1, index]),
            "resolution_limited": bool(census["ring_resolution_limited"][index]),
            "polish_converged": bool(census["polish_converged"][index]),
            "requested_displacement_m": float(census["requested_displacement"][index]),
            "cell_width_m": float(census["cell_width"][index]),
            "within_cell": bool(census["within_cell"][index]),
            "hessian_type": int(census["hessian_type"][index]),
            "hessian_type_agrees_o": bool(census["hessian_type_agrees"][0, index]),
            "hessian_type_agrees_x": bool(census["hessian_type_agrees"][1, index]),
            "polish_rejected_o": bool(census["polish_rejected"][0, index]),
            "polish_rejected_x": bool(census["polish_rejected"][1, index]),
            "representative_o": bool(census["representative_mask"][0, index]),
            "representative_x": bool(census["representative_mask"][1, index]),
            "gradient_norm": float(census["spline_gradient_norm"][index]),
            "hessian_determinant": float(census["spline_hessian_determinant"][index]),
        }
        diagnostic_rows.append(row)
    return {
        "authority": "spline_sampled_ring_containment",
        "containment_is_selection_rule": True,
        "precapacity_count": np.asarray(census["same_root_count"], dtype=np.int64),
        "capacity": np.asarray(census["capacity"], dtype=np.int64),
        "overflow": np.asarray(census["overflow"], dtype=bool),
        "admitted_candidates": retained_rows,
        "diagnostic_rows": diagnostic_rows,
        "spline_authored": bool(np.asarray(census["spline_authored"])),
    }


def _scalar_subset(values: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for name in names:
        if name in values:
            array = np.asarray(values[name])
            result[name] = array if array.ndim else array.item()
    return result


def _trip_topology(profile, start: dict[str, Any], previous_private) -> dict[str, Any]:
    operator = profile.operator
    state = jnp.asarray(start["state"])
    physical = state[: operator.physical_node_number]
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    masks, topology, _connected, axis_admitted = operator._fixed_design_read(
        physical, requested
    )
    grid_flux, _wall_flux = operator._fixed_design_topology.split_flux_map(physical)
    locator = operator._fixed_design_topology.grid
    wall = np.asarray(operator.wall.coordinate, dtype=np.float64)
    census = (
        _structured_census(locator, grid_flux, wall)
        if hasattr(locator, "candidate_census")
        else _legacy_census(locator, grid_flux, wall)
    )
    connectivity = jax.tree.map(
        np.asarray,
        operator._connectivity_read(physical, topology, classify=True),
    )
    margin = float(np.asarray(connectivity["class_margin"]))
    achieved_class = (
        "unresolved"
        if not np.isfinite(margin) and not np.isinf(margin)
        else ("diverted" if margin >= 0.0 else "limited")
    )
    private = np.asarray(masks.private_flux, dtype=bool).reshape(-1)
    grid_coordinate = np.asarray(operator.grid.coordinate, dtype=np.float64)
    changed = (
        np.flatnonzero(private != previous_private)
        if previous_private is not None
        else np.empty(0, dtype=np.int64)
    )
    changed_cells = [
        {
            "index": int(index),
            "position_m": grid_coordinate[index],
            "was_private": bool(previous_private[index]),
            "is_private": bool(private[index]),
        }
        for index in changed
    ]
    selected_axis = np.asarray(topology.axis, dtype=np.float64)
    selected_x = np.asarray(topology.x_point, dtype=np.float64)
    for candidate in census["admitted_candidates"]:
        point = np.asarray(candidate["position_m"], dtype=np.float64)
        selected = selected_axis if candidate["type"] == "o" else selected_x
        candidate["selected_primary"] = bool(
            np.linalg.norm(point - selected) <= FLOAT_TOLERANCE
        )
    connectivity_scalars = _scalar_subset(
        connectivity,
        (
            "class_margin",
            "u_wall",
            "u_xpoint",
            "wall_shadowed",
            "limiter_wall_node_index",
            "limiter_wall_node_arc",
            "limiter_wall_node_r",
            "limiter_wall_node_z",
            "limiter_wall_node_psi",
            "limiter_arc",
            "limiter_r",
            "limiter_z",
            "limiter_psi",
            "limiter_axis_flux_distance",
            "limiter_refinement_shift",
            "limiter_flux_from_global_surface",
            "limiter_selected_from_exact_nodes",
            "x_binding_state",
            "boundary_resolved",
        ),
    )
    return {
        "trip": start["trip"],
        "state_sha256": hashlib.sha256(
            np.asarray(start["state"], dtype=np.float64).tobytes()
        ).hexdigest(),
        "frozen_mask_cell_count": int(np.count_nonzero(start["mask"])),
        "axis_admitted": bool(np.asarray(axis_admitted)),
        "selected_axis_m": selected_axis,
        "selected_axis_value": float(np.asarray(topology.axis_flux)),
        "selected_x_m": selected_x,
        "selected_x_value": float(np.asarray(topology.x_point_flux)),
        "wall_tangency_m": np.asarray(topology.wall_point, dtype=np.float64),
        "wall_tangency_value": float(np.asarray(topology.wall_point_flux)),
        "limiter_minimum": connectivity_scalars,
        "class": achieved_class,
        "class_margin": margin,
        "private_mask_cell_count": int(np.count_nonzero(private)),
        "private_mask_changed_cell_count": int(changed.size),
        "private_mask_changed_cells": changed_cells,
        "census": census,
        "_private_mask": private,
    }


def _production_program(profile, target_current: float):
    def production(initial_flux):
        return profile.solve_branch(
            initial_flux,
            TopologyClass.DIVERTED,
            target_current=target_current,
            route="newton_krylov",
            tolerance=parity.FIXED_POINT_CRITERION,
            warmup=parity.WARMUP_SWEEPS,
            newton_steps=parity.NEWTON_STEPS,
            gmres_iterations=parity.GMRES_ITERATIONS,
            relaxation=parity.RELAXATION,
            step_cap=parity.STEP_CAP,
            active_set_steps=TRIP_BUDGET,
            stream_active_set=True,
        )

    return production


def measure(output: Path, label: str, base_revision: str, held_commit: str) -> dict:
    configure_dtypes()
    _require_h200()
    cache_root = Path(os.environ.get("NOVA_DIAG_CACHE", DEFAULT_CACHE))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = configure_persistent_compilation_cache(
        cache_root, minimum_compile_seconds=0.0
    )
    case, profile, target_current, carrier, policy = _profile_and_seed()
    reference = case["reference"]
    if (
        int(reference["shot"]) != REFERENCE_SHOT
        or int(reference["slice_index"]) != REFERENCE_SLICE
    ):
        raise RuntimeError(f"unexpected reference case {reference}")
    state = jnp.asarray(case["state"])
    trace = _TripTrace()
    originals = _instrument_active_set(trace)
    production = _production_program(profile, target_current)
    try:
        compile_started = time.perf_counter()
        executable = jax.jit(production).lower(state).compile()
        compile_wall = time.perf_counter() - compile_started
        solve_started = time.perf_counter()
        result = executable(state)
        result.equilibrium.flux.block_until_ready()
        solve_wall = time.perf_counter() - solve_started
    finally:
        _restore_active_set(*originals)
    history = result.equilibrium.fixed_point
    trip_count = int(np.asarray(history.active_set_iterations))
    if len(trace.starts) != trip_count or len(trace.ends) != trip_count:
        raise AssertionError(
            "callback counts disagree with the solver receipt: "
            f"starts={len(trace.starts)}, ends={len(trace.ends)}, result={trip_count}"
        )
    trips = []
    previous_private = None
    for start, end in zip(trace.starts, trace.ends, strict=True):
        if start["trip"] != end["trip"]:
            raise AssertionError(
                f"trip callback mismatch {start['trip']} != {end['trip']}"
            )
        topology = _trip_topology(profile, start, previous_private)
        previous_private = topology.pop("_private_mask")
        topology["residual_after_trip"] = end["residual"]
        topology["mask_difference_after_trip"] = end["mask_difference"]
        topology["attempted_newton_promotions"] = end["attempted_newton_promotions"]
        trips.append(topology)
    termination_code = int(np.asarray(history.termination_reason))
    termination = fixed_point.FixedPointTerminationReason(termination_code).name.lower()
    for trip in trips:
        trip["termination_after_trip"] = (
            termination if trip["trip"] == trip_count else "continue"
        )
    locator = profile.operator._fixed_design_topology.grid
    payload = {
        "schema": "nova.held_tip_trip_topology.v1",
        "complete": True,
        "captured_at": datetime.now(UTC).isoformat(),
        "label": label,
        "revision": _revision(),
        "base_revision": base_revision or None,
        "held_commit": held_commit or None,
        "reference": reference,
        "scheduler": _scheduler(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "device": str(jax.devices()[0]),
            "device_kind": jax.devices()[0].device_kind,
        },
        "persistent_compilation_cache": cache.receipt(),
        "carrier": carrier,
        "field_policy": policy,
        "source_locations": {},
        "provenance_complete": False,
        "compile_wall_s": compile_wall,
        "solve_wall_s": solve_wall,
        "result": {
            "active_set_trips": trip_count,
            "terminal_residual": float(np.asarray(history.residual)),
            "termination_code": termination_code,
            "termination_reason": termination,
            "requested_class": "diverted",
            "achieved_class": TopologyClass(
                int(np.asarray(result.achieved_class))
            ).name.lower(),
            "topology_consistent": bool(np.asarray(result.topology_consistent)),
            "converged": bool(np.asarray(result.converged)),
        },
        "trips": trips,
    }
    # Persist the completed numerical measurement before optional source
    # provenance inspection so an inspection failure cannot discard the arm.
    _write_json(output, payload)
    payload["source_locations"] = {
        "active_set_solver": _source_location(fixed_point._active_set_newton_krylov),
        "candidate_locator": _source_location(type(locator)),
        "candidate_selection": _source_location(
            profile.operator._fixed_design_topology.x_point_index
        ),
        "connectivity_read": _source_location(profile.operator._connectivity_read),
    }
    payload["provenance_complete"] = True
    _write_json(output, payload)
    print(
        "ARM_COMPLETE "
        + json.dumps(
            {
                "label": label,
                "revision": payload["revision"],
                "trips": trip_count,
                "residual": payload["result"]["terminal_residual"],
                "termination": termination,
                "class": payload["result"]["achieved_class"],
                "topology_consistent": payload["result"]["topology_consistent"],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def _candidate_distance(candidate: dict[str, Any], point: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(candidate["position_m"], dtype=np.float64) - point)
    )


def _first_lost_x(arm: dict[str, Any]) -> dict[str, Any] | None:
    trips = arm["trips"]
    for previous, current in zip(trips, trips[1:]):
        prior_x = np.asarray(previous["selected_x_m"], dtype=np.float64)
        admitted = [
            row
            for row in current["census"]["admitted_candidates"]
            if row["type"] == "x" and row["admitted"]
        ]
        if admitted:
            nearest_admitted = min(
                _candidate_distance(row, prior_x) for row in admitted
            )
            widths = [
                row.get("cell_width_m")
                for row in current["census"].get("diagnostic_rows", [])
                if row.get("cell_width_m") is not None
            ]
            tolerance = max(widths, default=0.061)
            if nearest_admitted <= tolerance:
                continue
        diagnostics = current["census"].get("diagnostic_rows", [])
        finite = [
            row
            for row in diagnostics
            if row.get("candidate") is not None
            and np.all(np.isfinite(np.asarray(row["candidate"][:2], dtype=float)))
        ]
        rejected = (
            min(
                finite,
                key=lambda row: np.linalg.norm(
                    np.asarray(row["candidate"][:2], dtype=float) - prior_x
                ),
            )
            if finite
            else None
        )
        rule = "unresolved_null_reporting"
        if rejected is not None:
            if rejected.get("ring_crossing_count") != 4:
                rule = "spline_sampled_ring_sign_count"
            elif rejected.get("within_cell") is False:
                rule = "polish_confined_to_detected_cell"
            elif rejected.get("hessian_type_agrees_x") is False:
                rule = "polarity_consistency"
            elif rejected.get("representative_x") is False:
                rule = "same_root_deduplication"
        return {
            "trip": current["trip"],
            "previous_selected_x_m": prior_x,
            "nearest_admitted_distance_m": (
                min(
                    (_candidate_distance(row, prior_x) for row in admitted),
                    default=None,
                )
            ),
            "rejecting_rule": rule,
            "rejected_candidate": rejected,
        }
    return None


def _first_departure(reference: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    comparable = min(len(reference["trips"]), len(arm["trips"]))
    quantities = (
        ("wall_minimum_position", "wall_tangency_m"),
        ("wall_minimum_value", "wall_tangency_value"),
        ("census_axis_value", "selected_axis_value"),
        ("census_x_value", "selected_x_value"),
        ("census_axis_position", "selected_axis_m"),
        ("census_x_position", "selected_x_m"),
        ("line_of_sight_or_connectivity", "class_margin"),
        ("private_mask", "private_mask_cell_count"),
    )
    for index in range(comparable):
        left = reference["trips"][index]
        right = arm["trips"][index]
        differences = []
        for meaning, key in quantities:
            if left[key] is None and right[key] is None:
                continue
            a = np.asarray(left[key], dtype=np.float64)
            b = np.asarray(right[key], dtype=np.float64)
            delta = float(np.max(np.abs(a - b)))
            if not np.isfinite(delta) or delta > FLOAT_TOLERANCE:
                differences.append(
                    {
                        "quantity": meaning,
                        "field": key,
                        "main": left[key],
                        "arm": right[key],
                        "absolute_delta": delta,
                    }
                )
        if differences:
            return {"trip": index + 1, "differences": differences}
    return {
        "trip": comparable + 1 if len(arm["trips"]) > comparable else None,
        "differences": [
            {
                "quantity": "additional_active_set_trip",
                "main_trip_count": len(reference["trips"]),
                "arm_trip_count": len(arm["trips"]),
            }
        ],
    }


def _selected_x_admission_audit(arm: dict[str, Any]) -> dict[str, Any]:
    admitted = []
    rejected = []
    for trip in arm["trips"]:
        selected = [
            row
            for row in trip["census"]["admitted_candidates"]
            if row["type"] == "x" and row["selected_primary"]
        ]
        if selected and selected[0]["admitted"]:
            candidate = selected[0]
            diagnostic = next(
                (
                    row
                    for row in trip["census"].get("diagnostic_rows", [])
                    if row.get("origin_index") == candidate.get("origin_index")
                    and row.get("representative_x")
                ),
                None,
            )
            admitted.append(
                {
                    "trip": trip["trip"],
                    "candidate": candidate,
                    "diagnostic": diagnostic,
                }
            )
        else:
            rejected.append(
                {
                    "trip": trip["trip"],
                    "selected_x_m": trip["selected_x_m"],
                    "selected_x_value": trip["selected_x_value"],
                }
            )
    return {
        "trip_count": len(arm["trips"]),
        "selected_x_admitted_trip_count": len(admitted),
        "first_rejected_trip": rejected[0] if rejected else None,
        "first_trip_selected_x_evidence": admitted[0] if admitted else None,
    }


def _first_class_difference(
    reference: dict[str, Any], arm: dict[str, Any]
) -> dict[str, Any] | None:
    for left, right in zip(reference["trips"], arm["trips"]):
        if left["class"] != right["class"]:
            return {
                "trip": right["trip"],
                "main_class": left["class"],
                "arm_class": right["class"],
                "main_class_margin": left["class_margin"],
                "arm_class_margin": right["class_margin"],
                "main_limiter_minimum": left["limiter_minimum"],
                "arm_limiter_minimum": right["limiter_minimum"],
            }
    return None


def _draw(arms: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(len(arms), 1, figsize=(14, 3.4 * len(arms)), sharex=False)
    if len(arms) == 1:
        axes = [axes]
    colors = {"limited": "#f5c16c", "diverted": "#8fd3c7", "unresolved": "#c7c7c7"}
    for axis, arm in zip(axes, arms, strict=True):
        trips = arm["trips"]
        for trip in trips:
            number = trip["trip"]
            axis.axvspan(
                number - 0.48,
                number + 0.48,
                color=colors[trip["class"]],
                alpha=0.22,
                linewidth=0,
            )
            for candidate in trip["census"]["admitted_candidates"]:
                if not candidate["admitted"]:
                    continue
                marker = "o" if candidate["type"] == "o" else "x"
                color = "#1f5a9d" if candidate["type"] == "o" else "#a83232"
                axis.scatter(
                    number,
                    candidate["position_m"][0],
                    marker=marker,
                    color=color,
                    alpha=0.35,
                    s=22,
                )
            axis.scatter(
                number,
                trip["selected_axis_m"][0],
                marker="o",
                facecolors="none",
                edgecolors="#08306b",
                linewidths=1.8,
                s=65,
                zorder=5,
            )
            axis.scatter(
                number,
                trip["selected_x_m"][0],
                marker="x",
                color="#7f0000",
                linewidths=2.0,
                s=65,
                zorder=5,
            )
        result = arm["result"]
        axis.set_title(
            f"{arm['label']}  {result['active_set_trips']} trips  "
            f"{result['termination_reason']}  "
            f"residual={result['terminal_residual']:.3e}"
        )
        axis.set_ylabel("candidate R [m]")
        axis.set_xticks([trip["trip"] for trip in trips])
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("active-set trip; background: green diverted, amber limited")
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color="#1f5a9d", label="admitted O"
        ),
        plt.Line2D(
            [], [], marker="x", linestyle="", color="#a83232", label="admitted X"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            color="#08306b",
            label="selected O",
        ),
        plt.Line2D(
            [], [], marker="x", linestyle="", color="#7f0000", label="selected X"
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def combine(inputs: list[Path], output: Path, figure: Path) -> dict[str, Any]:
    arms = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    by_label = {arm["label"]: arm for arm in arms}
    required = {"main", "containment", "polish", "integrated"}
    if set(by_label) != required:
        raise RuntimeError(f"arm labels {sorted(by_label)} != {sorted(required)}")
    main = by_label["main"]
    analysis = {
        "first_departure_from_main": {
            label: _first_departure(main, by_label[label])
            for label in ("containment", "polish", "integrated")
        },
        "containment_first_lost_x": _first_lost_x(by_label["containment"]),
        "containment_selected_x_admission": _selected_x_admission_audit(
            by_label["containment"]
        ),
        "containment_first_class_difference": _first_class_difference(
            main, by_label["containment"]
        ),
        "terminal_summary": {label: arm["result"] for label, arm in by_label.items()},
    }
    payload = {
        "schema": "nova.held_tip_trip_diagnosis.v1",
        "complete": True,
        "captured_at": datetime.now(UTC).isoformat(),
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in inputs],
        "analysis": analysis,
        "arms": by_label,
    }
    _write_json(output, payload)
    _draw(
        [by_label[label] for label in ("main", "containment", "polish", "integrated")],
        figure,
    )
    print(
        "DIAGNOSIS_COMPLETE "
        + json.dumps(
            {
                "output": str(output),
                "figure": str(figure),
                "trips": {
                    label: arm["result"]["active_set_trips"]
                    for label, arm in by_label.items()
                },
                "lost_x": analysis["containment_first_lost_x"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, required=True)
    measure_parser.add_argument("--label", required=True)
    measure_parser.add_argument("--base-revision", default="")
    measure_parser.add_argument("--held-commit", default="")
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--input", type=Path, action="append", required=True)
    combine_parser.add_argument("--output", type=Path, required=True)
    combine_parser.add_argument("--figure", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.action == "measure":
        measure(
            arguments.output,
            arguments.label,
            arguments.base_revision,
            arguments.held_commit,
        )
    else:
        combine(arguments.input, arguments.output, arguments.figure)


if __name__ == "__main__":
    main()
