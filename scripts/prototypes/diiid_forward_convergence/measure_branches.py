"""Measure the forward fixed-point descent on both topology branches.

For each declared DIII-D gate frame the script builds the profile the benchmark
builds, takes the bank's own cold-seed portfolio, and runs the registered
Newton-Krylov request from each branch's seed.  Per promotion it records the
relative residual, the plasma-mask difference, the mask area, the accepted and
attempted step counts and the globalisation factors.  At each staged state it
records the plasma-presence read: the magnetic axis and whether it lies inside
the governed wall, the area of the plasma support, and the toroidal current the
profile carries inside the confined core against the frame's target.

Three arms run per branch:

``default``       the registered request, unmodified.
``continued``     the same request with the mask-stability exits disabled, so the
                  loop spends its whole trip budget at a frozen mask instead of
                  declaring settlement.
``staged``        the registered request capped at ``k`` active-set trips, for
                  every ``k`` up to the default arm's trip count, which is how a
                  per-promotion state is obtained at all: the public receipt
                  persists the terminal state only.

The measurement is read-only: it imports the benchmark's builders and the public
solver, and it writes nothing into the repository.  A frame that fails before a
topology read is recorded as a refusal with the exception that stopped it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from benchmarks import diiid_forward_gs_match as bench
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from nova.media.poloidal import inside_wall_units
from terminal_class import classify_terminal

EXECUTION_FRAME_COUNT = bench.EXECUTION_FRAME_COUNT
PROMOTION_TOLERANCE = bench.GATE_RESIDUAL_TOLERANCE
STAGED_TRIP_LIMIT = 5


def _finite_list(values: Any) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64).ravel()
    return [float(value) if np.isfinite(value) else None for value in array]


def _int_list(values: Any) -> list[int]:
    array = np.asarray(values).ravel()
    return [int(value) for value in array]


def _wall_distance(boundary: np.ndarray, wall: np.ndarray) -> dict[str, Any]:
    """Return the closest approach of a closed boundary to the wall polyline."""

    if boundary.size == 0 or wall.size == 0:
        return {"min_distance_m": None, "nodes_within_1cm": 0}
    tree = cKDTree(np.asarray(wall, dtype=np.float64))
    distances, _ = tree.query(np.asarray(boundary, dtype=np.float64), k=1)
    return {
        "min_distance_m": float(np.min(distances)),
        "nodes_within_1cm": int(np.sum(distances <= 0.01)),
        "boundary_node_count": int(len(distances)),
    }


def _prepare_frame(frame: int, expansion: float):
    """Return the frame's row, profile, seed portfolio and conductor current."""

    paths = sorted(bench.DEFAULT_DATA.glob("*.parquet"))
    selected = bench.select_frames(
        paths, EXECUTION_FRAME_COUNT, bench.polarity_population()
    )
    match = [item for item in selected if item.frame == frame]
    if not match:
        raise SystemExit(f"frame {frame} is not in the declared cohort")
    selected_frame = match[0]
    row = bench._read(
        selected_frame.path,
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS,
    )
    row["_source_path"] = str(selected_frame.path)
    (
        profile,
        label_seed,
        label,
        wall,
        reliable,
        wall_statement,
    ) = bench.build_profile(row, frame, expansion)
    time_ms = float(row["efit_times"][frame])
    machine = bench.dataset_machine_description(
        row, source_row=str(row.get("_source_path", "corpus row"))
    )
    shipped_current = bench.shipped_current_at(
        row, machine.physical, bench.POLOIDAL_CONDUCTORS, time_ms
    )
    adapter = bench.complete_profile_current_adapter(
        profile,
        shipped_names=bench.POLOIDAL_CONDUCTORS,
        shipped_current_a=shipped_current,
        use_circuit=True,
    )
    profile = adapter.profile
    current = np.asarray(adapter.resolution.current(()), dtype=float)
    target_current_a = bench._target_current(row, time_ms)
    count = int(row["efit_lcfs_n"][frame])
    contour = np.c_[
        np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
    ]
    axis = np.asarray(
        (row["efit_r_axis"][frame], row["efit_z_axis"][frame]), dtype=float
    )
    saddle = contour[int(np.argmin(contour[:, 1]))]
    cold = profile.cold_seed_portfolio(
        target_current_a,
        axis,
        current=jnp.asarray(current),
        diverted_geometry=bench.SaddleSeedGeometry(tuple(axis), tuple(saddle)),
    )
    return {
        "row": row,
        "profile": profile,
        "wall": np.asarray(wall, dtype=float),
        "label": np.asarray(label, dtype=float),
        "reliable": int(reliable),
        "wall_statement": wall_statement,
        "cold": cold,
        "current": current,
        "target_current_a": float(target_current_a),
        "axis": axis,
        "saddle": np.asarray(saddle, dtype=float),
        "contour": contour,
        "label_seed": np.asarray(label_seed, dtype=float),
    }


def _cell_area(profile) -> float:
    radius = np.asarray(profile.lattice.radius, dtype=float)
    height = np.asarray(profile.lattice.height, dtype=float)
    return float(np.median(np.diff(radius)) * np.median(np.diff(height)))


def _presence(profile, flux_map, wall, mask, target_current_a) -> dict[str, Any]:
    """Return the plasma-presence read of one flux state."""

    area = float(np.sum(np.asarray(mask, dtype=bool))) * _cell_area(profile)
    try:
        observation = profile.current_moment_observation(
            jnp.asarray(np.asarray(flux_map, dtype=float).ravel()),
            support=MomentIntegralSupport.CONFINED_CORE,
        )
        confined_current = float(np.asarray(observation.plasma_current))
    except Exception as error:  # a failed read is recorded, never raised
        confined_current = float("nan")
        confined_error = f"{type(error).__name__}: {error}"
    else:
        confined_error = None
    axis = np.asarray(profile.operator.read(jnp.asarray(
        np.asarray(flux_map, dtype=float).ravel()))[1].axis, dtype=float)
    axis_inside = bool(
        inside_wall_units(np.asarray(axis, dtype=float)[None, :], wall)[0]
    )
    return {
        "axis_rz_m": [float(value) for value in axis],
        "axis_inside_wall": axis_inside,
        "plasma_area_m2": area,
        "confined_current_a": confined_current,
        "target_current_a": float(target_current_a),
        "confined_to_target_ratio": (
            abs(confined_current) / abs(target_current_a)
            if target_current_a and np.isfinite(confined_current)
            else None
        ),
        "confined_current_error": confined_error,
        "lattice_node_count": int(len(np.asarray(mask, dtype=bool))),
    }


def _solve(profile, seed, current, target_current_a, carrier: str, **overrides):
    request = bench._registered_solve_request(
        profile,
        seed,
        jnp.asarray(current),
        target_current_a,
        carrier_identity=carrier,
    )
    if overrides:
        request = dataclasses.replace(
            request, policy=dataclasses.replace(request.policy, **overrides)
        )
    started = time.perf_counter()
    receipt = profile.solve(request)
    receipt.equilibrium.flux.block_until_ready()
    return receipt, time.perf_counter() - started


def _record_receipt(receipt, wall_seconds, prepared, topology_class) -> dict[str, Any]:
    profile = prepared["profile"]
    wall = prepared["wall"]
    equilibrium = receipt.equilibrium
    residual_history = _finite_list(receipt.residual_history)
    terminal = float(equilibrium.fixed_point.residual)
    decisions, factors = receipt.globalisation_decisions
    topology = equilibrium.topology
    flux = np.asarray(equilibrium.flux[: profile.lattice.node_count], dtype=float)
    flux_grid = flux.reshape(profile.lattice.shape)
    boundary_rz = bench._assembled_boundary_geometry(
        flux_grid,
        profile.lattice.radius,
        profile.lattice.height,
        float(topology.boundary_flux),
        np.asarray(topology.axis, dtype=float),
    )[0]
    mask_history = _int_list(receipt.mask_history)
    try:
        masks = profile.operator.read(jnp.asarray(flux))[0]
        terminal_mask = np.asarray(masks.plasma, dtype=bool).ravel()
    except Exception:
        terminal_mask = np.zeros(profile.lattice.node_count, dtype=bool)
    presence = _presence(
        profile, flux_grid, wall, terminal_mask, prepared["target_current_a"]
    )
    record = {
        "status": "solved",
        "wall_seconds": wall_seconds,
        "residual_history": residual_history,
        "mask_history": mask_history,
        "amplitude_history": _finite_list(receipt.amplitude_history),
        "globalisation_decisions": _finite_list(decisions),
        "globalisation_factors": _finite_list(factors),
        "attempted_promotions": len(_finite_list(factors)),
        "accepted_promotions": int(
            sum(1 for value in _finite_list(decisions) if value)
        ),
        "termination_reason": int(receipt.termination_reason),
        "qualified": bool(np.asarray(receipt.qualified)),
        "terminal_residual": terminal if np.isfinite(terminal) else None,
        "terminal": classify_terminal(residual_history, float(terminal)),
        "active_set_iterations": int(
            getattr(equilibrium.fixed_point, "active_set_iterations", 0)
        ),
        "achieved_class": (
            TopologyClass.DIVERTED.name.lower()
            if bool(topology.diverted)
            else TopologyClass.LIMITED.name.lower()
        ),
        "topology_matches_request": bool(topology.diverted)
        == (topology_class is TopologyClass.DIVERTED),
        "x_point_rz_m": [float(value) for value in np.asarray(topology.x_point)],
        "x_point_inside_wall": bool(
            inside_wall_units(np.asarray(topology.x_point)[None, :], wall)[0]
        ),
        "boundary_flux_wb": float(topology.boundary_flux),
        "axis_flux_wb": float(topology.axis_flux),
        "wall_contact": _wall_distance(boundary_rz, wall),
        "presence": presence,
        "_flux_grid": flux_grid,
    }
    return record


def _branch_record(prepared, topology_class, expansion) -> dict[str, Any]:
    profile = prepared["profile"]
    cold = prepared["cold"]
    seed = np.asarray(cold.branches.flux[int(topology_class)], dtype=float)
    carrier = f"diiid-convergence-probe:{expansion}:{topology_class.name}"
    record: dict[str, Any] = {
        "requested_class": topology_class.name.lower(),
        "seed_identity_with_label": bool(
            np.array_equal(seed, prepared["label_seed"])
        ),
        "seed_l2_distance_from_label": float(
            np.linalg.norm(seed - prepared["label_seed"])
        ),
        "arms": {},
    }
    try:
        receipt, seconds = _solve(
            profile, seed, prepared["current"], prepared["target_current_a"],
            f"{carrier}:default",
        )
    except Exception as error:
        record.update(
            {
                "status": "refused",
                "refusal_type": type(error).__name__,
                "refusal_message": str(error),
                "terminal": {"class": "refusal", "evidence": str(error)},
                "traceback": traceback.format_exc()[-4000:],
            }
        )
        return record
    default = _record_receipt(receipt, seconds, prepared, topology_class)
    record["arms"]["default"] = default
    record["status"] = "solved"
    trips = max(1, min(int(default["active_set_iterations"]), STAGED_TRIP_LIMIT))
    try:
        receipt, seconds = _solve(
            profile, seed, prepared["current"], prepared["target_current_a"],
            f"{carrier}:continued",
            settled_exit=False, stagnation_stop=False, active_set_steps=16,
        )
        record["arms"]["continued"] = _record_receipt(
            receipt, seconds, prepared, topology_class
        )
    except Exception as error:
        record["arms"]["continued"] = {
            "status": "refused",
            "refusal_type": type(error).__name__,
            "refusal_message": str(error),
        }
    record["arms"]["staged"] = {}
    for steps in range(1, trips + 1):
        try:
            receipt, seconds = _solve(
                profile, seed, prepared["current"], prepared["target_current_a"],
                f"{carrier}:staged{steps}", active_set_steps=steps,
            )
            record["arms"]["staged"][str(steps)] = _record_receipt(
                receipt, seconds, prepared, topology_class
            )
        except Exception as error:
            record["arms"]["staged"][str(steps)] = {
                "status": "refused",
                "refusal_type": type(error).__name__,
                "refusal_message": str(error),
            }
    return record


def run_frame(frame: int, expansion: float, artifact_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    prepared = _prepare_frame(frame, expansion)
    profile = prepared["profile"]
    frame_record: dict[str, Any] = {
        "frame": frame,
        "shot": Path(prepared["row"]["_source_path"]).name,
        "time_ms": float(prepared["row"]["efit_times"][frame]),
        "pseudo_wall_expansion": expansion,
        "wall_statement": prepared["wall_statement"],
        "reliable_flux_surfaces": prepared["reliable"],
        "target_current_a": prepared["target_current_a"],
        "label_axis_rz_m": [float(value) for value in prepared["axis"]],
        "label_saddle_rz_m": [float(value) for value in prepared["saddle"]],
        "lattice_shape": list(profile.lattice.shape),
        "branches": {},
    }
    for topology_class in (TopologyClass.LIMITED, TopologyClass.DIVERTED):
        frame_record["branches"][topology_class.name.lower()] = _branch_record(
            prepared, topology_class, expansion
        )
    arrays: dict[str, np.ndarray] = {}
    for name, branch in frame_record["branches"].items():
        for arm_name, arm in (branch.get("arms") or {}).items():
            if not isinstance(arm, dict):
                continue
            grid = arm.pop("_flux_grid", None)
            if grid is None:
                continue
            arrays[f"{name}_{arm_name}_flux"] = grid
    arrays["wall"] = prepared["wall"]
    arrays["label"] = prepared["label"]
    arrays["radius"] = np.asarray(profile.lattice.radius)
    arrays["height"] = np.asarray(profile.lattice.height)
    arrays["label_boundary"] = prepared["contour"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_dir / f"frame-{frame}-states.npz", **arrays)
    frame_record["artifact"] = str(artifact_dir / f"frame-{frame}-states.npz")
    frame_record["frame_seconds"] = time.perf_counter() - started
    return frame_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--expansion",
        type=float,
        default=float(bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION),
    )
    arguments = parser.parse_args()
    configure_dtypes()
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "measurement": (
            "per-promotion forward descent on both topology branches, with the "
            "plasma-presence read of each terminal state"
        ),
        "expansion": arguments.expansion,
        "tolerance": PROMOTION_TOLERANCE,
        "frames": {},
    }
    for frame in arguments.frames:
        started = time.perf_counter()
        record = run_frame(frame, arguments.expansion, arguments.artifact_dir.resolve())
        receipt["frames"][str(frame)] = record
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(
            "FRAME "
            + json.dumps(
                {
                    "frame": frame,
                    "seconds": round(time.perf_counter() - started, 1),
                    "branches": {
                        name: {
                            "status": branch.get("status"),
                            "terminal": (
                                (branch.get("arms") or {})
                                .get("default", {})
                                .get("terminal", {})
                                .get("class")
                                if branch.get("status") == "solved"
                                else "refusal"
                            ),
                            "terminal_residual": (
                                (branch.get("arms") or {})
                                .get("default", {})
                                .get("terminal_residual")
                            ),
                            "presence": (branch.get("arms") or {})
                            .get("default", {})
                            .get("presence"),
                        }
                        for name, branch in record["branches"].items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()