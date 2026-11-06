"""Measure the forward fixed-point descent on both topology branches.

For each declared DIII-D gate frame the script builds the same profile the
benchmark builds, takes the bank's own cold-seed portfolio, and runs the
registered Newton-Krylov request from each branch's seed.  It records the
per-promotion relative residual, the per-promotion plasma-mask difference the
active set reports, the globalisation factors the inner iteration applied, and
the terminal class and topology read.

The measurement is read-only: it imports the benchmark's builders and the
public solver, and it writes nothing into the repository.  A frame that fails
before a topology read is recorded as a refusal with the exception that stopped
it, so a missing axis is a measurement rather than a lost run.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from benchmarks import diiid_forward_gs_match as bench
from nova.equilibrium.solve_request import ForwardSolveRequest  # noqa: F401
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes

EXECUTION_FRAME_COUNT = bench.EXECUTION_FRAME_COUNT
PROMOTION_TOLERANCE = bench.GATE_RESIDUAL_TOLERANCE


def _finite_list(values: Any) -> list[float]:
    array = np.asarray(values, dtype=np.float64).ravel()
    return [float(value) if np.isfinite(value) else None for value in array]


def _int_list(values: Any) -> list[int]:
    array = np.asarray(values).ravel()
    return [int(value) for value in array]


def classify_terminal(
    residuals: list[float | None], terminal: float | None
) -> dict[str, Any]:
    """Classify a descent as contraction stall, cycle, divergence or refusal.

    A refusal carries no finite terminal.  A divergence is a non-finite last
    promotion stream or a terminal above the first reading with growth after
    the descent's minimum.  A cycle is a period-2 repetition over the last four
    promotions at either of the two levels.  Everything else with a finite
    terminal above tolerance is a contraction stall, with the last-step ratio
    reported so a slow contraction is distinguishable from a flat one.
    """

    if terminal is None or not np.isfinite(terminal):
        return {"class": "refusal", "evidence": "no finite terminal residual"}
    finite = [value for value in residuals if value is not None]
    if not finite:
        return {"class": "refusal", "evidence": "no finite promotion recorded"}
    last = finite[-1]
    first = finite[0]
    if abs(last - terminal) > 1e-12 * max(1.0, abs(terminal)):
        finite = finite + [terminal]
    tail = finite[-4:]
    cycle = False
    cycle_detail = None
    if len(tail) == 4:
        even_gap = abs(tail[2] - tail[0]) / max(abs(tail[2]), 1e-300)
        odd_gap = abs(tail[3] - tail[1]) / max(abs(tail[3]), 1e-300)
        cycle = even_gap < 1e-3 and odd_gap < 1e-3 and tail[2] > PROMOTION_TOLERANCE
        cycle_detail = {
            "even_gap_relative": float(even_gap),
            "odd_gap_relative": float(odd_gap),
            "alternating_levels": tail,
        }
    ratios = [
        float(finite[index + 1] / finite[index])
        for index in range(len(finite) - 1)
        if finite[index] > 0.0
    ]
    last_ratio = ratios[-1] if ratios else None
    monotone = all(value < 1.0 for value in ratios[-3:]) if len(ratios) >= 3 else None
    if terminal <= PROMOTION_TOLERANCE:
        return {
            "class": "contracted",
            "evidence": "terminal residual within tolerance",
            "last_ratio": last_ratio,
        }
    if cycle:
        return {"class": "cycle", "evidence": cycle_detail, "last_ratio": last_ratio}
    minimum = min(finite)
    index_minimum = finite.index(minimum)
    growing_after_minimum = (
        index_minimum < len(finite) - 1 and finite[-1] > 2.0 * minimum
    )
    if not np.isfinite(last) or growing_after_minimum:
        return {
            "class": "divergence",
            "evidence": {
                "first": first,
                "last": last,
                "minimum": minimum,
                "index_of_minimum": index_minimum,
            },
            "last_ratio": last_ratio,
        }
    return {
        "class": "contraction_stall",
        "evidence": {
            "first": first,
            "last": last,
            "minimum": minimum,
            "last_three_ratios_below_one": monotone,
            "promotions": len(finite),
        },
        "last_ratio": last_ratio,
    }


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
    circuit_current = bench.circuit_current_map(shipped_current)
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
        "circuit_current": circuit_current,
    }


def _branch_record(
    prepared: dict[str, Any], topology_class: TopologyClass, expansion: float
) -> dict[str, Any]:
    profile = prepared["profile"]
    cold = prepared["cold"]
    seed = np.asarray(cold.branches.flux[int(topology_class)], dtype=float)
    label_seed = prepared["label_seed"]
    seed_identity = bool(np.array_equal(seed, label_seed))
    seed_distance = float(np.linalg.norm(seed - label_seed))
    request = bench._registered_solve_request(
        profile,
        seed,
        jnp.asarray(prepared["current"]),
        prepared["target_current_a"],
        carrier_identity=f"diiid-convergence-probe:{expansion}:{topology_class.name}",
    )
    record: dict[str, Any] = {
        "requested_class": topology_class.name.lower(),
        "seed_identity_with_label": seed_identity,
        "seed_l2_distance_from_label": seed_distance,
        "seed_l2_norm": float(np.linalg.norm(seed)),
        "policy": {
            "route": str(getattr(request.policy, "route", None)),
            "newton_steps": int(getattr(request.policy, "newton_steps", -1)),
            "gmres_iterations": int(getattr(request.policy, "gmres_iterations", -1)),
            "warmup": int(getattr(request.policy, "warmup", -1)),
            "relaxation": float(getattr(request.policy, "relaxation", float("nan"))),
            "step_cap": float(getattr(request.policy, "step_cap", float("nan"))),
            "tolerance": float(
                getattr(request.policy, "qualification_tolerance", float("nan"))
            ),
        },
    }
    started = time.perf_counter()
    try:
        receipt = profile.solve(request)
        receipt.equilibrium.flux.block_until_ready()
        wall_seconds = time.perf_counter() - started
        equilibrium = receipt.equilibrium
        residual_history = _finite_list(receipt.residual_history)
        terminal = float(equilibrium.fixed_point.residual)
        if not np.isfinite(terminal):
            terminal_value: float | None = None
        else:
            terminal_value = terminal
        decisions, factors = receipt.globalisation_decisions
        topology = equilibrium.topology
        diverted = bool(topology.diverted)
        flux = np.asarray(equilibrium.flux[: profile.lattice.node_count], dtype=float)
        flux_grid = flux.reshape(profile.lattice.shape)
        boundary_rz = bench._assembled_boundary_geometry(
            flux_grid,
            profile.lattice.radius,
            profile.lattice.height,
            float(topology.boundary_flux),
            np.asarray(topology.axis, dtype=float),
        )[0]
        record.update(
            {
                "status": "solved",
                "wall_seconds": wall_seconds,
                "residual_history": residual_history,
                "mask_history": _int_list(receipt.mask_history),
                "amplitude_history": _finite_list(receipt.amplitude_history),
                "globalisation_decisions": _finite_list(decisions),
                "globalisation_factors": _finite_list(factors),
                "termination_reason": int(receipt.termination_reason),
                "qualified": bool(np.asarray(receipt.qualified)),
                "terminal_residual": terminal_value,
                "terminal": classify_terminal(residual_history, terminal_value),
                "active_set_iterations": int(
                    getattr(equilibrium.fixed_point, "active_set_iterations", 0)
                ),
                "active_set_residuals": _finite_list(
                    getattr(equilibrium.fixed_point, "active_set_residuals", ())
                ),
                "active_set_mask_differences": _int_list(
                    getattr(equilibrium.fixed_point, "active_set_mask_differences", ())
                ),
                "achieved_diverted": diverted,
                "achieved_class": (
                    TopologyClass.DIVERTED.name.lower()
                    if diverted
                    else TopologyClass.LIMITED.name.lower()
                ),
                "topology_matches_request": diverted
                == (topology_class is TopologyClass.DIVERTED),
                "axis_rz_m": [float(value) for value in np.asarray(topology.axis)],
                "x_point_rz_m": [
                    float(value) for value in np.asarray(topology.x_point)
                ],
                "class_margin": float(topology.class_margin),
                "boundary_flux_wb": float(topology.boundary_flux),
                "axis_flux_wb": float(topology.axis_flux),
                "finite": bool(equilibrium.finite.passed),
                "boundary_point_count": int(len(boundary_rz)),
                "wall_contact": _wall_distance(boundary_rz, prepared["wall"]),
                "flux_shape": list(flux_grid.shape),
                "flux_minmax_wb": [
                    float(np.min(flux_grid)),
                    float(np.max(flux_grid)),
                ],
                "_flux_grid": flux_grid,
                "_boundary_rz": boundary_rz,
            }
        )
    except Exception as error:  # a refusal is a measurement, not a lost frame
        record.update(
            {
                "status": "refused",
                "wall_seconds": time.perf_counter() - started,
                "refusal_type": type(error).__name__,
                "refusal_message": str(error),
                "terminal": {"class": "refusal", "evidence": str(error)},
                "traceback": traceback.format_exc()[-4000:],
            }
        )
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
        for key in ("_flux_grid", "_boundary_rz"):
            value = branch.pop(key, None)
            if value is not None:
                arrays[f"{name}_{key.strip('_')}"] = value
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
        default=bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
    )
    arguments = parser.parse_args()
    configure_dtypes()
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "measurement": "per-promotion forward descent on both topology branches",
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
                            "terminal": branch.get("terminal", {}).get("class"),
                            "terminal_residual": branch.get("terminal_residual"),
                            "achieved": branch.get("achieved_class"),
                            "status": branch.get("status"),
                        }
                        for name, branch in record["branches"].items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    # Every frame shares one lattice shape, so the compiled programs are reused
    # across frames; clearing the caches here would only recompile them.


if __name__ == "__main__":
    main()
