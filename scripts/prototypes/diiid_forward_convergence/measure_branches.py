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
``staged``        the registered request capped at one active-set trip and at the
                  default arm's own count, which is how a state at a promotion
                  boundary is obtained at all: the public receipt persists the
                  terminal state only.  Intermediate counts are not re-run; the
                  default arm's history already carries them.

The measurement is read-only: it imports the benchmark's builders and the public
solver, and it writes nothing into the repository.  A frame that fails before a
topology read is recorded as a refusal with the exception that stopped it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from benchmarks import diiid_forward_gs_match as bench
from nova.equilibrium.domain import PlasmaDomain
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


ACCEPTED_DECISIONS = frozenset({1, 2, 3, 4})
REFUSED_DECISIONS = frozenset({5, 6})
NOT_EXECUTED_DECISION = -1


def _inner_step_counts(decisions, factors, fixed_point) -> dict[str, Any]:
    """Return the solver's own promotion counts beside the executed-step tally.

    ``receipt.globalisation_decisions`` is a fixed-shape trace padded with
    ``InnerIterationDecision.NOT_EXECUTED`` and a nan applied factor, so its
    length is the padding rather than the work done: counting it reports every
    slot as an attempted and an accepted promotion whether the solve ran one
    inner step or twenty-four. The solver accumulates the real counts while it
    runs and publishes them on the fixed point result, so those are read
    directly and the trace is tallied over its executed rows only, to
    cross-check them and to report the damping that was applied.
    """

    codes = [int(value) for value in np.asarray(decisions).ravel()]
    applied = np.asarray(factors, dtype=float).ravel()
    executed = [code for code in codes if code != NOT_EXECUTED_DECISION]
    return {
        "attempted_promotions": int(
            getattr(fixed_point, "attempted_newton_promotions", 0)
        ),
        "accepted_promotions": int(
            getattr(fixed_point, "accepted_newton_promotions", 0)
        ),
        "inner_step_trace": {
            "slots": len(codes),
            "executed": len(executed),
            "accepted": sum(code in ACCEPTED_DECISIONS for code in executed),
            "refused": sum(code in REFUSED_DECISIONS for code in executed),
            "damped": int(np.count_nonzero(np.isfinite(applied) & (applied != 1.0))),
        },
    }


def _counters_self_check() -> int:
    """Drive the promotion counter on traces whose answer is known by hand."""

    class Stub:
        attempted_newton_promotions = 2
        accepted_newton_promotions = 2

    nan = float("nan")
    cases = (
        ("padded only", [NOT_EXECUTED_DECISION] * 24, [nan] * 24, (0, 0, 0, 0)),
        (
            "two accepted",
            [1, 2, NOT_EXECUTED_DECISION, NOT_EXECUTED_DECISION],
            [1.0, 1.0, nan, nan],
            (2, 2, 0, 0),
        ),
        (
            "refused and damped",
            [5, 1, NOT_EXECUTED_DECISION],
            [0.25, 1.0, nan],
            (2, 1, 1, 1),
        ),
    )
    failures = []
    for name, decisions, factors, wanted in cases:
        counts = _inner_step_counts(decisions, factors, Stub())
        trace = counts["inner_step_trace"]
        reading = (
            trace["executed"],
            trace["accepted"],
            trace["refused"],
            trace["damped"],
        )
        verdict = "ok" if reading == wanted else f"FAIL expected {wanted}"
        print(
            f"{name:20s} executed={trace['executed']} accepted={trace['accepted']}"
            f" refused={trace['refused']} damped={trace['damped']}"
            f" solver=({counts['attempted_promotions']},"
            f"{counts['accepted_promotions']}) {verdict}"
        )
        if reading != wanted:
            failures.append(name)
        elif (counts["attempted_promotions"], counts["accepted_promotions"]) != (2, 2):
            print(f"{name:20s} FAIL the solver counts were not read")
            failures.append(name + ":solver")
    padding = [NOT_EXECUTED_DECISION] * 24
    previous_attempted = len(padding)
    previous_accepted = sum(1 for value in padding if value)
    print(
        f"the previous rule reads this all-padding trace as attempted="
        f"{previous_attempted} accepted={previous_accepted}, which is the padding"
    )
    if previous_accepted == 0:
        failures.append("padding")
    if failures:
        print(f"REFUSED {failures}")
        return 1
    print("COUNTERS SELF CHECK OK")
    return 0


def _prepare_frame(frame: int, expansion: float | None = None):
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


def _core_mask(profile, flux_map) -> np.ndarray:
    """Return the closed plasma cells the domain partition assigns to the core.

    The partition marks a cell CORE only when the field closes around the axis
    *and* the cell belongs to the axis's connected component, so its count is a
    closed-region area rather than an active-set size.
    """

    flat = jnp.asarray(np.asarray(flux_map, dtype=float).ravel())
    masks = profile.operator.read(flat)[0]
    return np.asarray(masks.label).ravel() == int(PlasmaDomain.CORE)


def _presence(profile, flux_map, wall, mask, target_current_a) -> dict[str, Any]:
    """Return the plasma-presence read of one flux state."""

    cells = np.asarray(mask, dtype=bool)
    area = float(np.sum(cells)) * _cell_area(profile)
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
    axis = np.asarray(
        profile.operator.read(jnp.asarray(np.asarray(flux_map, dtype=float).ravel()))[
            1
        ].axis,
        dtype=float,
    )
    axis_inside = bool(
        inside_wall_units(np.asarray(axis, dtype=float)[None, :], wall)[0]
    )
    return {
        "axis_rz_m": [float(value) for value in axis],
        "axis_inside_wall": axis_inside,
        "plasma_area_m2": area,
        "closed_core_cells": int(np.sum(cells)),
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
    mask_history = _int_list(receipt.changed_mask_entries)
    terminal_mask = _core_mask(profile, flux)
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
        **_inner_step_counts(decisions, factors, equilibrium.fixed_point),
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


def _harvest(node, path: str, arrays: dict[str, np.ndarray]) -> None:
    """Move every flux grid out of ``node`` into ``arrays``, wherever it sits.

    A staged arm nests one record per promotion, and each of those carries its
    own grid, so a single-level pop leaves arrays behind and the JSON write then
    fails on the first one it meets.  This walks to any depth and names each
    harvested grid by the path it came from.
    """

    if isinstance(node, dict):
        for key in list(node):
            value = node[key]
            if key == "_flux_grid":
                arrays[f"{path}._flux_grid"] = np.asarray(node.pop(key), dtype=float)
            else:
                _harvest(value, f"{path}.{key}", arrays)
    elif isinstance(node, list | tuple):
        for index, value in enumerate(node):
            _harvest(value, f"{path}[{index}]", arrays)


def _assert_json_clean(node, path: str = "$") -> None:
    """Raise naming the key path if any array survived the harvest."""

    if isinstance(node, dict):
        for key, value in node.items():
            _assert_json_clean(value, f"{path}.{key}")
    elif isinstance(node, list | tuple):
        for index, value in enumerate(node):
            _assert_json_clean(value, f"{path}[{index}]")
    elif isinstance(node, np.ndarray):
        raise TypeError(f"ndarray reached the receipt at {path}")


def _self_check() -> None:
    """Prove the harvest walks nested records and the guard still fires."""

    clean = {"a": {"staged": {"1": {"_flux_grid": np.zeros((2, 2))}}}}
    arrays: dict[str, np.ndarray] = {}
    _harvest(clean, "record", arrays)
    harvested = sorted(arrays)
    _assert_json_clean(clean)
    positive = (
        harvested == ["record.a.staged.1._flux_grid"] and not clean["a"]["staged"]["1"]
    )
    leaked = {"a": {"staged": {"1": {"_flux_grid": np.zeros(3)}}}}
    try:
        _assert_json_clean(leaked)
    except TypeError as error:
        negative = "a.staged.1._flux_grid" in str(error)
        negative_message = str(error)
    else:
        negative = False
        negative_message = "the guard did not fire"
    print(
        json.dumps(
            {
                "harvest_keys": harvested,
                "harvested_nested": bool(positive),
                "negative_control_fired": bool(negative),
                "negative_control_message": negative_message,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not positive:
        raise SystemExit("FAIL: the harvest did not reach the staged record")
    if not negative:
        raise SystemExit("FAIL: a leaked ndarray was not refused")
    print("PASS: nested grids are harvested; a leak is refused by key path")


def _branch_record(
    prepared, topology_class, expansion, persist, staged_steps, record=None
) -> dict[str, Any]:
    profile = prepared["profile"]
    cold = prepared["cold"]
    seed = np.asarray(cold.branches.flux[int(topology_class)], dtype=float)
    carrier = f"diiid-convergence-probe:{expansion}:{topology_class.name}"
    record: dict[str, Any] = {} if record is None else record
    record.update(
        {
            "requested_class": topology_class.name.lower(),
            "seed_identity_with_label": bool(
                np.array_equal(seed, prepared["label_seed"])
            ),
            "seed_l2_distance_from_label": float(
                np.linalg.norm(seed - prepared["label_seed"])
            ),
            "arms": {},
        }
    )
    try:
        receipt, seconds = _solve(
            profile,
            seed,
            prepared["current"],
            prepared["target_current_a"],
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
    persist("default", default)
    trips = max(1, min(int(default["active_set_iterations"]), STAGED_TRIP_LIMIT))
    try:
        receipt, seconds = _solve(
            profile,
            seed,
            prepared["current"],
            prepared["target_current_a"],
            f"{carrier}:continued",
            settled_exit=False,
            stagnation_stop=False,
            active_set_steps=16,
        )
        record["arms"]["continued"] = _record_receipt(
            receipt, seconds, prepared, topology_class
        )
        persist("continued", record["arms"]["continued"])
    except Exception as error:
        record["arms"]["continued"] = {
            "status": "refused",
            "refusal_type": type(error).__name__,
            "refusal_message": str(error),
        }
    record["arms"]["staged"] = {}
    # The staged arm exists to give a state at a promotion boundary, and each
    # distinct trip count costs its own compilation of the scan loop.  One
    # promotion is what separates the first accepted step from the terminal one;
    # the intermediate counts are already in the default arm's history.
    for steps in sorted({1, int(staged_steps)}):
        steps = min(steps, trips)
        try:
            receipt, seconds = _solve(
                profile,
                seed,
                prepared["current"],
                prepared["target_current_a"],
                f"{carrier}:staged{steps}",
                active_set_steps=steps,
            )
            record["arms"]["staged"][str(steps)] = _record_receipt(
                receipt, seconds, prepared, topology_class
            )
            persist(f"staged.{steps}", record["arms"]["staged"][str(steps)])
        except Exception as error:
            record["arms"]["staged"][str(steps)] = {
                "status": "refused",
                "refusal_type": type(error).__name__,
                "refusal_message": str(error),
            }
    return record


def run_frame(
    frame: int,
    expansion: float | None,
    artifact_dir: Path,
    receipt: dict[str, Any],
    output: Path,
    staged_steps: int,
    branches: tuple[TopologyClass, ...] = (
        TopologyClass.LIMITED,
        TopologyClass.DIVERTED,
    ),
) -> dict[str, Any]:
    started = time.perf_counter()
    prepared = _prepare_frame(frame, expansion)
    profile = prepared["profile"]
    base_arrays = {
        "wall": prepared["wall"],
        "label": prepared["label"],
        "radius": np.asarray(profile.lattice.radius),
        "height": np.asarray(profile.lattice.height),
        "label_boundary": prepared["contour"],
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
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
    receipt["frames"][str(frame)] = frame_record

    for topology_class in branches:
        branch_name = topology_class.name.lower()

        def persist(arm_name, arm_record, _branch=branch_name):
            """Write the receipt and this arm's grids the moment the arm lands."""

            arrays = dict(base_arrays)
            _harvest(
                arm_record,
                f"frames.{frame}.branches.{_branch}.arms.{arm_name}",
                arrays,
            )
            _assert_json_clean(frame_record, f"frames.{frame}")
            artifact = artifact_dir / f"frame-{frame}-{_branch}-{arm_name}.npz"
            np.savez_compressed(artifact, **arrays)
            arm_record["artifact"] = str(artifact)
            _assert_json_clean(receipt)
            output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "persisted": str(output),
                        "frame": frame,
                        "branch": _branch,
                        "arm": arm_name,
                        "artifact": str(artifact),
                        "seconds": round(time.perf_counter() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        # The branch record is attached before it is filled, so a persist call
        # made while an arm is still running writes that arm into the receipt
        # rather than an empty branch.
        holder = frame_record["branches"].setdefault(branch_name, {})
        _branch_record(
            prepared, topology_class, expansion, persist, staged_steps, holder
        )

    # Nothing should remain, but harvesting here makes the guarantee local to the
    # write rather than dependent on every arm having persisted.
    leftovers: dict[str, np.ndarray] = {}
    _harvest(frame_record, f"frames.{frame}", leftovers)
    frame_record["leftover_array_keys"] = sorted(leftovers)
    _assert_json_clean(frame_record, f"frames.{frame}")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    frame_record["frame_seconds"] = time.perf_counter() - started
    return frame_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=("limited", "diverted"),
        default=["limited", "diverted"],
        help=(
            "the topology branches to measure; a frame whose other branch is "
            "already in the receipt can be completed without re-solving it"
        ),
    )
    parser.add_argument(
        "--staged-steps",
        type=int,
        default=1,
        help=(
            "the extra active-set trip count to stage beyond the terminal one; "
            "each distinct value costs a fresh compilation of the loop"
        ),
    )
    parser.add_argument(
        "--pseudo-wall-expansion",
        dest="expansion",
        type=float,
        default=None,
        help="explicit pseudo-wall control; omit to use the governed limiter",
    )
    arguments = parser.parse_args()
    if arguments.self_check:
        _self_check()
        raise SystemExit(_counters_self_check())
    if not arguments.frames or arguments.output is None:
        parser.error("--frames and --output are required unless --self-check is given")
    configure_dtypes()
    arguments.branch_classes = [
        TopologyClass[name.upper()] for name in arguments.branches
    ]
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
        record = run_frame(
            frame,
            arguments.expansion,
            arguments.artifact_dir.resolve(),
            receipt,
            output,
            arguments.staged_steps,
            tuple(arguments.branch_classes),
        )
        receipt["frames"][str(frame)] = record
        receipt["peak_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
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
