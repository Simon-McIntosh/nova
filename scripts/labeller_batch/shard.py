#!/usr/bin/env python3
"""Write forward-solve steering sessions for one shard of MAST shots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    FIXED_POINT_CRITERION,
    _mast_case_from_selection,
    _passive_inclusive_case,
)
from benchmarks.forward_labeller_throughput import (
    KEYFRAME_SLICE,
    NEWTON_STEPS,
    SHOT_STORE,
    _centroid,
    _persisted_response_cache,
    _requested_class,
    _slices_seed,
)
from nova.equilibrium import fixed_point, reduced_newton
from nova.equilibrium.flux_surface_geometry import (
    FluxSurfaceGeometry,
    source_field_function,
)
from nova.equilibrium.solve_request import (
    ForwardSolveReceipt,
    ResolvedForwardSolveDefaults,
    resolve_forward_solve_policy,
)
from nova.equilibrium.steering_frames import (
    SteeringAction,
    assemble_frame,
    write_session,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(
    "/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_unified_6cam.json"
)
DEFAULT_COHORT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/"
    "physics-carried-playable-plasma/labeller-cohort-census.md"
)
EXPECTED_CORPUS_SHOTS = 8_012
EXPECTED_COHORT_SHOTS = 718
EXPECTED_LABELLABLE_SHOTS = 7_868
EXPECTED_EFM_SLICES = 639_041
EXPECTED_SHOTS_WITHOUT_EFM = 144
BRANCH_GUARD_TOLERANCE_M = 0.05


@dataclass(frozen=True)
class PreparedLabeller:
    """One shared operator and its immutable evidence."""

    profile: Any
    wall: np.ndarray
    carrier_evidence: dict[str, Any]
    policy_evidence: dict[str, Any]
    cache_directory: str
    setup_wall_seconds: float


def source_revision() -> str:
    """Return the revision that supplied this driver."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _cohort_shots(report: Path) -> set[int]:
    """Read the fixed cohort shot identities from its authoritative table."""
    text = report.read_text(encoding="utf-8")
    try:
        fixed = text.split("## Fixed shot split", 1)[1].split(
            "## Confounder subset rules", 1
        )[0]
    except IndexError as error:
        raise ValueError(f"cohort report has no fixed shot split: {report}") from error
    shots = {int(value) for value in re.findall(r"\b(\d{5}):\d+\b", fixed)}
    if len(shots) != EXPECTED_COHORT_SHOTS:
        raise ValueError(
            f"cohort report yielded {len(shots)} shots, expected "
            f"{EXPECTED_COHORT_SHOTS}"
        )
    return shots


def decoder_corpus(manifest: Path, cohort_report: Path) -> list[int]:
    """Return the unified-camera shots outside the fixed labeller cohort."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    shots = {int(window["shot_id"]) for window in payload["windows"]}
    result = sorted(shots - _cohort_shots(cohort_report))
    if len(result) != EXPECTED_CORPUS_SHOTS:
        raise ValueError(
            f"decoder corpus yielded {len(result)} shots, expected "
            f"{EXPECTED_CORPUS_SHOTS}"
        )
    return result


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Atomically write a JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_shards(
    shots: Sequence[int], directory: Path, count: int
) -> list[dict[str, Any]]:
    """Write balanced, deterministic shot lists and return their inventory."""
    if count < 1:
        raise ValueError("shard count must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(count - 1)))
    buckets = [[] for _ in range(count)]
    for index, shot in enumerate(shots):
        buckets[index % count].append(int(shot))
    inventory = []
    for index, bucket in enumerate(buckets):
        path = directory / f"shard-{index:0{width}d}.txt"
        path.write_text("".join(f"{shot}\n" for shot in bucket), encoding="utf-8")
        inventory.append(
            {
                "index": index,
                "path": str(path.resolve()),
                "shot_count": len(bucket),
                "first_shot": bucket[0] if bucket else None,
                "last_shot": bucket[-1] if bucket else None,
            }
        )
    return inventory


def prepare_labeller() -> PreparedLabeller:
    """Build the shared operator exactly as the throughput measurement does."""
    started = time.perf_counter()
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    case, context = _mast_case_from_selection(
        SHOT_STORE,
        {"shot": 22086, "slice_index": KEYFRAME_SLICE},
        {"note": "shared MAST operator construction"},
    )
    _passive_case, profile, policy_evidence = _passive_inclusive_case(
        case, context, response_cache
    )
    return PreparedLabeller(
        profile=profile,
        wall=np.asarray(case["wall_coordinate"], dtype=float),
        carrier_evidence=carrier_evidence,
        policy_evidence=policy_evidence,
        cache_directory=str(cache.directory),
        setup_wall_seconds=time.perf_counter() - started,
    )


def _solve_policy():
    """Return the resolved numerical policy used by every slice."""
    return resolve_forward_solve_policy(
        route="reduced_newton",
        overrides={
            "newton_steps": NEWTON_STEPS,
            "kernel_tolerance": FIXED_POINT_CRITERION,
        },
    )


def _forward_receipt(
    prepared: PreparedLabeller,
    result: reduced_newton.ReducedNewtonResult,
    *,
    requested_class,
    target_current: float,
    prescribed_current,
    solve_wall_seconds: float,
) -> ForwardSolveReceipt:
    """Lift the reduced result into the public steering-frame receipt."""
    residuals = jnp.asarray(result.active_set_residuals, dtype=jnp.float64)
    differences = jnp.asarray(result.active_set_mask_differences, dtype=jnp.int32)
    history = fixed_point.FixedPointResult(
        state=result.state,
        residual=jnp.asarray(result.terminal_residual, dtype=jnp.float64),
        trace=residuals,
        converged=jnp.asarray(result.converged),
        termination_reason=jnp.asarray(result.termination_reason, dtype=jnp.int32),
        active_set_iterations=jnp.asarray(
            result.active_set_iterations, dtype=jnp.int32
        ),
        active_set_residuals=residuals,
        active_set_mask_differences=differences,
        shadow_mask_changes=differences,
    )
    equilibrium = prepared.profile._receipt(
        result.state,
        history,
        requested_class,
        target_current,
        None,
        prescribed_current,
    )
    policy = _solve_policy()
    finite = bool(np.asarray(equilibrium.finite.passed))
    qualified = bool(
        result.converged
        and finite
        and float(result.terminal_residual) <= policy.qualification_tolerance
    )
    return ForwardSolveReceipt(
        terminal_state=equilibrium,
        qualified=qualified,
        termination_reason=history.termination_reason,
        residual_history=residuals,
        mask_history=differences,
        globalisation_decisions=(
            jnp.asarray(history.inner_iteration_decisions),
            jnp.asarray(history.inner_iteration_applied_factors),
        ),
        amplitude_history=jnp.atleast_1d(equilibrium.normalisation.amplitude),
        topology_read=equilibrium.topology,
        polish_receipt=None,
        compilation_cache_hit=False,
        wall_seconds=solve_wall_seconds,
        resolved_defaults=ResolvedForwardSolveDefaults.from_policy(
            policy,
            compilation_cache_directory=prepared.cache_directory,
        ),
    )


def _slice_inputs(group: zarr.Group, row: int) -> dict[str, Any] | None:
    """Return finite inputs for one reconstruction row, or no buildable row."""
    current = np.asarray(group["fcoil_c"][row], dtype=np.float64)
    scalars = {
        "time": float(group["time"][row]),
        "target_centroid_z": float(group["current_centrd_z"][row]),
        "reference_plasma_current": float(group["plasma_current_c"][row]),
    }
    if not np.all(np.isfinite(current)) or not all(
        np.isfinite(value) for value in scalars.values()
    ):
        return None
    return {**scalars, "current": current}


def _internal_geometry(prepared: PreparedLabeller, equilibrium, diverted: bool):
    """Produce the fixed-shape decoder and transport geometry blocks."""
    topology = equilibrium.topology
    axis = np.asarray(topology.axis, dtype=float)
    return FluxSurfaceGeometry.internal_geometry(
        prepared.profile.lattice,
        np.asarray(equilibrium.flux, dtype=float),
        source_field_function(
            prepared.profile.source,
            float(topology.flux_span),
        ),
        axis=(float(axis[0]), float(axis[1])),
        boundary_flux=float(topology.boundary_flux),
        n_surface=11,
        n_theta=64,
        n_rho=25,
        diverted=diverted,
    )


def label_shot(
    prepared: PreparedLabeller,
    shot: int,
    output_root: Path,
    *,
    program: reduced_newton.ReducedProgram | None,
    include_raster: bool,
    setup_wall_seconds: float,
) -> tuple[reduced_newton.ReducedProgram | None, dict[str, Any]]:
    """Label every buildable slice of one shot and persist one session."""
    session_path = output_root / f"{shot}.nc"
    manifest_path = output_root / f"{shot}.manifest.json"
    if session_path.is_file() and manifest_path.is_file():
        return program, {"shot": shot, "status": "skipped", "resumed": True}

    shot_started = time.perf_counter()
    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    frame_rows = []
    frames = []
    state = None
    for row in range(len(group["time"])):
        inputs = _slice_inputs(group, row)
        if inputs is None:
            frame_rows.append(
                {
                    "row": row,
                    "written": False,
                    "converged": False,
                    "qualified": False,
                    "exclusion": "non-finite fitted currents or reference scalars",
                    "target_source": "efm/current_centrd_z",
                    "branch_guard_ok": False,
                }
            )
            continue
        seed = _slices_seed(group, row, full_r, full_z)
        if not np.all(np.isfinite(seed)):
            frame_rows.append(
                {
                    "row": row,
                    "time": inputs["time"],
                    "written": False,
                    "converged": False,
                    "qualified": False,
                    "exclusion": "non-finite reconstruction flux seed",
                    "target_source": "efm/current_centrd_z",
                    "branch_guard_ok": False,
                }
            )
            continue
        if state is None:
            state = jnp.asarray(seed)
        requested_value = _requested_class(group, row)
        requested = jnp.asarray(requested_value, dtype=jnp.int8)
        target_current = abs(inputs["reference_plasma_current"])
        current = jnp.asarray(inputs["current"])
        started = time.perf_counter()
        result = reduced_newton.solve_reduced_newton(
            prepared.profile.operator,
            state,
            requested_class=requested,
            target_current=target_current,
            prescribed_current=current,
            tolerance=FIXED_POINT_CRITERION,
            newton_steps=NEWTON_STEPS,
            program=program,
            stream=False,
        )
        solve_wall_seconds = time.perf_counter() - started
        program = result.program
        state = result.state
        solve_receipt = _forward_receipt(
            prepared,
            result,
            requested_class=requested,
            target_current=target_current,
            prescribed_current=current,
            solve_wall_seconds=solve_wall_seconds,
        )
        equilibrium = solve_receipt.terminal_state
        geometry = _internal_geometry(
            prepared,
            equilibrium,
            diverted=requested_value == int(TopologyClass.DIVERTED),
        )
        frame = assemble_frame(
            solve_receipt,
            action=SteeringAction(
                name="label",
                delta=0.0,
                commanded_control_points=np.empty((0, 2), dtype=float),
            ),
            carrier_identity=response_carrier.DEFAULT_CARRIER.stem,
            applied_current=inputs["current"],
            internal_geometry=geometry,
            wall=prepared.wall,
        )
        achieved = _centroid(prepared.profile, result.state, target_current)
        centroid_error = achieved - inputs["target_centroid_z"]
        row_record = {
            "row": row,
            "time": inputs["time"],
            "written": True,
            "converged": bool(result.converged),
            "qualified": bool(solve_receipt.qualified),
            "terminal_residual": float(result.terminal_residual),
            "trips": int(result.active_set_iterations),
            "newton_steps": int(sum(result.newton_steps_per_trip)),
            "wall_seconds": solve_wall_seconds,
            "termination": result.termination_name,
            "conditioning_flag": False,
            "achieved_current_centroid_z": achieved,
            "target_current_centroid_z": inputs["target_centroid_z"],
            "centroid_error_m": centroid_error,
            "target_source": "efm/current_centrd_z",
            "branch_guard_ok": bool(
                np.isfinite(centroid_error)
                and abs(centroid_error) <= BRANCH_GUARD_TOLERANCE_M
            ),
        }
        frame_rows.append(row_record)
        frames.append(frame)
        print(
            "LABELLED "
            + json.dumps(
                {
                    "shot": shot,
                    "row": row,
                    "converged": row_record["converged"],
                    "qualified": row_record["qualified"],
                    "branch_guard_ok": row_record["branch_guard_ok"],
                    "wall_seconds": row_record["wall_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if not frames:
        raise RuntimeError(f"shot {shot} has no buildable EFM slices")
    temporary_stem = f"{shot}-partial-{uuid.uuid4().hex}"
    store = write_session(
        frames,
        filename=temporary_stem,
        dirname=str(output_root),
        time=[item["time"] for item in frame_rows if item.get("written")],
        include_raster=include_raster,
    )
    os.replace(Path(store.filepath), session_path)
    manifest = {
        "schema": "nova-forward-labeller-shot",
        "shot": shot,
        "status": "complete",
        "session": str(session_path.resolve()),
        "nova_revision": source_revision(),
        "carrier_identity": response_carrier.DEFAULT_CARRIER.stem,
        "carrier": prepared.carrier_evidence,
        "policy_digest": frames[0].policy_digest,
        "policy": _solve_policy().to_dict(),
        "constraint": {
            "mode": "diagnostic_branch_guard",
            "target_source": "efm/current_centrd_z",
            "tolerance_m": BRANCH_GUARD_TOLERANCE_M,
            "conditioning_flag": False,
        },
        "include_raster": include_raster,
        "setup_wall_seconds": setup_wall_seconds,
        "shot_wall_seconds": time.perf_counter() - shot_started,
        "slice_count": len(frame_rows),
        "written_slice_count": len(frames),
        "slices": frame_rows,
    }
    _write_json(manifest, manifest_path)
    return program, manifest


def run_shard(shots: Sequence[int], output_root: Path, *, include_raster: bool) -> int:
    """Prepare once, then label every requested shot with one carried program."""
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = prepare_labeller()
    program = None
    failures = []
    setup_unassigned = prepared.setup_wall_seconds
    for shot in shots:
        try:
            program, record = label_shot(
                prepared,
                int(shot),
                output_root,
                program=program,
                include_raster=include_raster,
                setup_wall_seconds=setup_unassigned,
            )
            if record["status"] != "skipped":
                setup_unassigned = 0.0
        except Exception as error:  # one shot must not strand the shard
            failure = {
                "schema": "nova-forward-labeller-shot",
                "shot": int(shot),
                "status": "failed",
                "nova_revision": source_revision(),
                "carrier_identity": response_carrier.DEFAULT_CARRIER.stem,
                "setup_wall_seconds": setup_unassigned,
                "failure": f"{type(error).__name__}: {error}",
                "slices": [],
            }
            _write_json(failure, output_root / f"{shot}.manifest.json")
            failures.append(failure)
            setup_unassigned = 0.0
            print(json.dumps(failure, sort_keys=True), flush=True)
    return 1 if failures else 0


def _shot_list(arguments) -> list[int]:
    shots = list(arguments.shots or ())
    if arguments.shot_list is not None:
        shots.extend(
            int(line.strip())
            for line in arguments.shot_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    result = list(dict.fromkeys(shots))
    if not result:
        raise ValueError("supply --shots or --shot-list")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path, nargs="?")
    parser.add_argument("--shots", type=int, nargs="*")
    parser.add_argument("--shot-list", type=Path)
    parser.add_argument("--include-raster", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--enumerate-corpus", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort-report", type=Path, default=DEFAULT_COHORT_REPORT)
    parser.add_argument("--write-shards", type=Path)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--plan-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run corpus planning, preparation, or one labelling shard."""
    arguments = _parser().parse_args(argv)
    if arguments.enumerate_corpus:
        corpus = decoder_corpus(arguments.manifest, arguments.cohort_report)
        inventory = None
        if arguments.write_shards is not None:
            if arguments.shard_count is None:
                raise ValueError("--write-shards requires --shard-count")
            inventory = write_shards(
                corpus, arguments.write_shards, arguments.shard_count
            )
        payload = {
            "schema": "nova-forward-labeller-plan",
            "corpus_shots": len(corpus),
            "scheduled_shots": len(corpus),
            "labellable_shots": EXPECTED_LABELLABLE_SHOTS,
            "known_shots_without_efm": EXPECTED_SHOTS_WITHOUT_EFM,
            "estimated_slices": EXPECTED_EFM_SLICES,
            "shard_count": arguments.shard_count,
            "shards": inventory,
        }
        if arguments.plan_output is not None:
            _write_json(payload, arguments.plan_output)
        print(json.dumps(payload, sort_keys=True))
        return 0

    if arguments.output_root is None:
        raise ValueError("output_root is required for preparation and labelling")
    shots = _shot_list(arguments)
    if arguments.prepare_only:
        prepared = prepare_labeller()
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "shots": shots,
                    "setup_wall_seconds": prepared.setup_wall_seconds,
                    "carrier_identity": response_carrier.DEFAULT_CARRIER.stem,
                    "response_shape": prepared.policy_evidence["response_shape"],
                    "cache_directory": prepared.cache_directory,
                },
                sort_keys=True,
            )
        )
        return 0
    return run_shard(
        shots,
        arguments.output_root,
        include_raster=arguments.include_raster,
    )


if __name__ == "__main__":
    raise SystemExit(main())
