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
import traceback
import uuid
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    FIXED_POINT_CRITERION,
    TOTAL_FLUX_FACTOR,
    _mast_case_from_selection,
    _passive_inclusive_case,
)
from benchmarks.forward_labeller_throughput import (
    KEYFRAME_SLICE,
    NEWTON_STEPS,
    SHOT_STORE,
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
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.steering_frames import (
    N_DIVERTOR_LEG_POINTS,
    N_DIVERTOR_LEGS,
    N_RHO,
    N_SURFACE,
    N_THETA,
    TORAX_PROFILE_FIELDS,
    SteeringAction,
    SteeringFrame,
    assemble_frame,
    policy_digest,
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
EXPECTED_CAMERA_FRAMES = 24_881_648
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


@dataclass(frozen=True)
class ShotWork:
    """One decoder-corpus shot and its total camera-frame demand."""

    shot: int
    camera_frames: int


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


def decoder_corpus(manifest: Path, cohort_report: Path) -> list[ShotWork]:
    """Return decoder shots ordered by descending camera-frame demand."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    cohort = _cohort_shots(cohort_report)
    frame_counts: dict[int, int] = {}
    for window in payload["windows"]:
        shot = int(window["shot_id"])
        if shot in cohort:
            continue
        frame_counts[shot] = frame_counts.get(shot, 0) + int(window["n_frames"])
    result = [
        ShotWork(shot=shot, camera_frames=frames)
        for shot, frames in sorted(
            frame_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    if len(result) != EXPECTED_CORPUS_SHOTS:
        raise ValueError(
            f"decoder corpus yielded {len(result)} shots, expected "
            f"{EXPECTED_CORPUS_SHOTS}"
        )
    frame_total = sum(item.camera_frames for item in result)
    if frame_total != EXPECTED_CAMERA_FRAMES:
        raise ValueError(
            f"decoder corpus yielded {frame_total} camera frames, expected "
            f"{EXPECTED_CAMERA_FRAMES}"
        )
    return result


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Atomically write a JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_shards(
    work: Sequence[ShotWork], directory: Path, count: int
) -> list[dict[str, Any]]:
    """Write contiguous priority shards and return their inventory."""
    if count < 1:
        raise ValueError("shard count must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(count - 1)))
    inventory = []
    for index in range(count):
        start = index * len(work) // count
        stop = (index + 1) * len(work) // count
        bucket = work[start:stop]
        path = directory / f"shard-{index:0{width}d}.txt"
        path.write_text("".join(f"{item.shot}\n" for item in bucket), encoding="utf-8")
        inventory.append(
            {
                "index": index,
                "path": str(path.resolve()),
                "shot_count": len(bucket),
                "camera_frames": sum(item.camera_frames for item in bucket),
                "first_shot": bucket[0].shot if bucket else None,
                "last_shot": bucket[-1].shot if bucket else None,
                "largest_frame_count": bucket[0].camera_frames if bucket else None,
                "smallest_frame_count": bucket[-1].camera_frames if bucket else None,
            }
        )
    return inventory


def tranche_inventory(
    shards: Sequence[dict[str, Any]], tranche_shards: int
) -> list[dict[str, Any]]:
    """Return cumulative priority coverage for consecutive shard tranches."""
    if tranche_shards < 1:
        raise ValueError("tranche shard count must be positive")
    total_shots = sum(int(item["shot_count"]) for item in shards)
    result = []
    cumulative_shots = 0
    cumulative_frames = 0
    for start in range(0, len(shards), tranche_shards):
        members = shards[start : start + tranche_shards]
        cumulative_shots += sum(int(item["shot_count"]) for item in members)
        cumulative_frames += sum(int(item["camera_frames"]) for item in members)
        estimated_slices = round(EXPECTED_EFM_SLICES * cumulative_shots / total_shots)
        result.append(
            {
                "tranche": len(result),
                "first_shard": int(members[0]["index"]),
                "last_shard": int(members[-1]["index"]),
                "cumulative_shots": cumulative_shots,
                "cumulative_camera_frames": cumulative_frames,
                "estimated_cumulative_slices": estimated_slices,
            }
        )
    return result


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
    """Return inputs only for a row carrying a finite reconstruction."""
    current = np.asarray(group["fcoil_c"][row], dtype=np.float64)
    scalars = {
        "time": float(group["time"][row]),
        "magnetic_axis_z": float(group["magnetic_axis_z"][row]),
        "target_centroid_z": float(group["current_centrd_z"][row]),
        "reference_plasma_current": float(group["plasma_current_c"][row]),
    }
    if not np.all(np.isfinite(current)) or not all(
        np.isfinite(value) for value in scalars.values()
    ):
        return None
    return {**scalars, "current": current}


def _update_slice_counts(manifest: dict[str, Any]) -> None:
    """Refresh disjoint admitted, converged, unconverged and excluded counts."""
    rows = manifest["slices"]
    excluded = sum(bool(item.get("excluded")) for item in rows)
    admitted = len(rows) - excluded
    converged = sum(
        bool(item.get("converged")) for item in rows if not item.get("excluded")
    )
    manifest.update(
        {
            "slice_count": len(rows),
            "admitted_slice_count": admitted,
            "written_slice_count": admitted,
            "converged_slice_count": converged,
            "unconverged_slice_count": admitted - converged,
            "excluded_slice_count": excluded,
        }
    )


def _persist_slice(manifest: dict[str, Any], path: Path, row: dict[str, Any]) -> None:
    """Checkpoint one slice record before work advances to the next row."""
    manifest["slices"].append(row)
    _update_slice_counts(manifest)
    _write_json(manifest, path)


def _masked_frame(
    prepared: PreparedLabeller,
    *,
    current: np.ndarray,
    wall_seconds: float,
    trips: int,
    template: SteeringFrame | None,
) -> SteeringFrame:
    """Return a typed frame whose solved geometry is explicitly absent."""
    action = SteeringAction(
        name="label",
        delta=0.0,
        commanded_control_points=np.empty((0, 2), dtype=float),
    )
    if template is not None:
        return template._replace(
            psi=np.full_like(np.asarray(template.psi), np.nan, dtype=float),
            psi_norm=np.full_like(np.asarray(template.psi_norm), np.nan, dtype=float),
            domain_label=np.zeros_like(
                np.asarray(template.domain_label), dtype=np.int8
            ),
            separatrix=np.full_like(
                np.asarray(template.separatrix), np.nan, dtype=float
            ),
            separatrix_vertex_count=np.int32(0),
            magnetic_axis_r=np.nan,
            magnetic_axis_z=np.nan,
            x_point_r=np.full_like(np.asarray(template.x_point_r), np.nan, dtype=float),
            x_point_z=np.full_like(np.asarray(template.x_point_z), np.nan, dtype=float),
            strike_points_r=np.full_like(
                np.asarray(template.strike_points_r), np.nan, dtype=float
            ),
            strike_points_z=np.full_like(
                np.asarray(template.strike_points_z), np.nan, dtype=float
            ),
            lcfs_r=np.full_like(np.asarray(template.lcfs_r), np.nan, dtype=float),
            lcfs_z=np.full_like(np.asarray(template.lcfs_z), np.nan, dtype=float),
            n_boundary_coords=np.int32(0),
            finite_mask=np.zeros_like(np.asarray(template.finite_mask), dtype=bool),
            coil_current=np.asarray(current, dtype=np.float64),
            compensating_current=np.zeros_like(
                np.asarray(template.compensating_current), dtype=np.float64
            ),
            action=action,
            wall_seconds=wall_seconds,
            trip_count=trips,
            flux_surface_psi=np.full_like(
                np.asarray(template.flux_surface_psi), np.nan, dtype=float
            ),
            flux_surface_r=np.full_like(
                np.asarray(template.flux_surface_r), np.nan, dtype=float
            ),
            flux_surface_z=np.full_like(
                np.asarray(template.flux_surface_z), np.nan, dtype=float
            ),
            **{
                name: np.full_like(
                    np.asarray(getattr(template, name)), np.nan, dtype=float
                )
                for name in TORAX_PROFILE_FIELDS
                if name != "rho_face_norm"
            },
            R_major=np.nan,
            a_minor=np.nan,
            B_0=np.nan,
            boundary_toroidal_flux=np.nan,
            magnetic_axis_z_scalar=np.nan,
            diverted=False,
            divertor_leg_r=np.full_like(
                np.asarray(template.divertor_leg_r), np.nan, dtype=float
            ),
            divertor_leg_z=np.full_like(
                np.asarray(template.divertor_leg_z), np.nan, dtype=float
            ),
            divertor_leg_finite=np.zeros_like(
                np.asarray(template.divertor_leg_finite), dtype=bool
            ),
        )

    radius, height, shape = prepared.profile.operator.raster_geometry()
    radial_count, vertical_count = tuple(int(slot) for slot in shape)
    policy = _solve_policy()
    resolved = ResolvedForwardSolveDefaults.from_policy(
        policy,
        compilation_cache_directory=prepared.cache_directory,
    )
    profile_channels = {
        name: np.full(N_RHO + 1, np.nan)
        for name in TORAX_PROFILE_FIELDS
        if name != "rho_face_norm"
    }
    return SteeringFrame(
        radius=np.asarray(radius),
        height=np.asarray(height),
        shape=np.asarray(shape, dtype=np.int32),
        psi=np.full((radial_count, vertical_count), np.nan),
        psi_norm=np.full((radial_count, vertical_count), np.nan),
        domain_label=np.zeros((radial_count, vertical_count), dtype=np.int8),
        separatrix=np.full((1, 2), np.nan),
        separatrix_vertex_count=np.int32(0),
        magnetic_axis_r=np.nan,
        magnetic_axis_z=np.nan,
        x_point_r=np.full(2, np.nan),
        x_point_z=np.full(2, np.nan),
        strike_points_r=np.full(2, np.nan),
        strike_points_z=np.full(2, np.nan),
        lcfs_r=np.full(1, np.nan),
        lcfs_z=np.full(1, np.nan),
        n_boundary_coords=np.int32(0),
        finite_mask=np.zeros(6, dtype=bool),
        coil_current=np.asarray(current, dtype=np.float64),
        compensating_current=np.empty(0, dtype=np.float64),
        action=action,
        wall_seconds=wall_seconds,
        trip_count=trips,
        carrier_identity=response_carrier.DEFAULT_CARRIER.stem,
        nova_version=resolved.nova_version,
        policy_digest=policy_digest(policy),
        flux_surface_psi_norm=np.linspace(0.0, 1.0, N_SURFACE),
        flux_surface_psi=np.full(N_SURFACE, np.nan),
        flux_surface_r=np.full((N_SURFACE, N_THETA), np.nan),
        flux_surface_z=np.full((N_SURFACE, N_THETA), np.nan),
        flux_surface_angle=np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False),
        rho_face_norm=np.linspace(0.0, 1.0, N_RHO + 1),
        **profile_channels,
        R_major=np.nan,
        a_minor=np.nan,
        B_0=np.nan,
        boundary_toroidal_flux=np.nan,
        magnetic_axis_z_scalar=np.nan,
        diverted=False,
        divertor_leg_r=np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan),
        divertor_leg_z=np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan),
        divertor_leg_finite=np.zeros(N_DIVERTOR_LEGS, dtype=bool),
    )


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


def _centroid_coordinates(
    prepared: PreparedLabeller, flux, target_current: float
) -> tuple[float, float]:
    """Return the solved plasma-current centroid in metres."""
    observation = prepared.profile.current_moment_observation(
        jnp.asarray(flux),
        support=MomentIntegralSupport.ALL_DOMAIN,
        target_current=target_current,
    )
    return (
        float(np.asarray(observation.centroid_r)),
        float(np.asarray(observation.centroid_z)),
    )


def _write_companion(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Atomically persist per-slice flux functions and centroid evidence."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(
        temporary,
        row=np.asarray([item["row"] for item in rows], dtype=np.int32),
        time=np.asarray([item["time"] for item in rows], dtype=np.float64),
        psi_norm=np.stack([item["psi_norm"] for item in rows]),
        p_prime=np.stack([item["p_prime"] for item in rows]),
        ff_prime=np.stack([item["ff_prime"] for item in rows]),
        current_centroid_r=np.asarray(
            [item["current_centroid_r"] for item in rows], dtype=np.float64
        ),
        current_centroid_z=np.asarray(
            [item["current_centroid_z"] for item in rows], dtype=np.float64
        ),
        reference_centroid_z=np.asarray(
            [item["reference_centroid_z"] for item in rows], dtype=np.float64
        ),
        branch_guard_ok=np.asarray(
            [item["branch_guard_ok"] for item in rows], dtype=bool
        ),
    )
    os.replace(temporary, path)


def label_shot(
    prepared: PreparedLabeller,
    shot: int,
    output_root: Path,
    *,
    program: reduced_newton.ReducedProgram | None,
    include_raster: bool,
    setup_wall_seconds: float,
    max_slices: int | None,
) -> tuple[reduced_newton.ReducedProgram | None, dict[str, Any]]:
    """Label every admitted slice while isolating failures within the shot."""
    session_path = output_root / f"{shot}.nc"
    companion_path = output_root / f"{shot}.npz"
    manifest_path = output_root / f"{shot}.manifest.json"
    if session_path.is_file() and manifest_path.is_file():
        return program, {"shot": shot, "status": "skipped", "resumed": True}

    shot_started = time.perf_counter()
    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    manifest = {
        "schema": "nova-forward-labeller-shot",
        "shot": shot,
        "status": "working",
        "session": str(session_path.resolve()),
        "companion": str(companion_path.resolve()),
        "nova_revision": source_revision(),
        "carrier_identity": response_carrier.DEFAULT_CARRIER.stem,
        "carrier": prepared.carrier_evidence,
        "policy_digest": policy_digest(_solve_policy()),
        "policy": _solve_policy().to_dict(),
        "constraint": {
            "mode": "diagnostic_branch_guard",
            "target_source": "efm/current_centrd_z",
            "tolerance_m": BRANCH_GUARD_TOLERANCE_M,
            "conditioning_flag": False,
        },
        "include_raster": include_raster,
        "setup_wall_seconds": setup_wall_seconds,
        "shot_wall_seconds": 0.0,
        "slice_count": 0,
        "admitted_slice_count": 0,
        "written_slice_count": 0,
        "converged_slice_count": 0,
        "unconverged_slice_count": 0,
        "excluded_slice_count": 0,
        "slices": [],
    }
    _write_json(manifest, manifest_path)
    frame_slots: list[SteeringFrame | None] = []
    frame_contexts: list[dict[str, Any]] = []
    companion_rows: list[dict[str, Any]] = []
    state = None
    admitted = 0
    for row in range(int(group["time"].shape[0])):
        inputs = _slice_inputs(group, row)
        if inputs is None:
            state = None
            _persist_slice(
                manifest,
                manifest_path,
                {
                    "row": row,
                    "time": float(group["time"][row]),
                    "written": False,
                    "excluded": True,
                    "converged": False,
                    "qualified": False,
                    "exclusion": "no reconstruction",
                    "target_source": "efm/current_centrd_z",
                    "branch_guard_ok": False,
                },
            )
            continue
        if max_slices is not None and admitted >= max_slices:
            break
        admitted += 1
        result = None
        started = time.perf_counter()
        try:
            seed = _slices_seed(group, row, full_r, full_z)
            if not np.all(np.isfinite(seed)):
                raise ValueError("non-finite reconstruction flux seed")
            if state is None:
                state = jnp.asarray(seed)
            requested_value = _requested_class(group, row)
            requested = jnp.asarray(requested_value, dtype=jnp.int8)
            target_current = abs(inputs["reference_plasma_current"])
            current = jnp.asarray(inputs["current"])
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
            program = result.program
            solve_wall_seconds = time.perf_counter() - started
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
            centroid_r, centroid_z = _centroid_coordinates(
                prepared, result.state, target_current
            )
            centroid_error = centroid_z - inputs["target_centroid_z"]
            branch_guard_ok = bool(
                np.isfinite(centroid_error)
                and abs(centroid_error) <= BRANCH_GUARD_TOLERANCE_M
            )
            row_record = {
                "row": row,
                "time": inputs["time"],
                "written": True,
                "excluded": False,
                "geometry_masked": False,
                "converged": bool(result.converged),
                "qualified": bool(solve_receipt.qualified),
                "terminal_residual": float(result.terminal_residual),
                "trips": int(result.active_set_iterations),
                "newton_steps": int(sum(result.newton_steps_per_trip)),
                "wall_seconds": solve_wall_seconds,
                "termination": result.termination_name,
                "conditioning_flag": False,
                "achieved_current_centroid_r": centroid_r,
                "achieved_current_centroid_z": centroid_z,
                "target_current_centroid_z": inputs["target_centroid_z"],
                "centroid_error_m": centroid_error,
                "target_source": "efm/current_centrd_z",
                "branch_guard_ok": branch_guard_ok,
            }
            frame_slots.append(frame)
            state = result.state
        except Exception as error:
            solve_wall_seconds = time.perf_counter() - started
            trips = int(result.active_set_iterations) if result is not None else 0
            row_record = {
                "row": row,
                "time": inputs["time"],
                "written": True,
                "excluded": False,
                "geometry_masked": True,
                "converged": False,
                "qualified": False,
                "trips": trips,
                "wall_seconds": solve_wall_seconds,
                "conditioning_flag": False,
                "exception": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "achieved_current_centroid_r": None,
                "achieved_current_centroid_z": None,
                "target_current_centroid_z": inputs["target_centroid_z"],
                "centroid_error_m": None,
                "target_source": "efm/current_centrd_z",
                "branch_guard_ok": False,
            }
            frame_slots.append(None)
            centroid_r = np.nan
            centroid_z = np.nan
            branch_guard_ok = False
            state = None

        frame_contexts.append(
            {
                "current": inputs["current"],
                "wall_seconds": solve_wall_seconds,
                "trips": int(row_record["trips"]),
            }
        )
        companion_rows.append(
            {
                "row": row,
                "time": inputs["time"],
                "psi_norm": np.asarray(group["psi_norm"], dtype=np.float64),
                "p_prime": -np.asarray(group["pprime"][row], dtype=np.float64)
                / TOTAL_FLUX_FACTOR,
                "ff_prime": -np.asarray(group["ffprime"][row], dtype=np.float64)
                / TOTAL_FLUX_FACTOR,
                "current_centroid_r": centroid_r,
                "current_centroid_z": centroid_z,
                "reference_centroid_z": inputs["target_centroid_z"],
                "branch_guard_ok": branch_guard_ok,
            }
        )
        _persist_slice(manifest, manifest_path, row_record)
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

    if not frame_slots:
        raise RuntimeError(f"shot {shot} has no admitted EFM slices")
    template = next((frame for frame in frame_slots if frame is not None), None)
    frames = [
        frame
        if frame is not None
        else _masked_frame(prepared, template=template, **context)
        for frame, context in zip(frame_slots, frame_contexts, strict=True)
    ]
    temporary_stem = f"{shot}-partial-{uuid.uuid4().hex}"
    store = write_session(
        frames,
        filename=temporary_stem,
        dirname=output_root,
        time=[item["time"] for item in manifest["slices"] if item.get("written")],
        include_raster=include_raster,
    )
    os.replace(Path(store.filepath), session_path)
    _write_companion(companion_rows, companion_path)
    manifest["status"] = "complete"
    manifest["shot_wall_seconds"] = time.perf_counter() - shot_started
    manifest["companion_slice_count"] = len(companion_rows)
    manifest["flux_function_grid_points"] = int(companion_rows[0]["psi_norm"].size)
    _write_json(manifest, manifest_path)
    return program, manifest


def run_shard(
    shots: Sequence[int],
    output_root: Path,
    *,
    include_raster: bool,
    max_slices: int | None,
) -> int:
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
                max_slices=max_slices,
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
                "traceback": traceback.format_exc(),
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
    parser.add_argument("--max-slices", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--enumerate-corpus", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort-report", type=Path, default=DEFAULT_COHORT_REPORT)
    parser.add_argument("--write-shards", type=Path)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--tranche-shards", type=int, default=8)
    parser.add_argument("--plan-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run corpus planning, preparation, or one labelling shard."""
    arguments = _parser().parse_args(argv)
    if arguments.max_slices is not None and arguments.max_slices < 1:
        raise ValueError("--max-slices must be positive")
    if arguments.enumerate_corpus:
        corpus = decoder_corpus(arguments.manifest, arguments.cohort_report)
        inventory = None
        if arguments.write_shards is not None:
            if arguments.shard_count is None:
                raise ValueError("--write-shards requires --shard-count")
            inventory = write_shards(
                corpus, arguments.write_shards, arguments.shard_count
            )
        tranches = (
            tranche_inventory(inventory, arguments.tranche_shards)
            if inventory is not None
            else None
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
            "tranche_shards": arguments.tranche_shards,
            "tranches": tranches,
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
        max_slices=arguments.max_slices,
    )


if __name__ == "__main__":
    raise SystemExit(main())
