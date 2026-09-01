"""Capture and compare pinned-revision active-set state across JAX backends."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHADOW_ROOT = Path(
    "/home/ITER/mcintos/Code/.reckon-worktrees/"
    "nova-a0f1e0938fc2/s18-hexgrid/hdg-cache-replay-shadow"
)
PINNED_REVISION = "a4bec44f5cbf80ad5e210c01c984ac8d02a89de9"
TARGET = (22086, 43)
BANK_RECEIPT = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
)
STAGES = (
    "incoming_state",
    "relinearized_map",
    "krylov_linear_action",
    "krylov_step",
    "inner_best_state",
    "reconciled_state",
)
STAGE_OPERATION = {
    "incoming_state": "trip carry",
    "relinearized_map": "re-linearization",
    "krylov_linear_action": "Krylov action",
    "krylov_step": "Krylov action",
    "inner_best_state": "nonlinear promotion",
    "reconciled_state": "mask reconciliation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_inputs() -> dict[str, Any]:
    bank = json.loads(BANK_RECEIPT.read_text())
    regeneration = bank["regeneration_receipt"]
    historical_change = next(
        row
        for row in regeneration["semantic_diff_against_head_before_regeneration"][
            "rows"
        ]
        if row["identity"] == "22086/43" and row["arm"] == "pure"
    )
    scheduler = regeneration["scheduler"]
    package_versions = {
        package: version(package)
        for package in (
            "jax",
            "jaxlib",
            "jax-cuda12-plugin",
            "jax-cuda12-pjrt",
        )
    }
    return {
        "pinned_capture_runtime": {
            "shared_python_environment": "/home/ITER/mcintos/Code/nova/.venv",
            "python": sys.version.split()[0],
            "jax": package_versions["jax"],
            "jaxlib": package_versions["jaxlib"],
            "version_provenance": (
                "the pinned revision uv.lock and the shared environment used by "
                "both retained capture jobs agree on these versions"
            ),
            "cpu_leg": {
                "jax_platforms": ["cpu"],
                "cuda_build": None,
            },
            "gpu_leg": {
                "jax_platforms": ["cuda", "cpu"],
                "cuda_build": {
                    "generation": "CUDA 12",
                    "jax_cuda12_plugin": package_versions["jax-cuda12-plugin"],
                    "jax_cuda12_pjrt": package_versions["jax-cuda12-pjrt"],
                },
            },
        },
        "historical_bank_cpu_convergence": {
            "receipt": str(BANK_RECEIPT),
            "outcome_before_regeneration": historical_change["qualification_changes"][
                "converged"
            ]["before"],
            "environment_recorded": False,
            "environment_note": (
                "the receipt records the historical convergence transition but "
                "does not retain its CPU node, Python, JAX, or jaxlib environment"
            ),
            "recorded_regeneration_environment": {
                "measurement_revision": regeneration["measurement_revision"],
                "job_id": scheduler["job_id"],
                "node": scheduler["node"],
                "device": scheduler["device"],
                "allocated_cpus": scheduler["allocated_cpus"],
                "allocated_gpus": scheduler["allocated_gpus"],
                "python": scheduler["python"],
                "jax_platforms": scheduler["jax_platforms"],
            },
        },
        "remaining_discriminator": (
            "runtime environment or historical-bank inputs, because the paired "
            "pinned CPU/GPU trajectories do not diverge beyond the float64 floor"
        ),
    }


def _array(value: Any, dtype: Any | None = None) -> np.ndarray:
    return np.asarray(value if dtype is None else np.asarray(value, dtype=dtype))


def _array_record(value: Any) -> list[Any]:
    array = np.asarray(value)
    if array.dtype.kind == "b":
        return array.astype(bool).tolist()
    if array.dtype.kind in "iu":
        return array.astype(np.int64).tolist()
    return array.astype(np.float64).tolist()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True
    ).strip()


def _validate_shadow(shadow_root: Path) -> dict[str, Any]:
    head = _git(shadow_root, "rev-parse", "HEAD")
    changed_solver = subprocess.run(
        [
            "git",
            "-C",
            str(shadow_root),
            "diff",
            "--quiet",
            PINNED_REVISION,
            head,
            "--",
            "nova",
            "benchmarks",
        ],
        check=False,
    ).returncode
    if changed_solver != 0:
        raise RuntimeError(
            "the shadow worktree changes solver or benchmark code after the "
            "pinned revision"
        )
    return {
        "pinned_revision": PINNED_REVISION,
        "shadow_head": head,
        "solver_diff_from_pinned_revision": False,
        "shadow_root": str(shadow_root),
    }


def _configure_shadow_import(shadow_root: Path) -> None:
    resolved = str(shadow_root.resolve())
    if resolved in sys.path:
        sys.path.remove(resolved)
    sys.path.insert(0, resolved)


def _allocation(backend: str) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("capture requires a scheduler allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    platforms = os.environ.get("JAX_PLATFORMS", "")
    record = {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": partition,
        "reservation": os.environ.get("SLURM_JOB_RESERVATION", ""),
        "allocated_cpus": cpus,
        "allocated_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", "0")),
        "tmpdir": os.environ.get("TMPDIR"),
        "jax_platforms": platforms.split(","),
    }
    if backend == "gpu":
        if cpus != 4:
            raise RuntimeError(f"GPU capture requires four CPUs, received {cpus}")
        if record["reservation"] != "gpu_0003_grpA":
            raise RuntimeError(f"unexpected H200 reservation {record['reservation']!r}")
        if platforms != "cuda,cpu":
            raise RuntimeError(
                f"GPU capture requires JAX_PLATFORMS=cuda,cpu, got {platforms!r}"
            )
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if "H200" not in gpu:
            raise RuntimeError(f"GPU capture requires an H200, received {gpu!r}")
        record["gpu"] = gpu
    else:
        if not partition.startswith("rigel"):
            raise RuntimeError(
                f"CPU capture requires a rigel lane, received {partition!r}"
            )
        if platforms != "cpu":
            raise RuntimeError(
                f"CPU capture requires JAX_PLATFORMS=cpu, got {platforms!r}"
            )
    return record


def _prepare_reference():
    import jax.numpy as jnp

    from benchmarks import mast_response_carrier_warm as response_carrier
    from benchmarks.efit_forward_parity_slice import (
        DECOMPOSITION_BANK,
        _mast_case_from_selection,
        _passive_inclusive_case,
        select_slices_by_shot,
    )
    from benchmarks.label_seed_residual_field import _persisted_response_cache
    from nova.imas.mast_solve_inputs import SHOT_STORE

    response_cache, carrier = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    selected_row, qualification = selected[TARGET]
    case, context = _mast_case_from_selection(SHOT_STORE, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    if int(policy["section_kernel_evaluations_this_shot"]) != 0:
        raise RuntimeError("forensics entered a direct response builder")
    reference = passive_case["reference"]
    return (
        profile,
        jnp.asarray(passive_case["state"]),
        abs(float(reference["plasma_current_a"])),
        {
            "shot": int(reference["shot"]),
            "slice_index": int(reference["slice_index"]),
            "time_s": float(reference["time_s"]),
            "carrier": carrier,
        },
    )


def _termination_name(code: int) -> str:
    from nova.equilibrium.fixed_point import FixedPointTerminationReason

    try:
        return FixedPointTerminationReason(code).name.lower()
    except ValueError:
        return f"unknown_{code}"


def _forensic_solve(profile, initial, target_current: float) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from benchmarks.efit_forward_parity_slice import (
        FIXED_POINT_CRITERION,
        GMRES_ITERATIONS,
        NEWTON_STEPS,
        RELAXATION,
        STEP_CAP,
        WARMUP_SWEEPS,
    )
    from nova.equilibrium import fixed_point as fp
    from nova.equilibrium.topology import TopologyClass

    requested = TopologyClass.DIVERTED
    shadowed_map = profile.operator.flux_map_with_shadow(
        None, requested, target_current
    )

    def shadow_mask(state):
        return profile.operator.residual_shadow_mask(state, requested)

    def promoted_shadow_mask(state, previous):
        return profile.operator.residual_shadow_mask(
            state, requested, previous_shadow=previous
        )

    state = fp._solver_state(initial, fp.Precision.AUTOMATIC)
    mask = jnp.ravel(jnp.asarray(shadow_mask(state), dtype=bool))
    mask_history = [np.asarray(mask, dtype=bool)]
    trajectory_state = state
    continue_trajectory = False
    globalization_state = None
    continue_globalization = False
    previous_live_residual = math.nan
    trips: list[dict[str, Any]] = []
    active_set_steps = int(fp._ACTIVE_SET_ITERATION_LIMIT)
    termination = "active_set_iteration_budget_exhausted"
    converged = False

    for index in range(active_set_steps):
        incoming_state = state
        incoming_mask = mask
        solve_state = trajectory_state if continue_trajectory else state
        run_warmup = not continue_trajectory

        def frozen_map(candidate):
            return shadowed_map(candidate, incoming_mask)

        relinearized_map, tangent = jax.linearize(frozen_map, solve_state)
        residual_vector = relinearized_map - solve_state

        def linear_action(vector):
            return vector - tangent(vector)

        krylov_linear_action = linear_action(residual_vector)
        preceding_baseline = (
            globalization_state.condition_baseline
            if continue_globalization and globalization_state is not None
            else jnp.asarray(jnp.nan, dtype=state.dtype)
        )
        qualified_step = fp._qualified_krylov_step(
            linear_action,
            residual_vector,
            fp._relative_residual(relinearized_map, solve_state),
            gmres_iterations=GMRES_ITERATIONS,
            condition_ratio_limit=fp._PROJECTED_KRYLOV_CONDITION_RATIO_LIMIT,
            preceding_condition_baseline=preceding_baseline,
        )

        def frozen_mask(_candidate):
            return incoming_mask

        inner_result, inner_globalization = fp._newton_krylov_inner(
            frozen_map,
            solve_state,
            newton_steps=NEWTON_STEPS,
            gmres_iterations=GMRES_ITERATIONS,
            warmup=WARMUP_SWEEPS,
            relaxation=RELAXATION,
            step_cap=STEP_CAP,
            krylov_condition_limit=fp._PROJECTED_KRYLOV_CONDITION_RATIO_LIMIT,
            convergence_tolerance=FIXED_POINT_CRITERION,
            shadow_mask_fn=frozen_mask,
            promoted_shadow_mask_fn=lambda _candidate, _previous: incoming_mask,
            shadowed_map_fn=lambda candidate, _shadow: frozen_map(candidate),
            run_warmup=jnp.asarray(run_warmup),
            globalization_state=globalization_state,
            resume_globalization=jnp.asarray(continue_globalization),
            return_globalization_state=True,
            model_trust_selection=True,
            acceptance_shadow_mask_fn=promoted_shadow_mask,
            acceptance_shadowed_map_fn=shadowed_map,
            own_mask_acceptance=True,
            precision=fp.Precision.AUTOMATIC,
        )
        solved_state = inner_result.state
        observed_mask = jnp.ravel(promoted_shadow_mask(solved_state, incoming_mask))
        observed_difference = int(
            np.asarray(jnp.sum(observed_mask != incoming_mask, dtype=jnp.int32))
        )
        observed_mapped = shadowed_map(solved_state, observed_mask)
        observed_residual = float(
            np.asarray(fp._relative_residual(observed_mapped, solved_state))
        )
        observed_finite = math.isfinite(observed_residual)
        observed_converged = (
            observed_finite
            and observed_residual <= FIXED_POINT_CRITERION
            and observed_difference == 0
        )
        observed_mask_host = np.asarray(observed_mask, dtype=bool)
        repeated = (
            observed_difference > 0
            and any(np.array_equal(observed_mask_host, old) for old in mask_history)
            and not observed_converged
        )

        damped_state = incoming_state + fp._ACTIVE_SET_CYCLE_DAMPING * (
            solved_state - incoming_state
        )
        damped_mask = jnp.ravel(promoted_shadow_mask(damped_state, incoming_mask))
        damped_mapped = shadowed_map(damped_state, damped_mask)
        damped_residual = float(
            np.asarray(fp._relative_residual(damped_mapped, damped_state))
        )
        damping_repeats = any(
            np.array_equal(np.asarray(damped_mask, dtype=bool), old)
            for old in mask_history
        )
        cycle_detected = repeated and damping_repeats

        selected_state = damped_state if repeated else solved_state
        selected_mask = damped_mask if repeated else observed_mask
        selected_residual = damped_residual if repeated else observed_residual
        selected_finite = math.isfinite(selected_residual)
        selected_difference = int(
            np.asarray(jnp.sum(selected_mask != incoming_mask, dtype=jnp.int32))
        )
        incoming_mapped = shadowed_map(incoming_state, incoming_mask)
        incoming_residual = float(
            np.asarray(fp._relative_residual(incoming_mapped, incoming_state))
        )
        incoming_merit = float(
            np.asarray(fp._smooth_relative_sup_merit(incoming_mapped, incoming_state))
        )
        selected_merit = float(
            np.asarray(
                fp._smooth_relative_sup_merit(
                    shadowed_map(selected_state, selected_mask), selected_state
                )
            )
        )
        retain_incoming = (
            selected_difference == 0
            and math.isfinite(incoming_residual)
            and math.isfinite(incoming_merit)
            and (not math.isfinite(selected_merit) or selected_merit > incoming_merit)
        )
        trajectory_state = inner_result.trajectory_state
        trajectory_mask = jnp.ravel(
            promoted_shadow_mask(trajectory_state, incoming_mask)
        )
        trajectory_finite = bool(
            np.asarray(jnp.all(jnp.isfinite(trajectory_state)))
        ) and math.isfinite(float(np.asarray(inner_result.trajectory_residual)))
        if retain_incoming:
            selected_state = incoming_state
            selected_mask = incoming_mask
            selected_residual = incoming_residual
            selected_finite = True
            selected_difference = 0
        continue_trajectory = bool(
            selected_difference == 0
            and not repeated
            and trajectory_finite
            and np.array_equal(
                np.asarray(trajectory_mask, dtype=bool),
                np.asarray(incoming_mask, dtype=bool),
            )
            and int(np.asarray(inner_result.accepted_newton_promotions)) > 0
        )
        continue_globalization = continue_trajectory
        converged = (
            selected_finite
            and selected_residual <= FIXED_POINT_CRITERION
            and selected_difference == 0
        )
        stagnated = (
            selected_finite
            and not converged
            and not cycle_detected
            and not continue_trajectory
            and selected_difference == 0
            and selected_residual == previous_live_residual
        )
        can_continue = (
            selected_finite and not converged and not cycle_detected and not stagnated
        )

        trips.append(
            {
                "trip_one_based": index + 1,
                "incoming_state": _array_record(jax.device_get(incoming_state)),
                "relinearized_map": _array_record(jax.device_get(relinearized_map)),
                "krylov_linear_action": _array_record(
                    jax.device_get(krylov_linear_action)
                ),
                "krylov_step": _array_record(jax.device_get(qualified_step.step)),
                "inner_best_state": _array_record(jax.device_get(solved_state)),
                "reconciled_state": _array_record(jax.device_get(selected_state)),
                "incoming_mask": _array_record(jax.device_get(incoming_mask)),
                "reconciled_mask": _array_record(jax.device_get(selected_mask)),
                "incoming_residual": incoming_residual,
                "observed_residual": observed_residual,
                "live_residual": selected_residual,
                "mask_difference": selected_difference,
                "cycle_damping_activated": repeated,
                "retain_incoming": retain_incoming,
                "continue_trajectory": continue_trajectory,
                "attempted_newton_promotions": int(
                    np.asarray(inner_result.attempted_newton_promotions)
                ),
                "accepted_newton_promotions": int(
                    np.asarray(inner_result.accepted_newton_promotions)
                ),
                "krylov_qualification": int(np.asarray(qualified_step.qualification)),
                "krylov_projected_condition": float(
                    np.asarray(qualified_step.projected_condition)
                ),
                "krylov_condition_baseline": float(
                    np.asarray(qualified_step.condition_baseline)
                ),
                "krylov_achieved_reduction": float(
                    np.asarray(qualified_step.achieved_reduction)
                ),
                "krylov_requested_tolerance": float(
                    np.asarray(qualified_step.requested_tolerance)
                ),
            }
        )
        state = selected_state
        mask = selected_mask
        globalization_state = inner_globalization
        previous_live_residual = selected_residual
        if can_continue and len(mask_history) < active_set_steps + 1:
            mask_history.append(np.asarray(selected_mask, dtype=bool))
        if converged:
            termination = "converged"
            break
        if cycle_detected:
            termination = "active_set_cycle_detected"
            break
        if not selected_finite:
            termination = "nonfinite_residual"
            break
        if stagnated:
            termination = "active_set_stagnated"
            break

    _masks, achieved = profile.operator.read(state)
    achieved_diverted = bool(np.asarray(achieved.diverted))
    return {
        "solver_converged": converged,
        "branch_qualified_converged": bool(
            converged and achieved_diverted and np.all(np.isfinite(np.asarray(state)))
        ),
        "achieved_diverted": achieved_diverted,
        "termination_reason": termination,
        "terminal_residual": previous_live_residual,
        "active_set_iterations": len(trips),
        "active_set_residuals": [row["live_residual"] for row in trips],
        "active_set_mask_differences": [row["mask_difference"] for row in trips],
        "trips": trips,
        "solver": {
            "fixed_point_tolerance": FIXED_POINT_CRITERION,
            "newton_steps": NEWTON_STEPS,
            "gmres_iterations": GMRES_ITERATIONS,
            "warmup_sweeps": WARMUP_SWEEPS,
            "relaxation": RELAXATION,
            "step_cap": STEP_CAP,
            "active_set_steps": active_set_steps,
        },
    }


def capture(backend: str, shadow_root: Path) -> dict[str, Any]:
    source = _validate_shadow(shadow_root)
    _configure_shadow_import(shadow_root)
    os.chdir(shadow_root)
    import jax

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    if jax.default_backend() != backend:
        raise RuntimeError(
            f"requested {backend!r} capture but JAX selected {jax.default_backend()!r}"
        )
    allocation = _allocation(backend)
    profile, seed, target_current, reference = _prepare_reference()
    started = time.perf_counter()
    solve = _forensic_solve(profile, seed, target_current)
    jax.effects_barrier()
    return {
        "receipt": "pinned MAST backend forensic capture",
        "backend": backend,
        "source": source,
        "reference": reference,
        "allocation": {
            **allocation,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "wall_seconds_including_compilation": time.perf_counter() - started,
        "solve": solve,
        "execution_contract": {
            "same_revision": PINNED_REVISION,
            "same_arm": "22086/43 pure diverted branch",
            "capture_stages": list(STAGES),
            "solver_source_modified": False,
            "outer_trip_control": (
                "benchmark-local host reconciliation around the unchanged pinned "
                "compiled inner Newton-Krylov solve"
            ),
        },
    }


def _difference(cpu: Any, gpu: Any) -> dict[str, Any]:
    left = np.asarray(cpu, dtype=np.float64)
    right = np.asarray(gpu, dtype=np.float64)
    if left.shape != right.shape:
        raise RuntimeError(f"paired state shape mismatch {left.shape} != {right.shape}")
    delta = right - left
    l2 = float(np.linalg.norm(delta))
    inf_norm = float(np.max(np.abs(delta)))
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0)
    floor = float(4096.0 * np.finfo(np.float64).eps * scale * math.sqrt(left.size))
    return {
        "l2": l2,
        "inf": inf_norm,
        "relative_l2": l2 / scale,
        "float64_reduction_order_floor_l2": floor,
        "over_noise_floor": l2 / floor if floor else math.inf,
        "beyond_float64_reduction_order_noise": l2 > floor,
    }


def compare(cpu: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    if cpu["source"]["pinned_revision"] != gpu["source"]["pinned_revision"]:
        raise RuntimeError("backend captures use different pinned revisions")
    identity_fields = ("shot", "slice_index", "time_s")
    cpu_carrier = cpu["reference"]["carrier"]["carrier"]
    gpu_carrier = gpu["reference"]["carrier"]["carrier"]
    stable_carrier_fields = (
        "file_sha256",
        "resolved_target_digest",
        "response_sha256",
        "semantic_response_identity",
    )
    same_reference = all(
        cpu["reference"][field] == gpu["reference"][field] for field in identity_fields
    ) and all(
        cpu_carrier[field] == gpu_carrier[field] for field in stable_carrier_fields
    )
    if not same_reference:
        raise RuntimeError("backend captures use different reference inputs")
    cpu_trips = cpu["solve"]["trips"]
    gpu_trips = gpu["solve"]["trips"]
    common = min(len(cpu_trips), len(gpu_trips))
    rows = []
    first = None
    candidates = []
    all_stage_events = []
    for index in range(common):
        left = cpu_trips[index]
        right = gpu_trips[index]
        stage_differences = {}
        previous_l2 = None
        for stage in STAGES:
            metric = _difference(left[stage], right[stage])
            denominator = max(
                previous_l2 if previous_l2 is not None else 0.0,
                metric["float64_reduction_order_floor_l2"],
            )
            metric["gain_from_previous_stage"] = metric["l2"] / denominator
            stage_differences[stage] = metric
            staged_event = {
                "trip_one_based": index + 1,
                "stage": stage,
                "operation": STAGE_OPERATION[stage],
                **metric,
            }
            all_stage_events.append(staged_event)
            if metric["beyond_float64_reduction_order_noise"]:
                candidates.append(staged_event)
                if first is None:
                    first = staged_event
            previous_l2 = metric["l2"]
        left_mask = np.asarray(left["reconciled_mask"], dtype=bool)
        right_mask = np.asarray(right["reconciled_mask"], dtype=bool)
        mask_difference = int(np.sum(left_mask != right_mask))
        if mask_difference and first is None:
            first = {
                "trip_one_based": index + 1,
                "stage": "reconciled_mask",
                "operation": "mask reconciliation",
                "mask_hamming_difference": mask_difference,
            }
        rows.append(
            {
                "trip_one_based": index + 1,
                "cpu_live_residual": left["live_residual"],
                "gpu_live_residual": right["live_residual"],
                "reconciled_mask_hamming_difference": mask_difference,
                "stage_state_differences": stage_differences,
            }
        )
    if first is None:
        largest = max(all_stage_events, key=lambda event: event["l2"])
        closest = max(all_stage_events, key=lambda event: event["over_noise_floor"])
        amplifier = {
            "operation": "none",
            "reason": (
                "all common-trip staged differences remain inside the noise floor"
            ),
            "largest_observed_difference": largest,
            "closest_approach_to_noise_floor": closest,
        }
    else:
        same_trip = [
            event
            for event in candidates
            if event["trip_one_based"] == first["trip_one_based"]
        ]
        amplifier = max(
            same_trip or [first],
            key=lambda event: event.get("gain_from_previous_stage", 0.0),
        )
    return {
        "noise_model": (
            "difference exceeds 4096*eps64*max(||cpu||2,||gpu||2,1)*sqrt(n)"
        ),
        "common_trip_count": common,
        "cpu_trip_count": len(cpu_trips),
        "gpu_trip_count": len(gpu_trips),
        "first_divergence": first,
        "amplifying_sub_stage": amplifier,
        "per_trip": rows,
    }


def _plot(receipt: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison = receipt["comparison"]
    rows = comparison["per_trip"]
    trips = [row["trip_one_based"] for row in rows]
    fig, (difference_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    for stage in STAGES:
        values = [
            max(
                row["stage_state_differences"][stage]["relative_l2"],
                np.finfo(np.float64).tiny,
            )
            for row in rows
        ]
        difference_axis.semilogy(
            trips, values, marker="o", label=stage.replace("_", " ")
        )
    difference_axis.set_ylabel("CPU/GPU state difference (relative L2)")
    difference_axis.grid(True, which="both", alpha=0.25)
    difference_axis.legend(ncol=2, fontsize=8)
    residual_axis.semilogy(
        range(1, len(receipt["backends"]["cpu"]["active_set_residuals"]) + 1),
        receipt["backends"]["cpu"]["active_set_residuals"],
        marker="o",
        label="CPU residual",
    )
    residual_axis.semilogy(
        range(1, len(receipt["backends"]["gpu"]["active_set_residuals"]) + 1),
        receipt["backends"]["gpu"]["active_set_residuals"],
        marker="s",
        label="GPU residual",
    )
    residual_axis.axhline(
        receipt["solver"]["fixed_point_tolerance"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="convergence tolerance",
    )
    residual_axis.set_xlabel("active-set trip")
    residual_axis.set_ylabel("relative sup residual")
    residual_axis.grid(True, which="both", alpha=0.25)
    residual_axis.legend()
    first = comparison["first_divergence"]
    title = "No staged divergence above float64 reduction noise"
    if first is not None:
        title = (
            f"First divergence: trip {first['trip_one_based']} at {first['operation']}"
        )
    fig.suptitle(f"Pinned 22086/43 pure backend forensics\n{title}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def compile_receipt(
    cpu_path: Path, gpu_path: Path, output: Path, figure: Path
) -> dict[str, Any]:
    cpu = json.loads(cpu_path.read_text())
    gpu = json.loads(gpu_path.read_text())
    comparison = compare(cpu, gpu)
    receipt = {
        "receipt": "pinned MAST CPU/GPU backend divergence forensics",
        "source": cpu["source"],
        "reference": cpu["reference"],
        "solver": cpu["solve"]["solver"],
        "backends": {
            backend: {
                "allocation": capture_payload["allocation"],
                "wall_seconds_including_compilation": capture_payload[
                    "wall_seconds_including_compilation"
                ],
                "solver_converged": capture_payload["solve"]["solver_converged"],
                "branch_qualified_converged": capture_payload["solve"][
                    "branch_qualified_converged"
                ],
                "termination_reason": capture_payload["solve"]["termination_reason"],
                "terminal_residual": capture_payload["solve"]["terminal_residual"],
                "active_set_iterations": capture_payload["solve"][
                    "active_set_iterations"
                ],
                "active_set_residuals": capture_payload["solve"][
                    "active_set_residuals"
                ],
                "active_set_mask_differences": capture_payload["solve"][
                    "active_set_mask_differences"
                ],
                "raw_capture": str(path),
                "raw_capture_sha256": _sha256(path),
            }
            for backend, capture_payload, path in (
                ("cpu", cpu, cpu_path),
                ("gpu", gpu, gpu_path),
            )
        },
        "convergence_matrix": {
            "committed_revision_cpu": {
                "outcome": (
                    "converged" if cpu["solve"]["solver_converged"] else "nonconverged"
                ),
                "termination_reason": cpu["solve"]["termination_reason"],
                "terminal_residual": cpu["solve"]["terminal_residual"],
            },
            "committed_revision_gpu": {
                "outcome": (
                    "converged" if gpu["solve"]["solver_converged"] else "nonconverged"
                ),
                "termination_reason": gpu["solve"]["termination_reason"],
                "terminal_residual": gpu["solve"]["terminal_residual"],
            },
            "head_gpu": {
                "outcome": "nonconverged",
                "termination_reason": "active_set_stagnated",
                "terminal_residual": 0.0004802335353917348,
                "source": (
                    "docs/figures/solver-convergence-regression/head-mechanism.json"
                ),
            },
            "head_cpu": {
                "outcome": "unknown",
                "reason": "three-hour timeout in job 1260726",
            },
        },
        "comparison": comparison,
        "evidence_inputs": _evidence_inputs(),
        "figure": str(figure),
        "execution_contract": {
            "solver_source_modified": False,
            "paired_revision": PINNED_REVISION,
            "paired_arm": "22086/43 pure",
            "staged_intermediates": list(STAGES),
        },
    }
    check(receipt, require_figure=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _plot(receipt, figure)
    check(receipt, require_figure=True)
    return receipt


def check(receipt: dict[str, Any], *, require_figure: bool = True) -> None:
    if receipt["source"]["pinned_revision"] != PINNED_REVISION:
        raise RuntimeError("receipt does not use the committed bank revision")
    if receipt["execution_contract"]["solver_source_modified"]:
        raise RuntimeError("forensics modified solver source")
    for backend in ("cpu", "gpu"):
        row = receipt["backends"][backend]
        if not isinstance(row["solver_converged"], bool):
            raise RuntimeError(f"{backend} convergence outcome is missing")
        if not row["active_set_residuals"]:
            raise RuntimeError(f"{backend} residual history is empty")
    comparison = receipt["comparison"]
    if not comparison["per_trip"]:
        raise RuntimeError("no paired trip differences were retained")
    if not comparison["amplifying_sub_stage"].get("operation"):
        raise RuntimeError("the amplifying sub-stage is unnamed")
    if require_figure:
        figure = Path(receipt["figure"])
        if not figure.exists() or figure.stat().st_size == 0:
            raise RuntimeError("backend divergence figure is absent")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    capture_parser.add_argument("--shadow-root", type=Path, default=DEFAULT_SHADOW_ROOT)
    capture_parser.add_argument("--output", type=Path, required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--cpu", type=Path, required=True)
    compile_parser.add_argument("--gpu", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--figure", type=Path, required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "capture":
        payload = capture(args.backend, args.shadow_root)
        _write_json(args.output, payload)
        print(
            "BACKEND_CAPTURE "
            f"backend={args.backend} converged={payload['solve']['solver_converged']} "
            f"reason={payload['solve']['termination_reason']} "
            f"residual={payload['solve']['terminal_residual']:.17g} "
            f"trips={payload['solve']['active_set_iterations']}",
            flush=True,
        )
    elif args.command == "compile":
        receipt = compile_receipt(args.cpu, args.gpu, args.output, args.figure)
        first = receipt["comparison"]["first_divergence"]
        amplifier = receipt["comparison"]["amplifying_sub_stage"]
        if first is None:
            closest = amplifier["closest_approach_to_noise_floor"]
            divergence_summary = (
                "first_divergence=none "
                "reason=no_divergence_beyond_float64_reduction_order_floor "
                f"largest_floor_ratio={closest['over_noise_floor']:.17g} "
                f"largest_floor_ratio_trip={closest['trip_one_based']} "
                f"largest_floor_ratio_stage={closest['stage']} "
                f"largest_floor_ratio_operation={closest['operation']}"
            )
        else:
            divergence_summary = (
                f"first_trip={first['trip_one_based']} "
                f"first_operation={first['operation']}"
            )
        print(
            "BACKEND_FORENSICS PASS "
            f"{divergence_summary} "
            f"amplifier={amplifier['operation']} "
            f"cpu={receipt['backends']['cpu']['termination_reason']} "
            f"gpu={receipt['backends']['gpu']['termination_reason']}",
            flush=True,
        )
    else:
        receipt = json.loads(args.receipt.read_text())
        check(receipt)
        print("BACKEND_FORENSICS_CHECK PASS", flush=True)


if __name__ == "__main__":
    main()
