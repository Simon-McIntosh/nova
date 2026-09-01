"""Measure the named kernels inside one batched solver-trip quantum.

The banked solve factorial remains the authority for the complete repeated
quantum.  This driver measures only the unresolved width-1024 kernel ratios,
then apportions the banked intercept without re-running its full solve.
Direct probe latency and normalized Amdahl ceilings remain separate fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.diiid_batched_throughput import build_workload
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/solver-trip-orchestration/trip-quantum.json"
DEFAULT_FIGURE = ROOT / "docs/figures/solver-trip-orchestration/trip-quantum.png"
SOURCE_PROFILE = (
    ROOT / "docs/figures/single-grid-solver-cutover/h200-solve-profile.json"
)
BANKED_RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20260901T053948341346-sse-h200-step-decomposition"
)
WIDTH = 1024
TRIPS = 16
COMPARISON_BASELINE_MS = 42.77
TARGET_FLOOR_MS = 1.7459246876209988


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready(value: Any) -> Any:
    leaves = jax.tree.leaves(value)
    if leaves:
        jax.block_until_ready(leaves[0])
    return value


def _distribution(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=float)
    first, third = np.percentile(values, (25.0, 75.0))
    median = float(np.median(values))
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_batch_seconds": median,
        "iqr_batch_seconds": float(third - first),
        "minimum_batch_seconds": float(values.min()),
        "maximum_batch_seconds": float(values.max()),
        "median_ms_per_member": 1.0e3 * median / WIDTH,
    }


def _measure(
    name: str,
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    repeats: int,
) -> dict[str, Any]:
    print(f"COMPILE_START name={name}", flush=True)
    started = time.perf_counter()
    compiled = jax.jit(function).lower(*arguments).compile()
    compile_seconds = time.perf_counter() - started
    print(f"COMPILE_DONE name={name} seconds={compile_seconds:.6f}", flush=True)
    started = time.perf_counter()
    _ready(compiled(*arguments))
    first_execute_seconds = time.perf_counter() - started
    samples = []
    for repeat in range(repeats):
        started = time.perf_counter()
        _ready(compiled(*arguments))
        samples.append(time.perf_counter() - started)
        print(
            f"SAMPLE_DONE name={name} repeat={repeat + 1}/{repeats} "
            f"seconds={samples[-1]:.9f}",
            flush=True,
        )
    return {
        "compile_seconds": compile_seconds,
        "first_execute_seconds": first_execute_seconds,
        "steady": _distribution(samples),
    }


def _scheduler() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    accepted = None
    if job_id:
        completed = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            fields = {
                token.split("=", 1)[0]: token.split("=", 1)[1]
                for token in completed.stdout.split()
                if "=" in token
            }
            accepted = fields.get("TimeLimit")
    return {
        "job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "accepted_time_limit": accepted,
        "temporary_directory": os.environ.get("TMPDIR"),
    }


def _device() -> dict[str, Any]:
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    return {
        "platform": device.platform,
        "kind": device.device_kind,
        "id": int(device.id),
        "host": platform.node(),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "memory_bytes_limit": int(stats.get("bytes_limit", 0)) or None,
        "peak_device_memory_bytes": int(
            stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0))
        ),
    }


def _banked_quantum(profile: dict[str, Any]) -> dict[str, Any]:
    factorial = next(
        item
        for item in profile["banked_solve_factorials"]
        if item["trip_counts"]["member_count"] == WIDTH
    )
    counts = factorial["trip_counts"]
    required = (
        counts["active_set_outer_iterations"]["minimum"],
        counts["active_set_outer_iterations"]["median"],
        counts["active_set_outer_iterations"]["maximum"],
        counts["attempted_newton_promotions"]["minimum"],
        counts["accepted_newton_promotions"]["minimum"],
        counts["configured_gmres_inner_iterations"],
    )
    if required != (TRIPS, float(TRIPS), TRIPS, TRIPS, TRIPS, 4):
        raise RuntimeError(f"banked trip census changed: {required!r}")
    batch_seconds = factorial["coefficients_batch_seconds"][
        "reconciliation_newton_setup_and_first_krylov"
    ]
    comparison_ceiling = (
        COMPARISON_BASELINE_MS
        * factorial["selected_policy_stage_share"][
            "reconciliation_newton_setup_and_first_krylov"
        ]
    )
    if not math.isclose(
        COMPARISON_BASELINE_MS - comparison_ceiling,
        TARGET_FLOOR_MS,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("landed comparison floor no longer matches the profile")
    return {
        "batch_seconds_for_sixteen_trips": batch_seconds,
        "batch_seconds_per_trip": batch_seconds / TRIPS,
        "measured_ms_per_member_per_trip": 1.0e3 * batch_seconds / TRIPS / WIDTH,
        "comparison_normalized_ms_per_slice_for_sixteen_trips": comparison_ceiling,
        "comparison_normalized_ms_per_slice_per_trip": comparison_ceiling / TRIPS,
        "trip_census": counts,
    }


def _programs(profile: Any, state: jax.Array, initial_mask: jax.Array | None = None):
    operator = profile.operator
    external = operator.external(profile.operator.external_current)
    external = jnp.broadcast_to(external, state.shape)
    if initial_mask is None:
        initial_mask = jax.vmap(operator.residual_shadow_mask)(state)
    _ready((state, external, initial_mask))

    def one_mask(candidate, shadow):
        observed = operator.residual_shadow_mask(candidate, previous_shadow=shadow)
        difference = jnp.sum(observed != shadow, dtype=jnp.int32)
        return observed, difference

    def frozen_map(candidate, shadow, source_flux):
        image = source_flux + operator.internal(candidate)
        return jnp.where(shadow, candidate, image)

    def mask_reconciliation(candidate, shadow, _source_flux):
        return jax.vmap(one_mask)(candidate, shadow)

    def relinearization(candidate, shadow, source_flux):
        def one(state_row, mask_row, external_row):
            mapped, _tangent = jax.linearize(
                lambda value: frozen_map(value, mask_row, external_row), state_row
            )
            return mapped - state_row

        return jax.vmap(one)(candidate, shadow, source_flux)

    def relinearization_and_first_action(candidate, shadow, source_flux):
        def one(state_row, mask_row, external_row):
            mapped, tangent = jax.linearize(
                lambda value: frozen_map(value, mask_row, external_row), state_row
            )
            residual = mapped - state_row
            return residual - tangent(residual)

        return jax.vmap(one)(candidate, shadow, source_flux)

    def fused_reconciliation_and_first_action(candidate, shadow, source_flux):
        def one(state_row, mask_row, external_row):
            observed, difference = one_mask(state_row, mask_row)
            mapped, tangent = jax.linearize(
                lambda value: frozen_map(value, observed, external_row), state_row
            )
            residual = mapped - state_row
            action = residual - tangent(residual)
            return action, difference

        return jax.vmap(one)(candidate, shadow, source_flux)

    return (
        (state, initial_mask, external),
        {
            "mask_reconciliation_gather_scatter_comparison": mask_reconciliation,
            "newton_relinearization": relinearization,
            "newton_relinearization_and_first_gmres_action": (
                relinearization_and_first_action
            ),
            "fused_reconciliation_and_first_gmres_action": (
                fused_reconciliation_and_first_action
            ),
        },
    )


def _attribution(
    probes: dict[str, Any], launch: dict[str, Any], banked: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    latency = {
        key: value["steady"]["median_batch_seconds"] for key, value in probes.items()
    }
    launch_seconds = launch["steady"]["median_batch_seconds"]
    mask_compute = max(
        latency["mask_reconciliation_gather_scatter_comparison"] - launch_seconds,
        0.0,
    )
    relinearization_compute = max(
        latency["newton_relinearization"] - launch_seconds, 0.0
    )
    first_action_compute = max(
        latency["newton_relinearization_and_first_gmres_action"]
        - latency["newton_relinearization"],
        0.0,
    )
    direct = {
        "mask_reconciliation_gather_scatter_comparison": mask_compute,
        "newton_relinearization": relinearization_compute,
        "preconditioner_assembly": 0.0,
        "first_gmres_action_synchronization": first_action_compute,
        "fixed_host_launch_overhead_per_trip": 0.0,
    }
    denominator = sum(direct.values())
    if denominator <= 0.0:
        raise RuntimeError("direct sub-stage probes have no positive latency")
    comparison_total = banked["comparison_normalized_ms_per_slice_for_sixteen_trips"]
    measured_total = banked["measured_ms_per_member_per_trip"]
    rows = []
    for name, seconds in direct.items():
        share = seconds / denominator
        rows.append(
            {
                "substage": name,
                "direct_probe_batch_seconds": seconds,
                "direct_probe_ms_per_member": 1.0e3 * seconds / WIDTH,
                "share_of_positive_direct_probe_latency": share,
                "attributed_measured_ms_per_member_per_trip": measured_total * share,
                "attributed_comparison_ms_per_slice_across_sixteen_trips": (
                    comparison_total * share
                ),
                "method": (
                    "positive direct probe ratio apportioned over the banked "
                    "factorial intercept"
                ),
            }
        )
    separate_seconds = (
        latency["mask_reconciliation_gather_scatter_comparison"]
        + latency["newton_relinearization_and_first_gmres_action"]
    )
    fused_seconds = latency["fused_reconciliation_and_first_gmres_action"]
    fusion_saving_seconds = max(separate_seconds - fused_seconds, 0.0)
    fusion_share = min(fusion_saving_seconds / denominator, 1.0)
    fusion = {
        "separate_probe_batch_seconds": separate_seconds,
        "fused_probe_batch_seconds": fused_seconds,
        "direct_fusion_saving_batch_seconds": fusion_saving_seconds,
        "direct_fusion_speedup_x": separate_seconds / fused_seconds,
        "normalized_comparison_ceiling_ms_per_slice": comparison_total * fusion_share,
        "scalar_compiled_dispatch_upper_bound_batch_seconds": launch_seconds,
        "host_launchs_in_production_solve": 1,
        "active_set_trips_inside_production_host_launch": TRIPS,
        "removable_host_launches_per_trip": 0,
        "interpretation": (
            "the production solve already places all trips inside one JIT host "
            "launch; the measured fusion delta is device-program composition, "
            "not removable Python launch latency"
        ),
    }
    return rows, fusion


def _candidates(
    attribution: list[dict[str, Any]], fusion: dict[str, Any], banked: dict[str, Any]
) -> list[dict[str, Any]]:
    ceilings = {
        row["substage"]: row["attributed_comparison_ms_per_slice_across_sixteen_trips"]
        for row in attribution
    }
    total = banked["comparison_normalized_ms_per_slice_for_sixteen_trips"]
    candidates = [
        (
            "fuse the complete reconciliation and Newton trip body",
            total,
            "structural upper bound: remove the complete banked quantum",
        ),
        (
            "reuse or hoist the Newton linearization",
            ceilings["newton_relinearization"],
            "width-1024 direct re-linearization ratio",
        ),
        (
            "fuse mask reconciliation with the Newton setup",
            ceilings["mask_reconciliation_gather_scatter_comparison"],
            "width-1024 direct gather/scatter/comparison ratio",
        ),
        (
            "schedule the first GMRES action with its linearization",
            ceilings["first_gmres_action_synchronization"],
            "width-1024 incremental first-action synchronization ratio",
        ),
        (
            "realize the measured explicit-program fusion",
            fusion["normalized_comparison_ceiling_ms_per_slice"],
            "separate versus fused width-1024 probe delta",
        ),
        (
            "remove per-trip host launches",
            0.0,
            "all sixteen production trips already execute inside one host launch",
        ),
        (
            "remove preconditioner assembly",
            0.0,
            "production GMRES supplies no preconditioner and assembles none",
        ),
    ]
    rows = []
    for candidate, ceiling, basis in candidates:
        ceiling = max(float(ceiling), 0.0)
        remaining = COMPARISON_BASELINE_MS - ceiling
        rows.append(
            {
                "candidate": candidate,
                "measured_ceiling_basis": basis,
                "maximum_removable_ms_per_slice": ceiling,
                "ideal_remaining_ms_per_slice": remaining,
                "margin_above_1_75_ms_floor": remaining - TARGET_FLOOR_MS,
                "reaches_1_75_ms_floor": math.isclose(
                    remaining, TARGET_FLOOR_MS, rel_tol=0.0, abs_tol=1.0e-12
                ),
            }
        )
    return sorted(
        rows, key=lambda row: row["maximum_removable_ms_per_slice"], reverse=True
    )


def _plot(receipt: dict[str, Any], output: Path) -> None:
    attribution = receipt["substage_attribution"]
    candidates = receipt["ranked_fusion_and_restructuring_candidates"]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    names = [row["substage"].replace("_", " ") for row in attribution]
    values = [row["attributed_measured_ms_per_member_per_trip"] for row in attribution]
    axes[0].barh(names, values, color="#4c78a8")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Attributed H200 ms per member per trip")
    axes[0].set_title("Width-1024 trip quantum")
    for index, value in enumerate(values):
        axes[0].text(value, index, f" {value:.3f}", va="center", fontsize=8)

    candidate_names = [row["candidate"] for row in candidates]
    ceilings = [row["maximum_removable_ms_per_slice"] for row in candidates]
    axes[1].barh(candidate_names, ceilings, color="#54a24b")
    axes[1].invert_yaxis()
    axes[1].axvline(
        COMPARISON_BASELINE_MS - TARGET_FLOOR_MS,
        color="#b279a2",
        linestyle="--",
        label="quantum to 1.75 ms floor",
    )
    axes[1].set_xlabel("Maximum removable ms per comparison slice")
    axes[1].set_title("Individual Amdahl ceilings")
    axes[1].legend(loc="lower right", fontsize=8)
    figure.suptitle("H200 solver-trip quantum decomposition")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(output: Path, figure: Path, repeats: int) -> dict[str, Any]:
    configure_dtypes()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("measurement requires a JAX GPU")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("measurement requires partition betelgeuse")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("measurement requires reservation gpu_0003_grpA")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("measurement requires TMPDIR=/tmp")

    landed = json.loads(SOURCE_PROFILE.read_text(encoding="utf-8"))
    banked = _banked_quantum(landed)
    workload, seed = build_workload()
    state = jnp.broadcast_to(seed, (WIDTH, *seed.shape))
    arguments, programs = _programs(workload, state)
    _ready(arguments)
    launch = _measure(
        "scalar_compiled_dispatch",
        lambda value: value + jnp.asarray(1, dtype=value.dtype),
        (jnp.asarray(0, dtype=jnp.int32),),
        repeats,
    )
    probes = {
        name: _measure(name, function, arguments, repeats)
        for name, function in programs.items()
    }
    attribution, fusion = _attribution(probes, launch, banked)
    candidates = _candidates(attribution, fusion, banked)
    receipt = {
        "schema": "nova.solver_trip_quantum_profile",
        "measurement_state": "complete",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _digest(Path(__file__)),
        },
        "scheduler": _scheduler(),
        "device": _device(),
        "configuration": {"width": WIDTH, "steady_samples_per_probe": repeats},
        "evidence_inputs": {
            "landed_stage_profile": {
                "path": str(SOURCE_PROFILE.relative_to(ROOT)),
                "sha256": _digest(SOURCE_PROFILE),
            },
            "banked_step_decomposition_run": str(BANKED_RUN),
            "reused_without_remeasurement": (
                "sixteen active-set trips, sixteen attempted and accepted Newton "
                "promotions, GMRES=4, and the complete factorial intercept"
            ),
        },
        "banked_quantum": banked,
        "direct_width_1024_probes": probes,
        "scalar_compiled_dispatch_probe": launch,
        "substage_attribution": attribution,
        "fusion_measurement": fusion,
        "ranked_fusion_and_restructuring_candidates": candidates,
        "verdict": {
            "headline": (
                "only complete trip-body collapse reaches the landed 1.75 ms "
                "floor; isolated sub-stage changes leave a positive margin"
            ),
            "preconditioner_assembly": (
                "zero: the production GMRES call supplies no preconditioner"
            ),
            "host_launch_overhead": (
                "zero per trip: all sixteen trips already execute inside one "
                "compiled host launch"
            ),
            "qualification": (
                "direct probes determine sub-stage ratios; the landed factorial "
                "intercept remains the additive solve-time authority"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict(receipt), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(receipt, figure)
    print(f"RECEIPT_WRITTEN={output}", flush=True)
    print(f"FIGURE_WRITTEN={figure}", flush=True)
    return receipt


def preflight() -> None:
    configure_dtypes()
    landed = json.loads(SOURCE_PROFILE.read_text(encoding="utf-8"))
    banked = _banked_quantum(landed)
    workload, seed = build_workload()
    state = jnp.broadcast_to(seed, (WIDTH, *seed.shape))
    initial_mask = jnp.zeros_like(state, dtype=bool)
    arguments, programs = _programs(workload, state, initial_mask)
    shaped = {
        name: jax.eval_shape(function, *arguments)
        for name, function in programs.items()
    }
    print(
        json.dumps(
            {
                "status": "preflight_complete",
                "width": WIDTH,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "programs": sorted(shaped),
                "banked_quantum": banked,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight()
    else:
        run(arguments.output, arguments.figure, arguments.repeats)


if __name__ == "__main__":
    main()
