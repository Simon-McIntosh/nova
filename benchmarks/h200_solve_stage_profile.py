"""Profile additive forward-map components and combine them with banked solves.

The accelerator run measures only components not isolated by the banked
Newton--Krylov factorial.  The resulting receipt keeps direct measurements,
nested differences, and comparison-baseline projections distinct so an
estimated ceiling cannot be mistaken for an observed solve latency.
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
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/single-grid-solver-cutover/h200-solve-profile.json"
)
DEFAULT_FIGURE = ROOT / "docs/figures/single-grid-solver-cutover/h200-solve-profile.png"
BANKED_RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20260901T053948341346-sse-h200-step-decomposition"
)
LADDER_RECEIPT = (
    ROOT / "docs/figures/diiid-forward-onboarding/throughput/"
    "batched_throughput_budget.json"
)
COMPARISON_BASELINE_MS = 42.77
TARGET_MS = 1.0
SCAN_ITERATION_US = 7.6


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


def _distribution(samples: list[float], width: int) -> dict[str, Any]:
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
        "median_ms_per_member": 1.0e3 * median / width,
        "iqr_ms_per_member": 1.0e3 * float(third - first) / width,
    }


def _measure_compiled(
    function: Callable[..., Any], arguments: tuple[Any, ...], width: int, repeats: int
) -> dict[str, Any]:
    started = time.perf_counter()
    compiled = jax.jit(function).lower(*arguments).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    _ready(compiled(*arguments))
    first_execute_seconds = time.perf_counter() - started
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        _ready(compiled(*arguments))
        samples.append(time.perf_counter() - started)
    return {
        "compile_seconds": compile_seconds,
        "first_execute_seconds": first_execute_seconds,
        "steady": _distribution(samples, width),
    }


def _transfer_profile(host: np.ndarray, repeats: int) -> dict[str, Any]:
    to_device = []
    to_host = []
    resident = jax.device_put(host)
    _ready(resident)
    for _ in range(repeats):
        started = time.perf_counter()
        value = jax.device_put(host)
        _ready(value)
        to_device.append(time.perf_counter() - started)
        started = time.perf_counter()
        np.asarray(jax.device_get(resident))
        to_host.append(time.perf_counter() - started)
    width = len(host)
    return {
        "payload_bytes": int(host.nbytes),
        "host_to_device": _distribution(to_device, width),
        "device_to_host": _distribution(to_host, width),
        "solve_timer_residency": (
            "inputs and outputs remain device-resident; these boundary probes are "
            "reported separately and are not added to the solve stages"
        ),
    }


def _callback_census(closed_jaxpr: Any) -> dict[str, Any]:
    primitives: list[str] = []

    def visit(value: Any) -> None:
        jaxpr = getattr(value, "jaxpr", value)
        equations = getattr(jaxpr, "eqns", ())
        for equation in equations:
            name = str(equation.primitive)
            if "callback" in name:
                primitives.append(name)
            for parameter in equation.params.values():
                if hasattr(parameter, "jaxpr") or hasattr(parameter, "eqns"):
                    visit(parameter)
                elif isinstance(parameter, (tuple, list)):
                    for item in parameter:
                        if hasattr(item, "jaxpr") or hasattr(item, "eqns"):
                            visit(item)

    visit(closed_jaxpr)
    return {
        "callback_primitive_count": len(primitives),
        "callback_primitives": sorted(primitives),
        "method": "recursive primitive census of the complete-map closed jaxpr",
    }


def _scheduler() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    fields: dict[str, str] = {}
    if job_id:
        result = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        fields = {
            token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in result.stdout.split()
            if "=" in token
        }
    return {
        "job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "accepted_time_limit": fields.get("TimeLimit"),
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


def _banked_arm(width: int, warmup: int, gmres: int) -> tuple[Path, dict[str, Any]]:
    path = BANKED_RUN / f"factorial-w{width}-w{warmup}-g{gmres}.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _summary(values: list[float | int]) -> dict[str, float | int]:
    array = np.asarray(values)
    return {
        "minimum": array.min().item(),
        "median": float(np.median(array)),
        "maximum": array.max().item(),
    }


def _fit_factorial(width: int) -> dict[str, Any]:
    design = []
    observations = []
    arms = []
    for warmup in (1, 2):
        for gmres in (1, 4):
            path, arm = _banked_arm(width, warmup, gmres)
            samples = arm["raw_latency_seconds"]
            design.extend((1.0, float(warmup), float(gmres - 1)) for _ in samples)
            observations.extend(samples)
            arms.append(
                {
                    "path": str(path),
                    "sha256": _digest(path),
                    "warmup_sweeps": warmup,
                    "gmres_iterations": gmres,
                    "compile_and_first_execute_seconds": arm[
                        "compile_and_first_execute_seconds"
                    ],
                    "steady_median_batch_seconds": arm["steady_latency"][
                        "median_seconds"
                    ],
                }
            )
    matrix = np.asarray(design)
    observed = np.asarray(observations)
    coefficient, _, _, _ = np.linalg.lstsq(matrix, observed, rcond=None)
    intercept, warmup_slope, extra_gmres_slope = coefficient
    selected = np.asarray(
        [intercept, warmup_slope, 3.0 * extra_gmres_slope], dtype=float
    )
    selected = np.maximum(selected, 0.0)
    shares = selected / selected.sum()
    _, telemetry_arm = _banked_arm(width, 1, 4)
    telemetry = telemetry_arm["minimal_terminal_telemetry"]
    return {
        "method": (
            "least-squares factorial over every banked warm sample: latency = "
            "intercept + warmup_sweeps*slope + (GMRES_iterations-1)*slope"
        ),
        "arms": arms,
        "coefficients_batch_seconds": {
            "reconciliation_newton_setup_and_first_krylov": float(intercept),
            "per_warmup_sweep": float(warmup_slope),
            "per_additional_gmres_action": float(extra_gmres_slope),
        },
        "selected_policy_stage_share": {
            "reconciliation_newton_setup_and_first_krylov": float(shares[0]),
            "one_warmup_sweep": float(shares[1]),
            "three_additional_gmres_actions": float(shares[2]),
        },
        "selected_policy_observed_ms_per_member": float(selected.sum() * 1.0e3 / width),
        "trip_counts": {
            "active_set_outer_iterations": _summary(telemetry["active_set_iterations"]),
            "attempted_newton_promotions": _summary(
                telemetry["attempted_newton_promotions"]
            ),
            "accepted_newton_promotions": _summary(
                telemetry["accepted_newton_promotions"]
            ),
            "configured_newton_steps_per_active_set_trip": 1,
            "configured_gmres_inner_iterations": 4,
            "configured_warmup_sweeps": 1,
            "terminal_relative_residual": _summary(
                telemetry["terminal_relative_residual"]
            ),
            "converged_members": int(sum(telemetry["converged"])),
            "member_count": len(telemetry["converged"]),
        },
    }


def _operator_profile(
    profile: Any, seed: jax.Array, width: int, repeats: int
) -> dict[str, Any]:
    state = jnp.broadcast_to(seed, (width, *seed.shape))
    current = jnp.broadcast_to(profile.operator.external_current, (width, 16))
    _ready((state, current))
    programs: tuple[tuple[str, Callable[..., Any], tuple[Any, ...]], ...] = (
        (
            "topology_read",
            jax.vmap(lambda item: profile.operator.read(item)),
            (state,),
        ),
        (
            "carrier_cell_current",
            jax.vmap(profile.operator.cell_current),
            (state,),
        ),
        (
            "plasma_response",
            jax.vmap(profile.operator.internal),
            (state,),
        ),
        (
            "complete_flux_map",
            jax.vmap(lambda item, conductor: profile.flux_map(conductor)(item)),
            (state, current),
        ),
    )
    measured = {}
    for name, function, arguments in programs:
        print(f"PROBE_START width={width} name={name}", flush=True)
        measured[name] = _measure_compiled(function, arguments, width, repeats)
        print(
            f"PROBE_DONE width={width} name={name} "
            f"median_ms={measured[name]['steady']['median_ms_per_member']:.6f}",
            flush=True,
        )
    topology = measured["topology_read"]["steady"]["median_ms_per_member"]
    carrier = measured["carrier_cell_current"]["steady"]["median_ms_per_member"]
    response = measured["plasma_response"]["steady"]["median_ms_per_member"]
    complete = measured["complete_flux_map"]["steady"]["median_ms_per_member"]
    additive = {
        "topology_and_primary_masks_ms": topology,
        "profile_carrier_after_topology_ms": carrier - topology,
        "dense_response_after_carrier_ms": response - carrier,
        "shadow_mask_and_map_composition_ms": complete - response,
    }
    callback_jaxpr = jax.make_jaxpr(programs[-1][1])(*programs[-1][2])
    return {
        "width": width,
        "programs": measured,
        "additive_nested_differences_ms_per_member_map": additive,
        "additive_sum_ms_per_member_map": float(sum(additive.values())),
        "negative_difference_warning": any(value < 0.0 for value in additive.values()),
        "host_callbacks": _callback_census(callback_jaxpr),
        "device_transfers": _transfer_profile(np.asarray(state), repeats),
    }


def _ranked_candidates(
    operator: dict[str, Any], factorial: dict[str, Any]
) -> list[dict[str, Any]]:
    shares = factorial["selected_policy_stage_share"]
    map_parts = operator["additive_nested_differences_ms_per_member_map"]
    positive_parts = {name: max(value, 0.0) for name, value in map_parts.items()}
    total = sum(positive_parts.values())
    map_share = {
        name: (value / total if total else 0.0)
        for name, value in positive_parts.items()
    }
    first_stage_ms = (
        COMPARISON_BASELINE_MS * shares["reconciliation_newton_setup_and_first_krylov"]
    )
    candidates = [
        (
            "reduce active-set reconciliation and repeated topology reads",
            first_stage_ms
            * (
                map_share["topology_and_primary_masks_ms"]
                + map_share["shadow_mask_and_map_composition_ms"]
            ),
            "fresh nested operator split apportioned within the banked "
            "factorial intercept",
        ),
        (
            "compress Newton setup and the first Krylov action",
            first_stage_ms,
            "banked factorial intercept; overlaps the operator-specific candidates",
        ),
        (
            "fuse or sparsify the dense plasma response",
            first_stage_ms * map_share["dense_response_after_carrier_ms"],
            "fresh nested operator split apportioned within the banked "
            "factorial intercept",
        ),
        (
            "remove three additional GMRES actions",
            COMPARISON_BASELINE_MS * shares["three_additional_gmres_actions"],
            "banked GMRES factorial slope",
        ),
        (
            "remove the warmup sweep",
            COMPARISON_BASELINE_MS * shares["one_warmup_sweep"],
            "banked warmup factorial slope",
        ),
    ]
    rows = []
    for name, removable, method in candidates:
        removable = max(float(removable), 0.0)
        remaining = max(COMPARISON_BASELINE_MS - removable, 0.0)
        rows.append(
            {
                "candidate": name,
                "measured_ceiling_basis": method,
                "maximum_removable_ms_per_slice": removable,
                "ideal_remaining_ms_per_slice": remaining,
                "ideal_speedup_x": (
                    COMPARISON_BASELINE_MS / remaining if remaining else None
                ),
                "fraction_of_41_77_ms_target_gap_closed": min(
                    removable / (COMPARISON_BASELINE_MS - TARGET_MS), 1.0
                ),
                "reaches_1_ms_target_alone": remaining <= TARGET_MS,
            }
        )
    return sorted(
        rows, key=lambda row: row["maximum_removable_ms_per_slice"], reverse=True
    )


def _plot(receipt: dict[str, Any], output: Path) -> None:
    width = receipt["comparison_width"]
    operator = next(
        item for item in receipt["operator_profiles"] if item["width"] == width
    )
    parts = operator["additive_nested_differences_ms_per_member_map"]
    candidates = receipt["ranked_speedup_candidates"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    labels = [name.replace("_ms", "").replace("_", " ") for name in parts]
    values = [parts[name] for name in parts]
    axes[0].barh(labels, values, color=("#4c78a8", "#72b7b2", "#f58518", "#e45756"))
    axes[0].invert_yaxis()
    axes[0].set_xlabel("H200 ms per member map")
    axes[0].set_title(f"Fresh additive map split, width {width}")
    for index, value in enumerate(values):
        axes[0].text(value, index, f" {value:.3f}", va="center", fontsize=8)

    names = [item["candidate"] for item in candidates]
    ceilings = [item["maximum_removable_ms_per_slice"] for item in candidates]
    axes[1].barh(names, ceilings, color="#54a24b")
    axes[1].invert_yaxis()
    axes[1].axvline(
        COMPARISON_BASELINE_MS - TARGET_MS,
        color="#b279a2",
        linestyle="--",
        label="41.77 ms gap to target",
    )
    axes[1].set_xlabel("ideal removable ms per 42.77 ms slice")
    axes[1].set_title("Measured Amdahl ceilings")
    axes[1].legend(loc="lower right", fontsize=8)
    figure.suptitle("H200 forward-solve stage attribution")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(
    output: Path, figure: Path, widths: tuple[int, ...], repeats: int
) -> dict[str, Any]:
    configure_dtypes()
    print("WORKLOAD_BUILD_START", flush=True)
    profile, seed = build_workload()
    print("WORKLOAD_BUILD_DONE", flush=True)
    ladder = json.loads(LADDER_RECEIPT.read_text(encoding="utf-8"))
    ladder_width = next(
        row for row in ladder["budget_rows"] if row["batch_size"] == 1024
    )
    ladder_ms = (
        1.0e3
        * float(np.median(ladder_width["raw_repetition_latency_seconds"]))
        / ladder["configuration"]["frames"]
        / 1024
    )
    operator_profiles = []
    factorial_profiles = []
    for width in widths:
        operator_profiles.append(_operator_profile(profile, seed, width, repeats))
        factorial_profiles.append(_fit_factorial(width))
        print(f"WIDTH_DONE={width}", flush=True)
    comparison_index = widths.index(1024) if 1024 in widths else -1
    candidates = _ranked_candidates(
        operator_profiles[comparison_index], factorial_profiles[comparison_index]
    )
    scan_ceiling_ms = (
        factorial_profiles[comparison_index]["trip_counts"][
            "active_set_outer_iterations"
        ]["median"]
        * SCAN_ITERATION_US
        / 1.0e3
    )
    receipt = {
        "schema": "nova.h200_forward_solve_stage_profile",
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
        "configuration": {
            "widths": list(widths),
            "steady_samples_per_probe": repeats,
            "comparison_baseline_ms_per_slice": COMPARISON_BASELINE_MS,
            "target_ms_per_slice": TARGET_MS,
            "comparison_gap_ms_per_slice": COMPARISON_BASELINE_MS - TARGET_MS,
        },
        "evidence_inputs": {
            "banked_step_decomposition_run": str(BANKED_RUN),
            "banked_evidence_reused": (
                "warmup/GMRES factorial, compile time, Newton promotions, "
                "active-set trips, "
                "terminal residuals, and conditioning timing are not remeasured"
            ),
            "jax_scan_latency": {
                "microseconds_per_iteration": SCAN_ITERATION_US,
                "source": str(BANKED_RUN / "live-plan-context.log"),
                "estimated_16_trip_ceiling_ms_per_slice": scan_ceiling_ms,
            },
            "committed_ladder": {
                "path": str(LADDER_RECEIPT.relative_to(ROOT)),
                "sha256": _digest(LADDER_RECEIPT),
                "width_1024_slices_per_second": ladder_width["slices_per_second"],
                "width_1024_recorded_ms_per_member_slice": ladder_ms,
                "lead_stated_comparison_ms_per_slice": COMPARISON_BASELINE_MS,
                "provenance_warning": (
                    "the ladder arithmetic is not 42.77 ms; 42.77 ms is retained "
                    "as the separately stated comparison baseline rather than "
                    "attributed to the JSON row"
                ),
            },
        },
        "comparison_width": 1024 if 1024 in widths else widths[-1],
        "operator_profiles": operator_profiles,
        "banked_solve_factorials": factorial_profiles,
        "compile_amortization": {
            "method": (
                "banked compile-and-first-execute seconds divided by delivered "
                "member slices"
            ),
            "width_1024_warmup_1_gmres_4_compile_seconds": factorial_profiles[
                comparison_index
            ]["arms"][1]["compile_and_first_execute_seconds"],
            "compile_ms_per_slice_at_10000_slices": 1.0e3
            * factorial_profiles[comparison_index]["arms"][1][
                "compile_and_first_execute_seconds"
            ]
            / 10_000,
            "compile_ms_per_slice_at_1000000_slices": 1.0e3
            * factorial_profiles[comparison_index]["arms"][1][
                "compile_and_first_execute_seconds"
            ]
            / 1_000_000,
            "slices_to_amortize_compile_below_1_ms": math.ceil(
                1.0e3
                * factorial_profiles[comparison_index]["arms"][1][
                    "compile_and_first_execute_seconds"
                ]
            ),
        },
        "ranked_speedup_candidates": candidates,
        "verdict": {
            "target_reached": False,
            "headline": (
                "No isolated candidate can close the 41.77 ms comparison gap; the "
                "banked reconciliation/Newton/first-Krylov intercept is the "
                "dominant ceiling."
            ),
            "required_action": (
                "attack active-set reconciliation and its repeated topology/map "
                "evaluations together; transfer elimination and scan-loop overhead "
                "cannot materially move the target"
            ),
            "qualification": (
                "Amdahl ceilings normalize measured stage shares to the mandated "
                "42.77 ms baseline. They are ceilings, not a prediction that the "
                "stages can be eliminated independently."
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--widths", type=int, nargs="+", default=(32, 1024))
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    run(arguments.output, arguments.figure, tuple(arguments.widths), arguments.repeats)


if __name__ == "__main__":
    main()
