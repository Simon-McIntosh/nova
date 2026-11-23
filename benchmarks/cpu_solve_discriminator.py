"""Measure the HEAD CPU cost of the banked MAST pure-arm solve route."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET = (22086, 43)
REFERENCE_REVISION = "a4bec44f5cbf80ad5e210c01c984ac8d02a89de9"
REFERENCE_JOB_ID = 1260944
REFERENCE_ELAPSED_SECONDS = 38 * 60
REFERENCE_RECEIPT = (
    ROOT / "docs/figures/solver-convergence-regression/backend-divergence.json"
)
DEFAULT_JSON = ROOT / "docs/figures/cpu-solve-viability/cpu-discriminator.json"
DEFAULT_FIGURE = ROOT / "docs/figures/cpu-solve-viability/cpu-discriminator.png"
THREAD_CONFIGURATION_JSON = (
    ROOT / "docs/figures/cpu-solve-viability/thread-configuration.json"
)
THREAD_CONFIGURATION_FIGURE = (
    ROOT / "docs/figures/cpu-solve-viability/thread-configuration.png"
)
THREAD_CONFIGURATION_TRIPS = 3
THREADED_XLA_FLAGS = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16"
THREADED_POOL_SETTINGS = {
    "OMP_NUM_THREADS": "16",
    "OPENBLAS_NUM_THREADS": "16",
    "MKL_NUM_THREADS": "16",
    "NUMEXPR_NUM_THREADS": "16",
}
EXPECTED_CARRIER_SHA256 = (
    "1da2b7bdb4a79d6b81513fa4aba909d318bd157c9b5453340b122bfd595428c9"
)
EXPECTED_CARRIER_IDENTITY = (
    "1d2c4a2b2f448ab8f1ae981031bbaf85fe4ee87f8ed9606fe6847d0fc9f1e994"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], text=True
    ).strip()


def _stage(name: str, started: float) -> float:
    elapsed = time.perf_counter() - started
    print(f"STAGE name={name} wall_seconds={elapsed:.6f}", flush=True)
    return elapsed


def _allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("capture requires a scheduler allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    memory_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if not partition.startswith("rigel"):
        raise RuntimeError(f"capture requires a rigel node, received {partition!r}")
    if cpus != 16:
        raise RuntimeError(f"capture requires 16 CPUs, received {cpus}")
    if memory_mb < 64 * 1024:
        raise RuntimeError(
            f"capture requires at least 64 GiB, received {memory_mb} MiB"
        )
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError(
            f"capture requires TMPDIR=/tmp, got {os.environ.get('TMPDIR')!r}"
        )
    if platforms != "cpu":
        raise RuntimeError(f"capture requires JAX_PLATFORMS=cpu, got {platforms!r}")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": partition,
        "allocated_cpus": cpus,
        "allocated_memory_gib": memory_mb / 1024.0,
        "allocated_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", "0")),
        "tmpdir": os.environ.get("TMPDIR"),
        "jax_platforms": platforms.split(","),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "step_id": os.environ.get("SLURM_STEP_ID"),
        "affinity_cpu_count": len(os.sched_getaffinity(0)),
        "python": sys.executable,
    }


@dataclass
class _CallbackCensus:
    """Retain ordered active-set callback timing without changing solver carry."""

    solve_started: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        self.solve_started = time.perf_counter()

    def callback(
        self,
        active,
        trip_index,
        mask_difference,
        live_residual,
        inner_iterations,
    ) -> None:
        callback_started = time.perf_counter()
        if not bool(active):
            return
        if self.solve_started is None:
            raise RuntimeError("active-set callback preceded the solve clock")
        event = {
            "trip_index_zero_based": int(trip_index),
            "mask_difference": int(mask_difference),
            "live_residual": float(live_residual),
            "inner_iterations": int(inner_iterations),
            "elapsed_since_solve_start_seconds": (
                callback_started - self.solve_started
            ),
        }
        event["host_callback_seconds"] = time.perf_counter() - callback_started
        with self._lock:
            event["callback_ordinal"] = len(self.events) + 1
            self.events.append(event)
        print(
            "CPU_DISCRIMINATOR_TRIP "
            f"callback={event['callback_ordinal']} "
            f"trip={event['trip_index_zero_based']} "
            f"residual={event['live_residual']:.17g} "
            f"elapsed={event['elapsed_since_solve_start_seconds']:.6f} "
            f"host_seconds={event['host_callback_seconds']:.6f}",
            flush=True,
        )


@dataclass
class _CompilationCensus:
    """Measure wall time spent in JAX backend compilation calls."""

    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def timed(self, compiler):
        original = compiler.backend_compile_and_load

        def observed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                module = args[1] if len(args) > 1 else kwargs.get("module")
                try:
                    module_name = str(module.operation.attributes["sym_name"])
                except AttributeError, KeyError, TypeError:
                    module_name = "unknown"
                with self._lock:
                    self.events.append(
                        {
                            "ordinal": len(self.events) + 1,
                            "module": module_name,
                            "wall_seconds": elapsed,
                        }
                    )

        return original, observed

    def receipt(self) -> dict[str, Any]:
        wall_seconds = [float(event["wall_seconds"]) for event in self.events]
        return {
            "method": (
                "sum of wall durations around "
                "jax._src.compiler.backend_compile_and_load"
            ),
            "call_count": len(self.events),
            "compile_seconds": float(sum(wall_seconds)),
            "maximum_compile_call_seconds": max(wall_seconds, default=0.0),
            "events": self.events,
        }


def _prepare_reference() -> tuple[Any, Any, float, dict[str, Any]]:
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

    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    carrier = carrier_evidence.get("carrier", carrier_evidence)
    if carrier["file_sha256"] != EXPECTED_CARRIER_SHA256:
        raise RuntimeError("persisted carrier file digest does not match the bank")
    if carrier["semantic_response_identity"] != EXPECTED_CARRIER_IDENTITY:
        raise RuntimeError(
            "persisted carrier semantic identity does not match the bank"
        )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    row, qualification = selected[TARGET]
    case, context = _mast_case_from_selection(SHOT_STORE, row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    if int(policy["section_kernel_evaluations_this_shot"]) != 0:
        raise RuntimeError("CPU discriminator entered a direct response builder")
    reference = passive_case["reference"]
    return (
        profile,
        jnp.asarray(passive_case["state"]),
        abs(float(reference["plasma_current_a"])),
        {
            "shot": int(reference["shot"]),
            "slice_index": int(reference["slice_index"]),
            "time_s": float(reference["time_s"]),
            "carrier": carrier_evidence,
            "loaded_from_persisted_carrier": True,
            "direct_green_operator_builder_entries": 0,
        },
    )


def _diverted_trip_events(
    events: list[dict[str, Any]], residuals: list[float]
) -> list[dict[str, Any]]:
    selected = []
    prior_ordinal = 0
    for trip_index, residual in enumerate(residuals):
        candidates = [
            event
            for event in events
            if event["trip_index_zero_based"] == trip_index
            and event["callback_ordinal"] > prior_ordinal
            and math.isclose(
                event["live_residual"], residual, rel_tol=1.0e-11, abs_tol=1.0e-14
            )
        ]
        if not candidates:
            raise RuntimeError(
                "no callback matches diverted trip "
                f"{trip_index + 1} residual {residual}"
            )
        event = min(candidates, key=lambda candidate: candidate["callback_ordinal"])
        selected.append(event)
        prior_ordinal = event["callback_ordinal"]
    previous = 0.0
    timed = []
    for one_based, event in enumerate(selected, start=1):
        elapsed = float(event["elapsed_since_solve_start_seconds"])
        timed.append(
            {
                "trip_one_based": one_based,
                "wall_seconds": elapsed - previous,
                "cumulative_wall_seconds": elapsed,
                "live_residual": event["live_residual"],
                "mask_difference": event["mask_difference"],
                "inner_iterations": event["inner_iterations"],
                "callback_ordinal": event["callback_ordinal"],
            }
        )
        previous = elapsed
    return timed


def _validate_thread_configuration(
    allocation: dict[str, Any], configuration: str
) -> None:
    if configuration not in {"baseline", "threaded"}:
        return
    for variable, expected in THREADED_POOL_SETTINGS.items():
        key = variable.lower()
        if allocation[key] != expected:
            raise RuntimeError(
                f"{configuration} capture requires {variable}={expected}, "
                f"received {allocation[key]!r}"
            )
    if allocation["affinity_cpu_count"] != 16:
        raise RuntimeError(
            f"{configuration} capture has affinity for "
            f"{allocation['affinity_cpu_count']} CPUs rather than 16"
        )
    flags = allocation["xla_flags"]
    if configuration == "baseline" and flags:
        raise RuntimeError(f"baseline capture requires empty XLA_FLAGS, got {flags!r}")
    if configuration == "threaded" and flags != THREADED_XLA_FLAGS:
        raise RuntimeError(
            "threaded capture requires the allocation-wide XLA CPU pool flags"
        )


def capture(
    output: Path,
    *,
    active_set_steps: int | None = None,
    configuration: str = "head",
) -> dict[str, Any]:
    import jax
    from jax._src import compiler

    from benchmarks import bank_revision_reproduction as bank_route
    from nova.equilibrium import fixed_point
    from nova.jax.config import configure_dtypes

    allocation = _allocation()
    _validate_thread_configuration(allocation, configuration)
    capture_started_epoch = time.time()
    stage_started = time.perf_counter()
    configure_dtypes()
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"capture selected JAX backend {jax.default_backend()!r}")
    configure_seconds = _stage("configure_cpu_backend", stage_started)

    stage_started = time.perf_counter()
    profile, seed, target_current, reference = _prepare_reference()
    preparation_seconds = _stage("load_persisted_carrier_and_case", stage_started)

    census = _CallbackCensus()
    compilation = _CompilationCensus()
    original_newton_krylov = fixed_point.newton_krylov
    original_callback = fixed_point._print_active_set_trip
    original_compiler, observed_compiler = compilation.timed(compiler)

    def observed_newton_krylov(*args, **kwargs):
        kwargs["stream_active_set"] = True
        if active_set_steps is not None:
            kwargs["active_set_steps"] = active_set_steps
        return original_newton_krylov(*args, **kwargs)

    fixed_point.newton_krylov = observed_newton_krylov
    fixed_point._print_active_set_trip = census.callback
    compiler.backend_compile_and_load = observed_compiler
    try:
        census.start()
        solve = bank_route._solve_pure_arm(
            profile,
            seed,
            target_current,
            bank_route._corroboration_module(ROOT),
        )
        jax.effects_barrier()
    finally:
        fixed_point.newton_krylov = original_newton_krylov
        fixed_point._print_active_set_trip = original_callback
        compiler.backend_compile_and_load = original_compiler
    solve_seconds = _stage(
        "bank_revision_reproduction_pure_arm",
        census.solve_started or time.perf_counter(),
    )
    residuals = list(solve.get("active_set_residuals") or [])
    per_trip = _diverted_trip_events(census.events, residuals)
    callback_seconds = [
        float(event["host_callback_seconds"]) for event in census.events
    ]
    payload = {
        "receipt": f"HEAD MAST CPU pure-arm {configuration} raw capture",
        "configuration": configuration,
        "source": {
            "revision": _git("rev-parse", "HEAD"),
            "root": str(ROOT),
            "nova_diff_stat": _git("diff", "--stat", "--", "nova"),
        },
        "allocation": {
            **allocation,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "reference": reference,
        "solve": solve,
        "per_trip": per_trip,
        "telemetry_callback_census": {
            "all_portfolio_callback_count": len(census.events),
            "diverted_callback_count": len(per_trip),
            "total_host_callback_seconds": float(sum(callback_seconds)),
            "mean_host_callback_seconds": (
                float(np.mean(callback_seconds)) if callback_seconds else 0.0
            ),
            "events": census.events,
        },
        "compilation": compilation.receipt(),
        "stage_wall_seconds": {
            "configure_cpu_backend": configure_seconds,
            "load_persisted_carrier_and_case": preparation_seconds,
            "bank_revision_reproduction_pure_arm": solve_seconds,
        },
        "execution_contract": {
            "capture_started_epoch": capture_started_epoch,
            "route": "benchmarks.bank_revision_reproduction._solve_pure_arm",
            "observational_override": "stream_active_set=True",
            "active_set_steps": active_set_steps,
            "solver_source_modified": False,
            "persisted_carrier_reused": True,
            "target": list(TARGET),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "CPU_DISCRIMINATOR_CAPTURE PASS "
        f"configuration={configuration} "
        f"wall_seconds={solve['solve_wall_seconds_including_compilation']:.6f} "
        f"compile_seconds={payload['compilation']['compile_seconds']:.6f} "
        f"trips={solve['trip_count']} "
        f"residual={solve['terminal_residual']:.17g} "
        f"termination={solve['termination_reason']}",
        flush=True,
    )
    return payload


_TIME_PATTERN = re.compile(
    r"^(?:(?P<days>\d+)-)?(?:(?P<hours>\d+):)?(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)$"
)


def _slurm_seconds(value: str) -> float:
    match = _TIME_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(f"unsupported Slurm time {value!r}")
    return (
        float(match.group("days") or 0) * 86400.0
        + float(match.group("hours") or 0) * 3600.0
        + float(match.group("minutes")) * 60.0
        + float(match.group("seconds"))
    )


def _sstat_samples(
    path: Path,
    job_id: int,
    capture_started_epoch: float,
    *,
    step_id: str = "0",
) -> dict[str, Any]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("sample_epoch|"):
            continue
        fields = line.split("|", 5)
        if len(fields) < 5:
            continue
        epoch, step, ave_cpu, ave_rss, max_rss = fields[:5]
        if step != f"{job_id}.{step_id}" or not ave_cpu:
            continue
        sample_epoch = float(epoch)
        cumulative_cpu_seconds = _slurm_seconds(ave_cpu)
        observable_wall = max(sample_epoch - capture_started_epoch, 0.0)
        if cumulative_cpu_seconds > 16.0 * (observable_wall + 60.0):
            continue
        rows.append(
            {
                "sample_epoch": sample_epoch,
                "step": step,
                "cumulative_cpu_seconds": cumulative_cpu_seconds,
                "average_rss": ave_rss,
                "maximum_rss": max_rss,
            }
        )
    if len(rows) < 2:
        raise RuntimeError(f"sstat retained only {len(rows)} usable process samples")
    deltas = []
    for left, right in zip(rows, rows[1:]):
        wall = right["sample_epoch"] - left["sample_epoch"]
        cpu = right["cumulative_cpu_seconds"] - left["cumulative_cpu_seconds"]
        if wall > 0 and cpu >= 0:
            deltas.append(cpu / wall)
    if not deltas:
        raise RuntimeError("sstat samples contain no positive observation intervals")
    observed_wall = rows[-1]["sample_epoch"] - capture_started_epoch
    average = rows[-1]["cumulative_cpu_seconds"] / observed_wall
    return {
        "method": (
            "effective process threads = delta cumulative AveCPU / delta sample wall; "
            "sstat step samples include the Python process and its XLA worker threads"
        ),
        "sample_count": len(rows),
        "interval_count": len(deltas),
        "average_effective_threads": average,
        "median_interval_effective_threads": float(np.median(deltas)),
        "maximum_interval_effective_threads": max(deltas),
        "allocated_threads": 16,
        "step": f"{job_id}.{step_id}",
        "samples": rows,
    }


def _reference() -> dict[str, Any]:
    receipt = json.loads(REFERENCE_RECEIPT.read_text())
    cpu = receipt["backends"]["cpu"]
    trip_count = int(cpu["active_set_iterations"])
    wall = float(cpu["wall_seconds_including_compilation"])
    amortized = wall / trip_count
    return {
        "revision": REFERENCE_REVISION,
        "job_id": REFERENCE_JOB_ID,
        "node": cpu["allocation"]["node"],
        "allocated_cpus": cpu["allocation"]["allocated_cpus"],
        "job_elapsed_seconds": REFERENCE_ELAPSED_SECONDS,
        "wall_seconds": wall,
        "trip_count": trip_count,
        "terminal_residual": cpu["terminal_residual"],
        "termination": cpu["termination_reason"],
        "per_trip_wall_seconds": [amortized] * trip_count,
        "per_trip_timing_basis": (
            "amortized total solve wall across seven trips because job 1260944 did "
            "not retain callback timestamps"
        ),
        "active_set_residuals": cpu["active_set_residuals"],
        "source": str(REFERENCE_RECEIPT),
        "source_sha256": _sha256(REFERENCE_RECEIPT),
    }


def _verdict(head_wall: float) -> tuple[str, str]:
    if head_wall > 3 * 3600:
        classification = "head-side"
        sentence = (
            "The greater-than-3-hour pathology is HEAD-side: the exact bank pure-arm "
            f"route took {head_wall / 3600.0:.2f} h at HEAD."
        )
    else:
        classification = "producer-specific"
        sentence = (
            "The greater-than-3-hour pathology is producer-specific: the exact bank "
            f"pure-arm route completed at HEAD in {head_wall / 60.0:.1f} min, so the "
            "pathology remains confined to generate_topology_visuals."
        )
    return classification, sentence


def _plot(receipt: dict[str, Any], figure: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    head = receipt["runs"]["head"]
    reference = receipt["runs"]["reference"]
    head_values = head["per_trip_wall_seconds"]
    reference_values = reference["per_trip_wall_seconds"]
    extent = max(len(head_values), len(reference_values))
    x = np.arange(1, extent + 1)
    width = 0.38
    fig, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    axis.bar(
        x[: len(reference_values)] - width / 2,
        reference_values,
        width,
        color="#A8B4C4",
        hatch="//",
        label="a4bec44f amortized reference",
    )
    axis.bar(
        x[: len(head_values)] + width / 2,
        head_values,
        width,
        color="#D36B3F",
        label="HEAD callback-to-callback",
    )
    axis.set_xlabel("active-set trip")
    axis.set_ylabel("wall seconds")
    axis.set_title("MAST 22086/43 CPU bank-route trip cost")
    axis.set_xticks(x)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.text(
        0.01,
        0.98,
        "Reference bars amortize the retained 2,200.212 s total; "
        "exact old trip timestamps were not recorded.",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)


def compile_receipt(
    raw_path: Path,
    sstat_path: Path,
    output: Path,
    figure: Path,
    *,
    job_elapsed: str,
    job_state: str,
    exit_marker: int,
) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text())
    solve = raw["solve"]
    per_trip = raw["per_trip"]
    head_wall = float(solve["solve_wall_seconds_including_compilation"])
    thread_usage = _sstat_samples(
        sstat_path,
        int(raw["allocation"]["job_id"]),
        float(raw["execution_contract"]["capture_started_epoch"]),
    )
    all_callbacks = raw["telemetry_callback_census"]["events"]
    diverted_completion = float(per_trip[-1]["cumulative_wall_seconds"])
    final_callback = max(
        float(event["elapsed_since_solve_start_seconds"]) for event in all_callbacks
    )
    limited_continuation = final_callback - diverted_completion
    post_callback_completion = head_wall - final_callback
    post_diverted_tail = head_wall - diverted_completion
    classification, sentence = _verdict(head_wall)
    receipt = {
        "receipt": "MAST 22086/43 CPU solve viability discriminator",
        "runs": {
            "head": {
                "revision": raw["source"]["revision"],
                "wall_seconds": head_wall,
                "per_trip_wall_seconds": [row["wall_seconds"] for row in per_trip],
                "per_trip": per_trip,
                "trip_count": solve["trip_count"],
                "terminal_residual": solve["terminal_residual"],
                "termination": solve["termination_reason"],
                "converged": solve["converged"],
                "allocation": raw["allocation"],
                "scheduler": {
                    "job_id": raw["allocation"]["job_id"],
                    "node": raw["allocation"]["node"],
                    "elapsed": job_elapsed,
                    "elapsed_seconds": _slurm_seconds(job_elapsed),
                    "state": job_state,
                    "exit_marker": exit_marker,
                    "limit_seconds": 4 * 3600,
                },
                "effective_xla_cpu_threads_from_sstat": thread_usage,
                "telemetry_callback_census": raw["telemetry_callback_census"],
                "stage_wall_seconds": raw["stage_wall_seconds"],
            },
            "reference": _reference(),
        },
        "dominant_cpu_cost": {
            "name": (
                "post-diverted portfolio tail: limited-branch continuation plus "
                "result materialization"
            ),
            "wall_seconds": post_diverted_tail,
            "share_of_solve_wall": post_diverted_tail / head_wall,
            "measurement": "ordered active-set callback timestamps",
            "components": {
                "limited_branch_callback_continuation_seconds": limited_continuation,
                "post_final_callback_completion_seconds": post_callback_completion,
                "selected_diverted_completion_seconds": diverted_completion,
                "final_portfolio_callback_seconds": final_callback,
            },
        },
        "verdict": {
            "classification": classification,
            "sentence": sentence,
        },
        "evidence_inputs": {
            "raw_capture": str(raw_path),
            "raw_capture_sha256": _sha256(raw_path),
            "sstat_samples": str(sstat_path),
            "sstat_samples_sha256": _sha256(sstat_path),
            "reference_receipt": str(REFERENCE_RECEIPT),
        },
        "execution_contract": {
            **raw["execution_contract"],
            "allocated_cpus": raw["allocation"]["allocated_cpus"],
            "allocated_memory_gib": raw["allocation"]["allocated_memory_gib"],
            "assigned_worktree_nova_diff_stat": _git("diff", "--stat", "--", "nova"),
            "launch_pattern": (
                "hardened sbatch with 60-second heartbeat, flushed stage timings, "
                "30-second sstat process sampling, and launch-then-harvest"
            ),
        },
        "figure": str(figure),
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
    head = receipt["runs"]["head"]
    reference = receipt["runs"]["reference"]
    if head["revision"] == REFERENCE_REVISION:
        raise RuntimeError("HEAD capture unexpectedly equals the reference revision")
    if head["allocation"]["allocated_cpus"] != 16:
        raise RuntimeError("HEAD capture does not record 16 allocated CPUs")
    if head["scheduler"]["limit_seconds"] != 4 * 3600:
        raise RuntimeError("HEAD capture does not record the four-hour limit")
    if (
        head["scheduler"]["exit_marker"] != 0
        or head["scheduler"]["state"] != "COMPLETED"
    ):
        raise RuntimeError("HEAD capture did not complete cleanly")
    if len(head["per_trip_wall_seconds"]) != int(head["trip_count"]):
        raise RuntimeError("HEAD per-trip timing census is incomplete")
    if len(reference["per_trip_wall_seconds"]) != int(reference["trip_count"]):
        raise RuntimeError("reference per-trip timing census is incomplete")
    if head["effective_xla_cpu_threads_from_sstat"]["sample_count"] < 2:
        raise RuntimeError("HEAD capture lacks sstat process utilisation samples")
    if not receipt["dominant_cpu_cost"]["name"]:
        raise RuntimeError("dominant CPU cost is unnamed")
    dominant = receipt["dominant_cpu_cost"]
    components = dominant["components"]
    component_sum = (
        components["limited_branch_callback_continuation_seconds"]
        + components["post_final_callback_completion_seconds"]
    )
    if not math.isclose(component_sum, dominant["wall_seconds"], abs_tol=1.0e-9):
        raise RuntimeError("dominant CPU tail components do not sum to its wall time")
    if not receipt["verdict"]["sentence"]:
        raise RuntimeError("CPU-pathology verdict is absent")
    if receipt["execution_contract"]["assigned_worktree_nova_diff_stat"]:
        raise RuntimeError("assigned worktree contains a change under nova")
    if require_figure:
        figure = Path(receipt["figure"])
        if not figure.exists() or figure.stat().st_size == 0:
            raise RuntimeError("CPU discriminator figure is absent")


def _relative_differences(observed: list[float], expected: list[float]) -> list[float]:
    return [
        abs(left - right) / max(abs(right), np.finfo(np.float64).tiny)
        for left, right in zip(observed, expected, strict=True)
    ]


def _thread_run(
    raw_path: Path,
    sstat_path: Path,
    *,
    job_elapsed: str,
    job_state: str,
    exit_marker: int,
) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text())
    step_id = str(raw["allocation"]["step_id"])
    usage = _sstat_samples(
        sstat_path,
        int(raw["allocation"]["job_id"]),
        float(raw["execution_contract"]["capture_started_epoch"]),
        step_id=step_id,
    )
    per_trip = raw["per_trip"]
    residuals = [float(row["live_residual"]) for row in per_trip]
    return {
        "configuration": raw["configuration"],
        "revision": raw["source"]["revision"],
        "per_trip_wall_seconds": [float(row["wall_seconds"]) for row in per_trip],
        "per_trip_residuals": residuals,
        "trip_count": len(per_trip),
        "terminal_residual": float(raw["solve"]["terminal_residual"]),
        "termination": raw["solve"]["termination_reason"],
        "solve_wall_seconds_including_compilation": float(
            raw["solve"]["solve_wall_seconds_including_compilation"]
        ),
        "compile_seconds": float(raw["compilation"]["compile_seconds"]),
        "compilation": raw["compilation"],
        "effective_xla_cpu_threads_from_sstat": usage,
        "allocation": raw["allocation"],
        "scheduler": {
            "job_id": raw["allocation"]["job_id"],
            "step_id": step_id,
            "node": raw["allocation"]["node"],
            "elapsed": job_elapsed,
            "elapsed_seconds": _slurm_seconds(job_elapsed),
            "state": job_state,
            "exit_marker": exit_marker,
            "limit_seconds": 3 * 3600,
        },
        "stage_wall_seconds": raw["stage_wall_seconds"],
        "raw_capture": str(raw_path),
        "raw_capture_sha256": _sha256(raw_path),
        "execution_contract": raw["execution_contract"],
    }


def _thread_verdict(threaded_per_trip: list[float], reference_seconds: float) -> dict:
    average = float(np.mean(threaded_per_trip))
    ratio = average / reference_seconds
    closes_gap = ratio <= 1.1
    if closes_gap:
        conclusion = (
            "threading alone closes the gap, so no material HEAD-side per-trip "
            "work increase remains"
        )
    else:
        conclusion = (
            "threading alone does not close the gap, so a HEAD-side per-trip work "
            "increase remains"
        )
    return {
        "threaded_mean_per_trip_seconds": average,
        "pinned_reference_seconds": reference_seconds,
        "threaded_to_reference_ratio": ratio,
        "threading_alone_closes_gap": closes_gap,
        "sentence": (
            f"With the 16-core XLA-CPU pool engaged, the first three HEAD trips "
            f"averaged {average:.1f} s against the {reference_seconds:.0f} s "
            f"pinned-revision reference ({ratio:.2f}x); {conclusion}."
        ),
    }


def _plot_thread_configuration(receipt: dict[str, Any], figure: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = receipt["configurations"]["baseline"]["per_trip_wall_seconds"]
    threaded = receipt["configurations"]["threaded"]["per_trip_wall_seconds"]
    reference = receipt["pinned_revision_reference"]["per_trip_wall_seconds"]
    x = np.arange(1, THREAD_CONFIGURATION_TRIPS + 1)
    width = 0.26
    fig, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    axis.bar(
        x - width,
        baseline,
        width,
        color="#D36B3F",
        label="HEAD control: XLA CPU pool unset",
    )
    axis.bar(
        x,
        threaded,
        width,
        color="#3F87B5",
        label="HEAD: 16-core XLA CPU pool",
    )
    axis.bar(
        x + width,
        reference,
        width,
        color="#A8B4C4",
        hatch="//",
        label="a4bec44f amortized reference",
    )
    axis.set_xlabel("active-set trip")
    axis.set_ylabel("wall seconds")
    axis.set_title("MAST 22086/43 XLA-CPU thread configuration")
    axis.set_xticks(x)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.text(
        0.01,
        0.98,
        receipt["verdict"]["sentence"],
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        wrap=True,
    )
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)


def compile_thread_configuration(
    baseline_raw_path: Path,
    threaded_raw_path: Path,
    sstat_path: Path,
    output: Path,
    figure: Path,
    *,
    job_elapsed: str,
    job_state: str,
    exit_marker: int,
) -> dict[str, Any]:
    baseline = _thread_run(
        baseline_raw_path,
        sstat_path,
        job_elapsed=job_elapsed,
        job_state=job_state,
        exit_marker=exit_marker,
    )
    threaded = _thread_run(
        threaded_raw_path,
        sstat_path,
        job_elapsed=job_elapsed,
        job_state=job_state,
        exit_marker=exit_marker,
    )
    banked_receipt = json.loads(DEFAULT_JSON.read_text())
    banked_head = banked_receipt["runs"]["head"]
    banked_per_trip = [
        float(value)
        for value in banked_head["per_trip_wall_seconds"][:THREAD_CONFIGURATION_TRIPS]
    ]
    banked_residuals = [
        float(row["live_residual"])
        for row in banked_head["per_trip"][:THREAD_CONFIGURATION_TRIPS]
    ]
    same_head_differences = _relative_differences(
        threaded["per_trip_residuals"], baseline["per_trip_residuals"]
    )
    baseline_to_banked_differences = _relative_differences(
        baseline["per_trip_residuals"], banked_residuals
    )
    threaded_to_banked_differences = _relative_differences(
        threaded["per_trip_residuals"], banked_residuals
    )
    pinned = _reference()
    reference_seconds = float(pinned["per_trip_wall_seconds"][0])
    receipt = {
        "receipt": "MAST 22086/43 XLA-CPU thread configuration",
        "configurations": {
            "baseline": baseline,
            "threaded": threaded,
        },
        "banked_head_baseline": {
            "source": str(DEFAULT_JSON),
            "source_sha256": _sha256(DEFAULT_JSON),
            "revision": banked_head["revision"],
            "per_trip_wall_seconds": banked_per_trip,
            "per_trip_residuals": banked_residuals,
            "average_effective_threads": banked_head[
                "effective_xla_cpu_threads_from_sstat"
            ]["average_effective_threads"],
            "xla_flags": banked_head["allocation"]["xla_flags"],
        },
        "pinned_revision_reference": {
            **pinned,
            "per_trip_wall_seconds": [reference_seconds] * THREAD_CONFIGURATION_TRIPS,
        },
        "residual_agreement": {
            "gate_basis": "fresh threaded capture versus fresh same-HEAD baseline",
            "relative_tolerance": 1.0e-9,
            "threaded_to_same_head_baseline_relative_differences": (
                same_head_differences
            ),
            "maximum_relative_difference": max(same_head_differences),
            "banked_revision_context": {
                "reason": (
                    "The banked HEAD capture predates stationary-point polishing "
                    "inside Topology.read, so its residual difference is revision "
                    "context rather than a threading semantic gate."
                ),
                "baseline_to_banked_relative_differences": (
                    baseline_to_banked_differences
                ),
                "threaded_to_banked_relative_differences": (
                    threaded_to_banked_differences
                ),
                "maximum_baseline_to_banked_relative_difference": max(
                    baseline_to_banked_differences
                ),
                "maximum_threaded_to_banked_relative_difference": max(
                    threaded_to_banked_differences
                ),
            },
        },
        "verdict": _thread_verdict(
            threaded["per_trip_wall_seconds"], reference_seconds
        ),
        "execution_contract": {
            "route": "benchmarks.bank_revision_reproduction._solve_pure_arm",
            "active_set_steps": THREAD_CONFIGURATION_TRIPS,
            "persisted_carrier_reused": True,
            "solver_source_modified": False,
            "allocated_cpus": 16,
            "allocated_memory_gib": 64.0,
            "threaded_xla_flags": THREADED_XLA_FLAGS,
            "threaded_pool_settings": THREADED_POOL_SETTINGS,
            "assigned_worktree_nova_diff_stat": _git("diff", "--stat", "--", "nova"),
            "launch_pattern": (
                "one hardened three-hour sbatch allocation with sequential control "
                "and threaded srun steps, heartbeat, 30-second sstat sampling, "
                "flushed stage timings, and launch-then-harvest"
            ),
        },
        "evidence_inputs": {
            "baseline_raw": str(baseline_raw_path),
            "baseline_raw_sha256": _sha256(baseline_raw_path),
            "threaded_raw": str(threaded_raw_path),
            "threaded_raw_sha256": _sha256(threaded_raw_path),
            "sstat_samples": str(sstat_path),
            "sstat_samples_sha256": _sha256(sstat_path),
        },
        "figure": str(figure),
    }
    check_thread_configuration(receipt, require_figure=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _plot_thread_configuration(receipt, figure)
    check_thread_configuration(receipt, require_figure=True)
    return receipt


def check_thread_configuration(
    receipt: dict[str, Any], *, require_figure: bool = True
) -> None:
    baseline = receipt["configurations"]["baseline"]
    threaded = receipt["configurations"]["threaded"]
    if baseline["configuration"] != "baseline":
        raise RuntimeError("control capture is not identified as baseline")
    if threaded["configuration"] != "threaded":
        raise RuntimeError("threaded capture is not identified as threaded")
    for run in (baseline, threaded):
        if run["trip_count"] != THREAD_CONFIGURATION_TRIPS:
            raise RuntimeError(
                "thread configuration capture did not retain three trips"
            )
        if len(run["per_trip_residuals"]) != THREAD_CONFIGURATION_TRIPS:
            raise RuntimeError("thread configuration residual census is incomplete")
        if run["compile_seconds"] <= 0 or run["compilation"]["call_count"] < 1:
            raise RuntimeError("thread configuration compile time was not measured")
        if run["effective_xla_cpu_threads_from_sstat"]["sample_count"] < 2:
            raise RuntimeError("thread configuration lacks sstat process samples")
        if run["scheduler"]["limit_seconds"] != 3 * 3600:
            raise RuntimeError(
                "thread configuration does not record the three-hour limit"
            )
        if run["scheduler"]["state"] != "COMPLETED":
            raise RuntimeError("thread configuration scheduler state is not completed")
        if run["scheduler"]["exit_marker"] != 0:
            raise RuntimeError("thread configuration batch exit marker is nonzero")
        if run["execution_contract"]["solver_source_modified"]:
            raise RuntimeError("thread configuration reports a solver source change")
    if baseline["scheduler"]["job_id"] != threaded["scheduler"]["job_id"]:
        raise RuntimeError("thread configurations did not share one allocation")
    if baseline["scheduler"]["node"] != threaded["scheduler"]["node"]:
        raise RuntimeError("thread configurations did not run on one node")
    if baseline["allocation"]["xla_flags"]:
        raise RuntimeError("baseline unexpectedly engaged explicit XLA CPU flags")
    if threaded["allocation"]["xla_flags"] != THREADED_XLA_FLAGS:
        raise RuntimeError("threaded capture did not retain the exact XLA CPU flags")
    for variable, expected in THREADED_POOL_SETTINGS.items():
        if threaded["allocation"][variable.lower()] != expected:
            raise RuntimeError(f"threaded capture did not retain {variable}={expected}")
    if receipt["residual_agreement"]["maximum_relative_difference"] >= 1.0e-9:
        raise RuntimeError("threaded residuals changed beyond reduction-order noise")
    if not receipt["verdict"]["sentence"]:
        raise RuntimeError("thread configuration verdict is absent")
    if receipt["execution_contract"]["assigned_worktree_nova_diff_stat"]:
        raise RuntimeError("assigned worktree contains a change under nova")
    if require_figure:
        figure = Path(receipt["figure"])
        if not figure.exists() or figure.stat().st_size == 0:
            raise RuntimeError("thread configuration figure is absent")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--output", type=Path, required=True)
    thread_capture_parser = subparsers.add_parser("thread-capture")
    thread_capture_parser.add_argument("--output", type=Path, required=True)
    thread_capture_parser.add_argument(
        "--configuration", choices=("baseline", "threaded"), required=True
    )
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--raw", type=Path, required=True)
    compile_parser.add_argument("--sstat", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    compile_parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    compile_parser.add_argument("--job-elapsed", required=True)
    compile_parser.add_argument("--job-state", required=True)
    compile_parser.add_argument("--exit-marker", type=int, required=True)
    thread_compile_parser = subparsers.add_parser("thread-compile")
    thread_compile_parser.add_argument("--baseline-raw", type=Path, required=True)
    thread_compile_parser.add_argument("--threaded-raw", type=Path, required=True)
    thread_compile_parser.add_argument("--sstat", type=Path, required=True)
    thread_compile_parser.add_argument(
        "--output", type=Path, default=THREAD_CONFIGURATION_JSON
    )
    thread_compile_parser.add_argument(
        "--figure", type=Path, default=THREAD_CONFIGURATION_FIGURE
    )
    thread_compile_parser.add_argument("--job-elapsed", required=True)
    thread_compile_parser.add_argument("--job-state", required=True)
    thread_compile_parser.add_argument("--exit-marker", type=int, required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--receipt", type=Path, default=DEFAULT_JSON)
    thread_check_parser = subparsers.add_parser("thread-check")
    thread_check_parser.add_argument(
        "--receipt", type=Path, default=THREAD_CONFIGURATION_JSON
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "capture":
        capture(args.output)
    elif args.command == "thread-capture":
        capture(
            args.output,
            active_set_steps=THREAD_CONFIGURATION_TRIPS,
            configuration=args.configuration,
        )
    elif args.command == "compile":
        receipt = compile_receipt(
            args.raw,
            args.sstat,
            args.output,
            args.figure,
            job_elapsed=args.job_elapsed,
            job_state=args.job_state,
            exit_marker=args.exit_marker,
        )
        usage = receipt["runs"]["head"]["effective_xla_cpu_threads_from_sstat"]
        print(
            "CPU_DISCRIMINATOR_COMPILE PASS "
            f"classification={receipt['verdict']['classification']} "
            f"average_effective_threads={usage['average_effective_threads']:.3f} "
            f"dominant_share={receipt['dominant_cpu_cost']['share_of_solve_wall']:.6f}",
            flush=True,
        )
    elif args.command == "check":
        receipt = json.loads(args.receipt.read_text())
        check(receipt)
        print("CPU_DISCRIMINATOR_CHECK PASS", flush=True)
    elif args.command == "thread-compile":
        receipt = compile_thread_configuration(
            args.baseline_raw,
            args.threaded_raw,
            args.sstat,
            args.output,
            args.figure,
            job_elapsed=args.job_elapsed,
            job_state=args.job_state,
            exit_marker=args.exit_marker,
        )
        baseline = receipt["configurations"]["baseline"]
        threaded = receipt["configurations"]["threaded"]
        baseline_threads = baseline["effective_xla_cpu_threads_from_sstat"][
            "average_effective_threads"
        ]
        threaded_threads = threaded["effective_xla_cpu_threads_from_sstat"][
            "average_effective_threads"
        ]
        residual_difference = receipt["residual_agreement"][
            "maximum_relative_difference"
        ]
        print(
            "CPU_THREAD_CONFIGURATION_COMPILE PASS "
            f"baseline_threads={baseline_threads:.3f} "
            f"threaded_threads={threaded_threads:.3f} "
            f"residual_relative_difference={residual_difference:.3e}",
            flush=True,
        )
    else:
        receipt = json.loads(args.receipt.read_text())
        check_thread_configuration(receipt)
        print("CPU_THREAD_CONFIGURATION_CHECK PASS", flush=True)


if __name__ == "__main__":
    main()
