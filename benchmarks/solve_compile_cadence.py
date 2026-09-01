"""Measure fresh-process compile cadence for the settled forward solve.

The campaign deliberately uses fresh child interpreters.  The first child sees
an empty persistent cache, the second changes only input values, and the third
changes the traced program while preserving its input and output shapes.  A
compile profiler trace supplies the nested XLA phase/pass view; those durations
are explicitly non-additive because compiler phases nest and parallel codegen
workers overlap.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/solver-trip-orchestration/compile-cadence.json"
BASELINE = ROOT / "docs/figures/single-grid-solver-cutover/h200-solve-profile.json"
DUAL_BRANCH = ROOT / "docs/figures/dual-branch-selection/two-branch-batch-cost.json"
WIDTH = 1024
POLICY = {
    "newton_steps": 1,
    "gmres_iterations": 4,
    "warmup": 1,
    "relaxation": 0.5,
    "step_cap": 0.25,
}
HEARTBEAT_SECONDS = 30


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_ready(value: Any) -> Any:
    import jax

    leaves = jax.tree.leaves(value)
    if leaves:
        jax.block_until_ready(leaves[0])
    return value


def _cache_monitor() -> dict[str, float | int]:
    import jax.monitoring as monitoring

    events: dict[str, float | int] = {"hits": 0, "saved_seconds": 0.0}

    def hit(event: str, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/cache_hits":
            events["hits"] = int(events["hits"]) + 1

    def saved(event: str, duration_secs: float, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/compile_time_saved_sec":
            events["saved_seconds"] = float(events["saved_seconds"]) + duration_secs

    monitoring.register_event_listener(hit)
    monitoring.register_event_duration_secs_listener(saved)
    return events


def _scheduler() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    accepted_time = None
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
            accepted_time = fields.get("TimeLimit")
    return {
        "job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "accepted_time_limit": accepted_time,
    }


def _device() -> dict[str, Any]:
    import jax
    import jaxlib

    device = jax.devices()[0]
    return {
        "host": platform.node(),
        "platform": device.platform,
        "kind": device.device_kind,
        "id": int(device.id),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "x64_enabled": bool(jax.config.x64_enabled),
    }


def _measure_child(
    output: Path,
    cache_dir: Path,
    variant: str,
    trace_dir: Path | None,
) -> None:
    import jax
    import jax.numpy as jnp

    from benchmarks.diiid_batched_throughput import build_workload
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    jax.config.update("jax_log_compiles", True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    cache_events = _cache_monitor()
    device = _device()
    if device["platform"] != "gpu" or "H200" not in device["kind"]:
        raise RuntimeError(f"reserved H200 required, got {device}")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the betelgeuse partition is required")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the gpu_0003_grpA reservation is required")

    setup_started = time.perf_counter()
    profile, seed = build_workload()
    initial = jnp.repeat(seed[None, :], WIDTH, axis=0)
    base_current = np.asarray(profile.operator.external_current, dtype=float)
    current = jnp.asarray(np.repeat(base_current[None, :], WIDTH, axis=0))
    if variant == "data_change":
        current = current * jnp.asarray(1.0 + 1.0e-9, dtype=current.dtype)
    _tree_ready((initial, current))
    setup_seconds = time.perf_counter() - setup_started

    def solve(state: Any, conductor: Any) -> Any:
        if variant == "code_change":
            scale = jnp.nextafter(
                jnp.asarray(1.0, dtype=state.dtype),
                jnp.asarray(2.0, dtype=state.dtype),
            )
            state = state * scale
        return profile.solve_batch(
            state,
            route="newton_krylov",
            current=conductor,
            **POLICY,
        )

    jitted = jax.jit(solve)
    trace_started = time.perf_counter()
    lowered = jitted.lower(initial, current)
    trace_and_lower_seconds = time.perf_counter() - trace_started
    stablehlo = lowered.as_text(dialect="stablehlo")
    stablehlo_sha256 = hashlib.sha256(stablehlo.encode()).hexdigest()

    compile_started = time.perf_counter()
    if trace_dir is None:
        compiled = lowered.compile()
    else:
        trace_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
            compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    del compiled
    _write_json(
        output,
        {
            "variant": variant,
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "driver_sha256": _sha256(Path(__file__)),
            "width": WIDTH,
            "policy": POLICY,
            "setup_seconds": setup_seconds,
            "trace_and_lower_seconds": trace_and_lower_seconds,
            "xla_compile_or_cache_load_seconds": compile_seconds,
            "stablehlo_sha256": stablehlo_sha256,
            "cache": {
                "directory": str(cache_dir),
                "hits": cache_events["hits"],
                "compile_seconds_saved": cache_events["saved_seconds"],
                "entry_count": sum(path.is_file() for path in cache_dir.rglob("*")),
                "bytes": sum(
                    path.stat().st_size
                    for path in cache_dir.rglob("*")
                    if path.is_file()
                ),
            },
            "device": device,
            "scheduler": _scheduler(),
            "compile_trace_directory": None if trace_dir is None else str(trace_dir),
        },
    )


def _trace_events(trace_dir: Path) -> list[dict[str, Any]]:
    candidates = sorted(trace_dir.rglob("*.trace.json.gz"))
    if not candidates:
        raise RuntimeError(f"no XLA profiler trace found below {trace_dir}")
    with gzip.open(candidates[-1], "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    aggregated: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0, 0.0])
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name = str(event.get("name", ""))
        duration = float(event["dur"]) / 1.0e6
        row = aggregated[name]
        row[0] = float(row[0]) + duration
        row[1] = int(row[1]) + 1
        row[2] = max(float(row[2]), duration)
    selected = {
        "trace_to_jaxpr",
        "lower_jaxpr_to_module",
        "backend_compile_and_load",
        "PjRtCApiClient::CompileAndLoad",
        "PJRT_Client_Compile",
        "PjRtStreamExecutorClient::CompileInternal",
        "OptimizeHloModule",
        "autotuner",
        "RunPreSchedulingPasses",
        "ScheduleGpuModule",
        "RunPostSchedulingPipelines",
        "CompileModuleToLlvmIr",
        "CompileSingleModule",
        "PTX->CUBIN",
    }
    normalized: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0, 0.0])
    for name, values in aggregated.items():
        for phase in selected:
            if (
                name == phase
                or name.startswith(f"{phase} ")
                or name.startswith(f"{phase}(")
            ):
                row = normalized[phase]
                row[0] = float(row[0]) + float(values[0])
                row[1] = int(row[1]) + int(values[1])
                row[2] = max(float(row[2]), float(values[2]))
                break
    rows = [
        {
            "event": name,
            "aggregate_duration_seconds": values[0],
            "event_count": values[1],
            "maximum_single_event_seconds": values[2],
        }
        for name, values in normalized.items()
    ]
    rows.sort(key=lambda row: row["aggregate_duration_seconds"], reverse=True)
    return rows


def _top_xla_passes(trace_dir: Path) -> list[dict[str, Any]]:
    candidates = sorted(trace_dir.rglob("*.trace.json.gz"))
    with gzip.open(candidates[-1], "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    skip = (
        "$",
        "Pjit",
        "PJRT",
        "Compile",
        "Execute",
        "Device",
        "GpuExecutable",
        "XLA GPU module",
    )
    aggregated: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0, 0.0])
    for event in trace["traceEvents"]:
        name = str(event.get("name", ""))
        if (
            event.get("ph") != "X"
            or "dur" not in event
            or name.startswith("jit_")
            or any(token in name for token in skip)
        ):
            continue
        lower = name.lower()
        if not any(
            token in lower
            for token in (
                "pipeline",
                "simplif",
                "fusion",
                "schedule",
                "remat",
                "autotun",
                "legal",
                "dce",
                "cse",
                "buffer-assignment",
            )
        ):
            continue
        duration = float(event["dur"]) / 1.0e6
        row = aggregated[name]
        row[0] = float(row[0]) + duration
        row[1] = int(row[1]) + 1
        row[2] = max(float(row[2]), duration)
    rows = [
        {
            "pass_or_pipeline": name,
            "aggregate_duration_seconds": values[0],
            "event_count": values[1],
            "maximum_single_event_seconds": values[2],
        }
        for name, values in aggregated.items()
    ]
    rows.sort(key=lambda row: row["aggregate_duration_seconds"], reverse=True)
    return rows[:20]


def _dual_branch_contribution() -> dict[str, Any]:
    receipt = json.loads(DUAL_BRANCH.read_text(encoding="utf-8"))
    rows = []
    for width_row in receipt["measurements"]:
        for cohort in width_row["seed_cohorts"]:
            single = cohort["single_pinned_branch"]["compile"][
                "lower_and_compile_seconds"
            ]
            dual = cohort["two_branch_portfolio"]["compile"][
                "lower_and_compile_seconds"
            ]
            rows.append(
                {
                    "batch_width": width_row["batch_width"],
                    "seed_cohort": cohort["seed_cohort"],
                    "single_pinned_compile_seconds": single,
                    "dual_branch_compile_seconds": dual,
                    "dual_over_single_ratio": dual / single,
                }
            )
    ratios = np.asarray([row["dual_over_single_ratio"] for row in rows])
    return {
        "source_receipt": str(DUAL_BRANCH.relative_to(ROOT)),
        "source_receipt_sha256": _sha256(DUAL_BRANCH),
        "source_job": receipt["environment"]["slurm_job_id"],
        "device": receipt["environment"]["device_kind"],
        "rows": rows,
        "ratio_summary": {
            "minimum": float(ratios.min()),
            "median": float(np.median(ratios)),
            "maximum": float(ratios.max()),
        },
        "verdict": (
            "stacking both pinned branches costs about 1.6x, not 2x, at the "
            "measured widths 4 and 16; it does not explain the width-1024 "
            "settled-loop compile by itself"
        ),
    }


def _launch_cadence() -> dict[str, Any]:
    return {
        "persistent_cache_recipe_census": {
            "forward_solve_launch_recipes_enabling_cache": [],
            "repository_matches": [
                "nova/biot/tiledassembly.py",
                "benchmarks/tiled_backend.py",
                "benchmarks/forward_solve_throughput.py",
                "benchmarks/observable_batch_acceptance.py",
            ],
            "interpretation": (
                "persistent caching exists for tiled Biot and explicit benchmark "
                "probes, but no forward-solve bank, small-lane, or pytest launch "
                "recipe enables jax_compilation_cache_dir"
            ),
        },
        "payments": [
            {
                "lane": "bank regeneration",
                "process_cadence": (
                    "one fresh Python process per benchmark invocation; a driver "
                    "that regenerates multiple frames reuses only executables whose "
                    "program shape remains identical inside that invocation"
                ),
                "payment": (
                    "one compile per distinct batch-width/program-shape bucket per "
                    "invocation; a restarted bank pays again"
                ),
                "persistent_cache_enabled": False,
            },
            {
                "lane": "small and focused benchmark lanes",
                "process_cadence": "each CLI invocation starts a fresh Python process",
                "payment": (
                    "once per distinct shape in each invocation, even when a prior "
                    "small lane compiled the same program"
                ),
                "persistent_cache_enabled": False,
            },
            {
                "lane": "pytest",
                "process_cadence": (
                    "one fresh process per pytest command; identical keys can share "
                    "the in-memory executable only within that test session"
                ),
                "payment": (
                    "focused and full-suite commands repay independently, as do "
                    "separately scheduled test shards"
                ),
                "persistent_cache_enabled": False,
            },
        ],
    }


def _run_child(command: list[str], log: Path, label: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", JAX_LOG_COMPILES="1"),
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            print(f"HEARTBEAT stage={label} elapsed_seconds={elapsed:.0f}", flush=True)
            time.sleep(HEARTBEAT_SECONDS)
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit {process.returncode}; see {log}")


def _jax_log_compile(log: Path) -> dict[str, Any]:
    """Return the named main-program timing rather than setup compilations."""
    text = log.read_text(encoding="utf-8")

    def last_seconds(pattern: str) -> float:
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"missing jit(solve) timing in {log}: {pattern}")
        return float(matches[-1])

    cache_keys = re.findall(
        r"Persistent compilation cache hit for 'jit_solve' with key '([^']+)'", text
    )
    return {
        "trace_seconds": last_seconds(
            r"Finished tracing solve for jit in ([0-9.]+) sec"
        ),
        "jaxpr_to_mlir_seconds": last_seconds(
            r"Finished jaxpr to MLIR module conversion jit\(solve\) in ([0-9.]+) sec"
        ),
        "xla_compile_or_cache_load_seconds": last_seconds(
            r"Finished XLA compilation of jit\(solve\) in ([0-9.]+) sec"
        ),
        "main_program_cache_hit": bool(cache_keys),
        "main_program_cache_key": cache_keys[-1] if cache_keys else None,
    }


def _campaign(output: Path, work: Path, *, reuse_measurements: bool = False) -> None:
    cache_dir = work / "persistent-cache"
    raw_dir = work / "raw"
    stages = (
        ("cold", True),
        ("data_change", False),
        ("code_change", False),
    )
    records = []
    if reuse_measurements:
        records = [
            json.loads((raw_dir / f"{variant}.json").read_text(encoding="utf-8"))
            for variant, _capture_trace in stages
        ]
    else:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for variant, capture_trace in stages:
            raw = raw_dir / f"{variant}.json"
            trace = work / "compile-trace" if capture_trace else None
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "measure",
                "--output",
                str(raw),
                "--cache-dir",
                str(cache_dir),
                "--variant",
                variant,
            ]
            if trace is not None:
                command += ["--trace-dir", str(trace)]
            _run_child(command, work / f"{variant}.log", variant)
            records.append(json.loads(raw.read_text(encoding="utf-8")))

    by_variant = {row["variant"]: row for row in records}
    cold = by_variant["cold"]
    data = by_variant["data_change"]
    code = by_variant["code_change"]
    for record in records:
        record["jax_log_compile"] = _jax_log_compile(work / f"{record['variant']}.log")
    cold_log = cold["jax_log_compile"]
    data_log = data["jax_log_compile"]
    code_log = code["jax_log_compile"]
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    banked_compile = baseline["compile_amortization"][
        "width_1024_warmup_1_gmres_4_compile_seconds"
    ]
    trace_dir = Path(cold["compile_trace_directory"])
    receipt = {
        "schema": "nova.solve_compile_cadence",
        "measurement_state": "complete",
        "source_revision": cold["source_revision"],
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "scheduler": cold["scheduler"],
        "device": cold["device"],
        "configuration": {"width": WIDTH, "policy": POLICY},
        "measurements": records,
        "headline": {
            "banked_compile_and_first_execute_seconds": banked_compile,
            "cold_trace_seconds": cold_log["trace_seconds"],
            "cold_jaxpr_to_mlir_seconds": cold_log["jaxpr_to_mlir_seconds"],
            "cold_xla_compile_seconds": cold_log["xla_compile_or_cache_load_seconds"],
            "cold_total_compile_seconds": cold_log["trace_seconds"]
            + cold_log["jaxpr_to_mlir_seconds"]
            + cold_log["xla_compile_or_cache_load_seconds"],
            "profiler_export_overhead_seconds": cold[
                "xla_compile_or_cache_load_seconds"
            ]
            - cold_log["xla_compile_or_cache_load_seconds"],
            "cache_warm_restart_trace_seconds": data_log["trace_seconds"],
            "cache_warm_restart_jaxpr_to_mlir_seconds": data_log[
                "jaxpr_to_mlir_seconds"
            ],
            "cache_warm_restart_xla_cache_load_seconds": data_log[
                "xla_compile_or_cache_load_seconds"
            ],
            "cache_warm_restart_total_compile_seconds": data_log["trace_seconds"]
            + data_log["jaxpr_to_mlir_seconds"]
            + data_log["xla_compile_or_cache_load_seconds"],
            "cache_warm_total_speedup": (
                cold_log["trace_seconds"]
                + cold_log["jaxpr_to_mlir_seconds"]
                + cold_log["xla_compile_or_cache_load_seconds"]
            )
            / (
                data_log["trace_seconds"]
                + data_log["jaxpr_to_mlir_seconds"]
                + data_log["xla_compile_or_cache_load_seconds"]
            ),
        },
        "cache_key_stability": {
            "pure_data_change": {
                "stablehlo_identical": cold["stablehlo_sha256"]
                == data["stablehlo_sha256"],
                "cache_hits": data["cache"]["hits"],
                "main_program_cache_hit": data_log["main_program_cache_hit"],
                "main_program_cache_key": data_log["main_program_cache_key"],
                "compile_seconds_saved": data["cache"]["compile_seconds_saved"],
                "verdict": (
                    "stable key: changed values with unchanged shapes reuse the "
                    "executable"
                ),
            },
            "code_change": {
                "stablehlo_identical": cold["stablehlo_sha256"]
                == code["stablehlo_sha256"],
                "cache_hits": code["cache"]["hits"],
                "main_program_cache_hit": code_log["main_program_cache_hit"],
                "main_program_cache_key": code_log["main_program_cache_key"],
                "compile_seconds_saved": code["cache"]["compile_seconds_saved"],
                "verdict": "changed traced StableHLO requires a distinct executable",
            },
            "version_scope": (
                "the key also includes backend/compiler options, device topology, "
                "jaxlib/XLA build and serialized computation; a jaxlib/XLA change "
                "must be treated as invalidating even if Python source is unchanged"
            ),
        },
        "compilation_time_breakdown": {
            "source": (
                "JAX_LOG_COMPILES plus JAX/XLA profiler trace around lowered.compile()"
            ),
            "interpretation": (
                "phase and pass events are nested and GPU codegen may run in "
                "parallel; aggregate event durations are diagnostic workload, not "
                "additive shares of wall clock"
            ),
            "phase_events": _trace_events(trace_dir),
            "top_pass_and_pipeline_events": _top_xla_passes(trace_dir),
            "raw_log": str(work / "cold.log"),
            "raw_trace": str(trace_dir),
        },
        "dual_branch_stacking": _dual_branch_contribution(),
        "lane_cadence": _launch_cadence(),
        "recommendations": [
            {
                "number": 1,
                "recommendation": (
                    "enable one versioned persistent compilation-cache directory "
                    "for bank regeneration and scheduled test lanes"
                ),
                "basis": (
                    "the measured warm-restart total replaces the measured cold "
                    "trace/lower plus XLA compile on every stable key"
                ),
            },
            {
                "number": 2,
                "recommendation": (
                    "bucket work by batch width and keep each bucket resident; do "
                    "not mix widths in fresh workers"
                ),
                "basis": (
                    "each width is a distinct program shape and therefore a "
                    "distinct cache entry even after persistent caching is enabled"
                ),
            },
            {
                "number": 3,
                "recommendation": (
                    "prioritize splitting or shrinking the settled-loop executable "
                    "only after persistent caching; dual-branch stacking alone is "
                    "not a twofold compile multiplier"
                ),
                "basis": (
                    "the independent H200 portfolio receipt measures a median "
                    f"{_dual_branch_contribution()['ratio_summary']['median']:.3f}x "
                    "dual-over-single compile ratio"
                ),
            },
        ],
        "evidence_inputs": {
            "banked_baseline": {
                "path": str(BASELINE.relative_to(ROOT)),
                "sha256": _sha256(BASELINE),
            },
            "raw_directory": str(raw_dir),
            "merged_stderr_logs": [str(work / f"{name}.log") for name, _ in stages],
        },
    }
    _write_json(output, receipt)
    print(f"RECEIPT_WRITTEN={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--cache-dir", type=Path, required=True)
    measure.add_argument(
        "--variant", choices=("cold", "data_change", "code_change"), required=True
    )
    measure.add_argument("--trace-dir", type=Path)
    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    campaign.add_argument("--work", type=Path, required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    assemble.add_argument("--work", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "measure":
        _measure_child(
            arguments.output,
            arguments.cache_dir,
            arguments.variant,
            arguments.trace_dir,
        )
    elif arguments.command == "campaign":
        _campaign(arguments.output, arguments.work)
    else:
        _campaign(arguments.output, arguments.work, reuse_measurements=True)


if __name__ == "__main__":
    main()
