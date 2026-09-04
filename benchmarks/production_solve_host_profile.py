"""Profile host participation in one warm production equilibrium solve.

The benchmark reproduces the banked MAST 22086/43 pure solve, profiles one
cache-warm invocation with cProfile, and records the logical device-to-host
payload of both the ordinary summary and the complete returned receipt.  A
separate mode supplies the same warm solve to an external native py-spy record.
"""

from __future__ import annotations

import argparse
import cProfile
from collections import Counter
from datetime import UTC, datetime
import hashlib
import html
import inspect
import json
import math
import os
from pathlib import Path
import platform
import pstats
import re
import socket
import subprocess
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import efit_forward_parity_slice as parity
from benchmarks.receipt_raster_check import _profile_and_seed
from nova.equilibrium import fixed_point
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/host-profile"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "profile.json"
DEFAULT_FIGURE = OUTPUT_ROOT / "host-profile.png"
DEFAULT_STATS = OUTPUT_ROOT / "production-solve.pstats"
DEFAULT_STATS_TEXT = OUTPUT_ROOT / "production-solve-cprofile.txt"
DEFAULT_MEASUREMENT = OUTPUT_ROOT / "cprofile-measurement.json"
DEFAULT_FLAME = OUTPUT_ROOT / "production-solve-native.svg"
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/"
    "production-solve-host-profile.md"
)
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile")
REFERENCE_SHOT = 22086
REFERENCE_SLICE = 43
ACTIVE_SET_TRIP_BUDGET = 24
EXPECTED_ACTIVE_SET_TRIPS = 7
TRACE_COMMIT = "a801e9a4834b06b44ee87a1c5934f97f783acb8e"
TRACE_RECEIPT_PATH = (
    "docs/figures/millisecond-converged-solve/trip-quantum/"
    "hessenberg-trace/profile.json"
)
TRACE_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/"
    "condition-discriminator-trace.md"
)
SUMMARY_FIELDS = (
    "active_set_iterations",
    "attempted_newton_promotions",
    "accepted_newton_promotions",
    "termination_reason",
    "residual",
    "converged",
    "maximum_projected_krylov_condition",
)


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_kinds": [getattr(device, "device_kind", None) for device in devices],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _scheduler() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "temporary_directory": os.environ.get("TMPDIR"),
    }


def _require_measurement_host() -> None:
    kinds = [str(getattr(device, "device_kind", "")) for device in jax.devices()]
    if jax.default_backend() != "gpu" or not any("H200" in kind for kind in kinds):
        raise RuntimeError(
            "host profiling requires the reserved H200; "
            f"backend={jax.default_backend()} devices={kinds}"
        )


def _source_anchor(function: Callable, needle: str) -> str:
    path = Path(inspect.getsourcefile(function) or "")
    lines, first = inspect.getsourcelines(function)
    offset = next((index for index, line in enumerate(lines) if needle in line), 0)
    try:
        display = path.relative_to(ROOT)
    except ValueError:
        display = path
    return f"{display}:{first + offset}"


def _production_program(profile: Any, target_current: float):
    def production(initial_flux):
        return profile.solve_branch(
            initial_flux,
            TopologyClass.DIVERTED,
            target_current=target_current,
            route="newton_krylov",
            tolerance=parity.FIXED_POINT_CRITERION,
            warmup=parity.WARMUP_SWEEPS,
            newton_steps=parity.NEWTON_STEPS,
            gmres_iterations=parity.GMRES_ITERATIONS,
            relaxation=parity.RELAXATION,
            step_cap=parity.STEP_CAP,
            active_set_steps=ACTIVE_SET_TRIP_BUDGET,
        )

    return production


def _ready(value: Any) -> Any:
    jax.block_until_ready(value)
    return value


def _solve_summary(result: Any) -> dict[str, Any]:
    history = result.equilibrium.fixed_point
    reason_value = int(np.asarray(history.termination_reason))
    return {
        "active_set_trips": int(np.asarray(history.active_set_iterations)),
        "attempted_newton_promotions": int(
            np.asarray(history.attempted_newton_promotions)
        ),
        "accepted_newton_promotions": int(
            np.asarray(history.accepted_newton_promotions)
        ),
        "termination_reason": fixed_point.FixedPointTerminationReason(
            reason_value
        ).name.lower(),
        "terminal_residual": float(np.asarray(history.residual)),
        "converged": bool(np.asarray(history.converged)),
        "maximum_projected_krylov_condition": float(
            np.asarray(history.maximum_projected_krylov_condition)
        ),
    }


def _stat_rows(stats_path: Path) -> list[dict[str, Any]]:
    stats = pstats.Stats(str(stats_path))
    rows = []
    for (filename, line, name), values in stats.stats.items():
        primitive_calls, calls, self_s, cumulative_s, _callers = values
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": name,
                "primitive_calls": primitive_calls,
                "calls": calls,
                "self_s": self_s,
                "cumulative_s": cumulative_s,
            }
        )
    return sorted(rows, key=lambda row: row["cumulative_s"], reverse=True)


def _write_stats_text(stats_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        pstats.Stats(str(stats_path), stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats()


def _leaf_bytes(leaf: Any) -> int:
    shape = tuple(getattr(leaf, "shape", ()))
    dtype = np.dtype(getattr(leaf, "dtype", np.asarray(leaf).dtype))
    return int(np.prod(shape, dtype=np.int64) if shape else 1) * dtype.itemsize


def _payload_category(path: str) -> str:
    if "fixed_point" in path and (".state" in path or path.endswith("[0]")):
        return "solver_state"
    if "fixed_point" in path:
        return "solver_history"
    if ".flux" in path:
        return "equilibrium_flux"
    return "branch_receipt"


def _transfer_census(result: Any, trips: int) -> dict[str, Any]:
    leaves_with_paths, _treedef = jax.tree_util.tree_flatten_with_path(result)
    rows = []
    categories: Counter[str] = Counter()
    for path, leaf in leaves_with_paths:
        if not hasattr(leaf, "dtype"):
            continue
        display = jax.tree_util.keystr(path)
        size = _leaf_bytes(leaf)
        category = _payload_category(display)
        rows.append(
            {
                "path": display,
                "shape": list(getattr(leaf, "shape", ())),
                "dtype": str(getattr(leaf, "dtype", "")),
                "bytes": size,
                "category": category,
            }
        )
        categories[category] += size
    started = time.perf_counter()
    jax.device_get(result)
    transfer_wall_s = time.perf_counter() - started
    summary = result.equilibrium.fixed_point
    summary_rows = []
    for name in SUMMARY_FIELDS:
        leaf = getattr(summary, name)
        summary_rows.append(
            {
                "field": name,
                "bytes": _leaf_bytes(leaf),
                "shape": list(getattr(leaf, "shape", ())),
                "dtype": str(getattr(leaf, "dtype", "")),
            }
        )
    total_bytes = sum(row["bytes"] for row in rows)
    summary_bytes = sum(row["bytes"] for row in summary_rows)
    return {
        "full_receipt": {
            "logical_buffer_count_per_solve": len(rows),
            "logical_bytes_per_solve": total_bytes,
            "logical_buffer_count_per_trip": len(rows) / trips,
            "logical_bytes_per_trip": total_bytes / trips,
            "bulk_device_get_wall_s": transfer_wall_s,
            "categories_bytes_per_solve": dict(categories),
            "leaves": rows,
        },
        "ordinary_summary": {
            "logical_buffer_count_per_solve": len(summary_rows),
            "logical_bytes_per_solve": summary_bytes,
            "logical_buffer_count_per_trip": len(summary_rows) / trips,
            "logical_bytes_per_trip": summary_bytes / trips,
            "fields": summary_rows,
        },
        "counting_contract": (
            "counts are logical device array buffers whose values cross to host; "
            "a backend may coalesce copies, so logical buffers are not claimed as "
            "independent DMA transactions"
        ),
    }


def _load_trace_receipt() -> tuple[dict[str, Any], str]:
    payload = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{TRACE_COMMIT}:{TRACE_RECEIPT_PATH}"],
        check=True,
        capture_output=True,
    ).stdout
    return json.loads(payload), _sha256_bytes(payload)


def _trace_evidence() -> dict[str, Any]:
    receipt, digest = _load_trace_receipt()
    exact = next(arm for arm in receipt["arms"] if arm["name"] == "exact_jacobi")
    trace = exact["device_trace"]
    host = trace["host_launch_api"]
    return {
        "receipt_commit": TRACE_COMMIT,
        "receipt_path": TRACE_RECEIPT_PATH,
        "receipt_sha256": digest,
        "report_path": str(TRACE_REPORT),
        "report_sha256": _sha256(TRACE_REPORT),
        "job_id": receipt["scheduler"]["job_id"],
        "synchronized_wall_s_per_trip": exact["trace_wall_s_per_observed_trip"],
        "summed_kernel_s_per_trip": trace["summed_kernel_s_per_observed_trip"],
        "host_launch_count_per_trip": host["host_correlation_count"],
        "host_gap_sum_s_per_trip": host["between_launch_gap_sum_s_per_observed_trip"],
        "activity_sum_s_per_trip": (
            trace["summed_kernel_s_per_observed_trip"]
            + host["between_launch_gap_sum_s_per_observed_trip"]
        ),
        "activity_sum_is_nonadditive": True,
    }


def _configure(cache_root: Path):
    configure_dtypes()
    _require_measurement_host()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = configure_persistent_compilation_cache(
        cache_root, minimum_compile_seconds=0.0
    )
    case, profile, target_current, carrier, policy = _profile_and_seed()
    reference = case["reference"]
    identity = (int(reference["shot"]), int(reference["slice_index"]))
    if identity != (REFERENCE_SHOT, REFERENCE_SLICE):
        raise RuntimeError(f"unexpected production identity {identity}")
    state = jnp.asarray(case["state"])
    function = jax.jit(_production_program(profile, target_current))
    started = time.perf_counter()
    executable = function.lower(state).compile()
    compile_wall_s = time.perf_counter() - started
    return case, carrier, policy, state, function, executable, cache, compile_wall_s


def measure(
    measurement_path: Path,
    stats_path: Path,
    stats_text_path: Path,
    cache_root: Path,
) -> dict[str, Any]:
    (
        case,
        carrier,
        policy,
        state,
        function,
        executable,
        cache,
        compile_wall_s,
    ) = _configure(cache_root)
    warm_started = time.perf_counter()
    warm_result = _ready(executable(state))
    warm_wall_s = time.perf_counter() - warm_started
    warm_summary = _solve_summary(warm_result)
    if warm_summary["active_set_trips"] != EXPECTED_ACTIVE_SET_TRIPS:
        raise RuntimeError(
            "production trip count changed from "
            f"{EXPECTED_ACTIVE_SET_TRIPS} to {warm_summary['active_set_trips']}"
        )

    profiler = cProfile.Profile()
    _ready(warm_result)
    profiler.enable()
    profiled_started = time.perf_counter()
    result = function(state)
    dispatched = time.perf_counter()
    _ready(result)
    synchronized = time.perf_counter()
    summary = _solve_summary(result)
    summarized = time.perf_counter()
    profiler.disable()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(stats_path))
    _write_stats_text(stats_path, stats_text_path)
    rows = _stat_rows(stats_path)
    trips = summary["active_set_trips"]

    transfer_started = time.perf_counter()
    transfer_result = function(state)
    _ready(transfer_result)
    transfer_solve_wall_s = time.perf_counter() - transfer_started
    transfers = _transfer_census(transfer_result, trips)

    measurement = {
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": _revision(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "runtime": _runtime(),
        "scheduler": _scheduler(),
        "production_identity": {
            "reference": case["reference"],
            "arm": "pure",
            "carrier": carrier,
            "field_policy": policy,
            "active_set_trip_budget": ACTIVE_SET_TRIP_BUDGET,
            "observed_active_set_trips": trips,
            "route": "ForwardProfile.solve_branch newton_krylov",
            "outer_boundary": "one jax.jit around the complete solve_branch call",
        },
        "compile": {
            "lower_and_compile_wall_s": compile_wall_s,
            "warm_execute_wall_s": warm_wall_s,
            "persistent_cache": cache.receipt(),
        },
        "profiled_solve": {
            "summary": summary,
            "wall_s": summarized - profiled_started,
            "wall_s_per_trip": (summarized - profiled_started) / trips,
            "dispatch_s": dispatched - profiled_started,
            "dispatch_s_per_trip": (dispatched - profiled_started) / trips,
            "compiled_wait_s": synchronized - dispatched,
            "compiled_wait_s_per_trip": (synchronized - dispatched) / trips,
            "summary_transfer_s": summarized - synchronized,
            "summary_transfer_s_per_trip": (summarized - synchronized) / trips,
            "compiled_wait_share": (synchronized - dispatched)
            / (summarized - profiled_started),
            "python_boundary_count_per_solve": 1,
            "python_boundary_count_per_trip": 1 / trips,
        },
        "transfer_census_solve_wall_s": transfer_solve_wall_s,
        "transfers": transfers,
        "cprofile": {
            "stats_path": str(stats_path),
            "stats_sha256": _sha256(stats_path),
            "text_path": str(stats_text_path),
            "text_sha256": _sha256(stats_text_path),
            "top_by_cumulative": rows[:80],
            "top_python_source_by_cumulative": [
                row
                for row in rows
                if row["file"].endswith(".py") and row["file"] != "~"
            ][:40],
        },
        "trace_comparison": _trace_evidence(),
        "source_anchors": {
            "compiled_outer_active_set_loop": _source_anchor(
                fixed_point._active_set_newton_krylov,
                "outer = jax.lax.fori_loop",
            ),
            "compiled_trip_conditional": _source_anchor(
                fixed_point._active_set_newton_krylov,
                "return jax.lax.cond",
            ),
        },
    }
    _write_json(measurement_path, measurement)
    print(f"MEASUREMENT_WRITTEN={measurement_path}", flush=True)
    return measurement


def pyspy_target(cache_root: Path) -> None:
    *_metadata, state, function, executable, _cache, _compile_wall = _configure(
        cache_root
    )
    warm = _ready(executable(state))
    warm_summary = _solve_summary(warm)
    print(
        "PYSPY_WARM_DONE "
        f"trips={warm_summary['active_set_trips']} "
        f"wall_boundary=complete_solve",
        flush=True,
    )
    started = time.perf_counter()
    result = _ready(function(state))
    summary = _solve_summary(result)
    print(
        "PYSPY_PROFILE_DONE "
        f"wall_s={time.perf_counter() - started:.9f} "
        f"trips={summary['active_set_trips']}",
        flush=True,
    )


_FLAME_TITLE = re.compile(
    r"<title>(?P<name>.*?) \((?P<samples>[0-9,]+) samples, "
    r"(?P<share>[0-9.]+)%\)</title>"
)


def _flame_summary(path: Path, exit_status: int) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {
            "status": "unavailable",
            "exit_status": exit_status,
            "path": str(path),
        }
    text = path.read_text(encoding="utf-8")
    rows = []
    for match in _FLAME_TITLE.finditer(text):
        rows.append(
            {
                "frame": html.unescape(match.group("name")),
                "samples": int(match.group("samples").replace(",", "")),
                "share": float(match.group("share")) / 100.0,
            }
        )
    rows.sort(key=lambda row: row["samples"], reverse=True)
    execution_tokens = (
        "execute",
        "pjrt",
        "streamexecutor",
        "blockhostuntilready",
        "synchronize",
        "cudaevent",
    )
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        lowered = row["frame"].lower()
        if not any(token in lowered for token in execution_tokens):
            continue
        previous = selected.get(row["frame"])
        if previous is None or row["samples"] > previous["samples"]:
            selected[row["frame"]] = row
    execution_rows = sorted(
        selected.values(), key=lambda row: row["samples"], reverse=True
    )
    return {
        "status": "captured" if exit_status == 0 else "captured_with_errors",
        "exit_status": exit_status,
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "frame_rows": len(rows),
        "top_cumulative_frames": rows[:60],
        "top_execution_frames": execution_rows[:40],
        "scope": (
            "one process containing executable restoration, one synchronized "
            "warm solve, and one synchronized profiled solve; native frames enabled"
        ),
    }


def _write_figure(receipt: dict[str, Any], path: Path) -> None:
    profile = receipt["profiled_solve"]
    trips = receipt["production_identity"]["observed_active_set_trips"]
    transfer = receipt["transfers"]["full_receipt"]
    categories = transfer["categories_bytes_per_solve"]
    trace = receipt["trace_comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    parts = [
        ("compiled runtime", profile["compiled_runtime_s_per_trip"]),
        ("Python wrapper", profile["python_wrapper_s_per_trip"]),
        ("summary transfer", profile["summary_transfer_s_per_trip"]),
    ]
    bottom = 0.0
    colors = ("tab:blue", "tab:orange", "tab:green")
    for (label, value), color in zip(parts, colors, strict=True):
        axes[0].bar(["production"], [value], bottom=bottom, label=label, color=color)
        bottom += value
    axes[0].axhline(
        trace["synchronized_wall_s_per_trip"],
        color="firebrick",
        linestyle="--",
        label="one-trip trace wall",
    )
    axes[0].set_ylabel("Wall seconds per observed trip")
    axes[0].set_title("Warm production boundary")
    axes[0].legend(fontsize=8)

    labels = list(categories)
    values = [categories[label] / trips for label in labels]
    axes[1].barh(
        [label.replace("_", " ") for label in labels],
        values,
        color="cadetblue",
    )
    axes[1].set_xlabel("Logical bytes per observed trip")
    axes[1].set_title("Complete receipt device-to-host payload")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _display_function(row: dict[str, Any]) -> str:
    filename = row["file"]
    try:
        filename = str(Path(filename).relative_to(ROOT))
    except ValueError, OSError:
        filename = Path(filename).name if filename != "~" else "native"
    return f"{filename}:{row['line']} {row['function']}"


def _mechanism(receipt: dict[str, Any]) -> dict[str, Any]:
    profile = receipt["profiled_solve"]
    trace = receipt["trace_comparison"]
    host_ceiling = (
        profile["python_wrapper_s_per_trip"] + profile["summary_transfer_s_per_trip"]
    )
    executable_gap = max(
        profile["compiled_runtime_s_per_trip"] - trace["synchronized_wall_s_per_trip"],
        0.0,
    )
    return {
        "finding": (
            "The warm production wall is not in a Python active-set driver. The "
            "complete solve has one Python dispatch boundary, while all observed "
            "active-set trips execute inside the compiled outer loop. The difference "
            "from the one-trip trace is therefore subsequent-trip work inside the "
            "full-solve executable."
        ),
        "source": receipt["source_anchors"]["compiled_outer_active_set_loop"],
        "compiled_runtime_share": profile["compiled_runtime_share"],
        "host_repair_ceiling_s_per_trip": host_ceiling,
        "host_repair_ceiling_share": host_ceiling / profile["wall_s_per_trip"],
        "subsequent_trip_executable_gap_s_per_trip": executable_gap,
        "repair": (
            "Trace and restructure the compiled outer-body program used for trips "
            "after the initial solve, or drive a common compiled trip executable "
            "with an explicit state/history carry. Receipt assembly and host "
            "transfers are not material repair targets."
        ),
        "estimated_saving_s_per_trip": executable_gap,
        "estimate_contract": (
            "upper-bound opportunity if later trips reach the synchronized initial-"
            "trip trace wall; it is not a measured speedup of an implemented repair"
        ),
    }


def _write_report(receipt: dict[str, Any], path: Path) -> None:
    profile = receipt["profiled_solve"]
    trips = receipt["production_identity"]["observed_active_set_trips"]
    transfer = receipt["transfers"]
    trace = receipt["trace_comparison"]
    mechanism = receipt["mechanism"]
    top_rows = receipt["cprofile"]["top_by_cumulative"][:15]
    flame = receipt["pyspy"]
    lines = [
        "# Production solve host profile",
        "",
        (
            f"SLURM job `{receipt['scheduler']['job_id']}` profiled one warm "
            "MAST 22086/43 pure production solve at "
            f"`{receipt['measurement_revision']}` "
            f"on `{receipt['runtime']['host']}`. The solve took "
            f"**{profile['wall_s']:.6f} s** over **{trips} active-set trips**, or "
            f"**{profile['wall_s_per_trip']:.6f} s/trip**."
        ),
        "",
        "## Boundary result",
        "",
        "| boundary | seconds / solve | seconds / trip | share of profiled wall |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("compiled/PJRT runtime", "compiled_runtime_s"),
        ("Python wrapper around the compiled call", "python_wrapper_s"),
        ("ordinary summary transfers", "summary_transfer_s"),
    ):
        value = profile[key]
        lines.append(
            f"| {label} | {value:.6f} | {value / trips:.6f} | "
            f"{100.0 * value / profile['wall_s']:.4f}% |"
        )
    lines.extend(
        [
            "",
            (
                "There is **one Python dispatch boundary for the complete solve**, "
                f"or {profile['python_boundary_count_per_trip']:.6f} boundaries per "
                "observed trip. The active-set loop is `jax.lax.fori_loop` at "
                f"`{receipt['source_anchors']['compiled_outer_active_set_loop']}`; "
                "the conditional that skips inactive trips is compiled as well at "
                f"`{receipt['source_anchors']['compiled_trip_conditional']}`."
            ),
            "",
            "## cProfile cumulative ranking",
            "",
            "| rank | function | calls | cumulative s | self s | cumulative s / trip |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_rows, 1):
        lines.append(
            f"| {rank} | `{_display_function(row)}` | {row['calls']} | "
            f"{row['cumulative_s']:.6f} | {row['self_s']:.6f} | "
            f"{row['cumulative_s'] / trips:.6f} |"
        )
    lines.extend(
        [
            "",
            "Cumulative rows are inclusive and therefore not additive. The explicit "
            "boundary timers above are the additive separation.",
            "",
            "## Device-to-host transfer census",
            "",
            "| payload | logical buffers / solve | bytes / solve | "
            "buffers / trip | bytes / trip |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("ordinary scalar summary", "ordinary_summary"),
        ("complete returned receipt", "full_receipt"),
    ):
        row = transfer[key]
        lines.append(
            f"| {label} | {row['logical_buffer_count_per_solve']} | "
            f"{row['logical_bytes_per_solve']} | "
            f"{row['logical_buffer_count_per_trip']:.3f} | "
            f"{row['logical_bytes_per_trip']:.3f} |"
        )
    lines.extend(
        [
            "",
            transfer["counting_contract"] + ".",
            "",
            "## Native py-spy evidence",
            "",
        ]
    )
    if flame["status"].startswith("captured"):
        lines.append(
            f"Native py-spy flame data was captured at `{flame['path']}` "
            f"({flame['size_bytes']} bytes, SHA-256 `{flame['sha256']}`, "
            f"{flame['sample_count']:,} samples, {flame['sample_errors']} unwind "
            "errors). The record "
            "contains executable restoration, one synchronized warm solve, and one "
            "synchronized measured solve; it is corroborative rather than an "
            "additive timer."
        )
        if flame["sampling_lag_warning"]:
            lines.append(
                "The sampler reported that native unwinding fell behind its 100 Hz "
                "target; frame shares are retained as qualified corroboration only."
            )
        lines.extend(
            [
                "",
                "| rank | cumulative frame | samples | share |",
                "|---:|---|---:|---:|",
            ]
        )
        for rank, row in enumerate(flame["top_execution_frames"][:15], 1):
            frame = row["frame"].replace("|", "\\|")
            lines.append(
                f"| {rank} | `{frame}` | {row['samples']} | "
                f"{100.0 * row['share']:.2f}% |"
            )
    else:
        lines.append(
            f"Native py-spy capture was unavailable (exit {flame['exit_status']}); "
            "the synchronized cProfile boundary remains the primary evidence."
        )
    lines.extend(
        [
            "",
            "## Comparison with the compiled one-trip trace",
            "",
            (
                f"Job `{trace['job_id']}` measured "
                f"**{trace['synchronized_wall_s_per_trip']:.6f} "
                "s** synchronized wall for the exact one-trip program, with "
                f"**{trace['summed_kernel_s_per_trip']:.6f} s** summed kernels, "
                f"**{trace['host_gap_sum_s_per_trip']:.6f} s** summed host gaps, and "
                f"**{trace['host_launch_count_per_trip']:,}** launches. Kernel and "
                "gap sums overlap in wall time and are not added as an elapsed-time "
                "estimate."
            ),
            "",
            "## Mechanism and repair",
            "",
            f"**Mechanism:** {mechanism['finding']} The source anchor is "
            f"`{mechanism['source']}`. The compiled runtime owns "
            f"**{100.0 * mechanism['compiled_runtime_share']:.4f}%** of the profiled "
            "wall. Eliminating every measured Python wrapper and summary-transfer "
            "cost could save at most "
            f"**{mechanism['host_repair_ceiling_s_per_trip']:.6f} "
            f"s/trip** ({100.0 * mechanism['host_repair_ceiling_share']:.4f}%).",
            "",
            f"**Recommended repair:** {mechanism['repair']}",
            "",
            (
                "The opportunity estimate is "
                f"**{mechanism['estimated_saving_s_per_trip']:.6f} "
                "s/trip**, the observed compiled-runtime gap to the synchronized "
                "one-trip "
                f"wall. This is an upper bound: {mechanism['estimate_contract']}."
            ),
            "",
            "## Evidence boundary",
            "",
            f"- cProfile stats: `{receipt['cprofile']['stats_path']}`.",
            f"- cProfile text: `{receipt['cprofile']['text_path']}`.",
            f"- Receipt: `{receipt['output_path']}`.",
            f"- Figure: `{receipt['figure_path']}`.",
            f"- Scheduler stdout: `{receipt['scheduler']['stdout_path']}`.",
            f"- Scheduler stderr: `{receipt['scheduler']['stderr_path']}`.",
            "- No `nova/` source file was changed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assemble(
    measurement_path: Path,
    output: Path,
    figure: Path,
    report: Path,
    flame_path: Path,
    pyspy_exit_status: int,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    receipt = json.loads(measurement_path.read_text(encoding="utf-8"))
    profile = receipt["profiled_solve"]
    execute_row = next(
        row
        for row in receipt["cprofile"]["top_by_cumulative"]
        if row["function"] == "__call__"
        and row["file"].endswith("jax/_src/interpreters/pxla.py")
    )
    trips = receipt["production_identity"]["observed_active_set_trips"]
    profile["compiled_runtime_s"] = execute_row["self_s"] + profile["compiled_wait_s"]
    profile["compiled_runtime_s_per_trip"] = profile["compiled_runtime_s"] / trips
    profile["python_wrapper_s"] = max(
        profile["dispatch_s"] - execute_row["self_s"], 0.0
    )
    profile["python_wrapper_s_per_trip"] = profile["python_wrapper_s"] / trips
    profile["compiled_runtime_share"] = (
        profile["compiled_runtime_s"] / profile["wall_s"]
    )
    profile["classification_contract"] = (
        "cProfile pxla.ExecuteReplicated.__call__ self time plus the explicit "
        "block_until_ready tail is compiled/PJRT runtime; remaining inclusive "
        "wrapper time is a conservative Python overhead ceiling"
    )
    receipt["pyspy"] = _flame_summary(flame_path, pyspy_exit_status)
    stdout_text = stdout_path.read_text(encoding="utf-8")
    receipt["pyspy"]["sampling_lag_warning"] = "behind in sampling" in stdout_text
    sample_match = re.search(r"Samples: ([0-9]+) Errors: ([0-9]+)", stdout_text)
    receipt["pyspy"]["sample_count"] = int(sample_match.group(1)) if sample_match else 0
    receipt["pyspy"]["sample_errors"] = (
        int(sample_match.group(2)) if sample_match else 0
    )
    receipt["scheduler"]["stdout_path"] = str(stdout_path)
    receipt["scheduler"]["stderr_path"] = str(stderr_path)
    receipt["mechanism"] = _mechanism(receipt)
    receipt["output_path"] = str(output)
    receipt["figure_path"] = str(figure)
    receipt["assembly_driver"] = {
        "path": str(Path(__file__).relative_to(ROOT)),
        "sha256": _sha256(Path(__file__)),
        "matches_measurement_driver": (
            _sha256(Path(__file__)) == receipt["driver"]["sha256"]
        ),
        "difference_scope": (
            "post-measurement boundary classification, native flame parsing, "
            "report wording, and figure styling; measurement execution is unchanged"
        ),
    }
    _write_figure(receipt, figure)
    receipt["figure_sha256"] = _sha256(figure)
    _write_json(output, receipt)
    _write_report(receipt, report)
    print(f"RECEIPT_WRITTEN={output}", flush=True)
    print(f"FIGURE_WRITTEN={figure}", flush=True)
    print(f"REPORT_WRITTEN={report}", flush=True)
    return receipt


def preflight() -> None:
    configure_dtypes()
    case, profile, target_current, _carrier, _policy = _profile_and_seed()
    state = jnp.asarray(case["state"])
    shaped = jax.eval_shape(_production_program(profile, target_current), state)
    print(
        json.dumps(
            {
                "status": "preflight_complete",
                "reference": case["reference"],
                "result_type": type(shaped).__name__,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("measure", "pyspy-target", "assemble", "preflight")
    )
    parser.add_argument("--measurement", type=Path, default=DEFAULT_MEASUREMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--stats-text", type=Path, default=DEFAULT_STATS_TEXT)
    parser.add_argument("--flame", type=Path, default=DEFAULT_FLAME)
    parser.add_argument("--pyspy-exit-status", type=int)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--stdout-path", type=Path)
    parser.add_argument("--stderr-path", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "preflight":
        preflight()
    elif arguments.mode == "measure":
        measure(
            arguments.measurement,
            arguments.stats,
            arguments.stats_text,
            arguments.cache_root,
        )
    elif arguments.mode == "pyspy-target":
        pyspy_target(arguments.cache_root)
    else:
        required = {
            "--pyspy-exit-status": arguments.pyspy_exit_status,
            "--stdout-path": arguments.stdout_path,
            "--stderr-path": arguments.stderr_path,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("assemble requires " + ", ".join(missing))
        assemble(
            arguments.measurement,
            arguments.output,
            arguments.figure,
            arguments.report,
            arguments.flame,
            arguments.pyspy_exit_status,
            arguments.stdout_path,
            arguments.stderr_path,
        )


if __name__ == "__main__":
    main()
