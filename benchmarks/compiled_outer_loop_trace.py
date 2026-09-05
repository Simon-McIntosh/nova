"""Trace the compiled active-set outer loop for one production solve.

The production solver implementation is left untouched.  This benchmark makes
an in-memory copy of the active-set entry point, inserts ordered host callbacks
at the outer-loop and active-body boundaries, and compiles that copy for the
banked MAST pure-arm invocation.  Callback timing is deliberately isolated from
the uninstrumented production timing retained in the receipt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import textwrap
import threading
import time
from typing import Any, Callable, Iterator

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
OUTPUT_ROOT = ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/outer-loop"
DEFAULT_OUTPUT = OUTPUT_ROOT / "profile.json"
DEFAULT_TABLE = OUTPUT_ROOT / "outer-iterations.csv"
DEFAULT_FIGURE = OUTPUT_ROOT / "outer-loop-trace.png"
DEFAULT_EVENT_LOG = OUTPUT_ROOT / "callback-events.jsonl"
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/"
    "compiled-outer-loop-trace.md"
)
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile")
HOST_PROFILE = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/host-profile/"
    "profile.json"
)
ONE_TRIP_PROFILE = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/"
    "hessenberg-trace/profile.json"
)
REFERENCE_SHOT = 22086
REFERENCE_SLICE = 43
ACTIVE_SET_BUDGET = 24
ONE_TRIP_ACTIVE_SET_BUDGET = 1
EXPECTED_ACTIVE_SET_TRIPS = 7
LINE_SEARCH_LADDER_LENGTH = len(fixed_point._BACKTRACKING_FACTORS)
ONE_TRIP_LINEAR_ACTIONS_PER_PROMOTION = 13


_EVENTS: list[dict[str, Any]] = []
_EVENT_LOCK = threading.Lock()
_EVENT_LOG: Path | None = None


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    cpu_devices = jax.devices("cpu")
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_kinds": [getattr(device, "device_kind", None) for device in devices],
        "cpu_callback_devices": [str(device) for device in cpu_devices],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
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
            "outer-loop tracing requires the reserved H200; "
            f"backend={jax.default_backend()} devices={kinds}"
        )
    if not jax.devices("cpu"):
        raise RuntimeError("ordered debug callbacks require a visible CPU device")


def _source_anchor(function: Callable, needle: str) -> str:
    path = Path(inspect.getsourcefile(function) or "")
    lines, first = inspect.getsourcelines(function)
    offset = next(
        (index for index, line in enumerate(lines) if needle in line),
        None,
    )
    if offset is None:
        raise RuntimeError(f"source anchor not found: {needle}")
    try:
        display = path.relative_to(ROOT)
    except ValueError:
        display = path
    return f"{display}:{first + offset}"


def _append_event(event: dict[str, Any]) -> None:
    with _EVENT_LOCK:
        event["sequence"] = len(_EVENTS)
        _EVENTS.append(event)
        if _EVENT_LOG is None:
            raise RuntimeError("callback event log is not configured")
        with _EVENT_LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_strict(event), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _reset_events(path: Path) -> None:
    global _EVENT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    with _EVENT_LOCK:
        _EVENTS.clear()
        _EVENT_LOG = path


def _record_outer_event(kind: str, index: Any, active: Any) -> None:
    _append_event(
        {
            "kind": kind,
            "timestamp_ns": time.perf_counter_ns(),
            "index": int(index),
            "active": bool(active),
        }
    )


def _record_active_result(
    index: Any,
    attempted: Any,
    accepted: Any,
    backtrack_counts: Any,
    decisions: Any,
    krylov_qualifications: Any,
    active_after: Any,
) -> None:
    timestamp_ns = time.perf_counter_ns()
    attempted_count = int(attempted)
    accepted_count = int(accepted)
    backtracks = np.asarray(backtrack_counts, dtype=np.int64)
    decision_values = np.asarray(decisions, dtype=np.int64)
    qualification_values = np.asarray(krylov_qualifications, dtype=np.int64)
    executed_receipt_rows = int(
        np.count_nonzero(
            decision_values != int(fixed_point.InnerIterationDecision.NOT_EXECUTED)
        )
    )
    if executed_receipt_rows != attempted_count:
        raise RuntimeError(
            "inner receipt row count does not match attempted promotions: "
            f"rows={executed_receipt_rows} attempted={attempted_count}"
        )
    event = {
        "kind": "solve_end",
        "timestamp_ns": timestamp_ns,
        "index": int(index),
        "active": bool(active_after),
        "attempted_promotions": attempted_count,
        "accepted_promotions": accepted_count,
        "executed_receipt_rows": executed_receipt_rows,
        "backtrack_counts": backtracks[:attempted_count].tolist(),
        "krylov_qualifications": qualification_values[:attempted_count].tolist(),
        "linear_actions": (
            executed_receipt_rows * ONE_TRIP_LINEAR_ACTIONS_PER_PROMOTION
        ),
        "line_search_grades": executed_receipt_rows * LINE_SEARCH_LADDER_LENGTH,
    }
    _append_event(event)


def _instrumented_active_set_entry() -> Callable:
    """Return a callback-instrumented in-memory copy of the production entry."""
    source = textwrap.dedent(inspect.getsource(fixed_point._active_set_newton_krylov))
    replacements = (
        (
            "def _active_set_newton_krylov(",
            "def _instrumented_active_set_newton_krylov(",
        ),
        (
            "    def outer_body(index, carry):\n        def solve_active(carry):\n",
            "    def outer_body(index, carry):\n"
            "        jax.debug.callback(\n"
            "            partial(_record_outer_event, 'outer_start'),\n"
            "            index, carry.active, ordered=True,\n"
            "        )\n"
            "        def solve_active(carry):\n"
            "            jax.debug.callback(\n"
            "                partial(_record_outer_event, 'solve_start'),\n"
            "                index, carry.active, ordered=True,\n"
            "            )\n",
        ),
        (
            "            next_presettlement = (\n",
            "            jax.debug.callback(\n"
            "                _record_active_result,\n"
            "                index,\n"
            "                inner_result.attempted_newton_promotions,\n"
            "                inner_result.accepted_newton_promotions,\n"
            "                inner_result.promotion_backtrack_counts,\n"
            "                inner_result.inner_iteration_decisions,\n"
            "                inner_result.inner_iteration_krylov_qualifications,\n"
            "                active,\n"
            "                ordered=True,\n"
            "            )\n"
            "            next_presettlement = (\n",
        ),
        (
            "        return jax.lax.cond("
            "carry.active, solve_active, lambda value: value, carry)\n",
            "        result = jax.lax.cond(\n"
            "            carry.active, solve_active, lambda value: value, carry\n"
            "        )\n"
            "        jax.debug.callback(\n"
            "            partial(_record_outer_event, 'outer_end'),\n"
            "            index, result.active, ordered=True,\n"
            "        )\n"
            "        return result\n",
        ),
    )
    for old, new in replacements:
        occurrences = source.count(old)
        if occurrences != 1:
            raise RuntimeError(
                "instrumentation source pattern changed: "
                f"expected one occurrence, found {occurrences}: {old!r}"
            )
        source = source.replace(old, new)
    namespace = dict(vars(fixed_point))
    namespace.update(
        {
            "partial": __import__("functools").partial,
            "_record_outer_event": _record_outer_event,
            "_record_active_result": _record_active_result,
        }
    )
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["_instrumented_active_set_newton_krylov"]


@contextmanager
def _patched_active_set_entry() -> Iterator[None]:
    original = fixed_point._active_set_newton_krylov
    fixed_point._active_set_newton_krylov = _instrumented_active_set_entry()
    try:
        yield
    finally:
        fixed_point._active_set_newton_krylov = original


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
            active_set_steps=ACTIVE_SET_BUDGET,
        )

    return production


def _ready(value: Any) -> Any:
    jax.block_until_ready(value)
    return value


def _solve_summary(result: Any) -> dict[str, Any]:
    history = result.equilibrium.fixed_point
    reason = fixed_point.FixedPointTerminationReason(
        int(np.asarray(history.termination_reason))
    )
    return {
        "active_set_trips": int(np.asarray(history.active_set_iterations)),
        "attempted_newton_promotions": int(
            np.asarray(history.attempted_newton_promotions)
        ),
        "accepted_newton_promotions": int(
            np.asarray(history.accepted_newton_promotions)
        ),
        "termination_reason": reason.name.lower(),
        "terminal_residual": float(np.asarray(history.residual)),
        "converged": bool(np.asarray(history.converged)),
    }


def _banked_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    host = json.loads(HOST_PROFILE.read_text(encoding="utf-8"))
    one_trip = json.loads(ONE_TRIP_PROFILE.read_text(encoding="utf-8"))
    exact = next(arm for arm in one_trip["arms"] if arm["name"] == "exact_jacobi")
    census = next(
        row for row in one_trip["evaluation_census"] if row["arm"] == "exact_jacobi"
    )
    return (
        {
            "path": str(HOST_PROFILE.relative_to(ROOT)),
            "sha256": _sha256(HOST_PROFILE),
            "job_id": host["scheduler"]["job_id"],
            "wall_s": host["profiled_solve"]["wall_s"],
            "wall_s_per_reported_trip": host["profiled_solve"]["wall_s_per_trip"],
            "reported_trips": host["production_identity"]["observed_active_set_trips"],
            "summary": host["profiled_solve"]["summary"],
        },
        {
            "path": str(ONE_TRIP_PROFILE.relative_to(ROOT)),
            "sha256": _sha256(ONE_TRIP_PROFILE),
            "job_id": one_trip["scheduler"]["job_id"],
            "synchronized_wall_s": exact["trace_wall_s_per_observed_trip"],
            "attempted_promotions": exact["solve"]["attempted_newton_promotions"],
            "accepted_promotions": exact["solve"]["accepted_newton_promotions"],
            "linear_actions": census["counts_total"]["linear_action"],
        },
    )


def _parse_outer_rows() -> list[dict[str, Any]]:
    by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in _EVENTS:
        by_index[event["index"]].append(event)
    expected_indices = list(range(1, ACTIVE_SET_BUDGET))
    if sorted(by_index) != expected_indices:
        raise RuntimeError(
            f"outer callback indices changed: {sorted(by_index)} != {expected_indices}"
        )
    rows = []
    for index in expected_indices:
        events = by_index[index]
        kinds = [event["kind"] for event in events]
        start = next(event for event in events if event["kind"] == "outer_start")
        end = next(event for event in events if event["kind"] == "outer_end")
        active = bool(start["active"])
        expected_kinds = (
            ["outer_start", "solve_start", "solve_end", "outer_end"]
            if active
            else ["outer_start", "outer_end"]
        )
        if kinds != expected_kinds:
            raise RuntimeError(
                f"callback order changed for outer index {index}: {kinds}"
            )
        row = {
            "outer_index": index,
            "active_at_entry": active,
            "active_after": bool(end["active"]),
            "outer_wall_s": (end["timestamp_ns"] - start["timestamp_ns"]) * 1.0e-9,
            "inner_callback_delta_s": None,
            "outer_minus_inner_callback_delta_s": None,
            "attempted_promotions": 0,
            "accepted_promotions": 0,
            "executed_receipt_rows": 0,
            "linear_actions": 0,
            "line_search_grades": 0,
            "backtrack_counts": [],
            "krylov_qualifications": [],
        }
        if active:
            solve_start = events[1]
            solve_end = events[2]
            inner_callback_delta = (
                solve_end["timestamp_ns"] - solve_start["timestamp_ns"]
            ) * 1.0e-9
            row.update(
                {
                    "inner_callback_delta_s": inner_callback_delta,
                    "outer_minus_inner_callback_delta_s": (
                        row["outer_wall_s"] - inner_callback_delta
                    ),
                    "attempted_promotions": solve_end["attempted_promotions"],
                    "accepted_promotions": solve_end["accepted_promotions"],
                    "executed_receipt_rows": solve_end["executed_receipt_rows"],
                    "linear_actions": solve_end["linear_actions"],
                    "line_search_grades": solve_end["line_search_grades"],
                    "backtrack_counts": solve_end["backtrack_counts"],
                    "krylov_qualifications": solve_end["krylov_qualifications"],
                }
            )
        rows.append(row)
    return rows


def _verify_event_log(path: Path) -> None:
    persisted = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if persisted != _strict(_EVENTS):
        raise RuntimeError(
            "persisted callback events do not match the in-memory trace: "
            f"persisted={len(persisted)} memory={len(_EVENTS)}"
        )


def _distribution(values: list[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size),
        "sum_s": float(data.sum()),
        "minimum_s": float(data.min()),
        "median_s": float(np.median(data)),
        "maximum_s": float(data.max()),
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "outer_index",
        "active_at_entry",
        "active_after",
        "outer_wall_s",
        "inner_callback_delta_s",
        "attempted_promotions",
        "accepted_promotions",
        "linear_actions",
        "line_search_grades",
    )
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append("" if value is None else str(value))
        lines.append(",".join(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(receipt: dict[str, Any], path: Path) -> None:
    rows = receipt["outer_iterations"]
    one_trip = receipt["banked_one_trip"]["synchronized_wall_s"]
    indices = [row["outer_index"] for row in rows]
    walls = [row["outer_wall_s"] for row in rows]
    colors = ["tab:blue" if row["active_at_entry"] else "0.72" for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), constrained_layout=True)
    axes[0].bar(indices, walls, color=colors)
    axes[0].axhline(
        one_trip,
        color="firebrick",
        linestyle="--",
        linewidth=1.5,
        label="compiled one-trip wall",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Callback-delimited wall (s, log scale)")
    axes[0].set_title("Every compiled outer iteration executes; inactive bodies skip")
    axes[0].legend()

    active = [row for row in rows if row["active_at_entry"]]
    axes[1].plot(
        [row["outer_index"] for row in active],
        [row["attempted_promotions"] for row in active],
        marker="o",
        label="promotion attempts",
    )
    axes[1].plot(
        [row["outer_index"] for row in active],
        [row["accepted_promotions"] for row in active],
        marker="s",
        label="accepted promotions",
    )
    axes[1].set_xlabel("Compiled outer-loop index")
    axes[1].set_ylabel("Receipt count")
    axes[1].set_title("Inner Newton work per active outer iteration")
    axes[1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(receipt: dict[str, Any], path: Path) -> None:
    counts = receipt["counts"]
    timing = receipt["timing"]
    budgets = receipt["budgets"]
    one_trip = receipt["banked_one_trip"]
    production = receipt["banked_production"]
    solve = receipt["instrumented_execution"]["summary"]
    mechanism = receipt["mechanism"]
    rows = receipt["outer_iterations"]
    lines = [
        "# Compiled outer-loop iteration trace",
        "",
        (
            f"SLURM job `{receipt['scheduler']['job_id']}` traced the exact banked "
            f"MAST 22086/43 pure production invocation at "
            f"`{receipt['measurement_revision']}` on `{receipt['runtime']['host']}`. "
            f"The active-set budget has **{counts['active_set_budget_positions']} "
            f"positions**: one initial frozen-mask solve followed by all "
            f"**{counts['outer_iterations']} compiled loop iterations**. Of those "
            f"loop iterations, **{counts['active_outer_iterations']} are active** and "
            f"**{counts['inactive_outer_iterations']} are inactive**. The initial "
            "solve plus the six active loop bodies are the seven trips in the "
            "production receipt."
        ),
        (
            "The instrumented and uninstrumented result summaries match exactly. "
            "This is an active-set-settled, nonconverged solve "
            f"(`converged={str(solve['converged']).lower()}`, termination "
            f"`{solve['termination_reason']}`, terminal residual "
            f"{solve['terminal_residual']:.9g}); it characterizes execution cost "
            "and is not a certificate-passing solve."
        ),
        "",
        "## Per-iteration evidence",
        "",
        (
            "Ordered `jax.debug.callback` timestamps delimit every outer-loop index. "
            "Those outer deltas are instrumented walls; the loop-carried state keeps "
            "the active rows sequential. The two callbacks placed inside "
            "`solve_active` do not form barriers around pure operations, which XLA "
            "may schedule before the first effect. Their small delta is reported only "
            "as an ordering diagnostic and is not used as an active-body wall. The "
            "uninstrumented 41.1 s production receipt remains the total-wall authority."
        ),
        "",
        "| outer | active in | active out | outer wall s | inner callback delta s | "
        "promotions attempted/accepted | linear actions | line-search grades |",
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        inner_callback_delta = (
            "—"
            if row["inner_callback_delta_s"] is None
            else f"{row['inner_callback_delta_s']:.6f}"
        )
        lines.append(
            f"| {row['outer_index']} | {str(row['active_at_entry']).lower()} | "
            f"{str(row['active_after']).lower()} | {row['outer_wall_s']:.6f} | "
            f"{inner_callback_delta} | {row['attempted_promotions']}/"
            f"{row['accepted_promotions']} | {row['linear_actions']} | "
            f"{row['line_search_grades']} |"
        )
    lines.extend(
        [
            "",
            (
                "Promotions are counted by non-padding inner-receipt rows. Linear "
                f"actions use the independently measured one-trip ratio of "
                f"{ONE_TRIP_LINEAR_ACTIONS_PER_PROMOTION} actions per promotion "
                f"({one_trip['linear_actions']} / "
                f"{one_trip['attempted_promotions']}); every promotion evaluates "
                f"the fixed {LINE_SEARCH_LADDER_LENGTH}-grade line-search ladder."
            ),
            "",
            "## Resolved budgets",
            "",
            "| program | active-set budget | Newton promotions / active trip | "
            "GMRES iterations / promotion | line-search grades / promotion |",
            "|---|---:|---:|---:|---:|",
            (
                f"| production solve | {budgets['production']['active_set_steps']} | "
                f"{budgets['production']['newton_steps']} | "
                f"{budgets['production']['gmres_iterations']} | "
                f"{budgets['production']['line_search_ladder_length']} |"
            ),
            (
                f"| compiled one-trip | {budgets['one_trip']['active_set_steps']} | "
                f"{budgets['one_trip']['newton_steps']} | "
                f"{budgets['one_trip']['gmres_iterations']} | "
                f"{budgets['one_trip']['line_search_ladder_length']} |"
            ),
            "",
            "## What the loop executes",
            "",
            (
                "The median active outer iteration is "
                f"**{timing['active_outer']['median_s']:.6f} "
                f"s**, **{timing['active_outer_to_one_trip_ratio']:.2f}×** the banked "
                f"**{one_trip['synchronized_wall_s']:.6f} s** one-trip executable "
                f"from job `{one_trip['job_id']}`. Active iterations range from "
                f"**{timing['active_outer']['minimum_s']:.6f} s** "
                f"({timing['minimum_active_to_one_trip_ratio']:.2f}×) "
                f"to **{timing['active_outer']['maximum_s']:.6f} s** "
                f"({timing['maximum_active_to_one_trip_ratio']:.2f}×). "
                "The uninstrumented production "
                f"receipt from job `{production['job_id']}` is "
                f"**{production['wall_s']:.6f} s** for seven reported trips, "
                f"**{production['wall_s_per_reported_trip']:.6f} s/trip**."
            ),
            (
                f"The {counts['inactive_outer_iterations']} inactive iterations still "
                "execute the compiled loop shell and conditional, but do not enter "
                f"`solve_active`: their callback-delimited walls total "
                f"**{timing['inactive_outer']['sum_s']:.6f} s** versus "
                f"**{timing['active_outer']['sum_s']:.6f} s** across active "
                "iterations. "
                "Because two callbacks dominate such tiny inactive rows, that total "
                "is an instrumentation upper bound, not a production saving claim."
            ),
            (
                "Every active outer iteration consumes its full 12-promotion receipt "
                "shape. After the settlement transition, no further inner Newton, "
                "GMRES, or line-search work runs; only the inactive loop shell "
                "remains. "
                "The settlement-proving active trip itself still reaches the full "
                "static inner budget before `carry.active` becomes false."
            ),
            "",
            "## Mechanism and implied repair",
            "",
            (
                f"**Mechanism:** the one-trip program never enters the outer loop: its "
                f"active-set budget is one, while the initial `solve_frozen` call is "
                f"outside the loop at `{receipt['source_anchors']['initial_solve']}`. "
                f"Production puts the next six active trips inside the compiled "
                f"`fori_loop` at `{receipt['source_anchors']['outer_loop']}`, whose "
                "body is guarded by "
                f"`{receipt['source_anchors']['active_conditional']}`. "
                "The callback rows show that the conditional skips inactive bodies. "
                f"The active loop iterations take "
                f"{timing['active_outer_to_one_trip_ratio']:.2f}× the one-trip wall "
                "at the median, so the production gap belongs to the looped program's "
                "active iterations, not Python dispatch or inactive padding."
            ),
            (
                "**Repair:** compile one common trip executable with an explicit "
                "state/history/globalization carry and drive repeated trips until "
                "settlement, so the initial and subsequent trips use the same fast "
                "program. A `lax.while_loop` exit can remove the inactive shell, but "
                "cannot close the measured active-trip gap by itself. Separately, stop "
                "the settlement-proving recovery sequence once its unchanged state and "
                "zero accepted promotions make settlement decidable."
            ),
            (
                "The common-trip opportunity is "
                f"**{mechanism['estimated_saving_s']:.6f} "
                "s/solve** across the six active loop bodies, computed from their "
                "callback-delimited walls against six banked one-trip walls. It is an "
                "upper-bound opportunity, not a speedup measured after a repair. The "
                f"inactive-loop exit opportunity is at most "
                f"**{mechanism['inactive_shell_upper_bound_s']:.6f} s/solve** and is "
                "callback-inflated."
            ),
            "",
            "## Evidence boundary",
            "",
            f"- Receipt: `{receipt['artifacts']['receipt']}`.",
            f"- Per-iteration CSV: `{receipt['artifacts']['table']}`.",
            f"- Figure: `{receipt['artifacts']['figure']}`.",
            f"- Durable callback event stream: `{receipt['artifacts']['event_log']}`.",
            f"- Scheduler stdout: `{receipt['log_paths']['stdout']}`.",
            f"- Scheduler stderr: `{receipt['log_paths']['stderr']}`.",
            "- No `nova/` source file was changed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    output: Path,
    table: Path,
    figure: Path,
    event_log: Path,
    report: Path,
    cache_root: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
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

    uninstrumented = jax.jit(_production_program(profile, target_current))
    started = time.perf_counter()
    uninstrumented_executable = uninstrumented.lower(state).compile()
    uninstrumented_compile_s = time.perf_counter() - started
    started = time.perf_counter()
    uninstrumented_result = _ready(uninstrumented_executable(state))
    uninstrumented_wall_s = time.perf_counter() - started
    uninstrumented_summary = _solve_summary(uninstrumented_result)

    with _patched_active_set_entry():
        jax.clear_caches()
        instrumented = jax.jit(_production_program(profile, target_current))
        started = time.perf_counter()
        instrumented_executable = instrumented.lower(state).compile()
        instrumented_compile_s = time.perf_counter() - started
        _reset_events(event_log)
        started = time.perf_counter()
        instrumented_result = _ready(instrumented_executable(state))
        instrumented_wall_s = time.perf_counter() - started
        instrumented_summary = _solve_summary(instrumented_result)

    _verify_event_log(event_log)
    rows = _parse_outer_rows()
    active_rows = [row for row in rows if row["active_at_entry"]]
    inactive_rows = [row for row in rows if not row["active_at_entry"]]
    if len(active_rows) + 1 != instrumented_summary["active_set_trips"]:
        raise RuntimeError(
            "active callback count does not match reported trips: "
            f"outer={len(active_rows)} "
            f"reported={instrumented_summary['active_set_trips']}"
        )
    if instrumented_summary != uninstrumented_summary:
        raise RuntimeError(
            "instrumentation changed the production result summary: "
            f"instrumented={instrumented_summary} "
            f"uninstrumented={uninstrumented_summary}"
        )
    if instrumented_summary["active_set_trips"] != EXPECTED_ACTIVE_SET_TRIPS:
        raise RuntimeError(
            "production trip count changed from "
            f"{EXPECTED_ACTIVE_SET_TRIPS} to "
            f"{instrumented_summary['active_set_trips']}"
        )

    active_attempts = sum(row["attempted_promotions"] for row in active_rows)
    active_accepted = sum(row["accepted_promotions"] for row in active_rows)
    initial_attempts = (
        instrumented_summary["attempted_newton_promotions"] - active_attempts
    )
    initial_accepted = (
        instrumented_summary["accepted_newton_promotions"] - active_accepted
    )
    banked_production, banked_one_trip = _banked_evidence()
    active_outer = _distribution([row["outer_wall_s"] for row in active_rows])
    inner_callback_delta = _distribution(
        [float(row["inner_callback_delta_s"]) for row in active_rows]
    )
    inactive_outer = _distribution([row["outer_wall_s"] for row in inactive_rows])
    common_trip_saving = sum(
        max(
            row["outer_wall_s"] - banked_one_trip["synchronized_wall_s"],
            0.0,
        )
        for row in active_rows
    )
    minimum_active_ratio = (
        active_outer["minimum_s"] / banked_one_trip["synchronized_wall_s"]
    )
    maximum_active_ratio = (
        active_outer["maximum_s"] / banked_one_trip["synchronized_wall_s"]
    )
    receipt: dict[str, Any] = {
        "schema": "nova.compiled_outer_loop_trace",
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": _revision(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
            "instrumentation": (
                "in-memory copy of fixed_point._active_set_newton_krylov with "
                "ordered benchmark-only debug callbacks"
            ),
        },
        "runtime": _runtime(),
        "scheduler": _scheduler(),
        "log_paths": {"stdout": str(stdout_path), "stderr": str(stderr_path)},
        "production_identity": {
            "reference": reference,
            "arm": "pure",
            "carrier": carrier,
            "field_policy": policy,
            "route": "ForwardProfile.solve_branch newton_krylov",
            "outer_boundary": "one jax.jit around the complete solve_branch call",
        },
        "budgets": {
            "production": {
                "active_set_steps": ACTIVE_SET_BUDGET,
                "newton_steps": parity.NEWTON_STEPS,
                "gmres_iterations": parity.GMRES_ITERATIONS,
                "line_search_ladder_length": LINE_SEARCH_LADDER_LENGTH,
            },
            "one_trip": {
                "active_set_steps": ONE_TRIP_ACTIVE_SET_BUDGET,
                "newton_steps": parity.NEWTON_STEPS,
                "gmres_iterations": parity.GMRES_ITERATIONS,
                "line_search_ladder_length": LINE_SEARCH_LADDER_LENGTH,
            },
        },
        "compile": {
            "uninstrumented_lower_and_compile_wall_s": uninstrumented_compile_s,
            "instrumented_lower_and_compile_wall_s": instrumented_compile_s,
            "persistent_cache": cache.receipt(),
        },
        "uninstrumented_execution": {
            "wall_s": uninstrumented_wall_s,
            "summary": uninstrumented_summary,
        },
        "instrumented_execution": {
            "wall_s": instrumented_wall_s,
            "summary": instrumented_summary,
            "timing_eligible_for_production_total": False,
        },
        "counts": {
            "active_set_budget_positions": ACTIVE_SET_BUDGET,
            "outer_iterations": len(rows),
            "active_outer_iterations": len(active_rows),
            "inactive_outer_iterations": len(inactive_rows),
            "reported_active_set_trips": instrumented_summary["active_set_trips"],
            "initial_trip_outside_outer_loop": 1,
        },
        "initial_trip": {
            "timed_by_outer_callbacks": False,
            "attempted_promotions": initial_attempts,
            "accepted_promotions": initial_accepted,
            "linear_actions": (
                initial_attempts * ONE_TRIP_LINEAR_ACTIONS_PER_PROMOTION
            ),
            "line_search_grades": initial_attempts * LINE_SEARCH_LADDER_LENGTH,
        },
        "outer_iterations": rows,
        "timing": {
            "active_outer": active_outer,
            "inner_callback_delta": inner_callback_delta,
            "inactive_outer": inactive_outer,
            "active_outer_to_one_trip_ratio": (
                active_outer["median_s"] / banked_one_trip["synchronized_wall_s"]
            ),
            "minimum_active_to_one_trip_ratio": minimum_active_ratio,
            "maximum_active_to_one_trip_ratio": maximum_active_ratio,
            "callback_contract": (
                "ordered callbacks delimit each outer-loop index; inner callback "
                "deltas are not body walls because XLA may schedule pure operations "
                "before the first inner effect; inactive outer deltas are upper bounds "
                "dominated by their two callbacks"
            ),
        },
        "banked_production": banked_production,
        "banked_one_trip": banked_one_trip,
        "source_anchors": {
            "initial_solve": _source_anchor(
                fixed_point._active_set_newton_krylov,
                "first_result, first_globalization = solve_frozen",
            ),
            "active_conditional": _source_anchor(
                fixed_point._active_set_newton_krylov,
                "return jax.lax.cond(carry.active",
            ),
            "outer_loop": _source_anchor(
                fixed_point._active_set_newton_krylov,
                "outer = jax.lax.fori_loop",
            ),
        },
        "mechanism": {
            "finding": (
                "The one-trip budget executes only the initial solve outside the "
                "compiled outer loop. Production's six subsequent active solves run "
                f"inside that loop and take {minimum_active_ratio:.2f} to "
                f"{maximum_active_ratio:.2f} "
                "one-trip walls each; inactive loop iterations take the conditional "
                "skip and do not execute inner work."
            ),
            "repair": (
                "Drive a common compiled trip executable with explicit state and "
                "history carry until settlement. A data-dependent outer exit removes "
                "inactive shell work only; stop the settlement-proving recovery "
                "sequence early as a separate inner-budget repair."
            ),
            "estimated_saving_s": common_trip_saving,
            "inactive_shell_upper_bound_s": inactive_outer["sum_s"],
            "estimate_contract": (
                "upper-bound opportunity if every active loop body reaches the banked "
                "one-trip wall; no implemented repair was measured"
            ),
        },
        "artifacts": {
            "receipt": str(output.resolve().relative_to(ROOT)),
            "table": str(table.resolve().relative_to(ROOT)),
            "figure": str(figure.resolve().relative_to(ROOT)),
            "event_log": str(event_log.resolve().relative_to(ROOT)),
            "report": str(report),
        },
    }
    _write_json(output, receipt)
    _write_table(table, rows)
    _write_figure(receipt, figure)
    _write_report(receipt, report)
    print(f"RECEIPT_WRITTEN={output}", flush=True)
    print(f"TABLE_WRITTEN={table}", flush=True)
    print(f"FIGURE_WRITTEN={figure}", flush=True)
    print(f"REPORT_WRITTEN={report}", flush=True)
    return receipt


def preflight() -> None:
    source = inspect.getsource(fixed_point._active_set_newton_krylov)
    instrumented = _instrumented_active_set_entry()
    print(
        json.dumps(
            {
                "status": "preflight_complete",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "instrumented_name": instrumented.__name__,
                "active_set_budget": ACTIVE_SET_BUDGET,
                "newton_steps": parity.NEWTON_STEPS,
                "gmres_iterations": parity.GMRES_ITERATIONS,
                "line_search_ladder_length": LINE_SEARCH_LADDER_LENGTH,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--event-log", type=Path, default=DEFAULT_EVENT_LOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--stdout-path", type=Path, default=Path("stdout.log"))
    parser.add_argument("--stderr-path", type=Path, default=Path("stderr.log"))
    arguments = parser.parse_args()
    if arguments.mode == "preflight":
        preflight()
        return
    run(
        arguments.output,
        arguments.table,
        arguments.figure,
        arguments.event_log,
        arguments.report,
        arguments.cache_root,
        arguments.stdout_path,
        arguments.stderr_path,
    )


if __name__ == "__main__":
    main()
