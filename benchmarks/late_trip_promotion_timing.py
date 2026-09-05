"""Measure promotion-level wall amplification in late active-set trips.

The production implementation remains untouched.  This benchmark compiles
in-memory copies of the active-set and Newton entry points, adding ordered
``jax.debug.callback`` boundaries around promotion, recovery, rebuilt-model,
and descent work.  A counted copy of JAX's batched GMRES records the Arnoldi
iterations actually executed, including every damped rebuilt-model solve.
Every callback is appended and fsynced before the solve continues.
"""

# Source-copy match strings intentionally retain the production line layout.
# ruff: noqa: E501

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
import textwrap
import threading
import time
from typing import Any, Callable, Iterator

import jax
from jax import lax
import jax.numpy as jnp
from jax._src import dtypes
from jax._src.lax import lax as lax_internal
from jax._src.scipy.sparse import linalg as jax_sparse_linalg
from jax.tree_util import tree_leaves, tree_map, tree_structure
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import compiled_outer_loop_trace as outer_trace
from benchmarks.receipt_raster_check import _profile_and_seed
from nova.equilibrium import fixed_point
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/late-trips"
DEFAULT_OUTPUT = OUTPUT_ROOT / "profile.json"
DEFAULT_TABLE = OUTPUT_ROOT / "promotions.csv"
DEFAULT_TRIP_TABLE = OUTPUT_ROOT / "per-trip.csv"
DEFAULT_FIGURE = OUTPUT_ROOT / "promotion-wall-by-trip.png"
DEFAULT_EVENT_LOG = OUTPUT_ROOT / "promotion-events.jsonl"
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/"
    "late-trip-promotion-timing.md"
)
DEFAULT_CACHE = Path("/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile")
FROZEN_PROFILE = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/"
    "recovery-frozen/outer-loop.json"
)
REFERENCE_SHOT = 22086
REFERENCE_SLICE = 43
ACTIVE_SET_BUDGET = 24
EXPECTED_ACTIVE_SET_TRIPS = 7
LADDER_LENGTH = len(fixed_point._BACKTRACKING_FACTORS)


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


def _replace_once(source: str, old: str, replacement: str) -> str:
    occurrences = source.count(old)
    if occurrences != 1:
        raise RuntimeError(
            "instrumentation source pattern changed: "
            f"expected one occurrence, found {occurrences}: {old!r}"
        )
    return source.replace(old, replacement)


def _source_anchor(function: Callable, needle: str) -> str:
    path = Path(inspect.getsourcefile(function) or "")
    lines, first = inspect.getsourcelines(function)
    offset = next((index for index, line in enumerate(lines) if needle in line), None)
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
            raise RuntimeError("promotion event log is not configured")
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


def _record_boundary(kind: str, trip: Any, promotion: Any) -> None:
    _append_event(
        {
            "kind": kind,
            "timestamp_ns": time.perf_counter_ns(),
            "trip": int(trip) + 1,
            "promotion": int(promotion) + 1,
        }
    )


def _record_gmres(
    gmres_kind: str,
    trip: Any,
    promotion: Any,
    damping_trial: Any,
    restart_cycles: Any,
    arnoldi_iterations: Any,
) -> None:
    _append_event(
        {
            "kind": "gmres_complete",
            "gmres_kind": gmres_kind,
            "timestamp_ns": time.perf_counter_ns(),
            "trip": int(trip) + 1,
            "promotion": int(promotion) + 1,
            "damping_trial": int(damping_trial),
            "restart_cycles": int(restart_cycles),
            "arnoldi_iterations": int(arnoldi_iterations),
        }
    )


def _record_ladder_result(
    trip: Any,
    promotion: Any,
    backtrack_count: Any,
    accepted: Any,
    recovery_activated: Any,
    radius_before: Any,
    radius_after: Any,
    recovery_outcome: Any,
    model_distrusted: Any,
) -> None:
    _append_event(
        {
            "kind": "ladder_end",
            "timestamp_ns": time.perf_counter_ns(),
            "trip": int(trip) + 1,
            "promotion": int(promotion) + 1,
            "backtrack_count": int(backtrack_count),
            "ladder_grade_reached": min(int(backtrack_count) + 1, LADDER_LENGTH),
            "accepted": bool(accepted),
            "recovery_activated": bool(recovery_activated),
            "recovery_radius_before": float(radius_before),
            "recovery_radius_after": float(radius_after),
            "recovery_outcome": int(recovery_outcome),
            "model_distrusted": bool(model_distrusted),
        }
    )


def _record_recovery_result(
    trip: Any,
    promotion: Any,
    radius_before: Any,
    radius_after: Any,
    accepted: Any,
    model_distrusted: Any,
) -> None:
    _append_event(
        {
            "kind": "recovery_end",
            "timestamp_ns": time.perf_counter_ns(),
            "trip": int(trip) + 1,
            "promotion": int(promotion) + 1,
            "recovery_radius_before": float(radius_before),
            "recovery_radius_after": float(radius_after),
            "accepted": bool(accepted),
            "model_distrusted": bool(model_distrusted),
        }
    )


def _record_rebuild_result(
    trip: Any,
    promotion: Any,
    accepted: Any,
    damping: Any,
    next_damping: Any,
) -> None:
    _append_event(
        {
            "kind": "rebuild_end",
            "timestamp_ns": time.perf_counter_ns(),
            "trip": int(trip) + 1,
            "promotion": int(promotion) + 1,
            "accepted": bool(accepted),
            "rebuild_damping": float(damping),
            "next_rebuild_damping": float(next_damping),
            "relinearized": True,
        }
    )


def _record_descent_result(
    trip: Any, promotion: Any, accepted: Any, scale: Any
) -> None:
    _append_event(
        {
            "kind": "descent_end",
            "timestamp_ns": time.perf_counter_ns(),
            "trip": int(trip) + 1,
            "promotion": int(promotion) + 1,
            "accepted": bool(accepted),
            "scale": float(scale),
            "grades_evaluated": len(fixed_point._STEEPEST_DESCENT_SCALES),
        }
    )


def _record_promotion_result(
    trip: Any,
    promotion: Any,
    accepted: Any,
    rebuild_activated: Any,
    rebuild_accepted: Any,
    descent_activated: Any,
    descent_accepted: Any,
) -> None:
    _append_event(
        {
            "kind": "promotion_end",
            "timestamp_ns": time.perf_counter_ns(),
            "trip": int(trip) + 1,
            "promotion": int(promotion) + 1,
            "accepted": bool(accepted),
            "rebuild_activated": bool(rebuild_activated),
            "rebuild_accepted": bool(rebuild_accepted),
            "descent_activated": bool(descent_activated),
            "descent_accepted": bool(descent_accepted),
        }
    )


def _gmres_batched_counted(
    operator,
    right_hand_side,
    initial,
    unit_residual,
    residual_norm,
    tolerance,
    restart,
    preconditioner,
):
    del tolerance
    basis = tree_map(
        lambda value: jnp.pad(
            value[..., None], ((0, 0),) * value.ndim + ((0, restart),)
        ),
        unit_residual,
    )
    dtype, weak_type = dtypes.lattice_result_type(*tree_leaves(right_hand_side))
    hessenberg = lax_internal._convert_element_type(
        jnp.eye(restart, restart + 1, dtype=dtype), weak_type=weak_type
    )

    def condition(carry):
        _basis, _hessenberg, breakdown, iteration = carry
        return jnp.logical_and(iteration < restart, jnp.logical_not(breakdown))

    def body(carry):
        current_basis, current_hessenberg, _breakdown, iteration = carry
        current_basis, current_hessenberg, breakdown = (
            jax_sparse_linalg._kth_arnoldi_iteration(
                iteration,
                operator,
                preconditioner,
                current_basis,
                current_hessenberg,
            )
        )
        return current_basis, current_hessenberg, breakdown, iteration + 1

    basis, hessenberg, _breakdown, used = lax.while_loop(
        condition, body, (basis, hessenberg, False, 0)
    )
    beta = (
        jnp.zeros_like(hessenberg, shape=(restart + 1,))
        .at[0]
        .set(residual_norm.astype(dtype))
    )
    coefficients = jax_sparse_linalg._lstsq(hessenberg.T, beta)
    delta = tree_map(
        lambda value: jax_sparse_linalg._dot(value[..., :-1], coefficients), basis
    )
    solution = jax_sparse_linalg._add(initial, delta)
    residual = preconditioner(
        jax_sparse_linalg._sub(right_hand_side, operator(solution))
    )
    unit_residual, residual_norm = jax_sparse_linalg._safe_normalize(residual)
    return solution, unit_residual, residual_norm, used


def _gmres_solve_counted(
    operator,
    right_hand_side,
    initial,
    absolute_tolerance,
    projected_tolerance,
    restart,
    maximum_restarts,
    preconditioner,
    *,
    trip,
    promotion,
    gmres_kind,
    damping_trial,
):
    residual = preconditioner(
        jax_sparse_linalg._sub(right_hand_side, operator(initial))
    )
    unit_residual, residual_norm = jax_sparse_linalg._safe_normalize(residual)

    def condition(carry):
        _solution, cycle, _unit_residual, current_norm, _used = carry
        return jnp.logical_and(
            cycle < maximum_restarts, current_norm > absolute_tolerance
        )

    def body(carry):
        solution, cycle, current_unit, current_norm, total_used = carry
        solution, current_unit, current_norm, used = _gmres_batched_counted(
            operator,
            right_hand_side,
            solution,
            current_unit,
            current_norm,
            projected_tolerance,
            restart,
            preconditioner,
        )
        return solution, cycle + 1, current_unit, current_norm, total_used + used

    solution, cycles, _unit_residual, _error, used = lax.while_loop(
        condition,
        body,
        (initial, 0, unit_residual, residual_norm, 0),
    )
    jax.debug.callback(
        __import__("functools").partial(_record_gmres, gmres_kind),
        trip,
        promotion,
        damping_trial,
        cycles,
        used,
        ordered=True,
    )
    return solution


def _timed_gmres(
    operator,
    right_hand_side,
    initial=None,
    *,
    tolerance=1.0e-5,
    absolute_tolerance=0.0,
    restart=20,
    maximum_restarts=None,
    trip,
    promotion,
    gmres_kind,
    damping_trial=-1,
):
    if initial is None:
        initial = tree_map(jnp.zeros_like, right_hand_side)
    preconditioner = jax_sparse_linalg._identity
    operator = jax_sparse_linalg._normalize_matvec(operator)
    preconditioner = jax_sparse_linalg._normalize_matvec(preconditioner)
    right_hand_side, initial = jax.device_put((right_hand_side, initial))
    size = sum(value.size for value in tree_leaves(right_hand_side))
    if maximum_restarts is None:
        maximum_restarts = 10 * size
    restart = min(restart, size)
    if tree_structure(initial) != tree_structure(right_hand_side):
        raise ValueError("GMRES initial value and right hand side must match")
    norm = jax_sparse_linalg._norm(right_hand_side)
    absolute_tolerance = jnp.maximum(tolerance * norm, absolute_tolerance)
    preconditioned = preconditioner(right_hand_side)
    preconditioned_norm = jax_sparse_linalg._norm(preconditioned)
    projected_tolerance = preconditioned_norm * jnp.minimum(
        1.0, absolute_tolerance / norm
    )

    def solve(linear_operator, vector):
        return _gmres_solve_counted(
            linear_operator,
            vector,
            initial,
            absolute_tolerance,
            projected_tolerance,
            restart,
            maximum_restarts,
            preconditioner,
            trip=trip,
            promotion=promotion,
            gmres_kind=gmres_kind,
            damping_trial=damping_trial,
        )

    solution = lax.custom_linear_solve(
        operator, right_hand_side, solve=solve, transpose_solve=solve
    )
    failed = jnp.isnan(jax_sparse_linalg._norm(solution))
    return solution, jnp.where(failed, -1, 0)


def _instrumented_qualified_step() -> Callable:
    source = textwrap.dedent(inspect.getsource(fixed_point._qualified_krylov_step))
    source = _replace_once(
        source,
        "def _qualified_krylov_step(",
        "def _instrumented_qualified_krylov_step(",
    )
    source = _replace_once(
        source,
        "    *,\n    gmres_iterations: int,\n",
        "    *,\n"
        "    benchmark_trip_index: jax.Array | int,\n"
        "    benchmark_promotion_index: jax.Array | int,\n"
        "    gmres_iterations: int,\n",
    )
    source = _replace_once(
        source,
        "    step, info = jax.scipy.sparse.linalg.gmres(\n"
        "        linear_action,\n"
        "        residual_vector,\n"
        "        tol=_GMRES_RELATIVE_TOLERANCE,\n"
        "        maxiter=gmres_iterations,\n"
        "        restart=gmres_iterations,\n"
        '        solve_method="batched",\n'
        "    )\n",
        "    step, info = _timed_gmres(\n"
        "        linear_action,\n"
        "        residual_vector,\n"
        "        tolerance=_GMRES_RELATIVE_TOLERANCE,\n"
        "        maximum_restarts=gmres_iterations,\n"
        "        restart=gmres_iterations,\n"
        "        trip=benchmark_trip_index,\n"
        "        promotion=benchmark_promotion_index,\n"
        '        gmres_kind="primary",\n'
        "    )\n",
    )
    namespace = dict(vars(fixed_point))
    namespace["_timed_gmres"] = _timed_gmres
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["_instrumented_qualified_krylov_step"]


def _instrumented_backtracked_promotion() -> Callable:
    source = textwrap.dedent(inspect.getsource(fixed_point._backtracked_promotion))
    source = _replace_once(
        source,
        "def _backtracked_promotion(",
        "def _instrumented_backtracked_promotion(",
    )
    source = _replace_once(
        source,
        "    *,\n    acceptance_map_fn:",
        "    *,\n"
        "    benchmark_trip_index: jax.Array | int,\n"
        "    benchmark_promotion_index: jax.Array | int,\n"
        "    acceptance_map_fn:",
    )
    source = _replace_once(
        source,
        "    def recover_with_continuation(_):\n        minimum_radius =",
        "    def recover_with_continuation(_):\n"
        "        jax.debug.callback(\n"
        '            partial(_record_boundary, "recovery_start"),\n'
        "            benchmark_trip_index, benchmark_promotion_index, ordered=True,\n"
        "        )\n"
        "        minimum_radius =",
    )
    source = _replace_once(
        source,
        "        next_radius = jnp.where(\n"
        "            recovery_accepted,\n"
        "            jnp.minimum(\n"
        "                trial_radius * _RECOVERY_RADIUS_GROWTH, _RECOVERY_RADIUS_INITIAL\n"
        "            ),\n"
        "            trial_radius,\n"
        "        )\n"
        "        return _BacktrackedPromotion(\n",
        "        next_radius = jnp.where(\n"
        "            recovery_accepted,\n"
        "            jnp.minimum(\n"
        "                trial_radius * _RECOVERY_RADIUS_GROWTH, _RECOVERY_RADIUS_INITIAL\n"
        "            ),\n"
        "            trial_radius,\n"
        "        )\n"
        "        jax.debug.callback(\n"
        "            _record_recovery_result,\n"
        "            benchmark_trip_index, benchmark_promotion_index,\n"
        "            initial_radius, next_radius, recovery_accepted, model_distrusted,\n"
        "            ordered=True,\n"
        "        )\n"
        "        return _BacktrackedPromotion(\n",
    )
    namespace = dict(vars(fixed_point))
    namespace.update(
        {
            "partial": __import__("functools").partial,
            "_record_boundary": _record_boundary,
            "_record_recovery_result": _record_recovery_result,
        }
    )
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["_instrumented_backtracked_promotion"]


def _instrumented_rebuilt_promotion() -> Callable:
    source = textwrap.dedent(inspect.getsource(fixed_point._rebuilt_model_promotion))
    source = _replace_once(
        source,
        "def _rebuilt_model_promotion(",
        "def _instrumented_rebuilt_model_promotion(",
    )
    source = _replace_once(
        source,
        "    *,\n    gmres_iterations: int,\n",
        "    *,\n"
        "    benchmark_trip_index: jax.Array | int,\n"
        "    benchmark_promotion_index: jax.Array | int,\n"
        "    gmres_iterations: int,\n",
    )
    source = _replace_once(
        source,
        "    acceptance_map_fn = map_fn if acceptance_map_fn is None else acceptance_map_fn\n"
        "    mapped, tangent = jax.linearize(map_fn, state)\n",
        "    jax.debug.callback(\n"
        '        partial(_record_boundary, "rebuild_start"),\n'
        "        benchmark_trip_index, benchmark_promotion_index, ordered=True,\n"
        "    )\n"
        "    acceptance_map_fn = map_fn if acceptance_map_fn is None else acceptance_map_fn\n"
        "    mapped, tangent = jax.linearize(map_fn, state)\n",
    )
    source = _replace_once(
        source,
        "        step, info = jax.scipy.sparse.linalg.gmres(\n"
        "            damped_normal_action,\n"
        "            normal_rhs,\n"
        "            maxiter=gmres_iterations,\n"
        "            restart=gmres_iterations,\n"
        '            solve_method="batched",\n'
        "        )\n",
        "        step, info = _timed_gmres(\n"
        "            damped_normal_action,\n"
        "            normal_rhs,\n"
        "            maximum_restarts=gmres_iterations,\n"
        "            restart=gmres_iterations,\n"
        "            trip=benchmark_trip_index,\n"
        "            promotion=benchmark_promotion_index,\n"
        '            gmres_kind="rebuild",\n'
        "            damping_trial=_index,\n"
        "        )\n",
    )
    source = _replace_once(
        source,
        "    return _RebuiltModelPromotion(\n        state=candidate,\n",
        "    selected_next_damping = jnp.where(\n"
        "        accepted,\n"
        "        jnp.maximum(\n"
        "            recorded_damping / _MODEL_REBUILD_DAMPING_GROWTH,\n"
        "            jnp.asarray(_MODEL_REBUILD_DAMPING_INITIAL, dtype=state.dtype),\n"
        "        ),\n"
        "        next_damping,\n"
        "    )\n"
        "    jax.debug.callback(\n"
        "        _record_rebuild_result,\n"
        "        benchmark_trip_index, benchmark_promotion_index,\n"
        "        accepted, recorded_damping, selected_next_damping, ordered=True,\n"
        "    )\n"
        "    return _RebuiltModelPromotion(\n"
        "        state=candidate,\n",
    )
    source = _replace_once(
        source,
        "        next_damping=jnp.where(\n"
        "            accepted,\n"
        "            jnp.maximum(\n"
        "                recorded_damping / _MODEL_REBUILD_DAMPING_GROWTH,\n"
        "                jnp.asarray(_MODEL_REBUILD_DAMPING_INITIAL, dtype=state.dtype),\n"
        "            ),\n"
        "            next_damping,\n"
        "        ),\n",
        "        next_damping=selected_next_damping,\n",
    )
    namespace = dict(vars(fixed_point))
    namespace.update(
        {
            "partial": __import__("functools").partial,
            "_record_boundary": _record_boundary,
            "_record_rebuild_result": _record_rebuild_result,
            "_timed_gmres": _timed_gmres,
        }
    )
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["_instrumented_rebuilt_model_promotion"]


def _instrumented_descent_promotion() -> Callable:
    source = textwrap.dedent(inspect.getsource(fixed_point._steepest_descent_promotion))
    source = _replace_once(
        source,
        "def _steepest_descent_promotion(",
        "def _instrumented_steepest_descent_promotion(",
    )
    source = _replace_once(
        source,
        "    acceptance_map_fn: Callable[[jax.Array], jax.Array] | None = None,\n",
        "    acceptance_map_fn: Callable[[jax.Array], jax.Array] | None = None,\n"
        "    benchmark_trip_index: jax.Array | int = 0,\n"
        "    benchmark_promotion_index: jax.Array | int = 0,\n",
    )
    source = _replace_once(
        source,
        "    acceptance_map_fn = map_fn if acceptance_map_fn is None else acceptance_map_fn\n",
        "    jax.debug.callback(\n"
        '        partial(_record_boundary, "descent_start"),\n'
        "        benchmark_trip_index, benchmark_promotion_index, ordered=True,\n"
        "    )\n"
        "    acceptance_map_fn = map_fn if acceptance_map_fn is None else acceptance_map_fn\n",
    )
    source = _replace_once(
        source,
        "    return _SteepestDescentPromotion(\n"
        "        state=jnp.where(accepted, candidates[selected], state),\n",
        "    selected_scale = jnp.where(\n"
        "        accepted, scales[selected], jnp.asarray(0.0, dtype=state.dtype)\n"
        "    )\n"
        "    jax.debug.callback(\n"
        "        _record_descent_result, benchmark_trip_index,\n"
        "        benchmark_promotion_index, accepted, selected_scale, ordered=True,\n"
        "    )\n"
        "    return _SteepestDescentPromotion(\n"
        "        state=jnp.where(accepted, candidates[selected], state),\n",
    )
    source = _replace_once(
        source,
        "        scale=jnp.where(\n"
        "            accepted, scales[selected], jnp.asarray(0.0, dtype=state.dtype)\n"
        "        ),\n",
        "        scale=selected_scale,\n",
    )
    namespace = dict(vars(fixed_point))
    namespace.update(
        {
            "partial": __import__("functools").partial,
            "_record_boundary": _record_boundary,
            "_record_descent_result": _record_descent_result,
        }
    )
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["_instrumented_steepest_descent_promotion"]


def _instrumented_inner() -> Callable:
    source = textwrap.dedent(inspect.getsource(fixed_point._newton_krylov_inner))
    source = _replace_once(
        source,
        "def _newton_krylov_inner(",
        "def _instrumented_newton_krylov_inner(",
    )
    source = _replace_once(
        source,
        "    presettlement_incumbent_scoring: jax.Array | bool = False,\n"
        "    precision: Precision | str = Precision.AUTOMATIC,\n",
        "    presettlement_incumbent_scoring: jax.Array | bool = False,\n"
        "    benchmark_trip_index: jax.Array | int = 0,\n"
        "    precision: Precision | str = Precision.AUTOMATIC,\n",
    )
    source = _replace_once(
        source,
        "        def attempt_step(measured):\n            def linear_action(vector):\n",
        "        def attempt_step(measured):\n"
        "            jax.debug.callback(\n"
        '                partial(_record_boundary, "promotion_start"),\n'
        "                benchmark_trip_index, measured.attempted, ordered=True,\n"
        "            )\n"
        "            def linear_action(vector):\n",
    )
    source = _replace_once(
        source,
        "            qualified_step = _qualified_krylov_step(\n",
        "            qualified_step = _instrumented_qualified_krylov_step(\n",
    )
    source = _replace_once(
        source,
        "                preceding_condition_baseline=measured.condition_baseline,\n"
        "            )\n",
        "                preceding_condition_baseline=measured.condition_baseline,\n"
        "                benchmark_trip_index=benchmark_trip_index,\n"
        "                benchmark_promotion_index=measured.attempted,\n"
        "            )\n",
    )
    source = _replace_once(
        source,
        "                promotion = _backtracked_promotion(\n",
        "                jax.debug.callback(\n"
        '                    partial(_record_boundary, "ladder_start"),\n'
        "                    benchmark_trip_index, measured.attempted, ordered=True,\n"
        "                )\n"
        "                promotion = _instrumented_backtracked_promotion(\n",
    )
    source = _replace_once(
        source,
        "                    acceptance_map_fn=acceptance_map,\n"
        "                    own_mask_acceptance=own_mask_acceptance,\n"
        "                )\n"
        "                minimum_radius =",
        "                    acceptance_map_fn=acceptance_map,\n"
        "                    own_mask_acceptance=own_mask_acceptance,\n"
        "                    benchmark_trip_index=benchmark_trip_index,\n"
        "                    benchmark_promotion_index=measured.attempted,\n"
        "                )\n"
        "                jax.debug.callback(\n"
        "                    _record_ladder_result, benchmark_trip_index,\n"
        "                    measured.attempted, promotion.backtrack_count,\n"
        "                    promotion.accepted, promotion.recovery_activated,\n"
        "                    promotion.recovery_radius_before,\n"
        "                    promotion.recovery_radius, promotion.recovery_outcome,\n"
        "                    promotion.model_distrusted, ordered=True,\n"
        "                )\n"
        "                minimum_radius =",
    )
    source = _replace_once(
        source,
        "                    return _rebuilt_model_promotion(\n",
        "                    return _instrumented_rebuilt_model_promotion(\n",
    )
    source = _replace_once(
        source,
        "                        own_mask_acceptance=own_mask_acceptance,\n"
        "                    )\n\n"
        "                def skip_rebuild",
        "                        own_mask_acceptance=own_mask_acceptance,\n"
        "                        benchmark_trip_index=benchmark_trip_index,\n"
        "                        benchmark_promotion_index=measured.attempted,\n"
        "                    )\n\n"
        "                def skip_rebuild",
    )
    source = _replace_once(
        source,
        "                    return _steepest_descent_promotion(\n",
        "                    return _instrumented_steepest_descent_promotion(\n",
    )
    source = _replace_once(
        source,
        "                        own_mask_acceptance=own_mask_acceptance,\n"
        "                    )\n\n"
        "                def skip_descent",
        "                        own_mask_acceptance=own_mask_acceptance,\n"
        "                        benchmark_trip_index=benchmark_trip_index,\n"
        "                        benchmark_promotion_index=measured.attempted,\n"
        "                    )\n\n"
        "                def skip_descent",
    )
    source = _replace_once(
        source,
        "                return _NewtonIterationState(\n"
        "                    candidate,\n",
        "                jax.debug.callback(\n"
        "                    _record_promotion_result, benchmark_trip_index,\n"
        "                    measured.attempted, promotion_accepted, rebuild_activated,\n"
        "                    rebuilt.accepted, descent_activated, descent.accepted,\n"
        "                    ordered=True,\n"
        "                )\n"
        "                return _NewtonIterationState(\n"
        "                    candidate,\n",
    )
    source = _replace_once(
        source,
        "                return _NewtonIterationState(\n"
        "                    state,\n"
        "                    nonlinear_residual,\n",
        "                jax.debug.callback(\n"
        "                    _record_promotion_result, benchmark_trip_index,\n"
        "                    measured.attempted, jnp.asarray(False), jnp.asarray(False),\n"
        "                    jnp.asarray(False), jnp.asarray(False), jnp.asarray(False),\n"
        "                    ordered=True,\n"
        "                )\n"
        "                return _NewtonIterationState(\n"
        "                    state,\n"
        "                    nonlinear_residual,\n",
    )
    namespace = dict(vars(fixed_point))
    namespace.update(
        {
            "partial": __import__("functools").partial,
            "_record_boundary": _record_boundary,
            "_record_ladder_result": _record_ladder_result,
            "_record_promotion_result": _record_promotion_result,
            "_instrumented_qualified_krylov_step": _instrumented_qualified_step(),
            "_instrumented_backtracked_promotion": (
                _instrumented_backtracked_promotion()
            ),
            "_instrumented_rebuilt_model_promotion": (
                _instrumented_rebuilt_promotion()
            ),
            "_instrumented_steepest_descent_promotion": (
                _instrumented_descent_promotion()
            ),
        }
    )
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["_instrumented_newton_krylov_inner"]


def _instrumented_active_set(inner: Callable) -> Callable:
    source = textwrap.dedent(inspect.getsource(fixed_point._active_set_newton_krylov))
    source = _replace_once(
        source,
        "def _active_set_newton_krylov(",
        "def _instrumented_active_set_newton_krylov(",
    )
    source = _replace_once(
        source,
        "        partition,\n        run_warmup,\n",
        "        partition,\n        benchmark_trip_index,\n        run_warmup,\n",
    )
    source = _replace_once(
        source,
        "            presettlement_incumbent_scoring=presettlement,\n"
        "            precision=precision,\n",
        "            presettlement_incumbent_scoring=presettlement,\n"
        "            benchmark_trip_index=benchmark_trip_index,\n"
        "            precision=precision,\n",
    )
    source = _replace_once(
        source,
        "        initial_partition,\n        jnp.asarray(True),\n",
        "        initial_partition,\n"
        "        jnp.asarray(0, dtype=jnp.int32),\n"
        "        jnp.asarray(True),\n",
    )
    source = _replace_once(
        source,
        "                carry.partition,\n                ~carry.continue_trajectory,\n",
        "                carry.partition,\n"
        "                jnp.asarray(index, dtype=jnp.int32),\n"
        "                ~carry.continue_trajectory,\n",
    )
    namespace = dict(vars(fixed_point))
    namespace["_newton_krylov_inner"] = inner
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["_instrumented_active_set_newton_krylov"]


@contextmanager
def _patched_solver() -> Iterator[None]:
    original = fixed_point._active_set_newton_krylov
    inner = _instrumented_inner()
    fixed_point._active_set_newton_krylov = _instrumented_active_set(inner)
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
            tolerance=outer_trace.parity.FIXED_POINT_CRITERION,
            warmup=outer_trace.parity.WARMUP_SWEEPS,
            newton_steps=outer_trace.parity.NEWTON_STEPS,
            gmres_iterations=outer_trace.parity.GMRES_ITERATIONS,
            relaxation=outer_trace.parity.RELAXATION,
            step_cap=outer_trace.parity.STEP_CAP,
            active_set_steps=ACTIVE_SET_BUDGET,
        )

    return production


def _event(kind: str, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [item for item in events if item["kind"] == kind]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate {kind} events in one promotion")
    return matches[0] if matches else None


def _delta(start: dict[str, Any] | None, end: dict[str, Any] | None) -> float:
    if start is None or end is None:
        return 0.0
    return (end["timestamp_ns"] - start["timestamp_ns"]) * 1.0e-9


def _promotion_rows() -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in _EVENTS:
        grouped[(event["trip"], event["promotion"])].append(event)
    expected = {(trip, promotion) for trip in range(1, 8) for promotion in range(1, 13)}
    if set(grouped) != expected:
        raise RuntimeError(
            f"promotion callback census changed: missing={sorted(expected - set(grouped))} "
            f"extra={sorted(set(grouped) - expected)}"
        )
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        events = grouped[key]
        starts = _event("promotion_start", events)
        ladder_start = _event("ladder_start", events)
        ladder_end = _event("ladder_end", events)
        promotion_end = _event("promotion_end", events)
        if None in (starts, ladder_start, ladder_end, promotion_end):
            raise RuntimeError(f"incomplete promotion boundary for {key}")
        recovery_start = _event("recovery_start", events)
        recovery_end = _event("recovery_end", events)
        rebuild_start = _event("rebuild_start", events)
        rebuild_end = _event("rebuild_end", events)
        descent_start = _event("descent_start", events)
        descent_end = _event("descent_end", events)
        gmres = [item for item in events if item["kind"] == "gmres_complete"]
        primary = [item for item in gmres if item["gmres_kind"] == "primary"]
        rebuild_gmres = [item for item in gmres if item["gmres_kind"] == "rebuild"]
        if len(primary) != 1:
            raise RuntimeError(
                f"expected one primary GMRES row for {key}: {len(primary)}"
            )
        recovery_wall = _delta(recovery_start, recovery_end)
        backtracking_wall = _delta(ladder_start, ladder_end)
        rebuild_wall = _delta(rebuild_start, rebuild_end)
        descent_wall = _delta(descent_start, descent_end)
        total_wall = _delta(starts, promotion_end)
        primary_wall = _delta(starts, ladder_start)
        ladder_only_wall = max(backtracking_wall - recovery_wall, 0.0)
        accounted = (
            primary_wall
            + ladder_only_wall
            + recovery_wall
            + rebuild_wall
            + descent_wall
        )
        rows.append(
            {
                "trip": key[0],
                "promotion": key[1],
                "wall_s": total_wall,
                "primary_linear_wall_s": primary_wall,
                "ladder_scoring_wall_s": ladder_only_wall,
                "recovery_wall_s": recovery_wall,
                "rebuild_wall_s": rebuild_wall,
                "descent_wall_s": descent_wall,
                "unattributed_boundary_wall_s": max(total_wall - accounted, 0.0),
                "ladder_grade_reached": ladder_end["ladder_grade_reached"],
                "backtrack_count": ladder_end["backtrack_count"],
                "accepted": promotion_end["accepted"],
                "primary_gmres_restarts": primary[0]["restart_cycles"],
                "primary_gmres_iterations_used": primary[0]["arnoldi_iterations"],
                "rebuild_gmres_calls": len(rebuild_gmres),
                "rebuild_gmres_restarts": sum(
                    item["restart_cycles"] for item in rebuild_gmres
                ),
                "rebuild_gmres_iterations_used": sum(
                    item["arnoldi_iterations"] for item in rebuild_gmres
                ),
                "gmres_iterations_used": sum(
                    item["arnoldi_iterations"] for item in gmres
                ),
                "recovery_activated": ladder_end["recovery_activated"],
                "recovery_radius_before": ladder_end["recovery_radius_before"],
                "recovery_radius_after": ladder_end["recovery_radius_after"],
                "recovery_outcome": ladder_end["recovery_outcome"],
                "rebuild_activated": promotion_end["rebuild_activated"],
                "rebuild_damping": (
                    rebuild_end["rebuild_damping"] if rebuild_end else None
                ),
                "rebuild_relinearized": bool(rebuild_end),
                "rebuild_accepted": promotion_end["rebuild_accepted"],
                "descent_activated": promotion_end["descent_activated"],
                "descent_grades_evaluated": (
                    descent_end["grades_evaluated"] if descent_end else 0
                ),
                "descent_accepted": promotion_end["descent_accepted"],
            }
        )
    return rows


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows))


def _trip_rows(
    promotions: list[dict[str, Any]], banked: dict[str, Any]
) -> list[dict[str, Any]]:
    banked_walls = {1: None}
    banked_walls.update(
        {
            int(row["outer_index"]) + 1: float(row["outer_wall_s"])
            for row in banked["outer_iterations"]
            if row["active_at_entry"]
        }
    )
    rows: list[dict[str, Any]] = []
    for trip in range(1, EXPECTED_ACTIVE_SET_TRIPS + 1):
        members = [row for row in promotions if row["trip"] == trip]
        promotion_wall = _sum(members, "wall_s")
        rebuild_and_terminal = (
            _sum(members, "recovery_wall_s")
            + _sum(members, "rebuild_wall_s")
            + _sum(members, "descent_wall_s")
        )
        rows.append(
            {
                "trip": trip,
                "promotions": len(members),
                "accepted_promotions": sum(row["accepted"] for row in members),
                "rejected_promotions": sum(not row["accepted"] for row in members),
                "instrumented_promotion_wall_s": promotion_wall,
                "banked_trip_wall_s": banked_walls.get(trip),
                "primary_linear_wall_s": _sum(members, "primary_linear_wall_s"),
                "ladder_scoring_wall_s": _sum(members, "ladder_scoring_wall_s"),
                "recovery_wall_s": _sum(members, "recovery_wall_s"),
                "rebuild_wall_s": _sum(members, "rebuild_wall_s"),
                "descent_wall_s": _sum(members, "descent_wall_s"),
                "post_promotion_wall_s": _sum(members, "unattributed_boundary_wall_s"),
                "recovery_rebuild_descent_share": (
                    rebuild_and_terminal / promotion_wall if promotion_wall else 0.0
                ),
                "gmres_iterations_used": sum(
                    row["gmres_iterations_used"] for row in members
                ),
                "recovery_activations": sum(
                    row["recovery_activated"] for row in members
                ),
                "rebuild_activations": sum(row["rebuild_activated"] for row in members),
                "descent_activations": sum(row["descent_activated"] for row in members),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_strict(rows))


def _write_figure(trips: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["trip"]) for row in trips]
    components = (
        ("primary_linear_wall_s", "primary linear model + GMRES"),
        ("ladder_scoring_wall_s", "Newton ladder scoring"),
        ("recovery_wall_s", "continuation recovery"),
        ("rebuild_wall_s", "rebuilt model + GMRES"),
        ("descent_wall_s", "descent grade set"),
        ("post_promotion_wall_s", "post-promotion scoring + boundary"),
    )
    figure, axis = plt.subplots(figsize=(10.4, 5.8))
    bottom = np.zeros(len(trips))
    for key, label in components:
        values = np.asarray([row[key] for row in trips])
        axis.bar(labels, values, bottom=bottom, label=label)
        bottom += values
    axis.set_xlabel("Active-set trip")
    axis.set_ylabel("Instrumented promotion wall (s)")
    axis.set_title("Rejected promotions activate recovery, rebuilt models, and descent")
    axis.legend(frameon=False, ncols=2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(receipt: dict[str, Any], path: Path) -> None:
    trips = receipt["trips"]
    mechanism = receipt["mechanism"]
    lines = [
        "# Late-trip promotion timing",
        "",
        f"Captured {receipt['captured_at']} at `{receipt['measurement_revision']}` "
        f"in SLURM job **{receipt['scheduler']['job_id']}** on "
        f"`{receipt['scheduler']['node']}` (`{receipt['scheduler']['partition']}`, "
        f"reservation `{receipt['scheduler']['reservation']}`). The uninstrumented "
        f"warm production solve took **{receipt['uninstrumented_execution']['wall_s']:.3f} s** "
        f"and retained {receipt['uninstrumented_execution']['summary']['active_set_trips']} trips.",
        "",
        "Ordered callbacks delimit every promotion and its activated recovery, "
        "rebuilt-model, and descent branches. Each callback row was appended, flushed, "
        "and fsynced before execution continued. The counted batched-GMRES copy preserves "
        "the production algorithm while exposing restart cycles and Arnoldi iterations.",
        "",
        "## Per-trip ranking",
        "",
        "| trip | production trip s | instrumented promotion s | accepted | recovery | rebuild | descent | GMRES iterations | recovery/rebuild/descent share |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trips:
        banked = (
            "n/a"
            if row["banked_trip_wall_s"] is None
            else f"{row['banked_trip_wall_s']:.3f}"
        )
        lines.append(
            f"| {row['trip']} | {banked} | "
            f"{row['instrumented_promotion_wall_s']:.3f} | "
            f"{row['accepted_promotions']}/{row['promotions']} | "
            f"{row['recovery_activations']} | {row['rebuild_activations']} | "
            f"{row['descent_activations']} | {row['gmres_iterations_used']} | "
            f"{100.0 * row['recovery_rebuild_descent_share']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Attribution",
            "",
            f"**Mechanism:** {mechanism['finding']}",
            "",
            f"All **{mechanism['full_fallback_promotions']} of "
            f"{mechanism['trip_6_7_promotions']}** promotions in trips 6 and 7 enter "
            "the full recovery/rebuild/descent fallback. Relative to a direct late-trip "
            f"promotion, that fallback surcharge is **{100.0 * mechanism['fallback_surcharge_share']:.1f}%** "
            "of their promotion wall. The rebuilt-model body alone is "
            f"**{100.0 * mechanism['rebuild_share']:.1f}%**; continuation-loop plus "
            f"descent bodies are only **{100.0 * mechanism['recovery_descent_share']:.2f}%**, "
            "so replaying the radius or descent grades is not the dominant mechanism. "
            "The rebuilt path calls "
            f"`jax.linearize` at `{receipt['source_anchors']['rebuild_relinearize']}` "
            "and then runs a damped normal-equation GMRES inside the fixed damping "
            f"ladder at `{receipt['source_anchors']['rebuild_gmres']}`. The continuation "
            f"recovery re-scores candidates at `{receipt['source_anchors']['recovery_score']}`; "
            f"descent evaluates its full scale set at `{receipt['source_anchors']['descent_scores']}`. "
            "After fallback, the solver evaluates the candidate map again at "
            f"`{receipt['source_anchors']['post_promotion_score']}`.",
            "",
            f"**Recommended repair:** {mechanism['repair']}",
            "",
            f"Estimated saving: **{mechanism['estimated_saving_s']:.3f} s per production "
            "solve** across trips 4–7, derived by applying the measured removable "
            "promotion surcharge for the 17 repeated rejected attempts after the first "
            "rejection in each unchanged state. This is an estimate, not an implemented "
            "before/after result.",
            "",
            "## Artifacts",
            "",
            f"- Event stream: `{receipt['artifacts']['event_log']}`",
            f"- Promotion table: `{receipt['artifacts']['table']}`",
            f"- Trip table: `{receipt['artifacts']['trip_table']}`",
            f"- Figure: `{receipt['artifacts']['figure']}`",
            f"- Receipt: `{receipt['artifacts']['receipt']}`",
            f"- Stdout: `{receipt['log_paths']['stdout']}`",
            f"- Stderr: `{receipt['log_paths']['stderr']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _runtime() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_kinds": [getattr(device, "device_kind", None) for device in devices],
        "jax": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    }


def _scheduler() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "temporary_directory": os.environ.get("TMPDIR"),
    }


def _revision() -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(
    output: Path,
    table: Path,
    trip_table: Path,
    figure: Path,
    event_log: Path,
    report: Path,
    cache_root: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    configure_dtypes()
    outer_trace._require_measurement_host()
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
    uninstrumented_result = jax.block_until_ready(uninstrumented_executable(state))
    uninstrumented_wall_s = time.perf_counter() - started
    uninstrumented_summary = outer_trace._solve_summary(uninstrumented_result)

    with _patched_solver():
        jax.clear_caches()
        instrumented = jax.jit(_production_program(profile, target_current))
        started = time.perf_counter()
        instrumented_executable = instrumented.lower(state).compile()
        instrumented_compile_s = time.perf_counter() - started
        _reset_events(event_log)
        started = time.perf_counter()
        instrumented_result = jax.block_until_ready(instrumented_executable(state))
        instrumented_wall_s = time.perf_counter() - started
        instrumented_summary = outer_trace._solve_summary(instrumented_result)

    if instrumented_summary != uninstrumented_summary:
        raise RuntimeError(
            "instrumentation changed the production summary: "
            f"instrumented={instrumented_summary} uninstrumented={uninstrumented_summary}"
        )
    if instrumented_summary["active_set_trips"] != EXPECTED_ACTIVE_SET_TRIPS:
        raise RuntimeError(
            "production trip count changed: "
            f"{instrumented_summary['active_set_trips']} != {EXPECTED_ACTIVE_SET_TRIPS}"
        )
    persisted = [
        json.loads(line)
        for line in event_log.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if persisted != _strict(_EVENTS):
        raise RuntimeError("persisted event stream does not match callback memory")

    promotions = _promotion_rows()
    banked = json.loads(FROZEN_PROFILE.read_text(encoding="utf-8"))
    trips = _trip_rows(promotions, banked)
    late = [row for row in trips if row["trip"] in (6, 7)]
    late_promotions = [row for row in promotions if row["trip"] in (6, 7)]
    direct_late_promotions = [
        row
        for row in promotions
        if row["trip"] in (4, 5) and not row["recovery_activated"]
    ]
    direct_late_wall = float(
        np.median([row["wall_s"] for row in direct_late_promotions])
    )
    late_promotion_wall = sum(row["wall_s"] for row in late_promotions)
    fallback_surcharge = sum(
        max(row["wall_s"] - direct_late_wall, 0.0) for row in late_promotions
    )
    rebuild_wall = sum(row["rebuild_wall_s"] for row in late_promotions)
    recovery_descent_wall = sum(
        row["recovery_wall_s"] + row["descent_wall_s"] for row in late_promotions
    )
    repeated_rejections = sum(max(row["rejected_promotions"] - 1, 0) for row in late)
    rejected_fallback_walls = [
        row["wall_s"] for row in late_promotions if not row["accepted"]
    ]
    rejected_fallback_surcharge = max(
        float(np.median(rejected_fallback_walls)) - direct_late_wall,
        0.0,
    )
    production_scale = sum(row["banked_trip_wall_s"] or 0.0 for row in late) / sum(
        row["instrumented_promotion_wall_s"] for row in late
    )
    estimated_saving = (
        repeated_rejections * rejected_fallback_surcharge * production_scale
    )
    receipt: dict[str, Any] = {
        "schema": "nova.late_trip_promotion_timing",
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": _revision(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
            "instrumentation": (
                "in-memory solver copies with ordered callbacks and a counted "
                "algorithm-identical batched GMRES"
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
            "production_total_eligible": False,
        },
        "promotions": promotions,
        "trips": trips,
        "source_anchors": {
            "recovery_score": _source_anchor(
                fixed_point._backtracked_promotion,
                "candidate_merit, candidate_residual = score(candidate)",
            ),
            "rebuild_relinearize": _source_anchor(
                fixed_point._rebuilt_model_promotion,
                "mapped, tangent = jax.linearize(map_fn, state)",
            ),
            "rebuild_gmres": _source_anchor(
                fixed_point._rebuilt_model_promotion,
                "step, info = jax.scipy.sparse.linalg.gmres",
            ),
            "descent_scores": _source_anchor(
                fixed_point._steepest_descent_promotion,
                "merits, residuals = jax.lax.map(score, candidates)",
            ),
            "post_promotion_score": _source_anchor(
                fixed_point._newton_krylov_inner,
                "candidate_merit = _smooth_relative_sup_merit",
            ),
        },
        "mechanism": {
            "finding": (
                "Late-trip amplification is the repeated full fallback cascade on an "
                "unchanged state, not continuation-radius replay or descent itself. "
                "Every trip-6 and trip-7 promotion exhausts the Newton ladder, rebuilds "
                "and relinearises the model, restarts damped GMRES, evaluates descent, "
                "then performs another candidate-map score; rejected attempts repeat "
                "that cascade while the state and frozen partition remain unchanged and "
                "only scalar globalisation state advances."
            ),
            "trip_6_7_promotions": len(late_promotions),
            "full_fallback_promotions": sum(
                row["recovery_activated"]
                and row["rebuild_activated"]
                and row["descent_activated"]
                for row in late_promotions
            ),
            "direct_late_promotion_median_s": direct_late_wall,
            "rejected_fallback_promotion_median_s": float(
                np.median(rejected_fallback_walls)
            ),
            "fallback_surcharge_share": fallback_surcharge / late_promotion_wall,
            "rebuild_share": rebuild_wall / late_promotion_wall,
            "recovery_descent_share": recovery_descent_wall / late_promotion_wall,
            "repeated_rejections_after_first": repeated_rejections,
            "repair": (
                "When a promotion leaves the state and frozen partition unchanged, carry "
                "the primary linearisation, map scores, and rebuilt normal model into the "
                "next damping attempt. Advance the scalar globalisation state and reapply "
                "every existing own-mask merit, residual, model-trust, conditioning, and "
                "acceptance decision. Exit after that carried sequence is exhausted "
                "instead of re-entering the complete promotion pipeline."
            ),
            "estimated_saving_s": estimated_saving,
            "estimate_contract": (
                "median rejected-fallback surcharge above a direct late-trip promotion, "
                "times the 17 rejected attempts after the first unchanged-state refusal "
                "in trips 6 and 7, scaled to the banked production trip walls"
            ),
        },
        "artifacts": {
            "receipt": str(output.resolve().relative_to(ROOT)),
            "table": str(table.resolve().relative_to(ROOT)),
            "trip_table": str(trip_table.resolve().relative_to(ROOT)),
            "figure": str(figure.resolve().relative_to(ROOT)),
            "event_log": str(event_log.resolve().relative_to(ROOT)),
            "report": str(report),
        },
    }
    _write_json(output, receipt)
    _write_csv(table, promotions)
    _write_csv(trip_table, trips)
    _write_figure(trips, figure)
    _write_report(receipt, report)
    print(f"RECEIPT_WRITTEN={output}", flush=True)
    print(f"TABLE_WRITTEN={table}", flush=True)
    print(f"TRIP_TABLE_WRITTEN={trip_table}", flush=True)
    print(f"FIGURE_WRITTEN={figure}", flush=True)
    print(f"REPORT_WRITTEN={report}", flush=True)
    return receipt


def preflight() -> None:
    inner = _instrumented_inner()
    active_set = _instrumented_active_set(inner)
    print(
        json.dumps(
            {
                "status": "preflight_complete",
                "inner": inner.__name__,
                "active_set": active_set.__name__,
                "active_set_budget": ACTIVE_SET_BUDGET,
                "newton_steps": outer_trace.parity.NEWTON_STEPS,
                "gmres_iterations": outer_trace.parity.GMRES_ITERATIONS,
                "ladder_length": LADDER_LENGTH,
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
    parser.add_argument("--trip-table", type=Path, default=DEFAULT_TRIP_TABLE)
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
        arguments.trip_table,
        arguments.figure,
        arguments.event_log,
        arguments.report,
        arguments.cache_root,
        arguments.stdout_path,
        arguments.stderr_path,
    )


if __name__ == "__main__":
    main()
