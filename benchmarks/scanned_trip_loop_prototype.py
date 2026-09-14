"""Measure the cost of replacing the outer active-set trip loop with a scan.

One whole-cell forward solve drives its bounded outer trip loop either with the
production ``jax.lax.fori_loop`` coordination or with an otherwise-identical
``jax.lax.scan`` over the same fixed number of trips.  The per-trip body
(frozen-mask Newton-Krylov solve, live-mask reconcile, next-partition read),
the carried state and the fixed telemetry slots are transcribed unchanged from
the production body in ``nova/equilibrium/fixed_point.py``; only the loop
driver changes.  Each arm is compiled on the same seed, then both are executed
from that seed so their terminal states and per-trip histories can be compared
side by side.

Part receipts are written under ``docs/figures/...`` as each arm lands so a
job expiry loses one arm rather than the run.  ``--combine`` reads an executed
pair from one platform and writes the comparison receipt and one SVG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
from time import perf_counter
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.solovev_certificate import (
    AXIS_M,
    _case,
    _certificate_compile_problem,
    _compiled_memory_fields,
    _is_diverted_case,
    _topology,
    _write_json,
)
from nova.equilibrium import fixed_point
from nova.equilibrium.fixed_point import (
    FixedPointResult,
    FixedPointTerminationReason,
    _ACTIVE_SET_CYCLE_DAMPING,
    _ActiveSetIterationState,
    _PROJECTED_KRYLOV_CONDITION_RATIO_LIMIT,
    _newton_krylov_inner,
    _print_active_set_trip,
    _relative_residual,
    _smooth_relative_sup_merit,
    _solver_state,
)
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.jax.config import (
    Precision,
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

try:
    from nova.media.ink import DEFAULT_INK
except Exception:  # pragma: no cover - rendering fallback only
    DEFAULT_INK = None

CASE = "weak-rotation-reactor-static"
REQUESTED_CELLS = -300

ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "docs/figures/millisecond-converged-solve/scan-prototype"
PARTS_ROOT = FIGURE_ROOT / "parts"


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _platform() -> str:
    backend = jax.default_backend()
    return "cuda" if backend in {"gpu", "cuda"} else "cpu"


def _lane() -> dict[str, Any]:
    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_default_backend": jax.default_backend(),
        "clip_mode": support_clip_mode(),
        "requested_cells": abs(REQUESTED_CELLS),
    }


def _hlo_instruction_count(hlo: str) -> int:
    """Count optimized HLO instructions: every result-producing line."""

    pattern = re.compile(r"^\s*(?:ROOT )?%\S+ = ")
    return sum(1 for line in hlo.splitlines() if pattern.match(line))


def _try_serialized_executable_size(compiled: Any) -> dict[str, Any]:
    """Report the AOT binary when the backend exposes serialisation."""

    try:
        executable = compiled.runtime_executable()
        serialized = executable.serialize()
        generated_size = executable.size_of_generated_code_in_bytes
        if callable(generated_size):
            generated_size = generated_size()
        generated = int(generated_size)
        return {
            "serialized_executable_bytes": len(serialized),
            "serialized_executable_mib": len(serialized) / 2**20,
            "size_of_generated_code_bytes": generated,
            "serialisation_error": None,
        }
    except Exception as error:  # backend may not offer serialisation
        return {
            "serialized_executable_bytes": None,
            "serialized_executable_mib": None,
            "size_of_generated_code_bytes": None,
            "serialisation_error": f"{type(error).__name__}: {error}",
        }


def _state_digest(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Scanned outer trip loop.
#
# The body below is the production ``_active_set_newton_krylov`` from
# ``nova/equilibrium/fixed_point.py`` (lines 3364-3872 at the base revision),
# transcribed unchanged except for the coordination line: the trips run under
# ``lax.scan`` over ``arange(1, active_set_steps)`` instead of
# ``fori_loop(1, active_set_steps, outer_body, outer)``.  The per-trip body,
# the carry and the telemetry slots are otherwise identical.
# ---------------------------------------------------------------------------


def scanned_active_set_newton_krylov(
    initial: jax.Array,
    *,
    newton_steps: int,
    gmres_iterations: int,
    warmup: int,
    relaxation: float,
    step_cap: float,
    krylov_condition_limit: float,
    convergence_tolerance: float,
    shadow_mask_fn: Callable[[jax.Array], jax.Array],
    promoted_shadow_mask_fn: Callable[[jax.Array, jax.Array], jax.Array],
    shadowed_map_fn: Callable[[jax.Array, jax.Array], jax.Array],
    active_set_steps: int,
    stream_active_set: bool,
    stream_inner_iterations: bool,
    stop_on_active_set_stagnation: bool,
    stop_on_active_set_settlement: bool,
    retain_outer_best_iterate: bool,
    continue_newton_trajectory: bool,
    continue_globalization_state: bool,
    model_trust_selection: bool,
    own_mask_acceptance: bool,
    presettlement_incumbent_scoring: bool,
    precision: Precision | str,
) -> FixedPointResult:
    """Reconcile bounded frozen-mask solves with their live active sets."""

    initial = _solver_state(initial, precision)
    partition_read = getattr(shadowed_map_fn, "_read_frozen_partition", None)
    partitioned_map = getattr(shadowed_map_fn, "_map_frozen_partition", None)
    partition_shadow = getattr(shadowed_map_fn, "_frozen_partition_shadow", None)
    freeze_topology = (
        partition_read is not None
        and partitioned_map is not None
        and partition_shadow is not None
    )
    if freeze_topology:
        initial_partition = partition_read(initial, None)
        initial_mask = jnp.ravel(
            jnp.asarray(partition_shadow(initial_partition), dtype=bool)
        )
    else:
        initial_mask = jnp.ravel(jnp.asarray(shadow_mask_fn(initial), dtype=bool))
        initial_partition = initial_mask
    history = jnp.zeros((active_set_steps + 1, initial_mask.size), dtype=bool)
    history = history.at[0].set(initial_mask)

    def solve_frozen(
        state,
        mask,
        partition,
        run_warmup,
        globalization_state=None,
        resume_globalization=False,
        presettlement=False,
    ):
        def frozen_map(candidate):
            if freeze_topology:
                return partitioned_map(candidate, partition)
            return shadowed_map_fn(candidate, mask)

        def frozen_mask(_candidate):
            return mask

        def frozen_acceptance_map(candidate, _mask):
            return frozen_map(candidate)

        def frozen_acceptance_mask(_candidate, _previous):
            return mask

        acceptance_mask = (
            frozen_acceptance_mask if freeze_topology else promoted_shadow_mask_fn
        )
        acceptance_map = frozen_acceptance_map if freeze_topology else shadowed_map_fn

        return _newton_krylov_inner(
            frozen_map,
            state,
            newton_steps=newton_steps,
            gmres_iterations=gmres_iterations,
            warmup=warmup,
            relaxation=relaxation,
            step_cap=step_cap,
            krylov_condition_limit=krylov_condition_limit,
            convergence_tolerance=convergence_tolerance,
            shadow_mask_fn=frozen_mask,
            stream_inner_iterations=stream_inner_iterations,
            run_warmup=run_warmup,
            globalization_state=globalization_state,
            resume_globalization=resume_globalization,
            return_globalization_state=True,
            model_trust_selection=model_trust_selection,
            acceptance_shadow_mask_fn=acceptance_mask,
            acceptance_shadowed_map_fn=acceptance_map,
            own_mask_acceptance=own_mask_acceptance,
            presettlement_incumbent_scoring=presettlement,
            precision=precision,
        )

    def mask_seen(mask, mask_history, history_count):
        populated = jnp.arange(mask_history.shape[0]) < history_count
        matches = jnp.all(mask_history == mask[None, :], axis=1)
        return jnp.any(populated & matches)

    def reconcile(
        index,
        state,
        mask,
        partition,
        mask_history,
        history_count,
        inner_result,
        inner_globalization,
        trip_active,
        previous_live_residual,
    ):
        solved_state = inner_result.state
        if freeze_topology:
            observed_partition = partition_read(solved_state, mask)
            observed_mask = jnp.ravel(
                jnp.asarray(partition_shadow(observed_partition), dtype=bool)
            )
        else:
            observed_mask = jnp.ravel(promoted_shadow_mask_fn(solved_state, mask))
            observed_partition = observed_mask
        observed_difference = jnp.sum(observed_mask != mask, dtype=jnp.int32)
        observed_mapped = (
            partitioned_map(solved_state, observed_partition)
            if freeze_topology
            else shadowed_map_fn(solved_state, observed_mask)
        )
        observed_residual = _relative_residual(observed_mapped, solved_state)
        observed_finite = jnp.isfinite(observed_residual)
        converged = (
            observed_finite
            & (observed_residual <= convergence_tolerance)
            & (observed_difference == 0)
        )
        repeated = (
            (observed_difference > 0)
            & mask_seen(observed_mask, mask_history, history_count)
            & ~converged
        )

        damped_state = state + _ACTIVE_SET_CYCLE_DAMPING * (solved_state - state)
        if freeze_topology:
            damped_mask = observed_mask
            damped_mapped = partitioned_map(damped_state, observed_partition)
        else:
            damped_mask = jnp.ravel(promoted_shadow_mask_fn(damped_state, mask))
            damped_mapped = shadowed_map_fn(damped_state, damped_mask)
        damped_residual = _relative_residual(damped_mapped, damped_state)
        damped_finite = jnp.isfinite(damped_residual)
        damping_repeats = mask_seen(damped_mask, mask_history, history_count)
        cycle_detected = repeated & damping_repeats

        selected_state = jnp.where(repeated, damped_state, solved_state)
        selected_mask = jnp.where(repeated, damped_mask, observed_mask)
        selected_partition = observed_partition
        selected_residual = jnp.where(repeated, damped_residual, observed_residual)
        selected_finite = jnp.where(repeated, damped_finite, observed_finite)
        selected_difference = jnp.sum(selected_mask != mask, dtype=jnp.int32)
        incoming_mapped = (
            partitioned_map(state, partition)
            if freeze_topology
            else shadowed_map_fn(state, mask)
        )
        incoming_residual = _relative_residual(incoming_mapped, state)
        incoming_merit = _smooth_relative_sup_merit(incoming_mapped, state)
        selected_mapped = (
            partitioned_map(selected_state, selected_partition)
            if freeze_topology
            else shadowed_map_fn(selected_state, selected_mask)
        )
        selected_merit = _smooth_relative_sup_merit(selected_mapped, selected_state)
        retain_incoming = (
            retain_outer_best_iterate
            & (selected_difference == 0)
            & jnp.isfinite(incoming_residual)
            & jnp.isfinite(incoming_merit)
            & (~jnp.isfinite(selected_merit) | (selected_merit > incoming_merit))
        )
        trajectory_state = inner_result.trajectory_state
        trajectory_mask = (
            observed_mask
            if freeze_topology
            else jnp.ravel(promoted_shadow_mask_fn(trajectory_state, mask))
        )
        trajectory_finite = jnp.all(jnp.isfinite(trajectory_state)) & jnp.isfinite(
            inner_result.trajectory_residual
        )
        selected_state = jnp.where(retain_incoming, state, selected_state)
        selected_mask = jnp.where(retain_incoming, mask, selected_mask)
        if freeze_topology:
            selected_partition = jax.tree.map(
                lambda incoming, observed: jnp.where(
                    retain_incoming, incoming, observed
                ),
                partition,
                selected_partition,
            )
        selected_residual = jnp.where(
            retain_incoming, incoming_residual, selected_residual
        )
        selected_finite = jnp.where(retain_incoming, True, selected_finite)
        selected_difference = jnp.where(retain_incoming, 0, selected_difference)
        continue_trajectory = (
            continue_newton_trajectory
            & (selected_difference == 0)
            & ~repeated
            & trajectory_finite
            & jnp.all(trajectory_mask == mask)
            & (inner_result.accepted_newton_promotions > 0)
        )
        continue_globalization = continue_trajectory & continue_globalization_state
        converged = (
            selected_finite
            & (selected_residual <= convergence_tolerance)
            & (selected_difference == 0)
        )
        stagnated = (
            stop_on_active_set_stagnation
            & selected_finite
            & ~converged
            & ~cycle_detected
            & ~continue_trajectory
            & (selected_difference == 0)
            & (selected_residual == previous_live_residual)
        )
        settled = (
            stop_on_active_set_settlement
            & own_mask_acceptance
            & selected_finite
            & ~converged
            & ~cycle_detected
            & (selected_difference == 0)
            & (inner_result.accepted_newton_promotions == 0)
            & jnp.all(selected_state == state)
        )
        if stream_active_set:
            jax.debug.callback(
                _print_active_set_trip,
                trip_active,
                jnp.asarray(index, dtype=jnp.int32),
                selected_difference,
                selected_residual,
                jnp.asarray(inner_result.attempted_newton_promotions, dtype=jnp.int32),
                ordered=True,
            )
        can_continue = (
            selected_finite & ~converged & ~cycle_detected & ~stagnated & ~settled
        )
        record_mask = can_continue & (history_count < mask_history.shape[0])
        mask_history = mask_history.at[history_count].set(
            jnp.where(record_mask, selected_mask, mask_history[history_count])
        )
        history_count = history_count + record_mask.astype(jnp.int32)
        active = can_continue & ((index + 1) < active_set_steps)
        return (
            selected_state,
            selected_mask,
            selected_partition,
            mask_history,
            history_count,
            selected_residual,
            selected_difference,
            repeated,
            active,
            converged,
            cycle_detected,
            ~selected_finite,
            stagnated,
            settled,
            trajectory_state,
            continue_trajectory,
            inner_globalization,
            continue_globalization,
        )

    first_result, first_globalization = solve_frozen(
        initial,
        initial_mask,
        initial_partition,
        jnp.asarray(True),
        presettlement=(
            jnp.asarray(True) if presettlement_incumbent_scoring else jnp.asarray(False)
        ),
    )
    (
        first_state,
        first_mask,
        first_partition,
        history,
        history_count,
        first_residual,
        first_difference,
        first_damping,
        first_active,
        first_converged,
        first_cycle,
        first_nonfinite,
        first_stagnated,
        first_settled,
        first_trajectory_state,
        first_continue_trajectory,
        first_globalization,
        first_continue_globalization,
    ) = reconcile(
        0,
        initial,
        initial_mask,
        initial_partition,
        history,
        jnp.asarray(1, dtype=jnp.int32),
        first_result,
        first_globalization,
        jnp.asarray(True),
        jnp.asarray(jnp.nan, dtype=initial.dtype),
    )
    first_presettlement = (
        jnp.asarray(first_difference != 0)
        if presettlement_incumbent_scoring
        else jnp.asarray(False)
    )
    outer = _ActiveSetIterationState(
        state=first_state,
        mask=first_mask,
        partition=first_partition,
        mask_history=history,
        mask_history_count=history_count,
        result=first_result,
        trajectory_state=first_trajectory_state,
        continue_trajectory=first_continue_trajectory,
        globalization_state=first_globalization,
        continue_globalization=first_continue_globalization,
        live_residual=first_residual,
        residuals=jnp.full(active_set_steps, jnp.nan, dtype=initial.dtype)
        .at[0]
        .set(first_residual),
        mask_differences=jnp.full(active_set_steps, -1, dtype=jnp.int32)
        .at[0]
        .set(first_difference),
        cycle_damping_activations=jnp.full(active_set_steps, -1, dtype=jnp.int32)
        .at[0]
        .set(first_damping.astype(jnp.int32)),
        iterations=jnp.asarray(1, dtype=jnp.int32),
        attempted_promotions=jnp.asarray(
            first_result.attempted_newton_promotions, dtype=jnp.int32
        ),
        accepted_promotions=jnp.asarray(
            first_result.accepted_newton_promotions, dtype=jnp.int32
        ),
        conditioning_count=jnp.asarray(
            first_result.krylov_conditioning_count, dtype=jnp.int32
        ),
        maximum_condition=jnp.asarray(
            first_result.maximum_projected_krylov_condition, dtype=initial.dtype
        ),
        active=first_active,
        converged=first_converged,
        cycle_detected=first_cycle,
        nonfinite=first_nonfinite,
        stagnated=first_stagnated,
        settled=first_settled,
        presettlement=first_presettlement,
    )

    def outer_body(index, carry):
        def solve_active(carry):
            solve_state = jnp.where(
                carry.continue_trajectory, carry.trajectory_state, carry.state
            )
            inner_result, inner_globalization = solve_frozen(
                solve_state,
                carry.mask,
                carry.partition,
                ~carry.continue_trajectory,
                carry.globalization_state,
                carry.continue_globalization,
                presettlement=carry.presettlement,
            )
            (
                state,
                mask,
                partition,
                mask_history,
                history_count,
                live_residual,
                mask_difference,
                damping_activated,
                active,
                converged,
                cycle_detected,
                nonfinite,
                stagnated,
                settled,
                trajectory_state,
                continue_trajectory,
                globalization_state,
                continue_globalization,
            ) = reconcile(
                index,
                carry.state,
                carry.mask,
                carry.partition,
                carry.mask_history,
                carry.mask_history_count,
                inner_result,
                inner_globalization,
                carry.active,
                carry.live_residual,
            )
            next_presettlement = (
                jnp.asarray(mask_difference != 0)
                if presettlement_incumbent_scoring
                else jnp.asarray(False)
            )
            return _ActiveSetIterationState(
                state=state,
                mask=mask,
                partition=partition,
                mask_history=mask_history,
                mask_history_count=history_count,
                result=inner_result,
                trajectory_state=trajectory_state,
                continue_trajectory=continue_trajectory,
                globalization_state=globalization_state,
                continue_globalization=continue_globalization,
                live_residual=live_residual,
                residuals=carry.residuals.at[index].set(live_residual),
                mask_differences=carry.mask_differences.at[index].set(mask_difference),
                cycle_damping_activations=(
                    carry.cycle_damping_activations.at[index].set(
                        damping_activated.astype(jnp.int32)
                    )
                ),
                iterations=carry.iterations + 1,
                attempted_promotions=(
                    carry.attempted_promotions
                    + jnp.asarray(
                        inner_result.attempted_newton_promotions, dtype=jnp.int32
                    )
                ),
                accepted_promotions=(
                    carry.accepted_promotions
                    + jnp.asarray(
                        inner_result.accepted_newton_promotions, dtype=jnp.int32
                    )
                ),
                conditioning_count=(
                    carry.conditioning_count
                    + jnp.asarray(
                        inner_result.krylov_conditioning_count, dtype=jnp.int32
                    )
                ),
                maximum_condition=jnp.maximum(
                    carry.maximum_condition,
                    jnp.asarray(
                        inner_result.maximum_projected_krylov_condition,
                        dtype=initial.dtype,
                    ),
                ),
                active=active,
                converged=converged,
                cycle_detected=cycle_detected,
                nonfinite=nonfinite,
                stagnated=stagnated,
                settled=settled,
                presettlement=next_presettlement,
            )

        return jax.lax.cond(carry.active, solve_active, lambda value: value, carry)

    trips = jnp.arange(1, active_set_steps)
    outer, _ = jax.lax.scan(
        lambda carry, index: (outer_body(index, carry), ()), outer, trips
    )
    reason = jnp.where(
        outer.converged,
        FixedPointTerminationReason.CONVERGED,
        jnp.where(
            outer.cycle_detected,
            FixedPointTerminationReason.ACTIVE_SET_CYCLE_DETECTED,
            jnp.where(
                outer.nonfinite,
                FixedPointTerminationReason.NONFINITE_RESIDUAL,
                jnp.where(
                    outer.settled,
                    FixedPointTerminationReason.ACTIVE_SET_SETTLED,
                    jnp.where(
                        outer.stagnated,
                        FixedPointTerminationReason.ACTIVE_SET_STAGNATED,
                        FixedPointTerminationReason.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED,
                    ),
                ),
            ),
        ),
    )
    return outer.result._replace(
        state=outer.state,
        residual=outer.live_residual,
        attempted_newton_promotions=outer.attempted_promotions,
        accepted_newton_promotions=outer.accepted_promotions,
        krylov_conditioning_count=outer.conditioning_count,
        maximum_projected_krylov_condition=outer.maximum_condition,
        converged=outer.converged,
        termination_reason=jnp.asarray(reason, dtype=jnp.int32),
        active_set_iterations=outer.iterations,
        active_set_residuals=outer.residuals,
        active_set_mask_differences=outer.mask_differences,
        active_set_cycle_damping_activations=outer.cycle_damping_activations,
    )


# ---------------------------------------------------------------------------
# Programs and measurement.
# ---------------------------------------------------------------------------


def make_programs(profile: Any, request: Any):
    """Return the production and scanned solve programs over the same map."""

    mapped = profile.flux_map(
        request.current,
        target_current=request.target_current,
        prescribed_current=request.prescribed_current,
    )
    shadowed_map = profile.operator.flux_map_with_shadow(
        request.current,
        target_current=request.target_current,
        prescribed_current=request.prescribed_current,
    )

    def shadow_mask(state):
        return profile.operator.residual_shadow_mask(state)

    def promoted_shadow_mask(state, previous):
        return profile.operator.residual_shadow_mask(state, previous_shadow=previous)

    options = request.policy.kernel_options()

    def production_solve(initial):
        return fixed_point.newton_krylov(
            mapped,
            initial,
            shadow_mask_fn=shadow_mask,
            promoted_shadow_mask_fn=promoted_shadow_mask,
            shadowed_map_fn=shadowed_map,
            **options,
        )

    def scanned_solve(initial):
        return scanned_active_set_newton_krylov(
            initial,
            shadow_mask_fn=shadow_mask,
            promoted_shadow_mask_fn=promoted_shadow_mask,
            shadowed_map_fn=shadowed_map,
            **options,
            krylov_condition_limit=_PROJECTED_KRYLOV_CONDITION_RATIO_LIMIT,
            stream_active_set=False,
            stream_inner_iterations=False,
            model_trust_selection=True,
            presettlement_incumbent_scoring=True,
            precision=Precision.AUTOMATIC,
        )

    return production_solve, scanned_solve


def _terminal_result(
    result: FixedPointResult, operator: Any, case_exact: Any
) -> dict[str, Any]:
    state = np.asarray(result.state, dtype=np.float64)
    topology = _topology(operator, state)
    axis_reference = (
        AXIS_M if _is_diverted_case(CASE) else np.asarray(case_exact.magnetic_axis)
    )
    axis_error = None
    if topology["axis_rz_m"] is not None:
        axis_error = float(
            np.linalg.norm(np.asarray(topology["axis_rz_m"]) - axis_reference)
        )
    return {
        "state_digest_sha256": _state_digest(state),
        "state": state.tolist(),
        "terminal_residual": float(np.asarray(result.residual)),
        "converged": bool(np.asarray(result.converged)),
        "termination_reason_int": int(np.asarray(result.termination_reason)),
        "termination_reason": FixedPointTerminationReason(
            int(np.asarray(result.termination_reason))
        ).name.lower(),
        "active_set_iterations": int(np.asarray(result.active_set_iterations)),
        "active_set_residuals": np.asarray(
            result.active_set_residuals, dtype=np.float64
        ).tolist(),
        "active_set_mask_differences": np.asarray(
            result.active_set_mask_differences, dtype=np.int64
        ).tolist(),
        "active_set_cycle_damping_activations": np.asarray(
            result.active_set_cycle_damping_activations, dtype=np.int64
        ).tolist(),
        "attempted_newton_promotions": int(
            np.asarray(result.attempted_newton_promotions)
        ),
        "accepted_newton_promotions": int(
            np.asarray(result.accepted_newton_promotions)
        ),
        "topology": topology,
        "axis_error": axis_error,
    }


def compile_census(
    arm: str,
    program: Callable[[jax.Array], FixedPointResult],
    seed: np.ndarray,
    dimensions: dict[str, Any],
    *,
    started: float,
) -> dict[str, Any]:
    lowered = jax.jit(program).lower(jnp.asarray(seed, dtype=jnp.float64))
    compiled = lowered.compile()
    hlo = compiled.as_text()
    analysis = _compiled_memory_fields(compiled.memory_analysis())
    serialisation = _try_serialized_executable_size(compiled)
    census = {
        "schema": "nova.scanned-trip-loop-compile-census",
        "arm": arm,
        "loop_driver": "scan" if arm == "scanned" else "fori_loop",
        "source_revision": _source_revision(),
        "lane": _lane(),
        "requested_cells": abs(REQUESTED_CELLS),
        "dimensions": dimensions,
        "compile_wall_seconds": perf_counter() - started,
        "optimised_hlo_bytes": len(hlo.encode("utf-8")),
        "instruction_count": _hlo_instruction_count(hlo),
        "total_hlo_lines": len(hlo.splitlines()),
        "memory_analysis": analysis,
        "serialisation": serialisation,
        "method": "jax.jit(program).lower(seed).compile().as_text()",
    }
    return census


def execute_arm(
    arm: str,
    program: Callable[[jax.Array], FixedPointResult],
    seed: np.ndarray,
    operator: Any,
    case_exact: Any,
    *,
    started: float,
    repeats: int = 3,
) -> dict[str, Any]:
    jitted = jax.jit(program)
    host_seed = jnp.asarray(seed, dtype=jnp.float64)
    warm_started = perf_counter()
    jax.block_until_ready(jitted(host_seed))
    warm_seconds = perf_counter() - warm_started
    timed = []
    for _ in range(repeats):
        run_started = perf_counter()
        result = jax.block_until_ready(jitted(host_seed))
        timed.append(perf_counter() - run_started)
    timed_seconds = np.asarray(timed, dtype=np.float64)
    return {
        "schema": "nova.scanned-trip-loop-arm-execution",
        "arm": arm,
        "loop_driver": "scan" if arm == "scanned" else "fori_loop",
        "source_revision": _source_revision(),
        "lane": _lane(),
        "warm_execution_seconds": warm_seconds,
        "timed_execution_seconds": timed_seconds.tolist(),
        "execution_min_seconds": float(np.min(timed_seconds)),
        "execution_median_seconds": float(np.median(timed_seconds)),
        "observe_started_wall_seconds": perf_counter() - started,
        "terminal": _terminal_result(result, operator, case_exact),
    }


def measurement_set() -> tuple[Any, np.ndarray, Any, Any, dict[str, Any]]:
    """Build the shared problem once and return every operand the arms need."""

    profile, seed, request, dimensions = _certificate_compile_problem(
        CASE, REQUESTED_CELLS
    )
    _carrier, _source, exact = _case(CASE)
    return profile, seed, request, exact, dimensions


def _strict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _strict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_strict(value) for value in obj]
    if isinstance(obj, tuple):
        return [_strict(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _residuals_or_nan(values: list[Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else value for value in values], dtype=np.float64
    )


def render_residual_svg(
    production: dict[str, Any], scanned: dict[str, Any], path: Path
) -> None:
    """Bank one per-trip residual SVG of both routes on shared axes."""

    production_residuals = _residuals_or_nan(
        production["terminal"]["active_set_residuals"]
    )
    scanned_residuals = _residuals_or_nan(scanned["terminal"]["active_set_residuals"])
    trips = np.arange(production_residuals.size)
    if DEFAULT_INK is None:
        production_color, scanned_color = "#3366cc", "#cc7722"
    else:
        production_color = DEFAULT_INK.flux_color
        scanned_color = DEFAULT_INK.thomson_secondary_color
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(
        trips,
        production_residuals,
        marker="o",
        markersize=5,
        linewidth=1.5,
        color=production_color,
        label="production fori_loop",
    )
    axis.semilogy(
        trips,
        scanned_residuals,
        marker="x",
        markersize=6,
        linewidth=1.5,
        linestyle="--",
        color=scanned_color,
        label="scanned trips",
    )
    axis.set_xlabel("trip index")
    axis.set_ylabel("live residual (relative)")
    axis.set_title("per-trip residual, whole-cell 300 cells")
    axis.legend(frameon=False)
    axis.grid(True, which="both", alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg")
    plt.close(figure)


def combine(
    production: dict[str, Any],
    scanned: dict[str, Any],
    dimensions: dict[str, Any],
) -> dict[str, Any]:
    production_terminal = production["terminal"]
    scanned_terminal = scanned["terminal"]
    production_census = production["census"]
    scanned_census = scanned["census"]
    production_state = np.asarray(production_terminal["state"], dtype=np.float64)
    scanned_state = np.asarray(scanned_terminal["state"], dtype=np.float64)
    state_bits_identical = (
        production_terminal["state_digest_sha256"]
        == scanned_terminal["state_digest_sha256"]
    )
    state_diff = np.max(np.abs(production_state - scanned_state))
    production_residuals = _residuals_or_nan(
        production_terminal["active_set_residuals"]
    )
    scanned_residuals = _residuals_or_nan(scanned_terminal["active_set_residuals"])
    residual_diff = np.abs(production_residuals - scanned_residuals)
    active_slots = np.isfinite(production_residuals) & np.isfinite(scanned_residuals)
    residual_sup = (
        float(np.max(residual_diff[active_slots])) if np.any(active_slots) else None
    )
    production_temp = production_census["memory_analysis"].get("temp_size_in_bytes", 0)
    scanned_temp = scanned_census["memory_analysis"].get("temp_size_in_bytes", 0)
    return {
        "schema": "nova.scanned-trip-loop-comparison",
        "source_revision": _source_revision(),
        "lane_from_production": production["lane"],
        "lane_from_scanned": scanned["lane"],
        "requested_cells": abs(REQUESTED_CELLS),
        "dimensions": dimensions,
        "terminal": {
            "production": {
                "residual": production_terminal["terminal_residual"],
                "converged": production_terminal["converged"],
                "iterations": production_terminal["active_set_iterations"],
                "termination_reason": production_terminal["termination_reason"],
                "axis_error": production_terminal["axis_error"],
            },
            "scanned": {
                "residual": scanned_terminal["terminal_residual"],
                "converged": scanned_terminal["converged"],
                "iterations": scanned_terminal["active_set_iterations"],
                "termination_reason": scanned_terminal["termination_reason"],
                "axis_error": scanned_terminal["axis_error"],
            },
            "comparison": {
                "terminal_state_bits_identical": state_bits_identical,
                "terminal_state_sup_difference": float(state_diff),
                "per_trip_residual_sup_difference": residual_sup,
                "terminal_residual_difference": float(
                    abs(
                        production_terminal["terminal_residual"]
                        - scanned_terminal["terminal_residual"]
                    )
                ),
                "mask_differences_identical": (
                    production_terminal["active_set_mask_differences"]
                    == scanned_terminal["active_set_mask_differences"]
                ),
                "cycle_damping_activations_identical": (
                    production_terminal["active_set_cycle_damping_activations"]
                    == scanned_terminal["active_set_cycle_damping_activations"]
                ),
            },
        },
        "census": {
            "production": production_census,
            "scanned": scanned_census,
            "comparison": {
                "instruction_count_ratio": (
                    scanned_census["instruction_count"]
                    / production_census["instruction_count"]
                ),
                "instruction_count_delta": (
                    scanned_census["instruction_count"]
                    - production_census["instruction_count"]
                ),
                "hlo_bytes_delta": (
                    scanned_census["optimised_hlo_bytes"]
                    - production_census["optimised_hlo_bytes"]
                ),
                "temporaries_temp_bytes_delta": scanned_temp - production_temp,
                "temporaries_temp_bytes_ratio": scanned_temp / production_temp,
                "serialized_executable_mib": {
                    "production": production_census["serialisation"].get(
                        "serialized_executable_mib"
                    ),
                    "scanned": scanned_census["serialisation"].get(
                        "serialized_executable_mib"
                    ),
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("production", "scanned"), default="both")
    parser.add_argument("--platform", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--execute", action="store_true", help="execute each arm from the seed"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="compare a platform's executed pair and render the SVG",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cells",
        type=int,
        default=-300,
        help="requested cells, negative for whole-cell (lower for smoke)",
    )
    arguments = parser.parse_args()

    global REQUESTED_CELLS
    REQUESTED_CELLS = arguments.cells

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root() / "scan-prototype"
    )
    if support_clip_mode() != "chord":
        set_support_clip_mode("chord")
    PARTS_ROOT.mkdir(parents=True, exist_ok=True)

    platform = arguments.platform or _platform()

    if arguments.combine:
        production_path = PARTS_ROOT / f"{platform}-production-execution.json"
        scanned_path = PARTS_ROOT / f"{platform}-scanned-execution.json"
        if not production_path.exists() or not scanned_path.exists():
            raise SystemExit(
                f"--combine requires {production_path.name} and {scanned_path.name}"
            )
        production = json.loads(production_path.read_text())
        scanned = json.loads(scanned_path.read_text())
        _, _, _, _, dimensions = measurement_set()
        _write_json(
            PARTS_ROOT / f"{platform}-comparison.json",
            _strict(combine(production, scanned, dimensions)),
        )
        svg_path = FIGURE_ROOT / f"per-trip-residual-{platform}.svg"
        render_residual_svg(production, scanned, svg_path)
        combined = {
            "figure": {
                "filesystem_path": str(svg_path.relative_to(ROOT)),
                "project_absolute_src": f"/nova/{svg_path.relative_to(ROOT / 'docs')}",
                "sha256": hashlib.sha256(svg_path.read_bytes()).hexdigest(),
            }
        }
        _write_json(PARTS_ROOT / f"{platform}-figure.json", _strict(combined))
        comparison_path = PARTS_ROOT / f"{platform}-comparison.json"
        comparison = json.loads(comparison_path.read_text())
        identical = comparison["terminal"]["comparison"][
            "terminal_state_bits_identical"
        ]
        print(
            f"COMBINED platform={platform} state_bits_identical={identical} "
            f"instruction_ratio="
            f"{comparison['census']['comparison']['instruction_count_ratio']:.4f}",
            flush=True,
        )
        return

    started = perf_counter()
    profile, seed, request, exact, dimensions = measurement_set()
    production_program, scanned_program = make_programs(profile, request)
    programs = {"production": production_program, "scanned": scanned_program}
    arm_names = (
        ("production", "scanned") if arguments.arm == "both" else (arguments.arm,)
    )
    for arm in arm_names:
        census = compile_census(arm, programs[arm], seed, dimensions, started=started)
        _write_json(PARTS_ROOT / f"{platform}-{arm}-census.json", _strict(census))
        temp_gib = census["memory_analysis"].get("temp_size_in_bytes", 0) / 2**30
        print(
            f"CENSUS platform={platform} arm={arm} "
            f"instructions={census['instruction_count']} temp={temp_gib:.4f}GiB",
            flush=True,
        )
        if arguments.execute:
            run = execute_arm(
                arm,
                programs[arm],
                seed,
                profile.operator,
                exact,
                started=started,
                repeats=arguments.repeats,
            )
            run["census"] = census
            _write_json(PARTS_ROOT / f"{platform}-{arm}-execution.json", _strict(run))
            print(
                f"EXECUTED platform={platform} arm={arm} "
                f"residual={run['terminal']['terminal_residual']} "
                f"converged={run['terminal']['converged']} "
                f"min={run['execution_min_seconds']:.4f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
