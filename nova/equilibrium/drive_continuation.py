"""Advance a large drive change in adaptive warm-started sub-steps.

A bank solve carries a converged state between drive slices. Two accepted
states define a secant in drive, removing the first-order seed drift without
another Jacobian or linear solve. The predicted state remains only a seed:
the solve's terminal fresh residual alone certifies convergence.

The remedy is drive continuation. A large drive change is split into sub-steps,
each a warm-started :func:`nova.equilibrium.support_read.solve_batch` from the
previous sub-step's converged state and, with opt-in reuse, last Newton factor.
A carried factor's
first trial must contract the fresh residual or the solve rebuilds it at the
incumbent. The fresh-residual line search and terminal certification still
decide which state is accepted.
The driver adapts the sub-step size from the outcome of each call, and it
certifies the state at the target drive against the unchanged fresh-read
residual of at most :data:`CERTIFICATION_TOLERANCE`.

Why the initial size is what it is
---------------------------------
:data:`INITIAL_RELATIVE_STEP` is 1e-3, the largest relative drive step the
warm-start record shows every lane converging, at three to six updates against
a sixty-update ceiling. A smaller initial size costs more sub-steps per decade
of drive for no extra robustness, and a larger one is the step the record shows
stalling.
"""

from typing import Callable, NamedTuple

import jax.numpy as jnp

from nova.equilibrium import support_read

#: Relative drive change of the first sub-step. See the module docstring.
INITIAL_RELATIVE_STEP = 1e-3

#: Factor applied to the sub-step after a sub-solve converges inside budget.
GROWTH = 2.0

#: Factor applied to the sub-step after a sub-solve misses its budget.
SHRINK = 0.5

#: Smallest sub-step the driver will attempt before reporting the change
#: unconverged. A step below this cannot buy progress the previous one missed.
MINIMUM_RELATIVE_STEP = 1e-9

#: Updates a sub-solve may take before the sub-step counts as a failure. The
#: warm-start record's worst converging slice at a 1e-3 step took six updates.
UPDATE_BUDGET = 6

#: Tolerance an intermediate sub-solve stops at. It only has to leave the next
#: sub-step a usable seed, which the record shows a residual two orders above
#: the certification budget already is.
INTERMEDIATE_TOLERANCE = 1e-6

#: The locked residual clause of the precision budget, applied to the state at
#: the target drive by the solve's own terminal fresh read.
CERTIFICATION_TOLERANCE = 1e-10


class SubStep(NamedTuple):
    """One warm-started sub-solve and the drive it was asked for."""

    drive: float
    change: float
    sub_step: float
    tolerance: float
    updates: int
    reads: int
    residual: float
    converged: bool
    final: bool


class DriveContinuationResult(NamedTuple):
    """The state reached at the target drive and how it was reached."""

    result: support_read.SolveResult
    drive: float
    certified: bool
    substeps: tuple[SubStep, ...]
    updates: int
    reads: int
    sub_step: float
    reason: str | None


def _batch_count(values) -> int:
    """Return a batch's update or read count.

    Every lane of one batch read and one batch update pays the whole call, so
    the batch's cost is the largest per-lane count rather than the sum.
    """
    return int(jnp.max(values))


def predict_seed(
    moments,
    drive: float,
    target_drive: float,
    *,
    previous_moments=None,
    previous_drive: float | None = None,
):
    """Extrapolate from two converged states, or retain the plain warm start.

    The caller supplies only accepted states of the same drive stream. A
    missing or repeated drive cannot define a secant and leaves the seed
    unchanged. Stationary points remain warm seeds for the next fresh read.
    """
    if previous_moments is None or previous_drive is None:
        return moments
    distance = jnp.asarray(drive - previous_drive)
    ratio = jnp.where(
        distance != 0,
        (target_drive - drive) / jnp.where(distance != 0, distance, 1.0),
        0.0,
    )
    return tuple(
        current + ratio * (current - previous)
        for current, previous in zip(moments, previous_moments, strict=True)
    )


def advance_drive(
    operands,
    cache,
    moments,
    seeds,
    seeds_previous,
    convention,
    policy,
    profile_at: Callable[[float], object],
    density_gradient,
    quadrature,
    *,
    start_drive: float,
    target_drive: float,
    axis_sign: float = -1.0,
    initial_relative_step: float = INITIAL_RELATIVE_STEP,
    growth: float = GROWTH,
    shrink: float = SHRINK,
    minimum_relative_step: float = MINIMUM_RELATIVE_STEP,
    update_budget: int = UPDATE_BUDGET,
    intermediate_tolerance: float = INTERMEDIATE_TOLERANCE,
    certification_tolerance: float = CERTIFICATION_TOLERANCE,
    max_substeps: int = 256,
    solve: Callable | None = None,
    solve_options: dict | None = None,
    previous_moments=None,
    previous_drive: float | None = None,
    factor_cache: support_read.NewtonFactor | None = None,
    factor_valid=True,
) -> DriveContinuationResult:
    """Reach ``target_drive`` from lanes converged at ``start_drive``.

    ``moments`` and ``seeds_previous`` are the warm start: the converged state
    and the stationary points its own read returned. ``profile_at`` builds the
    profile a drive asks for, so the driver never inspects the profile's
    internals. Two converged slices supplied through ``previous_moments`` and
    ``previous_drive`` seed the next slice by secant extrapolation. Accepted
    sub-steps extend that history; rejected attempts never enter it.

    Each sub-step solves at ``drive + change`` with ``tolerance`` set to the
    intermediate budget, except the sub-step that reaches the target, which is
    solved at :data:`CERTIFICATION_TOLERANCE` and whose terminal accepted state
    is therefore certified by the solve's own fresh read. A sub-solve that
    leaves any lane unconverged, or the batch above ``update_budget`` updates,
    is rejected: the drive does not advance, the sub-step is halved, and the
    same drive is retried. A sub-solve inside budget doubles the sub-step.

    ``solve`` is the solve the sub-steps call, defaulting to
    :func:`nova.equilibrium.support_read.solve_batch`; it exists so a caller
    can supply a version compiled for its own traced drive. ``solve_options``
    are passed through. Factor reuse is opt-in through
    ``solve_options={"reuse_factor": True}``: extra reads and updates can
    outweigh skipped assemblies (estimated H200 batch cost 2.84 s with fresh
    factors versus 5.73 s with carry over fourteen slices). With reuse enabled,
    ``factor_cache`` and ``factor_valid`` can prime the first sub-step from a
    preceding slice. Only accepted sub-steps replace that factor.
    """
    solve = solve or support_read.solve_batch
    options = dict(solve_options or {})
    span = abs(target_drive - start_drive)
    if span == 0:
        raise ValueError("the target drive equals the start drive")
    scale = abs(start_drive) or span
    sub_step = initial_relative_step * scale
    drive = float(start_drive)
    direction = 1.0 if target_drive > start_drive else -1.0
    state_moments = moments
    state_previous = seeds_previous
    substeps: list[SubStep] = []
    updates = reads = 0
    certified = False
    reason = None
    result = None
    while drive != float(target_drive):
        if len(substeps) >= max_substeps:
            reason = f"sub-step ceiling reached at drive {drive:.12g}"
            break
        remaining = abs(float(target_drive) - drive)
        attempted = sub_step
        change = min(attempted, remaining)
        final = change >= remaining
        tolerance = certification_tolerance if final else intermediate_tolerance
        next_drive = drive + direction * change
        predicted = predict_seed(
            state_moments,
            drive,
            next_drive,
            previous_moments=previous_moments,
            previous_drive=previous_drive,
        )
        attempt = solve(
            operands,
            cache,
            predicted,
            seeds,
            state_previous,
            convention,
            policy,
            profile_at(drive + direction * change),
            density_gradient,
            quadrature,
            axis_sign,
            max_steps=update_budget,
            tolerance=tolerance,
            **(
                {"factor_cache": factor_cache, "factor_valid": factor_valid}
                if options.get("reuse_factor", False) and factor_cache is not None
                else {}
            ),
            **options,
        )
        steps = _batch_count(attempt.steps)
        batch_reads = _batch_count(attempt.reads)
        residual = float(jnp.max(attempt.residual))
        converged = bool(jnp.all(attempt.converged))
        accepted = converged and residual <= tolerance
        updates += steps
        reads += batch_reads
        substeps.append(
            SubStep(
                drive + direction * change,
                change,
                attempted,
                tolerance,
                steps,
                batch_reads,
                residual,
                converged,
                final,
            )
        )
        if accepted:
            result = attempt
            previous_moments, previous_drive = state_moments, drive
            drive = float(target_drive) if final else next_drive
            state_moments = attempt.moments
            state_previous = attempt.support.stationary.position
            if options.get("reuse_factor", False) and hasattr(attempt, "factor"):
                factor_cache, factor_valid = attempt.factor, attempt.factor_valid
            if final:
                certified = residual <= certification_tolerance
                break
            sub_step = min(attempted * growth, scale)
        else:
            sub_step *= shrink
            if sub_step < minimum_relative_step * scale:
                reason = (
                    f"sub-step floor reached at drive {drive:.12g} with "
                    f"residual {residual:.6g}"
                )
                break
    if not certified and reason is None and result is None:
        reason = f"no sub-step solved at drive {drive:.12g}"
    return DriveContinuationResult(
        result,
        drive,
        certified,
        tuple(substeps),
        updates,
        reads,
        sub_step,
        reason,
    )
