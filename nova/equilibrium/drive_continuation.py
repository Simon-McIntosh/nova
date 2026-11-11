"""Advance a large drive change in adaptive warm-started sub-steps.

A bank solve carries a converged state between drive slices. Two accepted
states define a secant in drive, removing the first-order seed drift without
another Jacobian or linear solve. The predicted state remains only a seed:
the solve's terminal fresh residual alone certifies convergence.

The remedy is drive continuation. A large drive change is split into sub-steps,
each a warm-started :func:`nova.equilibrium.support_read.solve_batch` from the
previous sub-step's converged state. Nothing inside the solve changes: it is
called with the same Newton step, the same fresh-residual line search and the
same terminal certification; only the drive each call is asked for is smaller.
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

import jax
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

    drive: float | jax.Array
    change: float | jax.Array
    sub_step: float | jax.Array
    tolerance: float | jax.Array
    updates: int | jax.Array
    reads: int | jax.Array
    residual: float | jax.Array
    converged: bool | jax.Array
    final: bool | jax.Array


class DriveContinuationResult(NamedTuple):
    """The state reached at the target drive and how it was reached."""

    result: support_read.SolveResult
    drive: float | jax.Array
    certified: bool
    substeps: tuple[SubStep, ...]
    updates: int
    reads: int
    sub_step: float | jax.Array
    reason: str | None


def _batch_count(values) -> int:
    """Return a batch's update or read count.

    Every lane of one batch read and one batch update pays the whole call, so
    the batch's cost is the largest per-lane count rather than the sum.
    """
    return int(jnp.max(values))


def predict_seed(
    moments,
    drive: float | jax.Array,
    target_drive: float | jax.Array,
    *,
    previous_moments=None,
    previous_drive: float | jax.Array | None = None,
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
        current
        + ratio.reshape(ratio.shape + (1,) * (current.ndim - ratio.ndim))
        * (current - previous)
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
    profile_at: Callable[[float | jax.Array], object],
    density_gradient,
    quadrature,
    *,
    start_drive: float | jax.Array,
    target_drive: float | jax.Array,
    axis_sign: float = -1.0,
    initial_relative_step: float | jax.Array = INITIAL_RELATIVE_STEP,
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
    previous_drive: float | jax.Array | None = None,
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
    are passed through to it unchanged.

    Array start or target drives select independent lanes. ``profile_at`` then
    receives a drive vector and may return a callable from lane index to that
    lane's profile, as accepted by ``solve_batch``. Tolerances and sub-step
    telemetry are per-lane arrays; ``certified`` means every lane certified.
    A zero-span lane is read at its target once, then held. Scalar start and
    target drives keep the shared-step behaviour described above.
    """
    solve = solve or support_read.solve_batch
    options = dict(solve_options or {})
    if jnp.ndim(start_drive) or jnp.ndim(target_drive):
        return _advance_lanes(
            operands,
            cache,
            moments,
            seeds,
            seeds_previous,
            convention,
            policy,
            profile_at,
            density_gradient,
            quadrature,
            start_drive=start_drive,
            target_drive=target_drive,
            axis_sign=axis_sign,
            initial_relative_step=initial_relative_step,
            growth=growth,
            shrink=shrink,
            minimum_relative_step=minimum_relative_step,
            update_budget=update_budget,
            intermediate_tolerance=intermediate_tolerance,
            certification_tolerance=certification_tolerance,
            max_substeps=max_substeps,
            solve=solve,
            solve_options=options,
            previous_moments=previous_moments,
            previous_drive=previous_drive,
        )
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


def _advance_lanes(
    operands,
    cache,
    moments,
    seeds,
    seeds_previous,
    convention,
    policy,
    profile_at,
    density_gradient,
    quadrature,
    *,
    start_drive,
    target_drive,
    axis_sign,
    initial_relative_step,
    growth,
    shrink,
    minimum_relative_step,
    update_budget,
    intermediate_tolerance,
    certification_tolerance,
    max_substeps,
    solve,
    solve_options,
    previous_moments,
    previous_drive,
):
    """Keep accepted state, certification and secant history independently.

    Array drive calls pass an array to ``profile_at`` and an array tolerance
    to the solve. Certified lanes retain their state and ask for no updates.
    Scalar calls retain the shared-step contract of :func:`advance_drive`.
    """
    lanes = moments[0].shape[0]
    drive = jnp.broadcast_to(jnp.asarray(start_drive, moments[0].dtype), (lanes,))
    target = jnp.broadcast_to(jnp.asarray(target_drive, moments[0].dtype), (lanes,))
    span = jnp.abs(target - drive)
    scale = jnp.where(drive != 0, jnp.abs(drive), jnp.where(span > 0, span, 1.0))
    direction = jnp.sign(target - drive)
    sub_step = jnp.broadcast_to(jnp.asarray(initial_relative_step) * scale, (lanes,))
    done = jnp.zeros(lanes, bool)
    stopped = jnp.zeros(lanes, bool)
    state = moments
    stationary = seeds_previous
    older = moments if previous_moments is None else previous_moments
    old_drive = (
        drive
        if previous_drive is None
        else jnp.broadcast_to(jnp.asarray(previous_drive), (lanes,))
    )
    result = None
    substeps = []
    updates = reads = 0

    def choose(mask, candidate, incumbent):
        return jax.tree.map(
            lambda value, prior: jnp.where(
                mask.reshape((lanes,) + (1,) * (value.ndim - 1)), value, prior
            ),
            candidate,
            incumbent,
        )

    while bool(jnp.any(~done & ~stopped)) and len(substeps) < max_substeps:
        active = ~done & ~stopped
        remaining = jnp.abs(target - drive)
        change = jnp.where(active, jnp.minimum(sub_step, remaining), 0.0)
        final = change >= remaining
        next_drive = jnp.where(final & active, target, drive + direction * change)
        tolerance = jnp.where(final, certification_tolerance, intermediate_tolerance)
        predicted = choose(
            active,
            predict_seed(
                state,
                drive,
                next_drive,
                previous_moments=older,
                previous_drive=old_drive,
            ),
            state,
        )
        attempt = solve(
            operands,
            cache,
            predicted,
            seeds,
            stationary,
            convention,
            policy,
            profile_at(next_drive),
            density_gradient,
            quadrature,
            axis_sign,
            max_steps=update_budget,
            tolerance=jnp.where(active, tolerance, jnp.inf),
            **solve_options,
        )
        accepted = (
            active
            & attempt.converged
            & jnp.isfinite(attempt.residual)
            & (attempt.residual <= tolerance)
            & (attempt.steps <= update_budget)
        )
        updates += _batch_count(attempt.steps)
        reads += _batch_count(attempt.reads)
        substeps.append(
            SubStep(
                next_drive,
                change,
                sub_step,
                tolerance,
                attempt.steps,
                attempt.reads,
                attempt.residual,
                accepted,
                final & active,
            )
        )
        if result is None:
            result = attempt
        else:
            result = choose(accepted, attempt, result)
        older = choose(accepted, state, older)
        old_drive = jnp.where(accepted, drive, old_drive)
        state = choose(accepted, attempt.moments, state)
        stationary = choose(accepted, attempt.support.stationary.position, stationary)
        drive = jnp.where(accepted, next_drive, drive)
        done = done | (accepted & final)
        sub_step = jnp.where(
            active,
            jnp.where(
                accepted, jnp.minimum(sub_step * growth, scale), sub_step * shrink
            ),
            sub_step,
        )
        stopped = stopped | (
            active & ~accepted & (sub_step < minimum_relative_step * scale)
        )
    certified = bool(jnp.all(done))
    reason = (
        None
        if certified
        else (
            "sub-step floor reached"
            if bool(jnp.any(stopped))
            else "sub-step ceiling reached"
        )
    )
    return DriveContinuationResult(
        result, drive, certified, tuple(substeps), updates, reads, sub_step, reason
    )
