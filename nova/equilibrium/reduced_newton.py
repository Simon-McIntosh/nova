"""Plain Newton on the plasma-cell amplitudes behind one frozen shadow.

The free-boundary flux map writes one iterate as ``external + P @ j``, where
``P`` is the fixed plasma-to-node response the operator carries and ``j`` the
per-cell current the topology read drives.  ``j`` vanishes outside the
profile-owned support, so the map's whole nonlinearity factors through a few
hundred cell amplitudes rather than the flux nodes.

Posing the fixed point on those amplitudes gives a square system small enough
to carry a dense Jacobian.  Writing the flux-space map Jacobian as ``P M`` and
the amplitude-space map Jacobian as ``M P``, the two share every nonzero
eigenvalue, so a dense Newton step on the amplitudes is the flux-space Newton
step restricted to the response range: the directions a Krylov method must
discover are exactly the ones the amplitude coordinates already span, and the
null directions are removed rather than chased.

This module is a prototype that runs beside the production ladder without
changing it.  It reuses the production merit, the production backtracking
grades, the trip structure of one topology read per active-set trip, the
settled exit and the incumbent-mask acceptance, and replaces only the inner
solve: a dense ``jax.jacfwd`` Jacobian formed once per trip, a dense linear
solve per Newton step, and a backtracking line search that refreshes the
Jacobian when a whole grade ladder is rejected.

Three costs of driving that solve from the host are removed rather than
paid.  Scoring stops at the first grade below the incumbent merit, which is
the grade the selection takes in either case, so a step accepted at full
length evaluates the map once instead of eight times; the grades after the
first are scored together, so a refused first grade costs one further round
trip rather than one per remaining grade; and the residual sup norm the
receipt reads comes back from the program that already holds the residual
rather than from a reduction dispatched after it.

A trip closes inside one compiled program that
reconstructs the flux, promotes the residual shadow against the frozen one,
maps the promoted state, gathers the next trip's amplitudes and measures the
off-support leakage from a single topology read, with the frozen shadow as an
argument so every trip of one solve runs the same program.  Both predecessors
stay available for comparison, ``eager`` scoring and the ``dispatched``
boundary, because the semantics the fast routes must agree with is what those
routes compute.

When a constraint tuple is present, the merit and the exit test score the
augmented residual against the flux block's own normalised reference rather
than against the whole augmented vector.  A compensating unknown is
dimensionless and bounded by nothing, so a reference that admitted it would
let a warm-started loop whose unknown had accumulated toward order a hundred
inflate the denominator and loosen the convergence tolerance on both blocks by
that factor; scoring against the flux block alone removes that channel.  The
reduced route also publishes ``soft_mode_projection`` as ``None`` rather than
``NaN`` because a dense route carries no soft-mode projection, and refuses a
constrained solve that has no prescribed current to fold its compensating
currents into.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import os as _os
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintContext,
    ConstraintPair,
    ConstraintRecord,
    constraint_records,
    constraint_row_slices,
)
from nova.equilibrium.forward_operator import CellCurrentMoments
from nova.equilibrium.fixed_point import (
    _BACKTRACKING_FACTORS,
    _RELATIVE_SUP_MERIT_EXPONENT,
    _relative_residual,
    _smooth_relative_sup_merit,
    FIXED_POINT_RESIDUAL_TOLERANCE,
    FixedPointTerminationReason,
)


__all__ = [
    "ConstrainedReducedNewtonResult",
    "ReducedProgram",
    "ReducedCoordinates",
    "ReducedNewtonResult",
    "ReducedScores",
    "ReducedTrip",
    "reduced_coordinates",
    "solve_constrained_reduced_newton",
    "solve_constrained_reduced_newton_compiled",
    "solve_reduced_newton",
    "solve_reduced_newton_compiled",
]


ACTIVE_SET_STEPS = 16
NEWTON_STEPS = 12
SUPPORT_POLICY = "participation"
_ACTIVE_SUPPORT_FLOOR = 0.0

#: Optional per-solve stage recording for the keyframe driver.  When enabled,
#: the host loop records named boundaries (``convert``, ``shadow``, ``bind``,
#: ``gather``, ``trips``, ``records`` plus a ``tripN:*`` boundary per active-set
#: trip) against ``perf_counter`` so a measurement can separate host-side work
#: around the compiled program from the kernels themselves.  The recording
#: changes no number the solve produces; it is off unless the driver turns it
#: on.
_STAGE_TIMING_ENABLED = _os.environ.get("NOVA_REDUCED_STAGE_TIMING") == "1"
_STAGE_MARKS: list[tuple[str, float]] = []


def set_stage_timing(enabled: bool = True) -> None:
    """Enable or disable the named stage recording (measurement hook only)."""
    global _STAGE_TIMING_ENABLED
    _STAGE_TIMING_ENABLED = bool(enabled)


def clear_stage_marks() -> None:
    """Forget recorded stage boundaries before one solve's measurement."""
    _STAGE_MARKS.clear()


def reduced_stage_marks() -> list[tuple[str, float]]:
    """Return the stage boundaries recorded since the last clear, in order."""
    return list(_STAGE_MARKS)


def _stage_mark(name: str) -> None:
    """Record one named solve-stage boundary when stage timing is enabled."""
    if _STAGE_TIMING_ENABLED:
        _STAGE_MARKS.append((name, time.perf_counter()))


#: Score the first backtracking grade alone and, when it is refused, score
#: every remaining grade in one dispatch.  The production ladder takes the
#: first grade below the incumbent merit, so a grade after the accepted one
#: can only produce a value the selection discards, while a refused grade
#: leaves every later grade still to score and no ordering among them to
#: exploit.
LADDER_SCORING = "batched_tail"
#: Score the grades strictly one at a time, stopping at the first below the
#: incumbent merit.  Same selection, one synchronised round trip per grade.
SERIAL_LADDER_SCORING = "first_accept"
#: Score every grade before selecting, which is what the first prototype did
#: and what the banked receipt measured.
EAGER_LADDER_SCORING = "eager"
#: Close a trip inside one compiled program: reconstruct, promote the shadow,
#: map the promoted state, gather the next trip's amplitudes and measure the
#: off-support leakage share one topology read.
TRIP_BOUNDARY = "fused"
#: Close a trip through the separate calls the first prototype dispatched.
DISPATCHED_TRIP_BOUNDARY = "dispatched"
#: Hand the constraint tuple and the flux scale to every kernel as traced
#: arguments.  A commanded target then moves without changing the program's
#: identity, so a keyframe that re-solves the same rows at a new target
#: re-enters the program it already compiled.
TRACED_ROWS = "traced"
#: Bake the constraint tuple and the flux scale into the program when it is
#: built.  Same numbers, one program per commanded target; it stays reachable
#: because the semantics the traced route must agree with is what it computes.
CAPTURED_ROWS = "captured"


class ReducedCoordinates(NamedTuple):
    """Cell indices and moment leaves that carry one reduced state."""

    cells: jax.Array
    """Indices of the cells whose amplitudes are solved for."""

    leaves: tuple[str, ...]
    """Moment leaves carried per cell, in :class:`CellCurrentMoments` order."""

    cell_number: int
    """Total cell count of the operator's moment image."""

    @property
    def size(self) -> int:
        """Return the reduced state dimension."""
        return int(self.cells.size) * len(self.leaves)


class ReducedScores(NamedTuple):
    """Residual, residual sup norm and flux scores at one reduced state.

    The four are readings of a single write-then-read cycle, so one program
    computes the moments once and returns all four together.  The sup norm is
    a reduction over a residual that program already holds and nothing the
    solve branches on reads it, so dispatching it separately would spend a
    synchronised round trip on a number only the receipt reports.
    """

    residual: jax.Array
    residual_norm: float
    merit: float
    flux_residual: float


class ReducedTrip(NamedTuple):
    """Everything one active-set trip freezes before its Newton solve."""

    shadow: jax.Array
    base_state: jax.Array
    external: jax.Array
    coordinates: ReducedCoordinates


class ReducedNewtonStep(NamedTuple):
    """One accepted or refused Newton step of a reduced trip."""

    trip: int
    step: int
    reduced_residual: float
    flux_residual: float
    merit: float
    accepted_factor: float | None
    grades_tried: int
    map_evaluations: int
    jacobian_refreshed: bool
    wall_s: float


@dataclass
class ReducedNewtonResult:
    """Semantic outcome of one reduced-state active-set solve."""

    state: jax.Array
    terminal_residual: float
    active_set_iterations: int
    converged: bool
    termination_reason: int
    active_set_residuals: list[float] = field(default_factory=list)
    active_set_mask_differences: list[int] = field(default_factory=list)
    newton_steps_per_trip: list[int] = field(default_factory=list)
    jacobian_builds_per_trip: list[int] = field(default_factory=list)
    rejected_steps_per_trip: list[int] = field(default_factory=list)
    jacobian_wall_per_trip: list[float] = field(default_factory=list)
    newton_wall_per_trip: list[float] = field(default_factory=list)
    boundary_wall_per_trip: list[float] = field(default_factory=list)
    trip_wall_per_trip: list[float] = field(default_factory=list)
    map_evaluations_per_trip: list[int] = field(default_factory=list)
    reduced_dimension: int = 0
    support_cells: int = 0
    off_support_leakage: float = 0.0
    steps: list[ReducedNewtonStep] = field(default_factory=list)
    program: "ReducedProgram | None" = field(default=None, compare=False)

    @property
    def termination_name(self) -> str:
        """Return the production termination name of this solve."""
        return FixedPointTerminationReason(self.termination_reason).name.lower()


def _moment_leaves(operator) -> tuple[str, ...]:
    """Return the moment leaves a reduced state must carry to rebuild flux."""
    if getattr(operator, "use_linear_moments", False):
        return CellCurrentMoments._fields
    return ("cell_current",)


def _current_moments(operator, psi, requested_class, target_current):
    """Return the moments the production map images at one trial flux."""
    if target_current is None:
        return operator.cell_current_moments(psi, requested_class)
    moments, _amplitude = operator.normalised_current_moments(
        psi, target_current, requested_class
    )
    return moments


def reduced_coordinates(
    operator,
    state,
    *,
    requested_class=None,
    target_current=None,
    policy: str = SUPPORT_POLICY,
    floor: float = _ACTIVE_SUPPORT_FLOOR,
) -> ReducedCoordinates:
    """Select the cells whose amplitudes carry the trip's reduced state.

    ``participation`` takes the profile-owned support the domain labels
    define, which is the topological authority the map itself masks its
    moments with, so the reduction is exact whenever the labels hold.
    ``active`` takes only the cells carrying current at the trip's incoming
    state, which is smaller and exact whenever the current support does not
    grow inside the trip; the caller checks that with the leakage the solve
    reports.
    """
    leaves = _moment_leaves(operator)
    if policy == "participation":
        masks = operator.current_domain_masks(state, requested_class)
        selected = np.asarray(masks.profile_participation, dtype=bool)
    elif policy == "active":
        moments = _current_moments(operator, state, requested_class, target_current)
        amplitude = np.abs(np.asarray(moments.cell_current))
        selected = amplitude > floor * float(amplitude.max(initial=0.0))
    else:
        raise ValueError(f"unknown reduced-support policy {policy!r}")
    cells = jnp.asarray(np.flatnonzero(selected), dtype=jnp.int32)
    return ReducedCoordinates(cells, leaves, int(selected.size))


def _gather(coordinates: ReducedCoordinates, moments) -> jax.Array:
    """Project full-cell moments onto the reduced coordinates."""
    return jnp.concatenate(
        [getattr(moments, leaf)[coordinates.cells] for leaf in coordinates.leaves]
    )


def _scatter(coordinates: ReducedCoordinates, reduced: jax.Array):
    """Rebuild full-cell moments from one reduced state."""
    blocks = jnp.split(reduced, len(coordinates.leaves))
    zero = jnp.zeros(coordinates.cell_number, dtype=reduced.dtype)
    carried = {
        leaf: zero.at[coordinates.cells].set(block)
        for leaf, block in zip(coordinates.leaves, blocks, strict=True)
    }
    return CellCurrentMoments(
        *(carried.get(leaf, zero) for leaf in CellCurrentMoments._fields)
    )


def _timed(function, *arguments):
    """Return one device-synchronised call and its wall time."""
    start = time.perf_counter()
    value = function(*arguments)
    jax.block_until_ready(value)
    return value, time.perf_counter() - start


def _augmented_relative_residual(imaged, scored, reference):
    """Return the augmented sup residual over the flux block's own reference.

    The numerator is the same true max over both blocks the unaugmented exit
    test reads; the reference is the flux block alone, with the production
    floor, so a compensating unknown that has grown across warm starts cannot
    inflate the denominator and loosen the tolerance it gates.
    """
    return jnp.max(jnp.abs(imaged - scored)) / jnp.maximum(
        jnp.max(jnp.abs(reference)), 1.0e-30
    )


def _augmented_smooth_relative_sup_merit(imaged, scored, reference):
    """Return the augmented smooth p-norm merit over the flux reference.

    The residual numerator is the augmented vector's own; the reference is the
    flux block alone, with the same finite floor the production merit carries.
    The unknown is dimensionless and unbounded, so it must not set the scale
    the rest of the system is judged against.
    """
    residual_vector = jnp.ravel(imaged - scored)
    denominator_vector = jnp.concatenate(
        (jnp.ravel(reference), jnp.asarray([1.0e-30], dtype=reference.dtype))
    )
    return jnp.linalg.vector_norm(
        residual_vector, ord=_RELATIVE_SUP_MERIT_EXPONENT
    ) / jnp.linalg.vector_norm(denominator_vector, ord=_RELATIVE_SUP_MERIT_EXPONENT)


def _reduced_kernels(
    operator,
    coordinates: ReducedCoordinates,
    external,
    requested_class,
    target_current,
    augmentation: "_RowAugmentation | None" = None,
) -> dict[str, Callable[..., Any]]:
    """Build the jitted reduced map, residual, Jacobian and merit ladder.

    The trip's frozen shadow and its base state enter as arguments rather than
    captured constants, so every trip of one solve shares a single compiled
    program and the compilation is paid once.

    ``augmentation`` extends the reduced state by one compensating unknown per
    constraint row and the reduced residual by the authored rows themselves.
    Its absence is not a separate route: every kernel below then evaluates the
    unaugmented expression itself rather than a degenerate case of the
    augmented one, so the two entries return the same numbers.
    """

    def _split(reduced):
        """Return the amplitudes and the compensating unknowns of one state."""
        if augmentation is None:
            return reduced, None
        return reduced[: coordinates.size], reduced[coordinates.size :]

    normalise_current = target_current is not None
    trace_requested_class = requested_class is not None

    def _target(target_value):
        """Return the dynamic target when this program normalises current."""
        return target_value if normalise_current else None

    def _requested(requested_value):
        """Return the per-slice topology class when this route traces it."""
        return requested_value if trace_requested_class else requested_class

    def _bound(rows):
        """Return the constraint data this call evaluates its rows against.

        ``None`` selects the tuple the program was built around, which is what
        a solve that bakes its targets in passes.  A keyframe loop hands the
        edited tuple in instead and the kernel traces it, so a commanded
        target moves without changing the program's identity.
        """
        if augmentation is None:
            return None
        return augmentation.arguments if rows is None else rows

    def _image(moments, unknowns, bound, external_value):
        """Return the flux image of one moment set and its compensation."""
        image = external_value + operator.current_moment_image(moments)
        if augmentation is None:
            return image
        return image + augmentation.flux_delta(unknowns, bound.pairs)

    def _augment(residual, state, unknowns, shadow, bound):
        """Return the residual with the authored rows appended to it."""
        if augmentation is None:
            return residual, None
        rows = augmentation.rows(
            state, unknowns, shadow, bound.pairs, bound.target_current
        )
        return jnp.concatenate((residual, rows)), rows

    def _scores(state, mapped, unknowns, rows, bound):
        """Return the merit and relative residual the selection reads.

        Unaugmented these are the production flux-space scores.  With rows
        present they are the augmented analogue with the flux normalised by its
        own span beside the unknowns, so a row far from its target is visible
        to the line search instead of hidden under the flux block; the
        reference both score against is the flux block alone rather than the
        concatenated vector, because a normalised compensating unknown is
        bounded by nothing and a warm-started loop whose unknown had
        accumulated would otherwise inflate the denominator and loosen the
        tolerance on everything at once.
        """
        if augmentation is None:
            return (
                _smooth_relative_sup_merit(mapped, state),
                _relative_residual(mapped, state),
            )
        scale = bound.flux_scale
        scored = jnp.concatenate((state / scale, unknowns))
        imaged = jnp.concatenate((mapped / scale, unknowns - rows))
        reference = mapped / scale
        return (
            _augmented_smooth_relative_sup_merit(imaged, scored, reference),
            _augmented_relative_residual(imaged, scored, reference),
        )

    def reconstruct(
        reduced,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Return the flux one reduced state reconstructs behind the shadow."""
        amplitudes, unknowns = _split(reduced)
        external_value = external if external_value is None else external_value
        image = _image(
            _scatter(coordinates, amplitudes), unknowns, _bound(rows), external_value
        )
        return jnp.where(shadow, base_state, image)

    def reduced_map(
        reduced,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Return one write-then-read cycle of the reduced amplitudes."""
        moments = _current_moments(
            operator,
            reconstruct(
                reduced,
                shadow,
                base_state,
                external_value,
                target_value,
                requested_value,
                rows,
            ),
            _requested(requested_value),
            _target(target_value),
        )
        return _gather(coordinates, moments)

    def reduced_residual(
        reduced,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Return the reduced fixed-point residual ``u - R(u)`` and its rows."""
        amplitudes, unknowns = _split(reduced)
        if augmentation is None:
            return amplitudes - reduced_map(
                reduced,
                shadow,
                base_state,
                external_value,
                target_value,
                requested_value,
                rows,
            )
        bound = _bound(rows)
        state = reconstruct(
            reduced,
            shadow,
            base_state,
            external_value,
            target_value,
            requested_value,
            rows,
        )
        moments = _current_moments(
            operator, state, _requested(requested_value), _target(target_value)
        )
        residual, _rows = _augment(
            amplitudes - _gather(coordinates, moments), state, unknowns, shadow, bound
        )
        return residual

    def flux_scores(
        reduced,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Return the production merit and relative residual in flux space."""
        amplitudes, unknowns = _split(reduced)
        bound = _bound(rows)
        external_value = external if external_value is None else external_value
        state = reconstruct(
            reduced,
            shadow,
            base_state,
            external_value,
            target_value,
            requested_value,
            rows,
        )
        moments = _current_moments(
            operator, state, _requested(requested_value), _target(target_value)
        )
        image = _image(moments, unknowns, bound, external_value)
        mapped = jnp.where(shadow, state, image)
        scored = (
            None
            if augmentation is None
            else augmentation.rows(
                state, unknowns, shadow, bound.pairs, bound.target_current
            )
        )
        return _scores(state, mapped, unknowns, scored, bound)

    def ladder(
        reduced,
        step,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Score the production backtracking grades of one Newton step."""
        factors = jnp.asarray(_BACKTRACKING_FACTORS, dtype=reduced.dtype)
        candidates = reduced[None, :] + factors[:, None] * step[None, :]
        return jax.lax.map(
            lambda candidate: flux_scores(
                candidate,
                shadow,
                base_state,
                external_value,
                target_value,
                requested_value,
                rows,
            ),
            candidates,
        )

    def leakage(
        reduced,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Return the current the reduced map drives outside its support."""
        moments = _current_moments(
            operator,
            reconstruct(
                reduced,
                shadow,
                base_state,
                external_value,
                target_value,
                requested_value,
                rows,
            ),
            _requested(requested_value),
            _target(target_value),
        )
        current = moments.cell_current
        retained = jnp.zeros_like(current).at[coordinates.cells].set(1.0)
        excluded = jnp.max(jnp.abs(jnp.where(retained > 0.0, 0.0, current)))
        return excluded / jnp.maximum(jnp.max(jnp.abs(current)), 1.0e-30)

    def step_scores(
        reduced,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Return the reduced residual, its sup norm and the flux scores.

        The residual, its sup norm and the two flux scores are four readings
        of a single write-then-read cycle, so one program computes the
        moments once and the four readings share it.  Folding the norm in
        here is what keeps it off the host: the caller reads it beside the
        merit it was going to synchronise on anyway.
        """
        amplitudes, unknowns = _split(reduced)
        bound = _bound(rows)
        external_value = external if external_value is None else external_value
        state = reconstruct(
            reduced,
            shadow,
            base_state,
            external_value,
            target_value,
            requested_value,
            rows,
        )
        moments = _current_moments(
            operator, state, _requested(requested_value), _target(target_value)
        )
        image = _image(moments, unknowns, bound, external_value)
        mapped = jnp.where(shadow, state, image)
        residual, scored = _augment(
            amplitudes - _gather(coordinates, moments), state, unknowns, shadow, bound
        )
        merit, flux_residual = _scores(state, mapped, unknowns, scored, bound)
        return ReducedScores(
            residual,
            jnp.max(jnp.abs(residual)),
            merit,
            flux_residual,
        )

    def grade_scores(
        reduced,
        step,
        factor,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Score one backtracking grade and return the state it scored.

        The candidate amplitudes leave the program beside their scores, so a
        grade the caller accepts carries its own residual and merit into the
        next Newton step and no evaluation is repeated to recover them.  The
        grade enters as an argument, so every grade of every step runs the
        one compiled program.
        """
        candidate = reduced + factor * step
        return candidate, step_scores(
            candidate,
            shadow,
            base_state,
            external_value,
            target_value,
            requested_value,
            rows,
        )

    def tail_grades(
        reduced,
        step,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Score every backtracking grade after the first in one dispatch.

        A refused first grade leaves the remaining grades with no ordering to
        exploit: whichever of them the selection takes, the ones before it
        must be scored to know that they are refused.  Scoring them together
        pays one round trip for the whole tail instead of one per grade, and
        the candidates leave the program beside their scores exactly as the
        single-grade kernel returns them, so an accepted tail grade carries
        its own residual and merit into the next Newton step.
        """
        factors = jnp.asarray(_BACKTRACKING_FACTORS[1:], dtype=reduced.dtype)
        candidates = reduced[None, :] + factors[:, None] * step[None, :]
        return candidates, jax.lax.map(
            lambda candidate: step_scores(
                candidate,
                shadow,
                base_state,
                external_value,
                target_value,
                requested_value,
                rows,
            ),
            candidates,
        )

    def trip_boundary(
        reduced,
        shadow,
        base_state,
        external_value=None,
        target_value=None,
        requested_value=None,
        rows=None,
    ):
        """Close one trip inside a single program.

        Reconstructing the flux, promoting the residual shadow against the
        frozen one, mapping the promoted state, gathering the next trip's
        amplitudes and measuring the off-support leakage all read the same
        topology at the same state, so one program evaluates that read once
        and the frozen shadow enters as an argument rather than a constant,
        which keeps every trip of one solve on the same compiled program.
        """
        amplitudes, unknowns = _split(reduced)
        del amplitudes
        bound = _bound(rows)
        external_value = external if external_value is None else external_value
        state = reconstruct(
            reduced,
            shadow,
            base_state,
            external_value,
            target_value,
            requested_value,
            rows,
        )
        moments = _current_moments(
            operator, state, _requested(requested_value), _target(target_value)
        )
        promoted = jnp.ravel(
            jnp.asarray(
                operator.residual_shadow_mask(
                    state, _requested(requested_value), previous_shadow=shadow
                ),
                dtype=bool,
            )
        )
        image = _image(moments, unknowns, bound, external_value)
        mapped = jnp.where(promoted, state, image)
        current = moments.cell_current
        retained = jnp.zeros_like(current).at[coordinates.cells].set(1.0)
        excluded = jnp.max(jnp.abs(jnp.where(retained > 0.0, 0.0, current)))
        gathered = _gather(coordinates, moments)
        scored = (
            None
            if augmentation is None
            else augmentation.rows(
                state, unknowns, promoted, bound.pairs, bound.target_current
            )
        )
        return (
            state,
            promoted,
            jnp.sum(promoted != shadow),
            _scores(state, mapped, unknowns, scored, bound)[1],
            gathered if augmentation is None else jnp.concatenate((gathered, unknowns)),
            excluded / jnp.maximum(jnp.max(jnp.abs(current)), 1.0e-30),
        )

    def initial_gather(
        state, external_value=None, target_value=None, requested_value=None
    ):
        """Return the reduced amplitudes one flux state drives."""
        del external_value
        return _gather(
            coordinates,
            _current_moments(
                operator, state, _requested(requested_value), _target(target_value)
            ),
        )

    def newton_direction(jacobian, residual):
        """Return the dense plain-Newton step of the reduced system."""
        return -jnp.linalg.solve(jacobian, residual)

    return {
        "reconstruct": jax.jit(reconstruct),
        "reduced_map": jax.jit(reduced_map),
        "reduced_residual": jax.jit(reduced_residual),
        "jacobian": jax.jit(jax.jacfwd(reduced_residual, argnums=0)),
        "flux_scores": jax.jit(flux_scores),
        "ladder": jax.jit(ladder),
        "leakage": jax.jit(leakage),
        "direction": jax.jit(newton_direction),
        "step_scores": jax.jit(step_scores),
        "grade": jax.jit(grade_scores),
        "tail": jax.jit(tail_grades),
        "boundary": jax.jit(trip_boundary),
        "initial_gather": jax.jit(initial_gather),
    }


def _eager_grades(kernels, reduced, direction, merit, shadow, base_state, census):
    """Score every backtracking grade, then take the first below the merit.

    ``numpy.argmax`` over the below-merit flags returns the earliest grade
    that lowers the merit, so the grades after it are scored and discarded.
    """
    scored, _ = _timed(kernels["ladder"], reduced, direction, shadow, base_state)
    census["map_evaluations"] += len(_BACKTRACKING_FACTORS)
    merits = np.asarray(scored[0])
    below = np.isfinite(merits) & (merits < merit)
    accepted = int(np.argmax(below)) if bool(below.any()) else -1
    return accepted, len(_BACKTRACKING_FACTORS), None


def _host_scores(scores: ReducedScores) -> ReducedScores:
    """Return one device score tuple with its three scalars read together.

    The three scalars leave the device in one transfer, so the norm the
    receipt reports costs nothing beyond the merit the selection was going to
    synchronise on.
    """
    norm, merit, flux_residual = jax.device_get(
        (scores.residual_norm, scores.merit, scores.flux_residual)
    )
    return ReducedScores(
        scores.residual, float(norm), float(merit), float(flux_residual)
    )


def _lowers(merit: float, incumbent: float) -> bool:
    """Return whether one scored merit is admissible below the incumbent."""
    return bool(np.isfinite(merit)) and merit < incumbent


def _score_grade(kernels, reduced, direction, factor, shadow, base_state, census):
    """Score one backtracking grade and read its scalars to the host."""
    candidate, scores = kernels["grade"](
        reduced,
        direction,
        jnp.asarray(factor, dtype=reduced.dtype),
        shadow,
        base_state,
    )
    census["map_evaluations"] += 1
    return candidate, _host_scores(scores)


def _first_accept_grades(
    kernels, reduced, direction, merit, shadow, base_state, census
):
    """Score grades in order and stop at the first that lowers the merit.

    The selection is the one the eager ladder makes, so scoring stops where
    that selection is already decided.  An accepted grade returns the state
    it scored together with its residual and merit, which the next Newton
    step reads instead of evaluating the map again at the same point.  Every
    grade is its own dispatch, which is what the batched tail removes.
    """
    for index, factor in enumerate(_BACKTRACKING_FACTORS):
        candidate, scores = _score_grade(
            kernels, reduced, direction, factor, shadow, base_state, census
        )
        if _lowers(scores.merit, merit):
            return index, index + 1, (candidate, scores)
    return -1, len(_BACKTRACKING_FACTORS), None


def _batched_tail_grades(
    kernels, reduced, direction, merit, shadow, base_state, census
):
    """Score the first grade alone, then the whole refused tail at once.

    The selection is unchanged: the earliest grade whose merit falls below
    the incumbent, which is the grade the eager ladder and the serial route
    both take.  A step accepted at full length still costs the one evaluation
    the serial route pays, and a step whose first grade is refused pays one
    further round trip for the remaining grades instead of one apiece.
    """
    candidate, scores = _score_grade(
        kernels,
        reduced,
        direction,
        _BACKTRACKING_FACTORS[0],
        shadow,
        base_state,
        census,
    )
    if _lowers(scores.merit, merit):
        return 0, 1, (candidate, scores)
    candidates, tail = kernels["tail"](reduced, direction, shadow, base_state)
    census["map_evaluations"] += len(_BACKTRACKING_FACTORS) - 1
    norms, merits, flux_residuals = jax.device_get(
        (tail.residual_norm, tail.merit, tail.flux_residual)
    )
    below = np.isfinite(merits) & (merits < merit)
    if not bool(below.any()):
        return -1, len(_BACKTRACKING_FACTORS), None
    index = int(np.argmax(below))
    promotion = (
        candidates[index],
        ReducedScores(
            tail.residual[index],
            float(norms[index]),
            float(merits[index]),
            float(flux_residuals[index]),
        ),
    )
    return index + 1, index + 2, promotion


#: The grader each scoring policy drives its backtracking ladder through.
_GRADERS = {
    LADDER_SCORING: _batched_tail_grades,
    SERIAL_LADDER_SCORING: _first_accept_grades,
    EAGER_LADDER_SCORING: _eager_grades,
}


def _incumbent_scores(kernels, reduced, shadow, base_state, census, scoring):
    """Return the residual, its sup norm and the flux scores at one state."""
    if scoring != EAGER_LADDER_SCORING:
        census["map_evaluations"] += 1
        return _host_scores(kernels["step_scores"](reduced, shadow, base_state))
    residual, _ = _timed(kernels["reduced_residual"], reduced, shadow, base_state)
    scores, _ = _timed(kernels["flux_scores"], reduced, shadow, base_state)
    census["map_evaluations"] += 2
    return ReducedScores(
        residual,
        float(jnp.max(jnp.abs(residual))),
        float(scores[0]),
        float(scores[1]),
    )


def _plain_newton_trip(
    kernels: dict[str, Callable[..., Any]],
    reduced: jax.Array,
    shadow: jax.Array,
    base_state: jax.Array,
    *,
    trip: int,
    newton_steps: int,
    tolerance: float,
    steps: list[ReducedNewtonStep],
    scoring: str = LADDER_SCORING,
) -> tuple[jax.Array, dict[str, float]]:
    """Take plain Newton steps on the reduced state of one frozen trip.

    The Jacobian is formed once, reused while its steps are accepted, and
    refreshed only when a whole backtracking ladder is refused.  A ladder
    refused by a freshly built Jacobian ends the trip.
    """
    jacobian, jacobian_wall = _timed(kernels["jacobian"], reduced, shadow, base_state)
    census = {
        "steps": 0,
        "jacobian_builds": 1,
        "rejected": 0,
        "jacobian_wall": jacobian_wall,
        "newton_wall": 0.0,
        "map_evaluations": 0,
    }
    grader = _GRADERS[scoring]
    eager = scoring == EAGER_LADDER_SCORING
    factors = _BACKTRACKING_FACTORS
    fresh = True
    carried: ReducedScores | None = None
    for index in range(newton_steps):
        started = time.perf_counter()
        before = census["map_evaluations"]
        if carried is None:
            scores = _incumbent_scores(
                kernels, reduced, shadow, base_state, census, scoring
            )
        else:
            scores, carried = carried, None
        if not np.isfinite(scores.flux_residual) or scores.flux_residual <= tolerance:
            break
        accepted = -1
        refreshed = False
        direction = None
        promotion = None
        for _attempt in range(2):
            if eager:
                direction, _ = _timed(kernels["direction"], jacobian, scores.residual)
            else:
                direction = kernels["direction"](jacobian, scores.residual)
            accepted, _tried, promotion = grader(
                kernels, reduced, direction, scores.merit, shadow, base_state, census
            )
            if accepted >= 0:
                break
            census["rejected"] += 1
            if fresh:
                break
            jacobian, refresh_wall = _timed(
                kernels["jacobian"], reduced, shadow, base_state
            )
            census["jacobian_builds"] += 1
            census["jacobian_wall"] += refresh_wall
            fresh = True
            refreshed = True
        wall = time.perf_counter() - started
        census["newton_wall"] += wall
        record = ReducedNewtonStep(
            trip=trip,
            step=index,
            reduced_residual=scores.residual_norm,
            flux_residual=scores.flux_residual,
            merit=scores.merit,
            accepted_factor=None if accepted < 0 else float(factors[accepted]),
            grades_tried=(
                len(factors) * (2 if refreshed else 1)
                if accepted < 0
                else accepted + 1 + len(factors) * refreshed
            ),
            map_evaluations=census["map_evaluations"] - before,
            jacobian_refreshed=refreshed,
            wall_s=wall,
        )
        steps.append(record)
        if accepted < 0:
            break
        if promotion is None:
            reduced = reduced + factors[accepted] * direction
        else:
            reduced, carried = promotion
        fresh = False
        census["steps"] += 1
    return reduced, census


def _drive_trips(
    kernels: dict[str, Callable[..., Any]],
    state: jax.Array,
    reduced: jax.Array,
    shadow: jax.Array,
    *,
    tolerance: float,
    newton_steps: int,
    active_set_steps: int,
    fused: bool,
    scoring: str,
    regather: Callable[[jax.Array], jax.Array],
    dispatched_boundary: Callable[..., Any],
    stream: bool,
) -> dict[str, Any]:
    """Run the active-set trips of one reduced solve and report every census.

    One topology read opens each trip and freezes the residual shadow; the
    trip's Newton steps run entirely on the reduced state; the trip closes on
    the promoted shadow the settled state induces, and the solve exits when
    that shadow stops moving, which is the production settled exit on the
    incumbent mask.  The reduced state the trips carry is the plasma-cell
    amplitude vector on its own, or that vector extended by the compensating
    unknowns of a constraint tuple; the loop is the same either way because
    every difference between the two lives inside the kernels.
    """
    residuals: list[float] = []
    differences: list[int] = []
    per_trip_steps: list[int] = []
    per_trip_builds: list[int] = []
    per_trip_rejected: list[int] = []
    per_trip_jacobian_wall: list[float] = []
    per_trip_newton_wall: list[float] = []
    per_trip_boundary_wall: list[float] = []
    per_trip_wall: list[float] = []
    per_trip_maps: list[int] = []
    steps: list[ReducedNewtonStep] = []
    reason = FixedPointTerminationReason.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED
    converged = False
    terminal_residual = float("inf")
    leakage = 0.0

    for trip in range(active_set_steps):
        trip_started = time.perf_counter()
        _stage_mark(f"trip{trip}:start")
        if not fused:
            reduced = regather(state)
        reduced, census = _plain_newton_trip(
            kernels,
            reduced,
            shadow,
            state,
            trip=trip,
            newton_steps=newton_steps,
            tolerance=tolerance,
            steps=steps,
            scoring=scoring,
        )
        per_trip_steps.append(int(census["steps"]))
        per_trip_builds.append(int(census["jacobian_builds"]))
        per_trip_rejected.append(int(census["rejected"]))
        per_trip_jacobian_wall.append(float(census["jacobian_wall"]))
        per_trip_newton_wall.append(float(census["newton_wall"]))
        per_trip_maps.append(int(census["map_evaluations"]))
        boundary_started = time.perf_counter()
        if fused:
            closed, _ = _timed(kernels["boundary"], reduced, shadow, state)
            state, promoted, difference, observed, carried, excluded = closed
            difference = int(difference)
            observed = float(observed)
            leakage = max(leakage, float(excluded))
            reduced = carried
        else:
            state, promoted, difference, observed, excluded = dispatched_boundary(
                reduced, shadow, state
            )
            leakage = max(leakage, excluded)
        _stage_mark(f"trip{trip}:boundary")
        per_trip_boundary_wall.append(time.perf_counter() - boundary_started)
        per_trip_wall.append(time.perf_counter() - trip_started)
        residuals.append(observed)
        differences.append(difference)
        terminal_residual = observed
        if stream:
            print(
                f"REDUCED-TRIP {trip} residual={observed:.6e} "
                f"difference={difference} steps={census['steps']} "
                f"jacobians={census['jacobian_builds']} "
                f"wall={per_trip_wall[-1]:.4f}",
                flush=True,
            )
        shadow = promoted
        if np.isfinite(observed) and observed <= tolerance and difference == 0:
            converged = True
            reason = FixedPointTerminationReason.CONVERGED
            break
        if difference == 0:
            reason = FixedPointTerminationReason.ACTIVE_SET_SETTLED
            break

    return {
        "state": state,
        "reduced": reduced,
        "terminal_residual": terminal_residual,
        "converged": converged,
        "reason": reason,
        "residuals": residuals,
        "differences": differences,
        "newton_steps_per_trip": per_trip_steps,
        "jacobian_builds_per_trip": per_trip_builds,
        "rejected_steps_per_trip": per_trip_rejected,
        "jacobian_wall_per_trip": per_trip_jacobian_wall,
        "newton_wall_per_trip": per_trip_newton_wall,
        "boundary_wall_per_trip": per_trip_boundary_wall,
        "trip_wall_per_trip": per_trip_wall,
        "map_evaluations_per_trip": per_trip_maps,
        "off_support_leakage": leakage,
        "steps": steps,
    }


def _compiled_slice_solver(
    kernels: dict[str, Callable[..., Any]],
    *,
    tolerance: float,
    newton_steps: int,
    active_set_steps: int,
    initial_unknown: jax.Array | None = None,
) -> Callable[..., Any]:
    """Build one fixed-shape program for a complete reduced solve.

    The host route deliberately exposes one trip at a time so its decisions
    can be inspected.  This route keeps the same dense Newton and grade
    expressions, but carries the trip state through fixed ``fori_loop``
    bodies.  Once a trip settles or converges, its carry is masked and simply
    passes through the remaining budget.  The arrays returned by the program
    are the complete receipt history, so the caller performs one
    ``device_get`` for the slice rather than one read per trip.
    """
    factors = jnp.asarray(_BACKTRACKING_FACTORS, dtype=jnp.float64)

    def choose(reduced, jacobian, shadow, base_state, merit):
        direction = kernels["direction"](
            jacobian,
            kernels["step_scores"](reduced, shadow, base_state).residual,
        )
        candidates = reduced[None, :] + factors[:, None] * direction[None, :]
        scored = jax.lax.map(
            lambda candidate: kernels["step_scores"](candidate, shadow, base_state),
            candidates,
        )
        valid = jnp.isfinite(scored.merit) & (scored.merit < merit)
        accepted = jnp.argmax(valid.astype(jnp.int32))
        found = jnp.any(valid)
        return found, accepted, candidates[accepted]

    def trip_body(reduced, shadow, base_state):
        jacobian = kernels["jacobian"](reduced, shadow, base_state)

        def step_body(_index, carry):
            reduced, jacobian, active, step_count, builds, rejected = carry

            def run_step(carry):
                reduced, jacobian, _active, step_count, builds, rejected = carry
                scores = kernels["step_scores"](reduced, shadow, base_state)
                finished = jnp.isfinite(scores.flux_residual) & (
                    scores.flux_residual <= tolerance
                )

                def no_step(_):
                    return reduced, jacobian, False, step_count, builds, rejected

                def try_step(_):
                    first = choose(
                        reduced,
                        jacobian,
                        shadow,
                        base_state,
                        scores.merit,
                    )

                    def refresh(_):
                        refreshed = kernels["jacobian"](reduced, shadow, base_state)
                        selected = choose(
                            reduced,
                            refreshed,
                            shadow,
                            base_state,
                            scores.merit,
                        )
                        return selected, refreshed, 1

                    selected, selected_jacobian, refreshes = jax.lax.cond(
                        ~first[0], refresh, lambda _: (first, jacobian, 0), None
                    )
                    found, accepted, candidate = selected
                    return (
                        jnp.where(found, candidate, reduced),
                        selected_jacobian,
                        found,
                        step_count + found.astype(jnp.int32),
                        builds + refreshes,
                        rejected + (~found).astype(jnp.int32),
                    )

                return jax.lax.cond(finished, no_step, try_step, None)

            return jax.lax.cond(active, run_step, lambda value: value, carry)

        return jax.lax.fori_loop(
            0,
            newton_steps,
            step_body,
            (
                reduced,
                jacobian,
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )

    def solve(
        initial,
        shadow,
        external_value=None,
        target_value=None,
        requested_value=None,
    ):
        del external_value, target_value, requested_value
        reduced = kernels["initial_gather"](initial)
        if initial_unknown is not None:
            reduced = jnp.concatenate((reduced, initial_unknown))
        state = initial
        active = jnp.asarray(True)
        converged = jnp.asarray(False)
        reason = jnp.asarray(
            int(FixedPointTerminationReason.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED),
            dtype=jnp.int32,
        )
        terminal_residual = jnp.asarray(jnp.inf, dtype=initial.dtype)
        iterations = jnp.asarray(0, dtype=jnp.int32)
        converged_trip = jnp.asarray(-1, dtype=jnp.int32)
        leakage = jnp.asarray(0.0, dtype=initial.dtype)
        residuals = jnp.full((active_set_steps,), jnp.nan, dtype=initial.dtype)
        differences = jnp.full((active_set_steps,), -1, dtype=jnp.int32)
        trip_steps = jnp.zeros((active_set_steps,), dtype=jnp.int32)
        trip_builds = jnp.zeros((active_set_steps,), dtype=jnp.int32)
        trip_rejected = jnp.zeros((active_set_steps,), dtype=jnp.int32)
        trip_maps = jnp.zeros((active_set_steps,), dtype=jnp.int32)

        def outer_body(index, carry):
            (
                state,
                reduced,
                shadow,
                active,
                converged,
                reason,
                terminal_residual,
                iterations,
                converged_trip,
                leakage,
                residuals,
                differences,
                trip_steps,
                trip_builds,
                trip_rejected,
                trip_maps,
            ) = carry

            def run_trip(_):
                (
                    solved_reduced,
                    jacobian_active,
                    step_count,
                    trip_active,
                    builds,
                    rejected,
                ) = trip_body(reduced, shadow, state)
                del jacobian_active, trip_active
                closed = kernels["boundary"](solved_reduced, shadow, state)
                next_state, promoted, difference, observed, next_reduced, excluded = (
                    closed
                )
                converged_now = (
                    jnp.isfinite(observed) & (observed <= tolerance) & (difference == 0)
                )
                settled = difference == 0
                still_active = ~converged_now & ~settled
                next_reason = jnp.where(
                    converged_now,
                    int(FixedPointTerminationReason.CONVERGED),
                    jnp.where(
                        settled,
                        int(FixedPointTerminationReason.ACTIVE_SET_SETTLED),
                        reason,
                    ),
                )
                return (
                    next_state,
                    next_reduced,
                    promoted,
                    still_active,
                    converged | converged_now,
                    next_reason,
                    observed,
                    iterations + 1,
                    jnp.where(converged_now, index, converged_trip),
                    jnp.maximum(leakage, excluded),
                    residuals.at[index].set(observed),
                    differences.at[index].set(difference),
                    trip_steps.at[index].set(step_count),
                    trip_builds.at[index].set(builds),
                    trip_rejected.at[index].set(rejected),
                    trip_maps.at[index].set(step_count),
                )

            return jax.lax.cond(active, run_trip, lambda value: value, carry)

        return jax.lax.fori_loop(
            0,
            active_set_steps,
            outer_body,
            (
                state,
                reduced,
                shadow,
                active,
                converged,
                reason,
                terminal_residual,
                iterations,
                converged_trip,
                leakage,
                residuals,
                differences,
                trip_steps,
                trip_builds,
                trip_rejected,
                trip_maps,
            ),
        )

    return jax.jit(solve)


def _validate_solve_policy(ladder_scoring: str, trip_boundary: str) -> bool:
    """Return whether the trip boundary is fused, refusing unknown policies."""
    if ladder_scoring not in _GRADERS:
        raise ValueError(f"unknown ladder scoring {ladder_scoring!r}")
    if trip_boundary not in (TRIP_BOUNDARY, DISPATCHED_TRIP_BOUNDARY):
        raise ValueError(f"unknown trip boundary {trip_boundary!r}")
    return trip_boundary == TRIP_BOUNDARY


def solve_reduced_newton(
    operator,
    initial,
    *,
    requested_class=None,
    target_current=None,
    current=None,
    prescribed_current=None,
    tolerance: float = FIXED_POINT_RESIDUAL_TOLERANCE,
    newton_steps: int = NEWTON_STEPS,
    active_set_steps: int = ACTIVE_SET_STEPS,
    support_policy: str = SUPPORT_POLICY,
    support_floor: float = _ACTIVE_SUPPORT_FLOOR,
    ladder_scoring: str = LADDER_SCORING,
    trip_boundary: str = TRIP_BOUNDARY,
    program: "ReducedProgram | None" = None,
    stream: bool = False,
) -> ReducedNewtonResult:
    """Drive the reduced fixed point through the production trip structure.

    One topology read opens each trip and freezes the residual shadow; the
    trip's Newton steps run entirely on the reduced amplitudes; the trip closes
    on the promoted shadow the settled state induces, and the solve exits when
    that shadow stops moving, which is the production settled exit on the
    incumbent mask.  The reduced coordinates are selected once, at the seed, so
    that every trip shares one compiled program; each trip reports the current
    the map would drive outside them, which is zero when the selection holds.

    ``ladder_scoring`` selects between stopping at the first grade below the
    incumbent merit and scoring every grade before selecting; ``trip_boundary``
    selects between closing a trip in one compiled program and closing it
    through separate calls.  The alternatives compute the same decisions and
    exist so a measurement can read one against the other.
    """
    state = jnp.asarray(initial)
    target_current_value = (
        None if target_current is None else jnp.asarray(target_current)
    )
    # A carried program keeps the external flux it was built with.  When the
    # caller's conductor currents are the default ones (both ``None``) and the
    # program was itself built on the defaults, that external is exactly what
    # this call would recompute, so reusing it skips one per-keyframe device
    # evaluation.  A program built under explicit currents is never reused
    # this way.
    default_external = current is None and prescribed_current is None
    if program is not None and default_external and program.default_external:
        external = program.external
    else:
        external = operator.external(current, prescribed_current)
    _stage_mark("external")
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(state, requested_class), dtype=bool)
    )
    if program is None:
        coordinates = reduced_coordinates(
            operator,
            state,
            requested_class=requested_class,
            target_current=target_current_value,
            policy=support_policy,
            floor=support_floor,
        )
        program = ReducedProgram(
            coordinates=coordinates,
            external=external,
            kernels=_reduced_kernels(
                operator,
                coordinates,
                external,
                requested_class,
                target_current_value,
            ),
            row_count=0,
            operator_identity=id(operator),
            external_shape=tuple(external.shape),
            target_current_shape=(
                None
                if target_current_value is None
                else tuple(target_current_value.shape)
            ),
            requested_class_shape=(
                None if requested_class is None else tuple(np.shape(requested_class))
            ),
            default_external=default_external,
        )
    else:
        expected_target_shape = (
            None if target_current_value is None else tuple(target_current_value.shape)
        )
        expected_requested_shape = (
            None if requested_class is None else tuple(np.shape(requested_class))
        )
        if (
            program.operator_identity != id(operator)
            or program.external_shape != tuple(external.shape)
            or program.target_current_shape != expected_target_shape
            or program.requested_class_shape != expected_requested_shape
            or program.row_count != 0
        ):
            raise ValueError(
                "the reduced program does not match this operator's static shapes"
            )
    coordinates = program.coordinates
    kernels = _bind_dynamic_arguments(
        program.kernels, external, target_current_value, requested_class
    )
    fused = _validate_solve_policy(ladder_scoring, trip_boundary)

    def regather(state):
        """Return the amplitudes one flux state drives, outside a program."""
        moments = _current_moments(operator, state, requested_class, target_current)
        return _gather(coordinates, moments)

    def dispatched_boundary(reduced, shadow, base_state):
        """Close one trip through the separate calls the fused program folds."""
        leakage = float(kernels["leakage"](reduced, shadow, base_state))
        state, _ = _timed(kernels["reconstruct"], reduced, shadow, base_state)
        promoted = jnp.ravel(
            jnp.asarray(
                operator.residual_shadow_mask(
                    state, requested_class, previous_shadow=shadow
                ),
                dtype=bool,
            )
        )
        difference = int(jnp.sum(promoted != shadow))
        mapped = operator._exclude_shadow_residual(
            state,
            external + operator.internal(state, requested_class, target_current_value),
            requested_class,
            shadow=promoted,
        )
        observed = float(_relative_residual(mapped, state))
        return state, promoted, difference, observed, leakage

    driven = _drive_trips(
        kernels,
        state,
        kernels["initial_gather"](state) if fused else None,
        shadow,
        tolerance=tolerance,
        newton_steps=newton_steps,
        active_set_steps=active_set_steps,
        fused=fused,
        scoring=ladder_scoring,
        regather=regather,
        dispatched_boundary=dispatched_boundary,
        stream=stream,
    )
    return ReducedNewtonResult(
        state=driven["state"],
        terminal_residual=driven["terminal_residual"],
        active_set_iterations=len(driven["residuals"]),
        converged=driven["converged"],
        termination_reason=int(driven["reason"]),
        active_set_residuals=driven["residuals"],
        active_set_mask_differences=driven["differences"],
        newton_steps_per_trip=driven["newton_steps_per_trip"],
        jacobian_builds_per_trip=driven["jacobian_builds_per_trip"],
        rejected_steps_per_trip=driven["rejected_steps_per_trip"],
        jacobian_wall_per_trip=driven["jacobian_wall_per_trip"],
        newton_wall_per_trip=driven["newton_wall_per_trip"],
        boundary_wall_per_trip=driven["boundary_wall_per_trip"],
        trip_wall_per_trip=driven["trip_wall_per_trip"],
        map_evaluations_per_trip=driven["map_evaluations_per_trip"],
        reduced_dimension=coordinates.size,
        support_cells=int(coordinates.cells.size),
        off_support_leakage=driven["off_support_leakage"],
        steps=driven["steps"],
        program=program,
    )


class _RowArguments(NamedTuple):
    """The constraint data one solve varies without rebuilding its program.

    :class:`~nova.equilibrium.constraint.ConstraintPair` flattens to exactly
    the numerical leaves a keyframe edits - the row target, its tolerance and
    scale, the initial unknown, and the compensating direction with its
    ampere scale - and keeps the functional type and the solve policy in its
    tree structure.  Handing the tuple itself to a kernel as a traced
    argument therefore moves a commanded target without changing the
    program's identity, while a change of row kind or policy still builds the
    different program it is.  ``flux_scale`` travels beside it because it is
    read off the state the solve starts from and so moves with the keyframe.
    """

    pairs: tuple[ConstraintPair, ...]
    flux_scale: jax.Array
    target_current: Any


class ReducedProgram(NamedTuple):
    """One built reduced solve a later keyframe can re-enter.

    The coordinates and the external flux travel with the kernels because
    reuse is only meaningful while the layout holds: the amplitudes a later
    solve carries are indices into the same cells, and the flux those
    amplitudes reconstruct sits behind the same conductor state.  A caller
    that changes either builds a new program rather than re-entering this one.
    """

    coordinates: ReducedCoordinates
    external: jax.Array
    kernels: dict[str, Callable[..., Any]]
    row_count: int
    operator_identity: int = 0
    external_shape: tuple[int, ...] = ()
    target_current_shape: tuple[int, ...] | None = None
    requested_class_shape: tuple[int, ...] | None = None
    row_signature: tuple[tuple[str, str], ...] = ()
    #: Whether the carried external was computed from the default conductor
    #: currents (both ``None``), which is what makes reusing it on a later
    #: solve exact.  A program built under explicit currents always recomputes.
    default_external: bool = True
    slice_solver: Callable[..., Any] | None = None


def _bind_rows(
    kernels: dict[str, Callable[..., Any]], arguments: _RowArguments | None
) -> dict[str, Callable[..., Any]]:
    """Return the kernels with one solve's varying row data already bound.

    Binding here rather than threading the argument through the trip loop
    keeps the loop reading the same kernel signatures it read before, so the
    only thing the traced route changes is where a kernel finds its targets.
    ``direction`` and ``initial_gather`` never see a row, so they are handed
    back unbound.
    """
    if arguments is None:
        return kernels
    unbound = ("direction", "initial_gather")
    return {
        name: kernel if name in unbound else partial(kernel, rows=arguments)
        for name, kernel in kernels.items()
    }


def _bind_dynamic_arguments(
    kernels: dict[str, Callable[..., Any]],
    external: jax.Array,
    target_current: Any,
    requested_class: Any,
) -> dict[str, Callable[..., Any]]:
    """Bind per-slice field leaves as regular traced kernel arguments."""
    bound = {}
    for name, kernel in kernels.items():
        if name == "direction":
            bound[name] = kernel
            continue
        value = partial(kernel, external_value=external)
        if target_current is not None:
            value = partial(value, target_value=target_current)
        if requested_class is not None:
            value = partial(value, requested_value=requested_class)
        bound[name] = value
    return bound


@dataclass(frozen=True)
class _RowAugmentation:
    """The constraint block a reduced state and its residual are extended by.

    The reduced route poses its fixed point on amplitudes rather than on flux,
    so the flux a state reconstructs is a function of the unknowns rather than
    an unknown of its own.  A compensator whose flux image read the state would
    therefore make that reconstruction implicit, which is why only the
    circuit-current compensator is admitted here: its image is the prescribed
    response acting on a current, and it is linear in the unknown.
    """

    profile: Any
    pairs: tuple[ConstraintPair, ...]
    row_slices: tuple[slice, ...]
    row_count: int
    flux_scale: jax.Array
    requested_class: Any
    target_current: Any

    @property
    def arguments(self) -> "_RowArguments":
        """Return the row data a program bakes in when none is handed to it."""
        return _RowArguments(self.pairs, self.flux_scale, self.target_current)

    def flux_delta(self, unknowns, pairs) -> jax.Array:
        """Return the flux every compensating unknown drives at this state."""
        delta = 0.0
        for pair, row_slice in zip(pairs, self.row_slices, strict=True):
            delta = delta + pair.unknown.flux_delta(
                self.profile,
                None,
                pair.functional,
                pair.binding.payload,
                unknowns[row_slice],
            )
        return delta

    def rows(self, flux, unknowns, shadow, pairs, target_current) -> jax.Array:
        """Return every authored constraint residual row at one state."""
        context = ConstraintContext(flux, self.requested_class, target_current, shadow)
        return jnp.concatenate(
            tuple(
                jnp.ravel(
                    pair.functional.residual(
                        self.profile,
                        context,
                        unknowns[row_slice],
                        pair.binding.payload,
                        jnp.asarray(pair.binding.target),
                        jnp.asarray(pair.binding.scale),
                    )
                )
                for pair, row_slice in zip(pairs, self.row_slices, strict=True)
            )
        )


class _RowRecordView(NamedTuple):
    """The two members :func:`constraint_records` reads off a solved system.

    The reduced route's state is amplitudes and unknowns rather than flux and
    unknowns, so the terminal record is written from the flux that state
    reconstructs.  Sharing the reader keeps one definition of what a row
    achieved rather than a second copy that can drift from it.
    """

    row_slices: tuple[slice, ...]
    flux: jax.Array
    unknowns: jax.Array

    def split(self, state):
        """Return the terminal flux and unknowns this view already holds."""
        del state
        return self.flux, self.unknowns


@dataclass
class ConstrainedReducedNewtonResult(ReducedNewtonResult):
    """A reduced solve that also carried a tuple of constraint rows.

    ``state`` remains the terminal flux, so every field the unconstrained
    result declares means what it means there.  The compensating unknowns and
    the terminal row records are what the augmentation adds, together with the
    prescribed circuit currents the compensation implies, which is the vector a
    caller drives the machine with on the next keyframe.
    """

    compensating_unknown: jax.Array | None = None
    constraints: tuple[ConstraintRecord, ...] = ()
    prescribed_current: jax.Array | None = None
    row_count: int = 0
    program: "ReducedProgram | None" = None


def _row_augmentation(
    profile,
    pairs: tuple[ConstraintPair, ...],
    seed: jax.Array,
    *,
    requested_class,
    target_current,
) -> _RowAugmentation:
    """Validate one constraint tuple and bind it to this solve's flux scale."""
    for pair in pairs:
        if not isinstance(pair, ConstraintPair):
            raise TypeError("constraint_pairs must contain ConstraintPair values")
        if not isinstance(pair.unknown, CircuitCurrentUnknown):
            raise TypeError(
                "the reduced route admits circuit-current compensators only; a "
                "compensator whose flux image reads the state would make the "
                "amplitude reconstruction implicit"
            )
    row_slices = constraint_row_slices(pairs)
    return _RowAugmentation(
        profile=profile,
        pairs=pairs,
        row_slices=row_slices,
        row_count=row_slices[-1].stop,
        flux_scale=jnp.maximum(jnp.max(jnp.abs(seed)), jnp.finfo(seed.dtype).tiny),
        requested_class=requested_class,
        target_current=target_current,
    )


def solve_constrained_reduced_newton(
    profile,
    initial,
    *,
    constraint_pairs: tuple[ConstraintPair, ...] = (),
    requested_class=None,
    target_current=None,
    current=None,
    prescribed_current=None,
    tolerance: float = FIXED_POINT_RESIDUAL_TOLERANCE,
    newton_steps: int = NEWTON_STEPS,
    active_set_steps: int = ACTIVE_SET_STEPS,
    support_policy: str = SUPPORT_POLICY,
    support_floor: float = _ACTIVE_SUPPORT_FLOOR,
    ladder_scoring: str = LADDER_SCORING,
    trip_boundary: str = TRIP_BOUNDARY,
    row_arguments: str = TRACED_ROWS,
    program: ReducedProgram | None = None,
    stream: bool = False,
) -> ConstrainedReducedNewtonResult:
    """Drive the reduced fixed point with constraint rows on the same state.

    The compensating unknowns extend the amplitude vector and the authored
    constraint residuals extend the amplitude residual, so the square dense
    system one trip solves grows by one row and one column per constraint row
    and nothing else about the trip structure changes.  Scoring follows the
    augmented Newton-Krylov route in what it measures - the flux normalised by
    its own span beside the unknowns, so a row that is far from its target is
    visible to the line search rather than hidden under the flux block - but
    both scores divide by the flux block's own reference, because a normalised
    compensating unknown is bounded by nothing and must not set the scale the
    whole system is judged against.

    A constrained solve needs a prescribed current to fold its compensating
    currents into: either ``prescribed_current`` or an operator-side
    ``prescribed_current_field``.  With neither, the route refuses before any
    compiler runs, with the reason stated, rather than solving an equilibrium
    whose flux carries compensation its receipt does not record.  Rows' soft
    mode projections are published as ``None`` because a dense route computes
    none.

    An empty constraint tuple is the unconstrained solve.  Every kernel then
    evaluates the expression :func:`solve_reduced_newton` evaluates, so the two
    entries return the same numbers rather than nearly the same ones.

    ``row_arguments`` decides where a kernel finds its targets: traced, they
    arrive as an argument and a commanded target moves inside the program the
    caller already has; captured, they are baked in when the program is built
    and every new target is a new program.  The two compute the same numbers
    and the captured route stays reachable because that agreement is what
    makes the traced one safe to steer with.

    ``program`` is a built solve returned by an earlier call.  Handing it back
    is what turns a sequence of keyframes into one compiled program: the
    coordinates and the external flux come from it rather than being derived
    again, so the reduced state stays in the cells the program was built
    around.  The result carries its program either way, so a loop is a chain
    of calls each passing the last one's.
    """
    operator = profile.operator
    state = jnp.asarray(initial)
    pairs = tuple(constraint_pairs)
    target_current_value = (
        None if target_current is None else jnp.asarray(target_current)
    )
    _stage_mark("convert")
    if (
        pairs
        and prescribed_current is None
        and operator.prescribed_current_field is None
    ):
        raise ValueError(
            "a constrained reduced solve needs a prescribed current field to "
            "fold its rows' compensating currents into; pass prescribed_current "
            "or set the operator's prescribed field"
        )
    augmentation = (
        None
        if not pairs
        else _row_augmentation(
            profile,
            pairs,
            state,
            requested_class=requested_class,
            target_current=target_current_value,
        )
    )
    fused = _validate_solve_policy(ladder_scoring, trip_boundary)
    if pairs and not fused:
        raise ValueError(
            "the constrained reduced route closes its trips inside one program; "
            "the dispatched boundary rebuilds the reduced state from the flux "
            "alone and would discard the compensating unknowns"
        )
    if row_arguments not in (TRACED_ROWS, CAPTURED_ROWS):
        raise ValueError(f"unknown row arguments {row_arguments!r}")
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(state, requested_class), dtype=bool)
    )
    _stage_mark("shadow")
    default_external = current is None and prescribed_current is None
    if program is not None and default_external and program.default_external:
        external = program.external
    else:
        external = operator.external(current, prescribed_current)
    _stage_mark("external")
    row_signature = tuple(
        (
            type(pair.functional).__qualname__,
            type(pair.unknown).__qualname__,
        )
        for pair in pairs
    )
    if program is None:
        coordinates = reduced_coordinates(
            operator,
            state,
            requested_class=requested_class,
            target_current=target_current_value,
            policy=support_policy,
            floor=support_floor,
        )
        program = ReducedProgram(
            coordinates=coordinates,
            external=external,
            kernels=_reduced_kernels(
                operator,
                coordinates,
                external,
                requested_class,
                target_current_value,
                augmentation=augmentation,
            ),
            row_count=0 if augmentation is None else augmentation.row_count,
            operator_identity=id(operator),
            external_shape=tuple(external.shape),
            target_current_shape=(
                None
                if target_current_value is None
                else tuple(target_current_value.shape)
            ),
            requested_class_shape=(
                None if requested_class is None else tuple(np.shape(requested_class))
            ),
            row_signature=row_signature,
            default_external=default_external,
        )
    elif (
        program.operator_identity != id(operator)
        or program.external_shape != tuple(external.shape)
        or program.target_current_shape
        != (None if target_current_value is None else tuple(target_current_value.shape))
        or program.requested_class_shape
        != (None if requested_class is None else tuple(np.shape(requested_class)))
        or program.row_count != (0 if augmentation is None else augmentation.row_count)
        or program.row_signature != row_signature
    ):
        raise ValueError(
            "the reduced program does not match this operator's static shapes"
        )
    coordinates = program.coordinates
    kernels = _bind_dynamic_arguments(
        program.kernels, external, target_current_value, requested_class
    )
    kernels = _bind_rows(
        kernels,
        None
        if augmentation is None or row_arguments == CAPTURED_ROWS
        else augmentation.arguments,
    )
    _stage_mark("bind")

    def regather(state):
        """Return the amplitudes one flux state drives, outside a program."""
        moments = _current_moments(operator, state, requested_class, target_current)
        return _gather(coordinates, moments)

    def dispatched_boundary(reduced, shadow, base_state):
        """Close one trip through the separate calls the fused program folds."""
        leakage = float(kernels["leakage"](reduced, shadow, base_state))
        state, _ = _timed(kernels["reconstruct"], reduced, shadow, base_state)
        promoted = jnp.ravel(
            jnp.asarray(
                operator.residual_shadow_mask(
                    state, requested_class, previous_shadow=shadow
                ),
                dtype=bool,
            )
        )
        difference = int(jnp.sum(promoted != shadow))
        mapped = operator._exclude_shadow_residual(
            state,
            external + operator.internal(state, requested_class, target_current_value),
            requested_class,
            shadow=promoted,
        )
        observed = float(_relative_residual(mapped, state))
        return state, promoted, difference, observed, leakage

    amplitudes = kernels["initial_gather"](state) if fused else None
    if augmentation is not None:
        amplitudes = jnp.concatenate(
            (
                amplitudes,
                jnp.concatenate(
                    tuple(
                        jnp.ravel(jnp.asarray(pair.binding.initial_unknown))
                        for pair in pairs
                    )
                ),
            )
        )
    _stage_mark("gather")
    driven = _drive_trips(
        kernels,
        state,
        amplitudes,
        shadow,
        tolerance=tolerance,
        newton_steps=newton_steps,
        active_set_steps=active_set_steps,
        fused=fused,
        scoring=ladder_scoring,
        regather=regather,
        dispatched_boundary=dispatched_boundary,
        stream=stream,
    )
    _stage_mark("trips")
    unknowns = None if augmentation is None else driven["reduced"][coordinates.size :]
    records: tuple[ConstraintRecord, ...] = ()
    circuit_current = (
        None if prescribed_current is None else jnp.asarray(prescribed_current)
    )
    if circuit_current is None and operator.prescribed_current_field is not None:
        circuit_current = jnp.asarray(operator.prescribed_current_field.current)
    if augmentation is not None:
        records = constraint_records(
            profile,
            _RowRecordView(augmentation.row_slices, driven["state"], unknowns),
            driven["state"],
            pairs,
            jnp.full(augmentation.row_count, jnp.nan),
            requested_class=requested_class,
            target_current=target_current_value,
        )
        # A dense route has no soft-mode projection; a NaN would report a
        # number that does not exist rather than the absence a consumer can
        # distinguish.  Publish the typed absence.
        records = tuple(
            record._replace(soft_mode_projection=None) for record in records
        )
        if circuit_current is not None:
            for pair, record in zip(pairs, records, strict=True):
                circuit_current = circuit_current + (
                    jnp.asarray(pair.unknown.direction) @ record.physical_unknown
                )
    _stage_mark("records")
    return ConstrainedReducedNewtonResult(
        state=driven["state"],
        terminal_residual=driven["terminal_residual"],
        active_set_iterations=len(driven["residuals"]),
        converged=driven["converged"],
        termination_reason=int(driven["reason"]),
        active_set_residuals=driven["residuals"],
        active_set_mask_differences=driven["differences"],
        newton_steps_per_trip=driven["newton_steps_per_trip"],
        jacobian_builds_per_trip=driven["jacobian_builds_per_trip"],
        rejected_steps_per_trip=driven["rejected_steps_per_trip"],
        jacobian_wall_per_trip=driven["jacobian_wall_per_trip"],
        newton_wall_per_trip=driven["newton_wall_per_trip"],
        boundary_wall_per_trip=driven["boundary_wall_per_trip"],
        trip_wall_per_trip=driven["trip_wall_per_trip"],
        map_evaluations_per_trip=driven["map_evaluations_per_trip"],
        reduced_dimension=coordinates.size,
        support_cells=int(coordinates.cells.size),
        off_support_leakage=driven["off_support_leakage"],
        steps=driven["steps"],
        compensating_unknown=unknowns,
        constraints=records,
        prescribed_current=circuit_current,
        row_count=0 if augmentation is None else augmentation.row_count,
        program=program,
    )


def _compiled_output_fields(output):
    """Convert one compiled call's fixed-shape receipt arrays to host fields."""
    host = jax.device_get(output)
    iterations = int(host[7])
    return {
        "state": jnp.asarray(host[0]),
        "reduced": jnp.asarray(host[1]),
        "terminal_residual": float(host[6]),
        "active_set_iterations": iterations,
        "converged": bool(host[4]),
        "termination_reason": int(host[5]),
        "active_set_residuals": [float(value) for value in host[10][:iterations]],
        "active_set_mask_differences": [int(value) for value in host[11][:iterations]],
        "newton_steps_per_trip": [int(value) for value in host[12][:iterations]],
        "jacobian_builds_per_trip": [int(value) for value in host[13][:iterations]],
        "rejected_steps_per_trip": [int(value) for value in host[14][:iterations]],
        "map_evaluations_per_trip": [int(value) for value in host[15][:iterations]],
        "off_support_leakage": float(host[9]),
        "converged_trip": int(host[8]),
    }


def _compiled_program(
    operator,
    state,
    *,
    requested_class,
    target_current,
    external,
    program,
    augmentation=None,
):
    """Return the static program and the kernels for one compiled call."""
    row_count = 0 if augmentation is None else augmentation.row_count
    row_signature = (
        ()
        if augmentation is None
        else tuple(
            (
                type(pair.functional).__qualname__,
                type(pair.unknown).__qualname__,
            )
            for pair in augmentation.pairs
        )
    )
    target_shape = None if target_current is None else tuple(target_current.shape)
    requested_shape = (
        None if requested_class is None else tuple(np.shape(requested_class))
    )
    if program is None:
        coordinates = reduced_coordinates(
            operator,
            state,
            requested_class=requested_class,
            target_current=target_current,
            policy=SUPPORT_POLICY,
            floor=_ACTIVE_SUPPORT_FLOOR,
        )
        program = ReducedProgram(
            coordinates=coordinates,
            external=external,
            kernels=_reduced_kernels(
                operator,
                coordinates,
                external,
                requested_class,
                target_current,
                augmentation=augmentation,
            ),
            row_count=row_count,
            operator_identity=id(operator),
            external_shape=tuple(external.shape),
            target_current_shape=target_shape,
            requested_class_shape=requested_shape,
            row_signature=row_signature,
            default_external=False,
        )
    elif (
        program.operator_identity != id(operator)
        or program.external_shape != tuple(external.shape)
        or program.target_current_shape != target_shape
        or program.requested_class_shape != requested_shape
        or program.row_count != row_count
        or program.row_signature != row_signature
    ):
        raise ValueError(
            "the reduced program does not match this operator's static shapes"
        )
    return program, program.kernels


def _compiled_result(
    operator,
    initial,
    *,
    requested_class,
    target_current,
    external,
    program,
    augmentation,
    row_arguments,
    tolerance,
    newton_steps,
    active_set_steps,
):
    """Run a complete slice and synchronise its fixed-shape receipt once."""
    program, raw_kernels = _compiled_program(
        operator,
        initial,
        requested_class=requested_class,
        target_current=target_current,
        external=external,
        program=program,
        augmentation=augmentation,
    )
    kernels = _bind_dynamic_arguments(
        raw_kernels, external, target_current, requested_class
    )
    if augmentation is not None and row_arguments == TRACED_ROWS:
        kernels = _bind_rows(kernels, augmentation.arguments)
    initial_unknown = None
    if augmentation is not None:
        initial_unknown = jnp.concatenate(
            tuple(
                jnp.ravel(jnp.asarray(pair.binding.initial_unknown))
                for pair in augmentation.pairs
            )
        )
    solver = _compiled_slice_solver(
        kernels,
        tolerance=tolerance,
        newton_steps=newton_steps,
        active_set_steps=active_set_steps,
        initial_unknown=initial_unknown,
    )
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(initial, requested_class), dtype=bool)
    )
    output = solver(initial, shadow)
    fields = _compiled_output_fields(output)
    return fields, program._replace(slice_solver=solver)


def solve_reduced_newton_compiled(
    operator,
    initial,
    *,
    requested_class=None,
    target_current=None,
    current=None,
    prescribed_current=None,
    tolerance: float = FIXED_POINT_RESIDUAL_TOLERANCE,
    newton_steps: int = NEWTON_STEPS,
    active_set_steps: int = ACTIVE_SET_STEPS,
    program: "ReducedProgram | None" = None,
    stream: bool = False,
) -> ReducedNewtonResult:
    """Solve one slice in one compiled call with a fixed trip budget."""
    del stream
    state = jnp.asarray(initial)
    target_value = None if target_current is None else jnp.asarray(target_current)
    external = operator.external(current, prescribed_current)
    fields, program = _compiled_result(
        operator,
        state,
        requested_class=requested_class,
        target_current=target_value,
        external=external,
        program=program,
        augmentation=None,
        row_arguments=TRACED_ROWS,
        tolerance=tolerance,
        newton_steps=newton_steps,
        active_set_steps=active_set_steps,
    )
    return ReducedNewtonResult(
        state=fields["state"],
        terminal_residual=fields["terminal_residual"],
        active_set_iterations=fields["active_set_iterations"],
        converged=fields["converged"],
        termination_reason=fields["termination_reason"],
        active_set_residuals=fields["active_set_residuals"],
        active_set_mask_differences=fields["active_set_mask_differences"],
        newton_steps_per_trip=fields["newton_steps_per_trip"],
        jacobian_builds_per_trip=fields["jacobian_builds_per_trip"],
        rejected_steps_per_trip=fields["rejected_steps_per_trip"],
        map_evaluations_per_trip=fields["map_evaluations_per_trip"],
        reduced_dimension=program.coordinates.size,
        support_cells=int(program.coordinates.cells.size),
        off_support_leakage=fields["off_support_leakage"],
        program=program,
    )


def solve_constrained_reduced_newton_compiled(
    profile,
    initial,
    *,
    constraint_pairs: tuple[ConstraintPair, ...] = (),
    requested_class=None,
    target_current=None,
    current=None,
    prescribed_current=None,
    tolerance: float = FIXED_POINT_RESIDUAL_TOLERANCE,
    newton_steps: int = NEWTON_STEPS,
    active_set_steps: int = ACTIVE_SET_STEPS,
    row_arguments: str = TRACED_ROWS,
    program: ReducedProgram | None = None,
    stream: bool = False,
) -> ConstrainedReducedNewtonResult:
    """Solve one constrained slice in one compiled call."""
    del stream
    if not constraint_pairs:
        result = solve_reduced_newton_compiled(
            profile.operator,
            initial,
            requested_class=requested_class,
            target_current=target_current,
            current=current,
            prescribed_current=prescribed_current,
            tolerance=tolerance,
            newton_steps=newton_steps,
            active_set_steps=active_set_steps,
            program=program,
        )
        return ConstrainedReducedNewtonResult(**result.__dict__)
    if prescribed_current is None and profile.operator.prescribed_current_field is None:
        raise ValueError(
            "a constrained reduced solve needs a prescribed current field to "
            "fold its rows' compensating currents into; pass prescribed_current "
            "or set the operator's prescribed field"
        )
    if row_arguments not in (TRACED_ROWS, CAPTURED_ROWS):
        raise ValueError(f"unknown row arguments {row_arguments!r}")
    state = jnp.asarray(initial)
    target_value = None if target_current is None else jnp.asarray(target_current)
    pairs = tuple(constraint_pairs)
    augmentation = _row_augmentation(
        profile,
        pairs,
        state,
        requested_class=requested_class,
        target_current=target_value,
    )
    external = profile.operator.external(current, prescribed_current)
    fields, program = _compiled_result(
        profile.operator,
        state,
        requested_class=requested_class,
        target_current=target_value,
        external=external,
        program=program,
        augmentation=augmentation,
        row_arguments=row_arguments,
        tolerance=tolerance,
        newton_steps=newton_steps,
        active_set_steps=active_set_steps,
    )
    unknowns = fields["reduced"][program.coordinates.size :]
    records = constraint_records(
        profile,
        _RowRecordView(augmentation.row_slices, fields["state"], unknowns),
        fields["state"],
        pairs,
        jnp.full(augmentation.row_count, jnp.nan),
        requested_class=requested_class,
        target_current=target_value,
    )
    records = tuple(record._replace(soft_mode_projection=None) for record in records)
    circuit_current = (
        jnp.asarray(prescribed_current)
        if prescribed_current is not None
        else jnp.asarray(profile.operator.prescribed_current_field.current)
    )
    for pair, record in zip(pairs, records, strict=True):
        circuit_current = circuit_current + (
            jnp.asarray(pair.unknown.direction) @ record.physical_unknown
        )
    return ConstrainedReducedNewtonResult(
        state=fields["state"],
        terminal_residual=fields["terminal_residual"],
        active_set_iterations=fields["active_set_iterations"],
        converged=fields["converged"],
        termination_reason=fields["termination_reason"],
        active_set_residuals=fields["active_set_residuals"],
        active_set_mask_differences=fields["active_set_mask_differences"],
        newton_steps_per_trip=fields["newton_steps_per_trip"],
        jacobian_builds_per_trip=fields["jacobian_builds_per_trip"],
        rejected_steps_per_trip=fields["rejected_steps_per_trip"],
        map_evaluations_per_trip=fields["map_evaluations_per_trip"],
        reduced_dimension=program.coordinates.size,
        support_cells=int(program.coordinates.cells.size),
        off_support_leakage=fields["off_support_leakage"],
        compensating_unknown=unknowns,
        constraints=records,
        prescribed_current=circuit_current,
        row_count=augmentation.row_count,
        program=program,
    )
