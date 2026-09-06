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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
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
    "solve_reduced_newton",
]


ACTIVE_SET_STEPS = 16
NEWTON_STEPS = 12
SUPPORT_POLICY = "participation"
_ACTIVE_SUPPORT_FLOOR = 0.0

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

    def _image(moments, unknowns, bound):
        """Return the flux image of one moment set and its compensation."""
        image = external + operator.current_moment_image(moments)
        if augmentation is None:
            return image
        return image + augmentation.flux_delta(unknowns, bound.pairs)

    def _augment(residual, state, unknowns, shadow, bound):
        """Return the residual with the authored rows appended to it."""
        if augmentation is None:
            return residual, None
        rows = augmentation.rows(state, unknowns, shadow, bound.pairs)
        return jnp.concatenate((residual, rows)), rows

    def _scores(state, mapped, unknowns, rows, bound):
        """Return the merit and relative residual the selection reads.

        Unaugmented these are the production flux-space scores.  With rows
        present they are the same two functions of the state the augmented
        Newton-Krylov route scores: the flux normalised by its own span beside
        the unknowns, so a row far from its target is visible to the line
        search instead of hidden under the flux block.
        """
        if augmentation is None:
            return (
                _smooth_relative_sup_merit(mapped, state),
                _relative_residual(mapped, state),
            )
        scale = bound.flux_scale
        scored = jnp.concatenate((state / scale, unknowns))
        imaged = jnp.concatenate((mapped / scale, unknowns - rows))
        return (
            _smooth_relative_sup_merit(imaged, scored),
            _relative_residual(imaged, scored),
        )

    def reconstruct(reduced, shadow, base_state, rows=None):
        """Return the flux one reduced state reconstructs behind the shadow."""
        amplitudes, unknowns = _split(reduced)
        image = _image(_scatter(coordinates, amplitudes), unknowns, _bound(rows))
        return jnp.where(shadow, base_state, image)

    def reduced_map(reduced, shadow, base_state, rows=None):
        """Return one write-then-read cycle of the reduced amplitudes."""
        moments = _current_moments(
            operator,
            reconstruct(reduced, shadow, base_state, rows),
            requested_class,
            target_current,
        )
        return _gather(coordinates, moments)

    def reduced_residual(reduced, shadow, base_state, rows=None):
        """Return the reduced fixed-point residual ``u - R(u)`` and its rows."""
        amplitudes, unknowns = _split(reduced)
        if augmentation is None:
            return amplitudes - reduced_map(reduced, shadow, base_state, rows)
        bound = _bound(rows)
        state = reconstruct(reduced, shadow, base_state, rows)
        moments = _current_moments(operator, state, requested_class, target_current)
        residual, _rows = _augment(
            amplitudes - _gather(coordinates, moments), state, unknowns, shadow, bound
        )
        return residual

    def flux_scores(reduced, shadow, base_state, rows=None):
        """Return the production merit and relative residual in flux space."""
        amplitudes, unknowns = _split(reduced)
        bound = _bound(rows)
        state = reconstruct(reduced, shadow, base_state, rows)
        moments = _current_moments(operator, state, requested_class, target_current)
        image = _image(moments, unknowns, bound)
        mapped = jnp.where(shadow, state, image)
        scored = (
            None
            if augmentation is None
            else augmentation.rows(state, unknowns, shadow, bound.pairs)
        )
        return _scores(state, mapped, unknowns, scored, bound)

    def ladder(reduced, step, shadow, base_state, rows=None):
        """Score the production backtracking grades of one Newton step."""
        factors = jnp.asarray(_BACKTRACKING_FACTORS, dtype=reduced.dtype)
        candidates = reduced[None, :] + factors[:, None] * step[None, :]
        return jax.lax.map(
            lambda candidate: flux_scores(candidate, shadow, base_state, rows),
            candidates,
        )

    def leakage(reduced, shadow, base_state, rows=None):
        """Return the current the reduced map drives outside its support."""
        moments = _current_moments(
            operator,
            reconstruct(reduced, shadow, base_state, rows),
            requested_class,
            target_current,
        )
        current = moments.cell_current
        retained = jnp.zeros_like(current).at[coordinates.cells].set(1.0)
        excluded = jnp.max(jnp.abs(jnp.where(retained > 0.0, 0.0, current)))
        return excluded / jnp.maximum(jnp.max(jnp.abs(current)), 1.0e-30)

    def step_scores(reduced, shadow, base_state, rows=None):
        """Return the reduced residual, its sup norm and the flux scores.

        The residual, its sup norm and the two flux scores are four readings
        of a single write-then-read cycle, so one program computes the
        moments once and the four readings share it.  Folding the norm in
        here is what keeps it off the host: the caller reads it beside the
        merit it was going to synchronise on anyway.
        """
        amplitudes, unknowns = _split(reduced)
        bound = _bound(rows)
        state = reconstruct(reduced, shadow, base_state, rows)
        moments = _current_moments(operator, state, requested_class, target_current)
        image = _image(moments, unknowns, bound)
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

    def grade_scores(reduced, step, factor, shadow, base_state, rows=None):
        """Score one backtracking grade and return the state it scored.

        The candidate amplitudes leave the program beside their scores, so a
        grade the caller accepts carries its own residual and merit into the
        next Newton step and no evaluation is repeated to recover them.  The
        grade enters as an argument, so every grade of every step runs the
        one compiled program.
        """
        candidate = reduced + factor * step
        return candidate, step_scores(candidate, shadow, base_state, rows)

    def tail_grades(reduced, step, shadow, base_state, rows=None):
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
            lambda candidate: step_scores(candidate, shadow, base_state, rows),
            candidates,
        )

    def trip_boundary(reduced, shadow, base_state, rows=None):
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
        state = reconstruct(reduced, shadow, base_state, rows)
        moments = _current_moments(operator, state, requested_class, target_current)
        promoted = jnp.ravel(
            jnp.asarray(
                operator.residual_shadow_mask(
                    state, requested_class, previous_shadow=shadow
                ),
                dtype=bool,
            )
        )
        image = _image(moments, unknowns, bound)
        mapped = jnp.where(promoted, state, image)
        current = moments.cell_current
        retained = jnp.zeros_like(current).at[coordinates.cells].set(1.0)
        excluded = jnp.max(jnp.abs(jnp.where(retained > 0.0, 0.0, current)))
        gathered = _gather(coordinates, moments)
        scored = (
            None
            if augmentation is None
            else augmentation.rows(state, unknowns, promoted, bound.pairs)
        )
        return (
            state,
            promoted,
            jnp.sum(promoted != shadow),
            _scores(state, mapped, unknowns, scored, bound)[1],
            gathered if augmentation is None else jnp.concatenate((gathered, unknowns)),
            excluded / jnp.maximum(jnp.max(jnp.abs(current)), 1.0e-30),
        )

    def initial_gather(state):
        """Return the reduced amplitudes one flux state drives."""
        return _gather(
            coordinates,
            _current_moments(operator, state, requested_class, target_current),
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
    external = operator.external(current, prescribed_current)
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(state, requested_class), dtype=bool)
    )
    coordinates = reduced_coordinates(
        operator,
        state,
        requested_class=requested_class,
        target_current=target_current,
        policy=support_policy,
        floor=support_floor,
    )
    kernels = _reduced_kernels(
        operator, coordinates, external, requested_class, target_current
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
        mapped = operator.flux_map_with_shadow(
            current, requested_class, target_current, prescribed_current
        )(state, promoted)
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
        return _RowArguments(self.pairs, self.flux_scale)

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

    def rows(self, flux, unknowns, shadow, pairs) -> jax.Array:
        """Return every authored constraint residual row at one state."""
        context = ConstraintContext(
            flux, self.requested_class, self.target_current, shadow
        )
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
    augmented Newton-Krylov route: the merit and the relative residual are read
    on the flux normalised by its own span beside the unknowns, so a row that
    is far from its target is visible to the line search rather than hidden
    under the flux block.

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
    augmentation = (
        None
        if not pairs
        else _row_augmentation(
            profile,
            pairs,
            state,
            requested_class=requested_class,
            target_current=target_current,
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
    if program is None:
        external = operator.external(current, prescribed_current)
        coordinates = reduced_coordinates(
            operator,
            state,
            requested_class=requested_class,
            target_current=target_current,
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
                target_current,
                augmentation=augmentation,
            ),
            row_count=0 if augmentation is None else augmentation.row_count,
        )
    elif program.row_count != (0 if augmentation is None else augmentation.row_count):
        raise ValueError(
            "a reused program carries a fixed number of constraint rows; "
            f"it was built for {program.row_count} and this solve poses "
            f"{0 if augmentation is None else augmentation.row_count}"
        )
    coordinates = program.coordinates
    kernels = _bind_rows(
        program.kernels,
        None
        if augmentation is None or row_arguments == CAPTURED_ROWS
        else augmentation.arguments,
    )

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
        mapped = operator.flux_map_with_shadow(
            current, requested_class, target_current, prescribed_current
        )(state, promoted)
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
            target_current=target_current,
        )
        if circuit_current is not None:
            for pair, record in zip(pairs, records, strict=True):
                circuit_current = circuit_current + (
                    jnp.asarray(pair.unknown.direction) @ record.physical_unknown
                )
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
