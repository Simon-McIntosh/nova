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

Two costs of driving that solve from the host are removed rather than paid.
The backtracking grades are scored one at a time and scoring stops at the
first grade below the incumbent merit, which is the grade the selection takes
in either case, so a step accepted at full length evaluates the map once
instead of eight times.  A trip closes inside one compiled program that
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
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.forward_operator import CellCurrentMoments
from nova.equilibrium.fixed_point import (
    _BACKTRACKING_FACTORS,
    _relative_residual,
    _smooth_relative_sup_merit,
    FIXED_POINT_RESIDUAL_TOLERANCE,
    FixedPointTerminationReason,
)


__all__ = [
    "ReducedCoordinates",
    "ReducedNewtonResult",
    "ReducedTrip",
    "reduced_coordinates",
    "solve_reduced_newton",
]


ACTIVE_SET_STEPS = 16
NEWTON_STEPS = 12
SUPPORT_POLICY = "participation"
_ACTIVE_SUPPORT_FLOOR = 0.0

#: Score the backtracking grades one at a time and stop at the first that
#: lowers the merit.  The production ladder takes the first grade below the
#: incumbent merit, so scoring a later grade can only produce a value the
#: selection discards.
LADDER_SCORING = "first_accept"
#: Score every grade before selecting, which is what the first prototype did
#: and what the banked receipt measured.
EAGER_LADDER_SCORING = "eager"
#: Close a trip inside one compiled program: reconstruct, promote the shadow,
#: map the promoted state, gather the next trip's amplitudes and measure the
#: off-support leakage share one topology read.
TRIP_BOUNDARY = "fused"
#: Close a trip through the separate calls the first prototype dispatched.
DISPATCHED_TRIP_BOUNDARY = "dispatched"


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
) -> dict[str, Callable[..., Any]]:
    """Build the jitted reduced map, residual, Jacobian and merit ladder.

    The trip's frozen shadow and its base state enter as arguments rather than
    captured constants, so every trip of one solve shares a single compiled
    program and the compilation is paid once.
    """

    def reconstruct(reduced, shadow, base_state):
        """Return the flux one reduced state reconstructs behind the shadow."""
        image = external + operator.current_moment_image(_scatter(coordinates, reduced))
        return jnp.where(shadow, base_state, image)

    def reduced_map(reduced, shadow, base_state):
        """Return one write-then-read cycle of the reduced amplitudes."""
        moments = _current_moments(
            operator,
            reconstruct(reduced, shadow, base_state),
            requested_class,
            target_current,
        )
        return _gather(coordinates, moments)

    def reduced_residual(reduced, shadow, base_state):
        """Return the reduced fixed-point residual ``u - R(u)``."""
        return reduced - reduced_map(reduced, shadow, base_state)

    def flux_scores(reduced, shadow, base_state):
        """Return the production merit and relative residual in flux space."""
        state = reconstruct(reduced, shadow, base_state)
        moments = _current_moments(operator, state, requested_class, target_current)
        image = external + operator.current_moment_image(moments)
        mapped = jnp.where(shadow, state, image)
        return (
            _smooth_relative_sup_merit(mapped, state),
            _relative_residual(mapped, state),
        )

    def ladder(reduced, step, shadow, base_state):
        """Score the production backtracking grades of one Newton step."""
        factors = jnp.asarray(_BACKTRACKING_FACTORS, dtype=reduced.dtype)
        candidates = reduced[None, :] + factors[:, None] * step[None, :]
        return jax.lax.map(
            lambda candidate: flux_scores(candidate, shadow, base_state), candidates
        )

    def leakage(reduced, shadow, base_state):
        """Return the current the reduced map drives outside its support."""
        moments = _current_moments(
            operator,
            reconstruct(reduced, shadow, base_state),
            requested_class,
            target_current,
        )
        current = moments.cell_current
        retained = jnp.zeros_like(current).at[coordinates.cells].set(1.0)
        excluded = jnp.max(jnp.abs(jnp.where(retained > 0.0, 0.0, current)))
        return excluded / jnp.maximum(jnp.max(jnp.abs(current)), 1.0e-30)

    def step_scores(reduced, shadow, base_state):
        """Return the reduced residual, the merit and the flux residual at once.

        The reduced residual and the two flux scores are three readings of a
        single write-then-read cycle, so one program computes the moments
        once and the three readings share it.
        """
        state = reconstruct(reduced, shadow, base_state)
        moments = _current_moments(operator, state, requested_class, target_current)
        image = external + operator.current_moment_image(moments)
        mapped = jnp.where(shadow, state, image)
        return (
            reduced - _gather(coordinates, moments),
            _smooth_relative_sup_merit(mapped, state),
            _relative_residual(mapped, state),
        )

    def grade_scores(reduced, step, factor, shadow, base_state):
        """Score one backtracking grade and return the state it scored.

        The candidate amplitudes leave the program beside their scores, so a
        grade the caller accepts carries its own residual and merit into the
        next Newton step and no evaluation is repeated to recover them.  The
        grade enters as an argument, so every grade of every step runs the
        one compiled program.
        """
        candidate = reduced + factor * step
        residual, merit, flux_residual = step_scores(candidate, shadow, base_state)
        return candidate, residual, merit, flux_residual

    def trip_boundary(reduced, shadow, base_state):
        """Close one trip inside a single program.

        Reconstructing the flux, promoting the residual shadow against the
        frozen one, mapping the promoted state, gathering the next trip's
        amplitudes and measuring the off-support leakage all read the same
        topology at the same state, so one program evaluates that read once
        and the frozen shadow enters as an argument rather than a constant,
        which keeps every trip of one solve on the same compiled program.
        """
        state = reconstruct(reduced, shadow, base_state)
        moments = _current_moments(operator, state, requested_class, target_current)
        promoted = jnp.ravel(
            jnp.asarray(
                operator.residual_shadow_mask(
                    state, requested_class, previous_shadow=shadow
                ),
                dtype=bool,
            )
        )
        image = external + operator.current_moment_image(moments)
        mapped = jnp.where(promoted, state, image)
        current = moments.cell_current
        retained = jnp.zeros_like(current).at[coordinates.cells].set(1.0)
        excluded = jnp.max(jnp.abs(jnp.where(retained > 0.0, 0.0, current)))
        return (
            state,
            promoted,
            jnp.sum(promoted != shadow),
            _relative_residual(mapped, state),
            _gather(coordinates, moments),
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


def _first_accept_grades(
    kernels, reduced, direction, merit, shadow, base_state, census
):
    """Score grades in order and stop at the first that lowers the merit.

    The selection is the one the eager ladder makes, so scoring stops where
    that selection is already decided.  An accepted grade returns the state
    it scored together with its residual and merit, which the next Newton
    step reads instead of evaluating the map again at the same point.
    """
    for index, factor in enumerate(_BACKTRACKING_FACTORS):
        candidate, residual, candidate_merit, candidate_flux = kernels["grade"](
            reduced,
            direction,
            jnp.asarray(factor, dtype=reduced.dtype),
            shadow,
            base_state,
        )
        census["map_evaluations"] += 1
        value = float(candidate_merit)
        if np.isfinite(value) and value < merit:
            promotion = (candidate, residual, value, float(candidate_flux))
            return index, index + 1, promotion
    return -1, len(_BACKTRACKING_FACTORS), None


def _incumbent_scores(kernels, reduced, shadow, base_state, census, scoring):
    """Return the reduced residual, merit and flux residual at one state."""
    if scoring == LADDER_SCORING:
        residual, merit, flux_residual = kernels["step_scores"](
            reduced, shadow, base_state
        )
        census["map_evaluations"] += 1
        return residual, float(merit), float(flux_residual)
    residual, _ = _timed(kernels["reduced_residual"], reduced, shadow, base_state)
    scores, _ = _timed(kernels["flux_scores"], reduced, shadow, base_state)
    census["map_evaluations"] += 2
    return residual, float(scores[0]), float(scores[1])


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
    grader = _first_accept_grades if scoring == LADDER_SCORING else _eager_grades
    factors = _BACKTRACKING_FACTORS
    fresh = True
    carried: tuple[jax.Array, float, float] | None = None
    for index in range(newton_steps):
        started = time.perf_counter()
        before = census["map_evaluations"]
        if carried is None:
            residual, merit, flux_residual = _incumbent_scores(
                kernels, reduced, shadow, base_state, census, scoring
            )
        else:
            residual, merit, flux_residual = carried
            carried = None
        if not np.isfinite(flux_residual) or flux_residual <= tolerance:
            break
        accepted = -1
        refreshed = False
        direction = None
        promotion = None
        for _attempt in range(2):
            if scoring == LADDER_SCORING:
                direction = kernels["direction"](jacobian, residual)
            else:
                direction, _ = _timed(kernels["direction"], jacobian, residual)
            accepted, _tried, promotion = grader(
                kernels, reduced, direction, merit, shadow, base_state, census
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
            reduced_residual=float(jnp.max(jnp.abs(residual))),
            flux_residual=flux_residual,
            merit=merit,
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
            reduced, promoted_residual, promoted_merit, promoted_flux = promotion
            carried = (promoted_residual, promoted_merit, promoted_flux)
        fresh = False
        census["steps"] += 1
    return reduced, census


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
    if ladder_scoring not in (LADDER_SCORING, EAGER_LADDER_SCORING):
        raise ValueError(f"unknown ladder scoring {ladder_scoring!r}")
    if trip_boundary not in (TRIP_BOUNDARY, DISPATCHED_TRIP_BOUNDARY):
        raise ValueError(f"unknown trip boundary {trip_boundary!r}")
    fused = trip_boundary == TRIP_BOUNDARY
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

    carried_reduced = kernels["initial_gather"](state) if fused else None
    for trip in range(active_set_steps):
        trip_started = time.perf_counter()
        if fused:
            reduced = carried_reduced
        else:
            moments = _current_moments(operator, state, requested_class, target_current)
            reduced = _gather(coordinates, moments)
        reduced, census = _plain_newton_trip(
            kernels,
            reduced,
            shadow,
            state,
            trip=trip,
            newton_steps=newton_steps,
            tolerance=tolerance,
            steps=steps,
            scoring=ladder_scoring,
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
            state, promoted, difference, observed, carried_reduced, excluded = closed
            difference = int(difference)
            observed = float(observed)
            leakage = max(leakage, float(excluded))
        else:
            leakage = max(leakage, float(kernels["leakage"](reduced, shadow, state)))
            state, _ = _timed(kernels["reconstruct"], reduced, shadow, state)
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

    return ReducedNewtonResult(
        state=state,
        terminal_residual=terminal_residual,
        active_set_iterations=len(residuals),
        converged=converged,
        termination_reason=int(reason),
        active_set_residuals=residuals,
        active_set_mask_differences=differences,
        newton_steps_per_trip=per_trip_steps,
        jacobian_builds_per_trip=per_trip_builds,
        rejected_steps_per_trip=per_trip_rejected,
        jacobian_wall_per_trip=per_trip_jacobian_wall,
        newton_wall_per_trip=per_trip_newton_wall,
        boundary_wall_per_trip=per_trip_boundary_wall,
        trip_wall_per_trip=per_trip_wall,
        map_evaluations_per_trip=per_trip_maps,
        reduced_dimension=coordinates.size,
        support_cells=int(coordinates.cells.size),
        off_support_leakage=leakage,
        steps=steps,
    )
