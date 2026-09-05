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
    trip_wall_per_trip: list[float] = field(default_factory=list)
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
    }


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
    }
    factors = _BACKTRACKING_FACTORS
    fresh = True
    for index in range(newton_steps):
        started = time.perf_counter()
        residual, _ = _timed(kernels["reduced_residual"], reduced, shadow, base_state)
        scores, _ = _timed(kernels["flux_scores"], reduced, shadow, base_state)
        merit = float(scores[0])
        flux_residual = float(scores[1])
        if not np.isfinite(flux_residual) or flux_residual <= tolerance:
            break
        accepted = -1
        refreshed = False
        direction = None
        for _attempt in range(2):
            direction, _ = _timed(kernels["direction"], jacobian, residual)
            scored, _ = _timed(
                kernels["ladder"], reduced, direction, shadow, base_state
            )
            merits = np.asarray(scored[0])
            below = np.isfinite(merits) & (merits < merit)
            accepted = int(np.argmax(below)) if bool(below.any()) else -1
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
            jacobian_refreshed=refreshed,
            wall_s=wall,
        )
        steps.append(record)
        if accepted < 0:
            break
        reduced = reduced + factors[accepted] * direction
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
    shadowed_map = operator.flux_map_with_shadow(
        current, requested_class, target_current, prescribed_current
    )
    residuals: list[float] = []
    differences: list[int] = []
    per_trip_steps: list[int] = []
    per_trip_builds: list[int] = []
    per_trip_rejected: list[int] = []
    per_trip_jacobian_wall: list[float] = []
    per_trip_newton_wall: list[float] = []
    per_trip_wall: list[float] = []
    steps: list[ReducedNewtonStep] = []
    reason = FixedPointTerminationReason.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED
    converged = False
    terminal_residual = float("inf")
    leakage = 0.0

    for trip in range(active_set_steps):
        trip_started = time.perf_counter()
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
        )
        per_trip_steps.append(int(census["steps"]))
        per_trip_builds.append(int(census["jacobian_builds"]))
        per_trip_rejected.append(int(census["rejected"]))
        per_trip_jacobian_wall.append(float(census["jacobian_wall"]))
        per_trip_newton_wall.append(float(census["newton_wall"]))
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
        mapped = shadowed_map(state, promoted)
        observed = float(_relative_residual(mapped, state))
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
        trip_wall_per_trip=per_trip_wall,
        reduced_dimension=coordinates.size,
        support_cells=int(coordinates.cells.size),
        off_support_leakage=leakage,
        steps=steps,
    )
