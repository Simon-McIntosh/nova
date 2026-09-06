"""Production keyframe solver over one forward profile.

Shape control is an inverse-forward step. The inverse solves every free coil
current against the bounding-box flux and field rows; the forward solve then
runs on those prescribed currents, warm-started from the previous frame. A
second inverse-forward round is allowed when the first response leaves a
turning point more than the stated tolerance from its command. Constraint
pairs are deliberately absent from this path: they remain the diagnostic
placement mechanism, not the shape actuator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import jax.numpy as jnp

from nova.equilibrium.fixed_point import (
    FIXED_POINT_RESIDUAL_TOLERANCE,
    FixedPointResult,
)
from nova.equilibrium.forward import ForwardEquilibrium, ForwardProfile
from nova.equilibrium.reduced_newton import (
    ACTIVE_SET_STEPS,
    NEWTON_STEPS,
    solve_constrained_reduced_newton,
)
from nova.equilibrium.shape_inverse import (
    ShapeInverseResult,
    achieved_target,
    solve_shape_inverse,
    turning_point_error,
)

from apps.playable.session import SolveResult
from apps.playable.shape import PlasmaShape, move_bounding_box

#: A second inverse-forward round is taken above this point error [m].
TURNING_POINT_TOLERANCE = 2.0e-3
#: One initial response and at most one correction keep a keyframe bounded.
MAX_INVERSE_ROUNDS = 2


@dataclass(frozen=True)
class ForwardMachine:
    """One fixed carrier: a forward profile, its seed, wall and identity."""

    profile: ForwardProfile
    seed: np.ndarray
    wall: np.ndarray
    identity: str

    @property
    def circuit_count(self) -> int:
        """Return the number of prescribed circuits the carrier can drive."""
        field = self.profile.operator.prescribed_current_field
        return 0 if field is None else int(field.current.size)


@dataclass(frozen=True)
class InverseRoundReceipt:
    """One inverse current solve followed by one forward response."""

    inverse: ShapeInverseResult
    turning_point_error: float
    trips: int
    wall: float


@dataclass
class ProductionSolver:
    """Run bounded inverse-forward shape-control rounds on one machine."""

    machine: ForwardMachine
    route: str = "reduced_newton"
    newton_steps: int = 4
    gmres_iterations: int = 4
    active_set_steps: int = 2
    turning_point_tolerance: float = TURNING_POINT_TOLERANCE
    prescribed_current: np.ndarray = field(init=False, repr=False)
    last_target: object | None = field(init=False, default=None, repr=False)
    last_rounds: tuple[InverseRoundReceipt, ...] = field(
        init=False, default=(), repr=False
    )

    def __post_init__(self) -> None:
        """Validate the route and start from the carrier's own currents."""
        if self.route not in ("reduced_newton", "newton_krylov"):
            raise ValueError(
                f"unknown production route {self.route!r}; choose from "
                "('reduced_newton', 'newton_krylov')"
            )
        field_current = self.machine.profile.operator.prescribed_current_field
        if field_current is None:
            raise ValueError("shape control needs a prescribed current field")
        self.prescribed_current = np.asarray(field_current.current, dtype=float).copy()

    def _flux(self, previous: ForwardEquilibrium | None) -> np.ndarray:
        """Return the warm-start flux, or the machine seed for the prime."""
        return np.asarray(previous.flux) if previous is not None else self.machine.seed

    def _reduced(
        self,
        profile: ForwardProfile,
        flux: np.ndarray,
        prescribed_current: np.ndarray,
    ):
        """Run the reduced route with prescribed currents and no shape rows."""
        return solve_constrained_reduced_newton(
            profile,
            jnp.asarray(flux),
            constraint_pairs=(),
            current=None,
            prescribed_current=jnp.asarray(prescribed_current),
            tolerance=FIXED_POINT_RESIDUAL_TOLERANCE,
            newton_steps=NEWTON_STEPS,
            active_set_steps=ACTIVE_SET_STEPS,
            program=None,
        )

    @staticmethod
    def _reduced_receipt(profile: ForwardProfile, result) -> ForwardEquilibrium:
        """Return the equilibrium receipt a reduced result stands for.

        ``ForwardProfile.solve`` builds this receipt and drops the program its
        reduced result also carries, so the keyframe solver rebuilds the same
        receipt directly — the route's own entry is called with the program
        threaded through it instead of through the public wrapper.
        """
        residuals = jnp.asarray(result.active_set_residuals, dtype=jnp.float64)
        history = FixedPointResult(
            state=result.state,
            residual=jnp.asarray(result.terminal_residual, dtype=jnp.float64),
            trace=residuals,
            converged=jnp.asarray(result.converged),
            termination_reason=jnp.asarray(result.termination_reason, dtype=jnp.int32),
            active_set_iterations=jnp.asarray(
                result.active_set_iterations, dtype=jnp.int32
            ),
            active_set_residuals=residuals,
            active_set_mask_differences=jnp.asarray(
                result.active_set_mask_differences, dtype=jnp.int32
            ),
            shadow_mask_changes=jnp.asarray(
                result.active_set_mask_differences, dtype=jnp.int32
            ),
        )
        return profile._receipt(
            result.state,
            history,
            None,
            None,
            None,
            (None if result.prescribed_current is None else result.prescribed_current),
            constraints=result.constraints,
        )

    def _forward(
        self, profile: ForwardProfile, flux: np.ndarray, prescribed_current: np.ndarray
    ) -> tuple[ForwardEquilibrium, int, object | None]:
        """Run one unconstrained forward response on prescribed currents."""
        if self.route == "reduced_newton":
            result = self._reduced(profile, flux, prescribed_current)
            return (
                self._reduced_receipt(profile, result),
                int(result.active_set_iterations),
                result.program,
            )
        equilibrium = profile.solve(
            flux,
            route="newton_krylov",
            constraint_pairs=(),
            prescribed_current=prescribed_current,
            warmup=0,
            newton_steps=self.newton_steps,
            gmres_iterations=self.gmres_iterations,
            active_set_steps=self.active_set_steps,
            stop_on_active_set_settlement=False,
        )
        return (
            equilibrium,
            int(equilibrium.fixed_point.active_set_iterations),
            None,
        )

    def solve_target(
        self, previous: ForwardEquilibrium, target: object
    ) -> tuple[ForwardEquilibrium, object | None]:
        """Drive one fixed bounding-box target for at most two rounds."""
        profile = self.machine.profile
        flux = np.asarray(previous.flux)
        rounds = []
        equilibrium = previous
        program_out = None
        for _ in range(MAX_INVERSE_ROUNDS):
            round_started = perf_counter()
            inverse = solve_shape_inverse(
                profile,
                target,
                flux,
                prescribed_current=self.prescribed_current,
            )
            self.prescribed_current = inverse.currents
            equilibrium, round_trips, program_out = self._forward(
                profile, flux, self.prescribed_current
            )
            error = turning_point_error(profile, target, equilibrium.flux)
            rounds.append(
                InverseRoundReceipt(
                    inverse=inverse,
                    turning_point_error=error,
                    trips=round_trips,
                    wall=perf_counter() - round_started,
                )
            )
            if error <= self.turning_point_tolerance:
                break
            flux = np.asarray(equilibrium.flux)
        self.last_target = target
        self.last_rounds = tuple(rounds)
        return equilibrium, program_out

    def __call__(
        self,
        previous: ForwardEquilibrium | None,
        commanded: PlasmaShape,
        *,
        action: tuple[str, float] | None = None,
        program: object | None = None,
    ) -> SolveResult:
        """Warm-start and run the inverse-forward shape-control step.

        A prime uses the carrier currents unchanged. A shape action is applied
        to the achieved turning points of the previous frame, then one or two
        inverse-forward rounds drive that fixed target. Each forward solve is
        unconstrained and therefore publishes no compensating-row records.
        """
        del program
        profile = self.machine.profile
        flux = self._flux(previous)
        started = perf_counter()
        if previous is None or action is None:
            equilibrium, trips, program_out = self._forward(
                profile, flux, self.prescribed_current
            )
            self.last_target = achieved_target(profile, equilibrium.flux)
            self.last_rounds = ()
            return SolveResult(
                equilibrium,
                perf_counter() - started,
                trips,
                program=program_out,
                reused=False,
            )

        parameter, delta = action
        prior_shape = commanded.apply(parameter, -delta)
        target = move_bounding_box(
            achieved_target(profile, flux), prior_shape, parameter, delta
        )
        equilibrium, program_out = self.solve_target(previous, target)
        return SolveResult(
            equilibrium,
            perf_counter() - started,
            sum(item.trips for item in self.last_rounds),
            program=program_out,
            reused=False,
        )
