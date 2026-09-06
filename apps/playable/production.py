"""Production keyframe solver over one forward profile.

The production solve runs the constrained reduced route:
``solve_constrained_reduced_newton`` poses the current-centroid row on the
commanded bulk position beside the plasma-cell amplitudes, reads the
compensating circuit direction off the machine's response matrix once per
session (it is a property of the carrier, not of the frame), and returns a
:class:`~nova.equilibrium.reduced_newton.ReducedProgram` handle that the
session carries and hands back on every later solve.  With the row target
arriving as a traced argument, a keyframe chain re-enters one compiled
program from the second press on, so a steered solve runs at the warm-trip
figure rather than at a per-target compile.  The constrained Newton-Krylov
implementation stays reachable as the reference, selected by
``route="newton_krylov"``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import jax.numpy as jnp

from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintPair,
    CurrentCentroidConstraint,
    derive_circuit_compensators,
)
from nova.equilibrium.fixed_point import (
    FIXED_POINT_RESIDUAL_TOLERANCE,
    FixedPointResult,
)
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.forward import ForwardEquilibrium, ForwardProfile
from nova.equilibrium.reduced_newton import (
    ACTIVE_SET_STEPS,
    NEWTON_STEPS,
    TRACED_ROWS,
    solve_constrained_reduced_newton,
)

from apps.playable.session import SolveResult
from apps.playable.shape import PlasmaShape


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


def centroid_row(
    profile: ForwardProfile,
    commanded: PlasmaShape,
    *,
    tolerance: float = 1.0e-4,
    scale: float = 0.05,
) -> ConstraintPair:
    """Return the current-centroid row posed on the commanded bulk position."""
    return ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_r", "centroid_z"),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=CircuitCurrentUnknown(
            direction=np.zeros(
                (profile.operator.prescribed_current_field.circuit_count, 2)
            ),
            ampere_scale=np.asarray([1.0, 1.0]),
        ),
        binding=ConstraintBinding(
            target=np.asarray([commanded.axis_r, commanded.axis_z]),
            tolerance=np.full(2, tolerance),
            scale=np.full(2, scale),
            initial_unknown=np.asarray([0.0, 0.0]),
            payload=None,
            policy="imposed",
        ),
    )


def derive_direction(
    profile: ForwardProfile,
    row: ConstraintPair,
    flux: np.ndarray,
    *,
    circuits=None,
) -> tuple[ConstraintPair, ...]:
    """Return the row with the best-conditioned circuit direction applied."""
    derived, _selection = derive_circuit_compensators(
        profile, (row,), np.asarray(flux), circuits=circuits
    )
    return derived


@dataclass
class ProductionSolver:
    """Keyframe solver over one forward profile.

    The default route is the constrained reduced solve, whose compiled-program
    handle lets a session re-enter one program per keyframe chain;
    ``route="newton_krylov"`` keeps the constrained Newton-Krylov
    implementation reachable as the reference.  Both routes share the frozen
    compensating circuit direction read off the carrier's response matrix at
    construction.
    """

    machine: ForwardMachine
    route: str = "reduced_newton"
    newton_steps: int = 4
    gmres_iterations: int = 4
    active_set_steps: int = 2

    def __post_init__(self) -> None:
        """Freeze the compensating circuit direction read off the carrier."""
        if self.route not in ("reduced_newton", "newton_krylov"):
            raise ValueError(
                f"unknown production route {self.route!r}; choose from "
                "('reduced_newton', 'newton_krylov')"
            )
        commanded = PlasmaShape()
        row = centroid_row(self.machine.profile, commanded)
        circuits = (
            None
            if self.machine.circuit_count == 0
            else tuple(range(self.machine.circuit_count))
        )
        # The direction is a property of the response matrix evaluated at the
        # seed; the unknown's magnitude is re-solved every keyframe.
        self.frozen_pairs = derive_direction(
            self.machine.profile, row, self.machine.seed, circuits=circuits
        )

    def _pairs(self, commanded: PlasmaShape) -> tuple[ConstraintPair, ...]:
        """Return the frozen-row pairs re-bound to the commanded position."""
        return tuple(
            ConstraintPair(
                functional=pair.functional,
                unknown=pair.unknown,
                binding=replace(
                    pair.binding,
                    target=np.asarray([commanded.axis_r, commanded.axis_z]),
                ),
            )
            for pair in self.frozen_pairs
        )

    def _flux(self, previous: ForwardEquilibrium | None) -> np.ndarray:
        """Return the warm-start flux, or the machine seed for the prime."""
        return np.asarray(previous.flux) if previous is not None else self.machine.seed

    def _reduced(
        self,
        profile: ForwardProfile,
        flux: np.ndarray,
        commanded: PlasmaShape,
        program=None,
    ):
        """Run the constrained reduced solve with the traced row targets."""
        return solve_constrained_reduced_newton(
            profile,
            jnp.asarray(flux),
            constraint_pairs=self._pairs(commanded),
            current=None,
            prescribed_current=None,
            tolerance=FIXED_POINT_RESIDUAL_TOLERANCE,
            newton_steps=NEWTON_STEPS,
            active_set_steps=ACTIVE_SET_STEPS,
            row_arguments=TRACED_ROWS,
            program=program,
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

    def __call__(
        self,
        previous: ForwardEquilibrium | None,
        commanded: PlasmaShape,
        *,
        action: tuple[str, float] | None = None,
        program: object | None = None,
    ) -> SolveResult:
        """Warm-start from the previous equilibrium and solve the commanded set."""
        del action
        profile = self.machine.profile
        flux = self._flux(previous)
        started = perf_counter()
        if self.route == "reduced_newton":
            result = self._reduced(profile, flux, commanded, program)
            equilibrium = self._reduced_receipt(profile, result)
            trips = int(result.active_set_iterations)
            program_out = result.program
            reused = program is not None
        else:
            equilibrium = profile.solve(
                flux,
                route="newton_krylov",
                constraint_pairs=self._pairs(commanded),
                warmup=0,
                newton_steps=self.newton_steps,
                gmres_iterations=self.gmres_iterations,
                active_set_steps=self.active_set_steps,
                stop_on_active_set_settlement=False,
            )
            trips = int(equilibrium.fixed_point.active_set_iterations)
            program_out, reused = None, False
        wall = perf_counter() - started
        return SolveResult(equilibrium, wall, trips, program=program_out, reused=reused)
