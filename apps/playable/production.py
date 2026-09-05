"""Production keyframe solver over one forward profile.

The first implementation of the keyframe solve protocol is the constrained
Newton-Krylov route: a current-centroid row is posed on the commanded bulk
position, its compensating circuit direction is read off the machine's
response matrix once per session (it is a property of the carrier, not of the
frame), and the solve warm-starts from the previous equilibrium.  It runs at
seconds per keyframe today; the constrained reduced route replaces it behind
the same protocol without the app changing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintPair,
    CurrentCentroidConstraint,
    derive_circuit_compensators,
)
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.forward import ForwardEquilibrium, ForwardProfile

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
    """Constrained Newton-Krylov keyframes on one forward profile."""

    machine: ForwardMachine
    newton_steps: int = 4
    gmres_iterations: int = 4
    active_set_steps: int = 2

    def __post_init__(self) -> None:
        """Freeze the compensating circuit direction read off the carrier."""
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

    def __call__(
        self,
        previous: ForwardEquilibrium | None,
        commanded: PlasmaShape,
        *,
        action: tuple[str, float] | None = None,
    ) -> SolveResult:
        """Warm-start from the previous equilibrium and solve the commanded set."""
        del action
        profile = self.machine.profile
        flux = np.asarray(previous.flux) if previous is not None else self.machine.seed
        started = perf_counter()
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
        wall = perf_counter() - started
        trips = int(equilibrium.fixed_point.active_set_iterations)
        return SolveResult(equilibrium, wall, trips)
