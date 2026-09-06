"""Contracts for the constrained Newton-Krylov entry point."""

from __future__ import annotations

import numpy as np
import pytest

from nova.equilibrium import fixed_point
from nova.equilibrium.constraint import (
    ConstraintBinding,
    ConstraintMultiplier,
    ConstraintPair,
    CurrentCentroidConstraint,
)
from nova.equilibrium.forward_operator import PrescribedCurrentField
from nova.equilibrium.observation import MomentIntegralSupport

from tests.test_reduced_newton import machine as machine_fixture


@pytest.fixture(scope="module")
def solovev_machine():
    """Reuse the established Solovev machine once for this module."""
    return machine_fixture.__wrapped__()


def _centroid_pair(profile, flux):
    """Bind a centroid row at the value reached by the Solovev seed."""
    observation = profile.current_moment_observation(
        flux, support=MomentIntegralSupport.ALL_DOMAIN
    )
    target = float(np.asarray(observation.centroid_z))
    scale = float(np.ptp(np.asarray(profile.lattice.height)))
    return ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=ConstraintMultiplier(multiplier_scale=[1.0]),
        binding=ConstraintBinding(
            target=[target],
            tolerance=[1.0e-8],
            scale=[scale],
            initial_unknown=[0.0],
        ),
    )


def test_krylov_constraints_refuse_without_a_prescribed_field(solovev_machine):
    """A circuit compensator must have a field whose current it can drive."""
    profile, seed = solovev_machine
    pair = _centroid_pair(profile, seed)
    saved = profile.operator.prescribed_current_field
    profile.operator.prescribed_field = None
    try:
        with pytest.raises(ValueError, match="prescribed current field"):
            profile._solve_augmented_constraints(
                seed,
                None,
                constraint_pairs=(pair,),
                warmup=0,
                gmres_iterations=2,
                active_set_steps=1,
                stop_on_active_set_settlement=False,
            )
    finally:
        profile.operator.prescribed_field = saved


def test_krylov_constraints_reach_solver_with_an_operator_prescribed_field(
    solovev_machine, monkeypatch
):
    """The same Solovev request reaches Krylov when the field is available."""
    profile, seed = solovev_machine
    response = np.concatenate(
        (
            np.asarray(profile.operator.grid.source_target),
            np.asarray(profile.operator.wall.source_target),
        )
    )
    profile.operator.prescribed_field = PrescribedCurrentField(
        response=response,
        current=np.zeros(response.shape[1]),
    )
    pair = _centroid_pair(profile, seed)

    class SolverReached(RuntimeError):
        pass

    def reach_solver(*_args, **_kwargs):
        raise SolverReached

    monkeypatch.setattr(fixed_point, "newton_krylov", reach_solver)
    with pytest.raises(SolverReached):
        profile._solve_augmented_constraints(
            seed,
            None,
            constraint_pairs=(pair,),
            warmup=0,
            gmres_iterations=2,
            active_set_steps=1,
            stop_on_active_set_settlement=False,
        )
