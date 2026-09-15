"""CPU contract for the synthetic outboard-edge demonstration."""

from __future__ import annotations

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from benchmarks.edge_constraint_demonstration import (
        _edge_pair,
        outboard_midplane_edge_radius,
    )
    from nova.equilibrium import reduced_newton
    from nova.equilibrium.constraint import ConstraintContext
    from nova.equilibrium.forward_operator import PrescribedCurrentField
    from nova.jax.config import configure_dtypes

    from tests.test_reduced_newton import machine  # noqa: F401


SOLVE_TOLERANCE = 1.0e-8
NEWTON_STEPS = 24
OUTWARD_COMMAND_M = 0.005
TRIP_CAP_A = 1.0e-3


def _prescribed(profile) -> None:
    """Attach every fixture conductor image to one prescribed-current field."""
    operator = profile.operator
    response = jnp.concatenate(
        (
            jnp.asarray(operator.grid.source_target),
            jnp.asarray(operator.wall.source_target),
        )
    )
    operator.prescribed_field = PrescribedCurrentField(
        response=response,
        current=jnp.zeros(response.shape[1], dtype=jnp.float64),
    )


def test_outward_limited_edge_refuses_when_no_bounded_grade_lowers_merit(machine):
    """A displaced limited edge stays open when every bounded grade is worse."""
    configure_dtypes()
    profile, seed = machine
    _prescribed(profile)
    field = profile.operator.prescribed_current_field
    free = reduced_newton.solve_reduced_newton(
        profile.operator,
        jnp.asarray(seed),
        prescribed_current=field.current,
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert free.converged
    _masks, topology = profile.operator.read(jnp.asarray(free.state))
    chord_height = float(np.asarray(topology.axis)[1])
    edge_radius = outboard_midplane_edge_radius(
        profile, free.state, height_m=chord_height
    )
    lattice = profile.lattice
    assert float(lattice.radius[0]) < edge_radius < float(lattice.radius[-1])
    repeated = outboard_midplane_edge_radius(profile, free.state, height_m=chord_height)
    assert repeated == edge_radius

    pair, _selection = _edge_pair(
        profile,
        free.state,
        point_rz_m=np.asarray([edge_radius + OUTWARD_COMMAND_M, chord_height]),
        flux_span_wb=abs(float(np.asarray(topology.flux_span))),
        requested_class=None,
        target_current=None,
        prescribed_current=field.current,
        program=free.program,
        circuits=range(field.current.size),
    )
    initial_residual = np.asarray(
        pair.functional.observed(
            profile,
            ConstraintContext(jnp.asarray(free.state), None, None, None),
            pair.binding.payload,
        )
    )
    assert initial_residual.shape == (1,)
    assert abs(float(initial_residual[0])) > float(
        np.asarray(pair.binding.tolerance)[0]
    )

    result = reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        constraint_pairs=(pair,),
        prescribed_current=field.current,
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
        active_set_steps=1,
        constraint_current_step_cap=TRIP_CAP_A,
    )
    applied = abs(float(np.asarray(result.constraints[0].physical_unknown)[0]))
    assert applied == 0.0
    assert result.newton_steps_per_trip == [0]
    assert result.termination_name == "sufficient_decrease_refused"
    assert result.qualified
    assert not result.converged
    assert result.refusal_reason is None
    assert not bool(np.asarray(result.constraints[0].qualified)[0])
