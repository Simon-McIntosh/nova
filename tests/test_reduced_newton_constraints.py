"""Constraint rows on the reduced-amplitude plain-Newton route.

The machine is the bootstrapped Solov'ev free-boundary problem
:mod:`tests.test_reduced_newton` builds, with the same ring of external
conductors offered a second time as a prescribed response so a compensating
circuit current has something to act through.  The rows and their compensating
unknowns are the ones the augmented Newton-Krylov route already carries; what
is pinned here is that they can extend the reduced state instead, which is
what puts a steered keyframe on the millisecond route.

Four contracts.  An empty constraint tuple is the unconstrained solve, number
for number, because the augmented entry then evaluates the unaugmented
expressions rather than a degenerate case of the augmented ones.  A row
imposed at the value the free solve already reaches costs no compensating
current and no Newton step.  A row whose target is moved is reached, and the
autodiff response the compensator steers through equals a central difference
of the same observation.  And a moved target re-solved from the previous
augmented state costs fewer trips than the same target from the cold seed,
which is the warm-start contract a keyframe budget rests on.

The response check is taken at the converged free equilibrium and never at the
analytic seed.  The seed puts the material boundary exactly on a seed flux
surface, so every wall node is a maximiser and the boundary read is a kink
rather than a strict extremum; a central difference there returns the mean of
two different one-sided slopes, which is not a derivative of anything.  The
converged state has a strict extremum and the read is smooth, which the
fixture asserts rather than assumes.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.equilibrium import reduced_newton
    from nova.equilibrium.constraint import (
        ConstraintBinding,
        ConstraintMultiplier,
        ConstraintPair,
        CurrentCentroidConstraint,
        ProfileAmplitudeUnknown,
    )
    from nova.equilibrium.forward_operator import PrescribedCurrentField
    from nova.equilibrium.observation import MomentIntegralSupport
    from nova.equilibrium.reduced_newton import ReducedNewtonResult

    from tests.test_reduced_newton import machine  # noqa: F401


#: Relative fixed-point residual every solve here is driven to.  The augmented
#: residual carries the scaled constraint rows beside the flux, so a row that
#: clears this figure sits within this fraction of its declared scale.
SOLVE_TOLERANCE = 1.0e-8
#: Newton budget per trip.  The prototype holds one Jacobian for a whole trip,
#: so its inner iteration is a chord method and needs the wider budget the
#: unconstrained contract already measured on this machine.
NEWTON_STEPS = 24
#: Distance the commanded current centroid is moved, in metres.  It is a sixth
#: of one lattice cell, which is a keyframe-sized nudge rather than a
#: re-design of the equilibrium.
CENTROID_MOVE = 1.0e-2
#: Agreement required between the commanded and the achieved centroid.
CENTROID_AGREEMENT = 1.0e-6
#: Compensating unknown admitted as zero, as a fraction of its own current
#: scale, when the row is imposed at the value the free solve already reaches.
UNMOVED_COMPENSATION = 1.0e-10
#: Relative agreement required between the row's autodiff response and a
#: central difference of the same observation.
RESPONSE_AGREEMENT = 1.0e-6
#: Flux perturbation the central difference is taken over, as a fraction of
#: the flux span.  The observation reads the plasma domain labels, and those
#: are a step function of the flux that the tangent carries no derivative of,
#: so the step has to stay inside one labelling; the test asserts that it does
#: rather than trusting the figure.
RESPONSE_STEP = 1.0e-6
#: Wall-flux spread required before the boundary read is treated as a strict
#: extremum, as a fraction of the flux span.
STRICT_EXTREMUM_SPREAD = 1.0e-3
#: Wall-flux spread below which every node is a maximiser and the read is a
#: kink.  The analytic seed sits here by construction.
KINK_SPREAD = 1.0e-12


def _prescribed(profile):
    """Offer the machine's own conductors as a prescribed response.

    The bootstrapped ring already couples to every flux node, so the same
    coupling read back off the operator is the response a compensating current
    acts through.  Its stored current is zero, which leaves the free solve
    exactly the solve the unconstrained contract measures.
    """
    operator = profile.operator
    response = jnp.concatenate(
        (
            jnp.asarray(operator.grid.source_target),
            jnp.asarray(operator.wall.source_target),
        )
    )
    operator.prescribed_field = PrescribedCurrentField(
        response=response, current=jnp.zeros(response.shape[1])
    )
    return operator.prescribed_current_field


def _wall_flux(profile, state):
    """Return the flux sampled at every material boundary node."""
    return np.asarray(state)[profile.lattice.node_count :]


def _centroid(profile, flux):
    """Return the vertical current centroid [m] of one flux state."""
    return float(
        np.asarray(
            profile.current_moment_observation(
                jnp.asarray(flux), support=MomentIntegralSupport.ALL_DOMAIN
            ).centroid_z
        )
    )


def _centroid_pair(profile, flux, target):
    """Return the vertical centroid row with a matrix-led compensator.

    The seed pair carries a multiplier rather than a circuit, so the
    derivation supplies both the direction and the current amplitude that
    moves the row by one declared scale: the direction decides which circuits
    compensate and the amplitude decides how the augmented system is
    conditioned.
    """
    scale = float(np.ptp(np.asarray(profile.lattice.height)))
    seeded = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=ConstraintMultiplier(multiplier_scale=jnp.asarray([1.0])),
        binding=ConstraintBinding(
            target=jnp.asarray([target]),
            tolerance=jnp.asarray([CENTROID_AGREEMENT]),
            scale=jnp.asarray([scale]),
            initial_unknown=jnp.asarray([0.0]),
        ),
    )
    (derived,), selection = profile.derived_constraint_pairs(
        (seeded,), jnp.asarray(flux)
    )
    return derived, selection


@pytest.fixture(scope="module")
def steered(machine):  # noqa: F811
    """Return the free equilibrium a steered keyframe starts from.

    The free solve runs before any row is posed, so the target every contract
    below commands is a value this machine actually reaches rather than a
    number invented for the test.
    """
    profile, seed = machine
    _prescribed(profile)
    free = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert free.converged
    span = float(np.ptp(np.asarray(free.state)))
    assert np.ptp(_wall_flux(profile, seed)) <= KINK_SPREAD * span
    assert np.ptp(_wall_flux(profile, free.state)) >= STRICT_EXTREMUM_SPREAD * span
    return profile, seed, free


def test_an_empty_tuple_reproduces_the_unconstrained_solve(steered):
    """No rows means no augmentation: the two entries return one answer.

    Every kernel of the augmented builder evaluates the unaugmented
    expression when the tuple is empty, so the agreement required here is
    equality and not a tolerance.  The wall clocks are excluded because they
    measure this machine's scheduler rather than the solve.
    """
    profile, seed, _free = steered
    unconstrained = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    constrained = reduced_newton.solve_constrained_reduced_newton(
        profile,
        seed,
        constraint_pairs=(),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert constrained.compensating_unknown is None
    assert constrained.constraints == ()
    assert constrained.row_count == 0
    compared = 0
    for item in dataclasses.fields(ReducedNewtonResult):
        if item.name.endswith("_wall_per_trip"):
            continue
        left = getattr(unconstrained, item.name)
        right = getattr(constrained, item.name)
        if item.name == "steps":
            assert len(left) == len(right)
            for one, other in zip(left, right, strict=True):
                assert one._replace(wall_s=0.0) == other._replace(wall_s=0.0)
        elif item.name == "state":
            assert np.array_equal(np.asarray(left), np.asarray(right))
        else:
            assert left == right, item.name
        compared += 1
    assert compared == len(dataclasses.fields(ReducedNewtonResult)) - 4


def test_a_row_at_the_free_centroid_costs_nothing(steered):
    """Imposing a row where the solve already sits moves no current.

    The compensating unknown enters at zero and the augmented residual is
    already inside the tolerance, so the trip takes no Newton step, the solve
    closes on an unmoved mask, and the current the row asks the machine for is
    exactly none.
    """
    profile, _seed, free = steered
    achieved = _centroid(profile, free.state)
    pair, _selection = _centroid_pair(profile, free.state, achieved)
    result = reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert result.converged
    assert result.active_set_iterations == 1
    assert result.newton_steps_per_trip == [0]
    record = result.constraints[0]
    ampere_scale = float(np.asarray(pair.unknown.ampere_scale)[0])
    assert abs(float(np.asarray(record.normalized_unknown)[0])) <= UNMOVED_COMPENSATION
    assert abs(float(np.asarray(record.physical_unknown)[0])) <= (
        UNMOVED_COMPENSATION * abs(ampere_scale)
    )
    assert abs(_centroid(profile, result.state) - achieved) <= CENTROID_AGREEMENT


def test_the_row_response_matches_a_central_difference(steered):
    """The response the compensator steers through is the observation's own.

    The matrix the direction is read off is the derivative of the registered
    observation with respect to every prescribed circuit current, taken at
    fixed plasma state, so the central difference that certifies it perturbs
    the flux by the same response and re-reads the same observation.  The step
    stays inside one plasma labelling, which the labels themselves confirm:
    the tangent carries no derivative of a label change, so a step that moved
    one would be comparing two different questions.
    """
    profile, _seed, free = steered
    achieved = _centroid(profile, free.state)
    pair, _selection = _centroid_pair(profile, free.state, achieved)
    field = profile.operator.prescribed_current_field
    direction = np.asarray(pair.unknown.direction)[:, 0]
    image = np.asarray(field.response) @ direction
    span = float(np.ptp(np.asarray(free.state)))
    step = RESPONSE_STEP * span / float(np.max(np.abs(image)))

    tangent = float(
        np.ravel(
            np.asarray(profile.constraint_response_matrix((pair,), free.state))
            @ direction
        )[0]
    )
    state = np.asarray(free.state)
    forward = state + step * image
    backward = state - step * image
    labels = profile.operator.current_domain_masks(jnp.asarray(state), None)
    for probe in (forward, backward):
        probed = profile.operator.current_domain_masks(jnp.asarray(probe), None)
        assert np.array_equal(
            np.asarray(labels.profile_participation),
            np.asarray(probed.profile_participation),
        )
    difference = (_centroid(profile, forward) - _centroid(profile, backward)) / (
        2.0 * step
    )
    assert abs(difference) > 0.0
    assert abs(tangent - difference) <= RESPONSE_AGREEMENT * abs(difference)


def test_a_moved_target_is_reached_and_warm_starting_costs_fewer_trips(steered):
    """A commanded centimetre is delivered, and the warm start is cheaper.

    The same moved target is solved twice: once from the converged free
    equilibrium the previous keyframe left behind, and once from the cold
    analytic seed.  Both must land on the commanded centroid; the warm start
    must reach it in fewer active-set trips, which is the whole reason a
    steered session re-solves from its own previous state.
    """
    profile, seed, free = steered
    commanded = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, commanded)
    common = dict(
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    warm = reduced_newton.solve_constrained_reduced_newton(
        profile, free.state, **common
    )
    cold = reduced_newton.solve_constrained_reduced_newton(profile, seed, **common)
    for result in (warm, cold):
        assert result.converged
        assert abs(_centroid(profile, result.state) - commanded) <= CENTROID_AGREEMENT
        assert (
            abs(float(np.asarray(result.constraints[0].physical_residual)[0]))
            <= CENTROID_AGREEMENT
        )
    assert warm.active_set_iterations < cold.active_set_iterations
    compensating = float(np.asarray(warm.constraints[0].physical_unknown)[0])
    assert compensating != 0.0
    assert np.all(np.isfinite(np.asarray(warm.prescribed_current)))


def test_the_public_route_carries_the_rows_into_a_receipt(steered):
    """A constrained keyframe is reachable by naming the route, not the module.

    The gate that refused augmented constraints off the Newton-Krylov route
    now names the routes that carry a compensating unknown vector, so the same
    row solved through ``ForwardProfile.solve`` returns an ordinary forward
    equilibrium whose terminal record says what the row achieved and whose
    prescribed currents already carry the compensation.
    """
    profile, _seed, free = steered
    commanded = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, commanded)
    equilibrium = profile.solve(
        free.state,
        route="reduced_newton",
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert bool(np.asarray(equilibrium.fixed_point.converged))
    record = equilibrium.constraints[0]
    assert bool(np.asarray(record.qualified)[0])
    assert abs(_centroid(profile, equilibrium.flux) - commanded) <= CENTROID_AGREEMENT
    with pytest.raises(ValueError, match="compensating unknown vector"):
        profile.solve(
            free.state, route="picard", constraint_pairs=(pair,), evaluations=1
        )


def test_the_reduced_route_refuses_a_state_reading_compensator(steered):
    """Only a compensator whose flux image is explicit is admitted.

    The reduced state is amplitudes, so the flux is a function of the
    unknowns rather than an unknown of its own.  A compensator that read the
    state to build its own flux image would make that reconstruction
    implicit, and the route says so instead of solving a different problem.
    """
    profile, _seed, free = steered
    pair = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=ProfileAmplitudeUnknown(
            component="pressure_gradient", amplitude_scale=jnp.asarray([1.0])
        ),
        binding=ConstraintBinding(
            target=jnp.asarray([0.0]),
            tolerance=jnp.asarray([CENTROID_AGREEMENT]),
            scale=jnp.asarray([1.0]),
            initial_unknown=jnp.asarray([0.0]),
        ),
    )
    with pytest.raises(TypeError, match="circuit-current compensators only"):
        reduced_newton.solve_constrained_reduced_newton(
            profile, free.state, constraint_pairs=(pair,)
        )
