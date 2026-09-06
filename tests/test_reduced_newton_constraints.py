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
    from nova.equilibrium.fixed_point import _relative_residual
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import PrescribedCurrentField
    from nova.equilibrium.observation import MomentIntegralSupport
    from nova.equilibrium.reduced_newton import ReducedNewtonResult
    from nova.equilibrium.solve_request import (
        ExplicitSolveSeed,
        ForwardSolveReceipt,
        ForwardSolveRequest,
    )

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
#: rather than trusting the figure.  Measured on this machine the quotient
#: converges on the tangent at first order and cleanly: relative differences
#: of 2.7e-03, 2.7e-04, 2.6e-05, 2.6e-06, 2.6e-07 and 2.8e-08 at fractions of
#: 1e-03 down to 1e-08, with no cancellation floor reached.  This fraction
#: therefore sits a factor of four inside the agreement below rather than at
#: the edge of it.
RESPONSE_STEP = 1.0e-7
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
        if item.name.endswith("_wall_per_trip") or item.name == "program":
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
    assert compared == sum(
        not item.name.endswith("_wall_per_trip") and item.name != "program"
        for item in dataclasses.fields(ReducedNewtonResult)
    )


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


def test_a_moved_target_is_reached_and_warm_starting_costs_less(steered):
    """A commanded centimetre is delivered, and the warm start is cheaper.

    The same moved target is solved twice: once from the converged free
    equilibrium the previous keyframe left behind, and once from the cold
    analytic seed.  Both must land on the commanded centroid, and the warm
    start must cost less to get there, which is the whole reason a steered
    session re-solves from its own previous state.

    Cheaper is counted in Newton steps rather than in active-set trips.  A
    trip ends when the residual shadow stops moving, and on this machine the
    shadow the analytic seed induces is already the converged one, so both
    arms settle in a single trip and the trip count cannot separate them; the
    dense Newton steps inside that trip are where the difference lives.  The
    trip count is still required not to grow, so a warm start that moved the
    shadow would be caught.
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
    assert warm.active_set_iterations <= cold.active_set_iterations
    assert sum(warm.newton_steps_per_trip) < sum(cold.newton_steps_per_trip)
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


#: Agreement required between a scored scalar read out of a program that
#: traces its constraint targets and one that folds them in as constants.
#: The two evaluate the same expressions, and XLA is free to reassociate the
#: reductions behind the merit and the relative residual differently in each;
#: measured on this machine that moves the terminal residual by 6.7e-09 of
#: itself.  This figure brackets that reassociation two decades wide rather
#: than predicting it, and no decision reads the numbers it covers.
SCORE_REASSOCIATION = 1.0e-6
#: Kernels a constrained solve drives on this machine.  Each must compile
#: once across a sequence of moved targets that re-enters one program.
STEERED_KERNELS = ("initial_gather", "jacobian", "direction", "grade", "boundary")


def _cache_sizes(program):
    """Return how many programs each kernel of one built solve has compiled."""
    return {
        name: kernel._cache_size()
        for name, kernel in program.kernels.items()
        if hasattr(kernel, "_cache_size")
    }


def test_consecutive_moved_targets_share_one_compiled_program(steered):
    """Moving a commanded target re-enters the program the caller already has.

    The constraint tuple flattens to its own numerical leaves, so handing it
    to every kernel as a traced argument keeps the target out of the
    program's identity.  Three commanded targets in a row, each re-solved
    from the state and the compensation the last one left behind and each
    handed back the program the last one built, therefore compile every
    kernel exactly once rather than once per target.
    """
    profile, _seed, free = steered
    achieved = _centroid(profile, free.state)
    state, program, unknown = free.state, None, None
    reached = []
    for index in range(3):
        commanded = achieved + (index + 1) * CENTROID_MOVE
        pair, _selection = _centroid_pair(profile, state, commanded)
        if unknown is not None:
            pair = dataclasses.replace(
                pair,
                binding=dataclasses.replace(pair.binding, initial_unknown=unknown),
            )
        result = reduced_newton.solve_constrained_reduced_newton(
            profile,
            state,
            constraint_pairs=(pair,),
            program=program,
            tolerance=SOLVE_TOLERANCE,
            newton_steps=NEWTON_STEPS,
        )
        assert result.converged
        reached.append(_centroid(profile, result.state) - commanded)
        state, program, unknown = (
            result.state,
            result.program,
            result.compensating_unknown,
        )
    assert np.max(np.abs(reached)) <= CENTROID_AGREEMENT
    sizes = _cache_sizes(program)
    assert set(STEERED_KERNELS) <= set(sizes)
    assert max(sizes.values()) == 1, sizes
    assert all(sizes[name] == 1 for name in STEERED_KERNELS), sizes


def test_compiled_constrained_slice_preserves_the_host_flags(steered):
    """The fixed-budget constrained call keeps the host route's decisions."""
    profile, _seed, free = steered
    target = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, target)
    common = dict(
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=1,
        active_set_steps=1,
        prescribed_current=jnp.asarray(
            profile.operator.prescribed_current_field.current
        ),
    )
    host = reduced_newton.solve_constrained_reduced_newton(
        profile, free.state, **common
    )
    compiled = reduced_newton.solve_constrained_reduced_newton_compiled(
        profile, free.state, **common
    )
    assert compiled.converged == host.converged
    assert compiled.termination_reason == host.termination_reason
    assert compiled.active_set_iterations == host.active_set_iterations
    assert compiled.active_set_mask_differences == host.active_set_mask_differences


def test_prescribed_currents_reuse_one_constrained_program(steered):
    """A constrained program accepts a new prescribed current vector as data."""
    profile, _seed, free = steered
    target = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, target)
    base = np.asarray(profile.operator.prescribed_current_field.current, dtype=float)
    edited = base.copy()
    edited[0] += 1.0e-6
    common = dict(
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=1,
    )
    first = reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        prescribed_current=jnp.asarray(base),
        **common,
    )
    reused = reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        prescribed_current=jnp.asarray(edited),
        program=first.program,
        **common,
    )
    reference = reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        prescribed_current=jnp.asarray(edited),
        **common,
    )
    assert np.array_equal(np.asarray(reused.state), np.asarray(reference.state))
    assert len(reused.steps) == len(reference.steps)
    for left, right in zip(reused.steps, reference.steps, strict=True):
        assert left._replace(wall_s=0.0) == right._replace(wall_s=0.0)
    assert reused.terminal_residual == reference.terminal_residual
    sizes = _cache_sizes(first.program)
    assert max(sizes.values()) == 1, sizes


def test_a_traced_target_decides_what_a_captured_one_decides(steered):
    """Where a kernel finds its target changes no decision and no answer.

    Baking the tuple into the program is the reference the traced route has
    to reproduce: the same commanded target solved both ways must walk the
    same trips, accept the same grades and land on the same flux, bit for
    bit, or the saving is being taken out of the answer.

    Every scored scalar is bracketed rather than pinned, and every decision is
    pinned rather than bracketed.  A target that arrives as a program
    parameter and a target folded in as a constant leave XLA free to
    reassociate the reductions that produce the merit and the two residuals,
    and measured on this machine that moves the residual a trip reports by
    about 7e-09 of itself.  The solve stops on that number against a
    tolerance fifteen decades above the motion, so the trips it takes, the
    grades it accepts and the flux it lands on are unmoved, which is what the
    exact comparisons below require.
    """
    profile, _seed, free = steered
    commanded = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, commanded)
    common = dict(
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    captured = reduced_newton.solve_constrained_reduced_newton(
        profile, free.state, row_arguments=reduced_newton.CAPTURED_ROWS, **common
    )
    traced = reduced_newton.solve_constrained_reduced_newton(
        profile, free.state, row_arguments=reduced_newton.TRACED_ROWS, **common
    )
    assert np.array_equal(np.asarray(captured.state), np.asarray(traced.state))
    assert np.array_equal(
        np.asarray(captured.compensating_unknown),
        np.asarray(traced.compensating_unknown),
    )
    for name in (
        "active_set_iterations",
        "converged",
        "termination_reason",
        "active_set_mask_differences",
        "newton_steps_per_trip",
        "jacobian_builds_per_trip",
        "rejected_steps_per_trip",
        "map_evaluations_per_trip",
    ):
        assert getattr(captured, name) == getattr(traced, name), name
    for name in ("terminal_residual", "off_support_leakage"):
        left, right = getattr(captured, name), getattr(traced, name)
        assert abs(left - right) <= SCORE_REASSOCIATION * max(abs(left), 1.0), name
    assert len(captured.active_set_residuals) == len(traced.active_set_residuals)
    for left, right in zip(
        captured.active_set_residuals, traced.active_set_residuals, strict=True
    ):
        assert abs(left - right) <= SCORE_REASSOCIATION * max(abs(left), 1.0)
    assert len(captured.steps) == len(traced.steps)
    for one, other in zip(captured.steps, traced.steps, strict=True):
        for name in ("trip", "step", "accepted_factor", "grades_tried"):
            assert getattr(one, name) == getattr(other, name), name
        for name in ("map_evaluations", "jacobian_refreshed"):
            assert getattr(one, name) == getattr(other, name), name
        for name in ("reduced_residual", "flux_residual", "merit"):
            left, right = getattr(one, name), getattr(other, name)
            assert abs(left - right) <= SCORE_REASSOCIATION * abs(left), name


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


def test_row_response_through_the_routes_own_augmentation_matches_a_central_difference(
    steered,
):
    """The row the merged solve drives responds as a central difference reads.

    The response certified is the one the route's own program solves with: the
    Jacobian it forms at the terminal augmented state carries the row's
    derivative with respect to the compensating unknown, and the central
    difference perturbs that unknown through the same program's reconstruct, so
    the observation read is the same function the solve minimised.  The
    perturbation stays inside one plasma labelling, which the labels confirm.
    """
    profile, _seed, free = steered
    commanded = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, commanded)
    result = reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert result.converged
    coordinates = result.program.coordinates
    operator = profile.operator
    terminal = jnp.asarray(result.state)
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(terminal, None), dtype=bool)
    )
    moments = operator.cell_current_moments(terminal)
    amplitudes = reduced_newton._gather(coordinates, moments)
    unknown = float(np.asarray(result.compensating_unknown)[0])
    kernels = result.program.kernels

    def flux_at(value):
        reduced = jnp.concatenate(
            (amplitudes, jnp.asarray([value], dtype=amplitudes.dtype))
        )
        return kernels["reconstruct"](reduced, shadow, terminal)

    span = float(np.ptp(np.asarray(result.state)))
    probe = 1.0e-3 * max(abs(unknown), 1.0)
    unit_flux = (
        float(jnp.max(jnp.abs(flux_at(unknown + probe) - flux_at(unknown)))) / probe
    )
    # Perturb the flux by a tenth of the step the response matrix contract
    # uses, so the labels cannot move and the quotient sits decades inside the
    # agreement it is certified against.
    step = (RESPONSE_STEP / 10.0) * span / max(unit_flux, 1.0e-12)
    labels = operator.current_domain_masks(terminal, None)
    for value in (unknown + step, unknown - step):
        probed = operator.current_domain_masks(flux_at(value), None)
        assert np.array_equal(
            np.asarray(labels.profile_participation),
            np.asarray(probed.profile_participation),
        )
    difference = (
        _centroid(profile, flux_at(unknown + step))
        - _centroid(profile, flux_at(unknown - step))
    ) / (2.0 * step)
    reduced = jnp.concatenate(
        (amplitudes, jnp.asarray([unknown], dtype=amplitudes.dtype))
    )
    jacobian = kernels["jacobian"](reduced, shadow, terminal)
    scale = float(np.asarray(pair.binding.scale)[0])
    row_block = float(
        np.ravel(np.asarray(jacobian[coordinates.size :, coordinates.size :]))[0]
    )
    # rows = (observed - target) / scale, so d(observed)/dc = scale * d(rows)/dc.
    tangent = scale * row_block
    assert abs(difference) > 0.0
    assert abs(tangent - difference) <= RESPONSE_AGREEMENT * abs(difference)


def test_the_line_search_follows_the_augmented_merit_when_the_blocks_disagree(steered):
    """Where the flux block is done and the row block is not, the row decides.

    At the converged free state a moved row leaves the flux block already
    inside the tolerance while the augmented merit is still far above it.  A
    ladder that scored the flux alone would declare the state converged and
    take no step, so the only thing that can make the route move is the row
    block, and the grade it accepts is the one the augmented merit accepts.
    """
    profile, _seed, free = steered
    assert free.terminal_residual <= SOLVE_TOLERANCE
    commanded = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, commanded)
    result = reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert result.converged
    assert result.newton_steps_per_trip and result.newton_steps_per_trip[0] >= 1
    first = result.steps[0]
    assert first.accepted_factor == 1.0
    # The augmented exit metric at the incumbent was above the tolerance the
    # free solve had cleared, so the row block is what the step is for.
    assert first.flux_residual > SOLVE_TOLERANCE

    # The full-length candidate is where the blocks disagree.  The flux block
    # alone at the incumbent is done, and its own merit at the candidate is
    # worse than the incumbent, so a ladder that scored the flux alone would
    # refuse this grade; the augmented merit at the candidate is below the
    # incumbent because the row block improves, so the augmented selection
    # accepts it - and the step the route actually took is grade zero.
    coordinates = result.program.coordinates
    operator = profile.operator
    base_state = jnp.asarray(free.state)
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(base_state, None), dtype=bool)
    )
    amplitudes = reduced_newton._gather(
        coordinates, operator.cell_current_moments(base_state)
    )
    incumbent = jnp.concatenate(
        (amplitudes, jnp.asarray([0.0], dtype=amplitudes.dtype))
    )
    kernels = result.program.kernels
    direction = kernels["direction"](
        kernels["jacobian"](incumbent, shadow, base_state),
        kernels["reduced_residual"](incumbent, shadow, base_state),
    )
    candidate = incumbent + direction
    augmented_incumbent = kernels["step_scores"](incumbent, shadow, base_state)
    augmented_candidate = kernels["step_scores"](candidate, shadow, base_state)
    assert float(augmented_candidate.merit) < float(augmented_incumbent.merit)
    # The flux-only program carries no unknown block, so it scores the
    # amplitudes alone and the flux part of the augmented route's direction.
    unaugmented = reduced_newton._reduced_kernels(
        operator, coordinates, operator.external(), None, None
    )
    candidate_flux = amplitudes + direction[: coordinates.size]
    flux_only_incumbent = unaugmented["flux_scores"](amplitudes, shadow, base_state)
    flux_only_candidate = unaugmented["flux_scores"](candidate_flux, shadow, base_state)
    assert float(flux_only_incumbent[1]) <= SOLVE_TOLERANCE
    assert float(flux_only_candidate[0]) > float(flux_only_incumbent[0])


def test_a_large_compensating_unknown_does_not_loosen_the_scoring(steered):
    """Holding the augmented residual fixed, a hundred-fold unknown moves nothing.

    The exit test and the merit divide by a reference.  When that reference is
    the concatenated augmented vector, a normalised compensating unknown of
    order one hundred sets it and divides the residuals down by that factor, so
    the tolerance the whole system stops on loosens as a keyframe loop's
    unknown accumulates.  This contract builds the scored and imaged vectors
    exactly as the route's scoring builds them, holds the residual and the flux
    reference fixed while the unknown block grows from zero to one hundred, and
    requires the bounded scoring to return identical merit and exit values at
    every unknown, while the unbounded expressions the augmented route used to
    evaluate would have loosened by the same amount.
    """
    profile, _seed, free = steered
    commanded = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, commanded)
    result = reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert result.converged
    scale = float(np.max(np.abs(np.asarray(free.state))))
    rows = float(np.asarray(result.constraints[0].scaled_residual)[0])
    flux_scored = jnp.asarray(result.state, dtype=jnp.float64) / scale
    flux_imaged = flux_scored + 1.0e-8
    reference = flux_imaged

    def scored_imaged(unknown):
        scored = jnp.concatenate((flux_scored, jnp.asarray([unknown])))
        imaged = jnp.concatenate((flux_imaged, jnp.asarray([unknown - rows])))
        return scored, imaged

    bounded = {}
    for unknown in (0.0, 1.0, 10.0, 100.0):
        scored, imaged = scored_imaged(unknown)
        bounded[unknown] = (
            reduced_newton._augmented_smooth_relative_sup_merit(
                imaged, scored, reference
            ),
            reduced_newton._augmented_relative_residual(imaged, scored, reference),
        )
    for unknown in (1.0, 10.0, 100.0):
        assert float(bounded[unknown][0]) == float(bounded[0.0][0])
        assert float(bounded[unknown][1]) == float(bounded[0.0][1])
    # The unbounded expressions the merged code replaced loosen with the
    # unknown; asserting they do shows the bounded scoring is doing the work.
    old = _relative_residual(*scored_imaged(100.0))
    old_base = _relative_residual(*scored_imaged(0.0))
    assert float(old) <= 0.2 * float(old_base)


def test_the_reduced_route_publishes_soft_mode_projection_as_a_typed_absence(steered):
    """No soft-mode projection exists on a dense route; the record says so.

    The Krylov augmented solver publishes the projection it measured; the dense
    reduced route has none, and a NaN would read as a number rather than as the
    absence it is.  The rest of the record stays real.
    """
    profile, _seed, free = steered
    achieved = _centroid(profile, free.state)
    pair, _selection = _centroid_pair(profile, free.state, achieved + CENTROID_MOVE)
    result = reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert result.converged
    assert all(record.soft_mode_projection is None for record in result.constraints)
    record = result.constraints[0]
    assert bool(np.asarray(record.qualified)[0])
    assert np.isfinite(float(np.asarray(record.physical_residual)[0]))
    assert np.all(np.isfinite(np.asarray(result.prescribed_current)))


def test_a_constrained_solve_without_a_prescribed_field_is_refused(steered):
    """Compensation with nowhere to fold into is refused, with the reason stated.

    A circuit compensator's flux image is the machine's prescribed response, so
    with neither ``prescribed_current`` nor an operator-side field a constrained
    solve has no machine to drive and no receipt to fold into; the route refuses
    up front rather than letting the individual compensator raise mid-trace.
    """
    profile, _seed, free = steered
    achieved = _centroid(profile, free.state)
    pair, _selection = _centroid_pair(profile, free.state, achieved + CENTROID_MOVE)
    saved = profile.operator.prescribed_field
    profile.operator.prescribed_field = None
    try:
        with pytest.raises(ValueError, match="prescribed current field"):
            reduced_newton.solve_constrained_reduced_newton(
                profile,
                free.state,
                constraint_pairs=(pair,),
                tolerance=SOLVE_TOLERANCE,
                newton_steps=NEWTON_STEPS,
            )
    finally:
        profile.operator.prescribed_field = saved


def test_reduced_requests_with_constraints_never_reach_the_krylov_solver(
    steered, monkeypatch
):
    """The widened accelerated gate cannot hand a reduced request to Krylov.

    ``solve`` routes the reduced routes to their own entry before the shared
    ladder is consulted, and the ladder itself refuses a constrained reduced
    route outright, so the Krylov augmented solver is unreachable from a
    ``reduced_newton`` request no matter how the routing evolves.
    """
    profile, _seed, free = steered
    commanded = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, commanded)

    def fail(*args, **kwargs):
        raise AssertionError(
            "a reduced_newton request must never reach the Krylov augmented solver"
        )

    monkeypatch.setattr(ForwardProfile, "_solve_augmented_constraints", fail)
    equilibrium = profile.solve(
        free.state,
        route="reduced_newton",
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert bool(np.asarray(equilibrium.fixed_point.converged))
    with pytest.raises(ValueError, match="compensating unknown vector"):
        profile._solve_accelerated(
            "reduced_newton", free.state, None, constraint_pairs=(pair,)
        )


def test_the_request_path_reports_the_reduced_active_set_history(steered):
    """A typed reduced request selects the reduced active-set arrays.

    The request-path history selection covers the reduced routes beside the
    Krylov route, so the receipt reports the per-trip residuals and mask
    differences the reduced solve publishes rather than a plain flux trace.
    """
    profile, _seed, free = steered
    commanded = _centroid(profile, free.state) + CENTROID_MOVE
    pair, _selection = _centroid_pair(profile, free.state, commanded)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="pfs-reduced-history",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(np.asarray(free.state)),
        policy_overrides={
            "route": "reduced_newton",
            "kernel_tolerance": SOLVE_TOLERANCE,
            "newton_steps": NEWTON_STEPS,
            "compilation_cache": False,
        },
        constraint_pairs=(pair,),
    )
    receipt = profile.solve(request)
    assert isinstance(receipt, ForwardSolveReceipt)
    history = receipt.equilibrium.fixed_point
    assert len(receipt.residual_history) >= 1
    np.testing.assert_array_equal(
        np.asarray(receipt.residual_history),
        np.asarray(history.active_set_residuals),
    )
    np.testing.assert_array_equal(
        np.asarray(receipt.mask_history),
        np.asarray(history.active_set_mask_differences),
    )
