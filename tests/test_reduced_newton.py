"""Reduced-amplitude plain Newton against the production Newton-Krylov ladder.

The machine is the same bootstrapped Solov'ev free-boundary problem the
forward-solve contract uses: a ring of external conductors fitted to hold an
analytic seed, driven by an edge-vanishing absolute source so the wall-limited
branch attracts. It is a real free-boundary map built from Nova's Green
kernels, not an algebraic identity, so a fixed point reached on the plasma-cell
amplitudes is evidence about the map rather than about the parameterisation.

Pinned here: the reduced coordinates carry the whole plasma current the map
drives; reconstructing flux from them reproduces the production map exactly;
the reduced plain-Newton solve lands on the production Newton-Krylov solve's
fixed point to within the solver tolerance; scoring the refused grade tail in
one dispatch returns bitwise what the per-grade dispatches return and walks
the same solve; and the residual sup norm the receipt reports comes back from
the scoring kernel rather than from a reduction dispatched after it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.equilibrium import reduced_newton
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.fixed_point import _BACKTRACKING_FACTORS, _relative_residual
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.jax.config import configure_dtypes


P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_FIELD_FUNCTION = 5.0
CONDUCTORS = 16
#: Relative fixed-point residual both routes are driven to.
SOLVE_TOLERANCE = 1.0e-8
#: Sup-norm flux agreement between the two terminal states, as a fraction of
#: the flux span. Each route stops as soon as its own residual clears the
#: tolerance above, so their terminals differ by the map's amplification of
#: that residual rather than by round-off; the pin brackets that amplification
#: at two decades rather than predicting it.
FIXED_POINT_AGREEMENT = 100.0 * SOLVE_TOLERANCE
#: Newton budget each arm is given. The prototype holds one Jacobian for a
#: whole trip and refreshes it only when a grade ladder is refused, so its
#: inner iteration is a chord method and contracts linearly where an
#: exact-tangent Newton contracts quadratically; measured on this machine the
#: reduced factor is about a quarter per step, so it needs the wider budget to
#: reach the same tolerance from the same seed.
REDUCED_NEWTON_STEPS = 24
#: Agreement required between the fused trip boundary and the dispatched calls
#: it replaces, on the residual each trip observes and on the terminal flux as
#: a fraction of its span. The two compute the same expressions and differ only
#: where one compiled program reassociates what several evaluate separately, so
#: these bracket that reassociation rather than predict it.
BOUNDARY_RESIDUAL_AGREEMENT = 1.0e-9
BOUNDARY_FLUX_AGREEMENT = 1.0e-9
PRODUCTION_NEWTON_STEPS = 12
GMRES_ITERATIONS = 12


def _terms():
    """Return the Solov'ev quartic, offset and vertical coefficients."""
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius, height):
    """Return the analytic seed flux [Wb] the conductors are fitted to."""
    alpha, offset, beta = _terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _wall_loop(points=61):
    """Return a material boundary lying on one seed flux surface."""
    alpha, offset, beta = _terms()
    wall_flux = _solovev(AXIS_RADIUS, 0.0) - SEED_SPAN
    inner, outer = np.sqrt(np.sort(np.roots([alpha, offset, -wall_flux])))
    centre, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
    angle = 2 * np.pi * np.arange(points) / points
    radius = centre + half * np.cos(angle)
    argument = np.clip((wall_flux - _solovev(radius, 0.0)) / beta, 0.0, None)
    return np.c_[radius, np.sign(np.sin(angle)) * np.sqrt(argument)], wall_flux


def _green_block(target, source, section=0.05):
    """Return the total-flux coupling [Wb/A] of one source set on one target."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


def _flat_profile(amplitude):
    """Return a constant absolute gradient."""

    def gradient(psi_norm):
        """Return the constant value at every normalised flux."""
        return jnp.full_like(jnp.asarray(psi_norm, dtype=jnp.float64), amplitude)

    return gradient


def _edge_vanishing_profile(amplitude):
    """Return an absolute gradient that falls linearly to zero at the edge."""

    def gradient(psi_norm):
        """Return the tapered value at one normalised flux."""
        return amplitude * (1.0 - jnp.clip(jnp.asarray(psi_norm), 0.0, 1.0))

    return gradient


@pytest.fixture(scope="module")
def machine():
    """Return the bootstrapped free-boundary solve and its analytic seed."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.6, 1.42, 15), np.linspace(-0.42, 0.42, 15))
    coordinate = lattice.coordinate
    wall, wall_flux = _wall_loop()
    seed_flux = _solovev(coordinate[:, 0], coordinate[:, 1])
    wall_seed = _solovev(wall[:, 0], wall[:, 1])
    inside = seed_flux >= wall_flux

    angle = 2 * np.pi * np.arange(CONDUCTORS) / CONDUCTORS
    conductor = np.c_[1.0 + 0.62 * np.cos(angle), 0.62 * np.sin(angle)]
    coupling = {
        "plasma_to_grid": _green_block(coordinate, coordinate),
        "plasma_to_wall": _green_block(wall, coordinate),
        "source_to_grid": _green_block(coordinate, conductor),
        "source_to_wall": _green_block(wall, conductor),
    }

    def build(core, current):
        """Return the solve for one declared source and conductor state."""
        return ForwardProfile.from_lattice(
            lattice,
            ForwardSource(core=core, boundary_field_function=BOUNDARY_FIELD_FUNCTION),
            external_current=current,
            wall_coordinate=wall,
            polarity=1,
            inside_material=inside,
            **coupling,
        )

    seed = jnp.asarray(np.r_[seed_flux, wall_seed])
    flat = build(
        DomainProfile(p_prime=_flat_profile(P_PRIME), ff_prime=_flat_profile(FF_PRIME)),
        np.zeros(CONDUCTORS),
    )
    cell_current = np.asarray(flat.operator.cell_current(seed))
    target = np.r_[
        seed_flux - coupling["plasma_to_grid"] @ cell_current,
        wall_seed - coupling["plasma_to_wall"] @ cell_current,
    ]
    weight = np.r_[inside.astype(float), np.ones(len(wall))]
    matrix = np.r_[coupling["source_to_grid"], coupling["source_to_wall"]]
    current = np.linalg.lstsq(matrix * weight[:, None], target * weight, rcond=None)[0]

    profile = build(
        DomainProfile(
            p_prime=_edge_vanishing_profile(2.0 * DRIVE * P_PRIME),
            ff_prime=_edge_vanishing_profile(2.0 * DRIVE * FF_PRIME),
        ),
        current,
    )
    return profile, seed


def test_reduced_coordinates_carry_the_whole_plasma_current(machine):
    """The profile-owned support holds every cell the map drives current in."""
    profile, seed = machine
    operator = profile.operator
    coordinates = reduced_newton.reduced_coordinates(operator, seed)
    assert coordinates.leaves == ("cell_current",)
    assert coordinates.cell_number == operator.grid.node_number
    assert 0 < coordinates.size < operator.node_number
    current = np.asarray(operator.cell_current(seed))
    carried = np.zeros(coordinates.cell_number, dtype=bool)
    carried[np.asarray(coordinates.cells)] = True
    assert np.count_nonzero(current) > 0
    assert not np.any(current[~carried])


def test_reconstruction_reproduces_the_production_flux_map(machine):
    """Flux rebuilt from the reduced amplitudes equals the production map."""
    profile, seed = machine
    operator = profile.operator
    coordinates = reduced_newton.reduced_coordinates(operator, seed)
    shadow = jnp.ravel(jnp.asarray(operator.residual_shadow_mask(seed), dtype=bool))
    kernels = reduced_newton._reduced_kernels(
        operator, coordinates, operator.external(), None, None
    )
    moments = operator.cell_current_moments(seed)
    reduced = reduced_newton._gather(coordinates, moments)
    rebuilt = kernels["reconstruct"](reduced, shadow, seed)
    expected = operator.flux_map_with_shadow()(seed, shadow)
    span = float(jnp.max(jnp.abs(expected)))
    assert float(jnp.max(jnp.abs(rebuilt - expected))) <= 1.0e-12 * span


@pytest.mark.slow
def test_reduced_newton_reaches_the_production_fixed_point(machine):
    """Both routes converge, to the same flux, within the solver tolerance."""
    profile, seed = machine
    production = profile.solve(
        seed,
        route="newton_krylov",
        convergence_tolerance=SOLVE_TOLERANCE,
        newton_steps=PRODUCTION_NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    )
    reduced = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        tolerance=SOLVE_TOLERANCE,
        newton_steps=REDUCED_NEWTON_STEPS,
    )
    assert bool(np.asarray(production.fixed_point.converged))
    assert reduced.converged
    assert reduced.termination_name == "converged"
    assert reduced.off_support_leakage == 0.0

    mapped = profile.operator.flux_map()
    for state in (production.flux, reduced.state):
        assert float(_relative_residual(mapped(state), state)) <= SOLVE_TOLERANCE
    span = float(jnp.max(jnp.abs(production.flux)))
    difference = float(jnp.max(jnp.abs(reduced.state - production.flux)))
    assert difference <= FIXED_POINT_AGREEMENT * span


@pytest.mark.slow
def test_dense_jacobian_is_square_on_the_reduced_state(machine):
    """One trip's Jacobian is a dense square matrix of the reduced dimension."""
    profile, seed = machine
    operator = profile.operator
    coordinates = reduced_newton.reduced_coordinates(operator, seed)
    shadow = jnp.ravel(jnp.asarray(operator.residual_shadow_mask(seed), dtype=bool))
    kernels = reduced_newton._reduced_kernels(
        operator, coordinates, operator.external(), None, None
    )
    reduced = reduced_newton._gather(coordinates, operator.cell_current_moments(seed))
    jacobian = kernels["jacobian"](reduced, shadow, seed)
    assert jacobian.shape == (coordinates.size, coordinates.size)
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    residual = kernels["reduced_residual"](reduced, shadow, seed)
    direction = kernels["direction"](jacobian, residual)
    assert float(jnp.max(jnp.abs(jacobian @ direction + residual))) <= 1.0e-6 * float(
        jnp.max(jnp.abs(residual))
    )


@pytest.mark.slow
def test_ladder_scoring_accepts_the_grade_the_eager_ladder_selects(machine):
    """Stopping at the first grade below the merit selects the eager grade.

    The eager ladder scores every backtracking grade and then takes the
    earliest one below the incumbent merit, so a route that stops scoring at
    that grade must reach the same decision on every step of a whole solve,
    and the two solves must then walk the same trips to the same flux.
    """
    profile, seed = machine
    common = dict(
        tolerance=SOLVE_TOLERANCE,
        newton_steps=REDUCED_NEWTON_STEPS,
        trip_boundary=reduced_newton.DISPATCHED_TRIP_BOUNDARY,
    )
    eager = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        ladder_scoring=reduced_newton.EAGER_LADDER_SCORING,
        **common,
    )
    scored = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        ladder_scoring=reduced_newton.LADDER_SCORING,
        **common,
    )
    assert eager.steps
    assert len(scored.steps) == len(eager.steps)
    for taken, reference in zip(scored.steps, eager.steps, strict=True):
        assert (taken.trip, taken.step) == (reference.trip, reference.step)
        assert taken.accepted_factor == reference.accepted_factor
        assert taken.grades_tried == reference.grades_tried
        assert taken.jacobian_refreshed == reference.jacobian_refreshed
    assert scored.newton_steps_per_trip == eager.newton_steps_per_trip
    assert scored.active_set_mask_differences == (eager.active_set_mask_differences)
    assert scored.termination_name == eager.termination_name
    span = float(jnp.max(jnp.abs(eager.state)))
    difference = float(jnp.max(jnp.abs(scored.state - eager.state)))
    assert difference <= 1.0e-12 * span


@pytest.mark.slow
def test_ladder_scoring_evaluates_one_map_per_accepted_step(machine):
    """A step accepted at full length costs one map evaluation, not eight.

    The eager route evaluates the map for the reduced residual, again for the
    incumbent merit and once per grade; the scoring route reads all of them
    from the grade it accepts, so an accepting step evaluates the map once and
    only the trip's opening state is scored separately.
    """
    profile, seed = machine
    common = dict(
        tolerance=SOLVE_TOLERANCE,
        newton_steps=REDUCED_NEWTON_STEPS,
        trip_boundary=reduced_newton.DISPATCHED_TRIP_BOUNDARY,
    )
    eager = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        ladder_scoring=reduced_newton.EAGER_LADDER_SCORING,
        **common,
    )
    scored = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        ladder_scoring=reduced_newton.LADDER_SCORING,
        **common,
    )
    full_length = [step for step in scored.steps if step.accepted_factor == 1.0]
    assert full_length
    assert all(step.map_evaluations == 1 for step in full_length[1:])
    assert sum(scored.map_evaluations_per_trip) < sum(eager.map_evaluations_per_trip)


@pytest.mark.slow
def test_fused_trip_boundary_reproduces_the_dispatched_boundary(machine):
    """One compiled trip close returns what the separate calls returned.

    The promoted shadow, the mask difference against the frozen shadow and the
    live residual of the promoted map are the boundary's decisions, and they
    are compared for exact equality against the calls the dispatched boundary
    makes at the same reduced state and the same frozen shadow.  The next
    trip's amplitudes and the off-support leakage are carried values rather
    than decisions and are compared to a tolerance: one program computing the
    whole boundary is free to reassociate arithmetic that separate programs
    evaluate one expression at a time, and that reassociation moves the last
    bits of a sum without moving anything the solve decides.
    """
    profile, seed = machine
    operator = profile.operator
    coordinates = reduced_newton.reduced_coordinates(operator, seed)
    shadow = jnp.ravel(jnp.asarray(operator.residual_shadow_mask(seed), dtype=bool))
    kernels = reduced_newton._reduced_kernels(
        operator, coordinates, operator.external(), None, None
    )
    reduced = kernels["initial_gather"](seed)
    direction = kernels["direction"](
        kernels["jacobian"](reduced, shadow, seed),
        kernels["reduced_residual"](reduced, shadow, seed),
    )
    for factor in (0.0, 1.0):
        moved = reduced + factor * direction
        state, promoted, difference, residual, gathered, leakage = kernels["boundary"](
            moved, shadow, seed
        )
        expected_state = kernels["reconstruct"](moved, shadow, seed)
        expected_promoted = jnp.ravel(
            jnp.asarray(
                operator.residual_shadow_mask(
                    expected_state, None, previous_shadow=shadow
                ),
                dtype=bool,
            )
        )
        mapped = operator.flux_map_with_shadow()(expected_state, expected_promoted)
        assert bool(jnp.array_equal(state, expected_state))
        assert bool(jnp.array_equal(promoted, expected_promoted))
        assert int(difference) == int(jnp.sum(expected_promoted != shadow))
        assert float(residual) == float(_relative_residual(mapped, expected_state))
        expected_gather = reduced_newton._gather(
            coordinates, operator.cell_current_moments(expected_state)
        )
        span = float(jnp.max(jnp.abs(expected_gather)))
        assert float(jnp.max(jnp.abs(gathered - expected_gather))) <= 1.0e-12 * span
        assert float(leakage) == pytest.approx(
            float(kernels["leakage"](moved, shadow, seed)), abs=1.0e-15
        )


@pytest.mark.slow
def test_fused_trip_boundary_solves_to_the_dispatched_terminal_state(machine):
    """Closing trips in one program walks the same trips to the same flux.

    Every decision is compared for equality: the trips taken, the mask
    difference each one promoted, the Newton steps inside it, the termination
    and the leakage.  The residual each trip observed and the terminal flux
    are compared to a tolerance, because a single program is free to
    reassociate arithmetic that separate dispatches evaluate one expression at
    a time.  Measured on the H200 bank rows that reassociation moves the
    terminal flux by 3e-16 to 3e-15 of its span; the pins below hold it far
    inside what would change a decision.
    """
    profile, seed = machine
    common = dict(
        tolerance=SOLVE_TOLERANCE,
        newton_steps=REDUCED_NEWTON_STEPS,
        ladder_scoring=reduced_newton.LADDER_SCORING,
    )
    dispatched = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        trip_boundary=reduced_newton.DISPATCHED_TRIP_BOUNDARY,
        **common,
    )
    fused = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        trip_boundary=reduced_newton.TRIP_BOUNDARY,
        **common,
    )
    assert fused.active_set_iterations == dispatched.active_set_iterations
    assert fused.active_set_mask_differences == dispatched.active_set_mask_differences
    assert fused.newton_steps_per_trip == dispatched.newton_steps_per_trip
    assert fused.termination_name == dispatched.termination_name
    assert fused.off_support_leakage == dispatched.off_support_leakage
    assert fused.active_set_residuals == pytest.approx(
        dispatched.active_set_residuals, rel=BOUNDARY_RESIDUAL_AGREEMENT
    )
    span = float(jnp.max(jnp.abs(dispatched.state)))
    difference = float(jnp.max(jnp.abs(fused.state - dispatched.state)))
    assert difference <= BOUNDARY_FLUX_AGREEMENT * span


def test_batched_tail_scores_the_grades_the_serial_ladder_dispatches(machine):
    """One batched tail returns what the per-grade dispatches return, bitwise.

    The tail exists to remove round trips, not to change arithmetic: it builds
    the same candidate amplitudes from the same factors and drives them
    through the same scoring program, one lane at a time.  Every leaf is
    compared for exact equality against the single-grade kernel, because a
    tail that moved a last bit could move a merit across the incumbent and
    change which grade the selection takes.
    """
    profile, seed = machine
    operator = profile.operator
    coordinates = reduced_newton.reduced_coordinates(operator, seed)
    shadow = jnp.ravel(jnp.asarray(operator.residual_shadow_mask(seed), dtype=bool))
    kernels = reduced_newton._reduced_kernels(
        operator, coordinates, operator.external(), None, None
    )
    reduced = kernels["initial_gather"](seed)
    direction = kernels["direction"](
        kernels["jacobian"](reduced, shadow, seed),
        kernels["reduced_residual"](reduced, shadow, seed),
    )
    candidates, tail = kernels["tail"](reduced, direction, shadow, seed)
    factors = _BACKTRACKING_FACTORS[1:]
    assert candidates.shape == (len(factors), coordinates.size)
    for index, factor in enumerate(factors):
        expected_candidate, expected = kernels["grade"](
            reduced, direction, jnp.asarray(factor, dtype=reduced.dtype), shadow, seed
        )
        assert bool(jnp.array_equal(candidates[index], expected_candidate))
        for lane, reference in zip(tail, expected, strict=True):
            assert bool(jnp.array_equal(lane[index], reference))


def test_step_kernel_returns_the_residual_sup_norm(machine):
    """The norm the receipt reads comes back with the residual, not after it.

    Nothing the solve branches on reads the reduced residual's sup norm, so
    reducing it in a dispatch of its own would spend a synchronised round trip
    per Newton step on a number only the receipt reports.  The scoring kernel
    already holds the residual, and the value it returns is the value a
    separate reduction would have produced.
    """
    profile, seed = machine
    operator = profile.operator
    coordinates = reduced_newton.reduced_coordinates(operator, seed)
    shadow = jnp.ravel(jnp.asarray(operator.residual_shadow_mask(seed), dtype=bool))
    kernels = reduced_newton._reduced_kernels(
        operator, coordinates, operator.external(), None, None
    )
    reduced = kernels["initial_gather"](seed)
    scores = kernels["step_scores"](reduced, shadow, seed)
    assert float(scores.residual_norm) == float(jnp.max(jnp.abs(scores.residual)))
    assert bool(
        jnp.array_equal(
            scores.residual, kernels["reduced_residual"](reduced, shadow, seed)
        )
    )
    solved = reduced_newton.solve_reduced_newton(
        operator, seed, tolerance=SOLVE_TOLERANCE, newton_steps=1, active_set_steps=1
    )
    assert solved.steps
    assert solved.steps[0].reduced_residual == float(scores.residual_norm)


@pytest.mark.slow
def test_batched_tail_scoring_walks_the_serial_route(machine):
    """Batching the refused tail leaves every decision and the flux unmoved.

    The two routes differ only in how many dispatches a refused first grade
    costs, so the grade each step accepts, the trips taken, the masks promoted
    and the termination must be equal, and the terminal flux bitwise equal:
    where a first grade accepts the two routes execute the same programs, and
    where it is refused the tail scores the same candidates through the same
    program one lane at a time.
    """
    profile, seed = machine
    common = dict(
        tolerance=SOLVE_TOLERANCE,
        newton_steps=REDUCED_NEWTON_STEPS,
        trip_boundary=reduced_newton.TRIP_BOUNDARY,
    )
    serial = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        ladder_scoring=reduced_newton.SERIAL_LADDER_SCORING,
        **common,
    )
    batched = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        ladder_scoring=reduced_newton.LADDER_SCORING,
        **common,
    )
    assert serial.steps
    assert len(batched.steps) == len(serial.steps)
    for taken, reference in zip(batched.steps, serial.steps, strict=True):
        assert (taken.trip, taken.step) == (reference.trip, reference.step)
        assert taken.accepted_factor == reference.accepted_factor
        assert taken.grades_tried == reference.grades_tried
        assert taken.jacobian_refreshed == reference.jacobian_refreshed
        assert taken.reduced_residual == reference.reduced_residual
        assert taken.merit == reference.merit
        assert taken.flux_residual == reference.flux_residual
    assert batched.active_set_iterations == serial.active_set_iterations
    assert batched.newton_steps_per_trip == serial.newton_steps_per_trip
    assert batched.active_set_mask_differences == serial.active_set_mask_differences
    assert batched.termination_name == serial.termination_name
    assert batched.terminal_residual == serial.terminal_residual
    assert bool(jnp.array_equal(batched.state, serial.state))
