"""Shape-control rows on the bootstrapped Solov'ev free-boundary machine.

The machine is the same one the forward-solve contract uses: a ring of
external conductors fitted to hold an analytic seed, driven by an
edge-vanishing absolute source so the wall-limited branch attracts.  The
conductors are given a prescribed-current field with a zero baseline, so a
compensating circuit current is a pure actuator handle on a real Green
response rather than an algebraic identity.

Pinned here: the Miller parametrisation reproduces the shape figures it was
handed; the constraint-response matrix of every shape row equals a central
difference along the same circuit's flux image; a constrained solve drives the
isoflux, stationary-point and wall-gap rows onto their commanded targets; and
a moved target re-solved from the previous equilibrium costs fewer trips than
the same target solved from the seed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.equilibrium import (
        CircuitCurrentUnknown,
        ConstraintBinding,
        ConstraintPair,
        IsofluxConstraint,
        WallGapConstraint,
        WallGapTarget,
        XPointConstraint,
        compensator_rule_name,
        constraint_response_matrix,
        derive_circuit_compensators,
        miller_boundary_points,
    )
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.constraint import ConstraintContext
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import PrescribedCurrentField
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.jax.config import configure_dtypes


P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_FIELD_FUNCTION = 5.0
CONDUCTORS = 16
NEWTON_STEPS = 12
GMRES_ITERATIONS = 12
#: Poloidal control points on the Miller target boundary. Four rows leave the
#: sixteen-conductor response matrix comfortably over-determined, so the
#: singular-distribution rule has room to move one row without dragging the
#: others.
CONTROL_POINTS = 4
#: Commanded moves. Both are a few percent of the minor radius, the scale a
#: shape controller actually steers in, and small enough that the previous
#: equilibrium is a good warm start for the next.
VERTICAL_STEP = 0.02
ELONGATION_STEP = 0.05


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
    lattice = FluxLattice(np.linspace(0.6, 1.42, 25), np.linspace(-0.42, 0.42, 25))
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
    # A zero-baseline prescribed field leaves the map untouched and hands the
    # compensating directions the conductors' own Green response.
    profile.operator.prescribed_field = PrescribedCurrentField(
        response=jnp.asarray(
            np.r_[coupling["source_to_grid"], coupling["source_to_wall"]]
        ),
        current=jnp.zeros(CONDUCTORS),
    )
    return profile, seed


@pytest.fixture(scope="module")
def converged(machine):
    """Return the unconstrained equilibrium every commanded move starts from."""
    profile, seed = machine
    return profile.solve(
        seed,
        route="newton_krylov",
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    )


def _shape(profile, flux):
    """Return the achieved boundary extremes and the Miller figures they imply."""
    boundary = _boundary_polygon(profile, flux)
    radius, height = boundary[:, 0], boundary[:, 1]
    inner, outer = float(np.min(radius)), float(np.max(radius))
    lower, upper = float(np.min(height)), float(np.max(height))
    minor = 0.5 * (outer - inner)
    return {
        "geometric_radius": 0.5 * (outer + inner),
        "geometric_height": 0.5 * (upper + lower),
        "minor_radius": minor,
        "elongation": 0.5 * (upper - lower) / minor,
        "triangularity": (0.5 * (outer + inner) - float(radius[int(np.argmax(height))]))
        / minor,
    }


def _boundary_polygon(profile, flux, angles=181):
    """Ray-cast the boundary contour outward from the magnetic axis."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    axis = np.asarray(topology.axis, dtype=float)
    level = float(np.asarray(topology.boundary_flux))
    polarity = float(profile.operator.polarity)
    theta = 2.0 * np.pi * np.arange(angles) / angles
    lattice = profile.lattice
    span = 0.5 * min(
        float(lattice.radius[-1] - lattice.radius[0]),
        float(lattice.height[-1] - lattice.height[0]),
    )
    grid = np.asarray(flux)[: lattice.node_count].reshape(lattice.shape)
    points = []
    for angle in theta:
        ray = np.c_[np.cos(angle) * np.ones(1), np.sin(angle) * np.ones(1)][0]
        low, high = 0.0, span
        for _step in range(60):
            middle = 0.5 * (low + high)
            value = float(np.asarray(_sample(profile, grid, axis + middle * ray)))
            if polarity * (value - level) > 0.0:
                low = middle
            else:
                high = middle
        points.append(axis + 0.5 * (low + high) * ray)
    return np.asarray(points)


def _sample(profile, grid, point):
    """Return the interpolated flux the constraint rows read at one point."""
    from nova.equilibrium.constraint import sample_lattice_flux

    return sample_lattice_flux(profile.lattice, jnp.asarray(grid), jnp.asarray(point))


def _reference_point(profile, flux):
    """Return the wall point whose flux sets the limited boundary level."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    return np.asarray(topology.wall_point, dtype=float)


def _isoflux_pair(profile, flux, points, *, scale, tolerance):
    """Return an isoflux pair whose direction the response matrix decided."""
    rows = int(np.shape(points)[0])
    payload = jnp.concatenate(
        (jnp.asarray(points), jnp.asarray(_reference_point(profile, flux))[None])
    )
    pair = ConstraintPair(
        functional=IsofluxConstraint(point_count=rows, reference="reference_point"),
        unknown=CircuitCurrentUnknown(
            direction=jnp.zeros((CONDUCTORS, rows)).at[0].set(1.0),
            ampere_scale=jnp.full((rows,), 1.0e3),
        ),
        binding=ConstraintBinding(
            target=jnp.zeros(rows),
            tolerance=jnp.full((rows,), tolerance),
            scale=jnp.full((rows,), scale),
            initial_unknown=jnp.zeros(rows),
            payload=payload,
        ),
    )
    (derived,), selection = derive_circuit_compensators(
        profile, (pair,), jnp.asarray(flux), circuits=range(CONDUCTORS)
    )
    return derived, selection


def test_miller_boundary_reproduces_the_shape_figures_it_was_handed():
    """The parametrisation returns the extent, elongation and triangularity set."""
    configure_dtypes()
    points = np.asarray(
        miller_boundary_points(
            geometric_radius=1.0,
            geometric_height=0.05,
            minor_radius=0.3,
            elongation=1.8,
            triangularity=0.4,
            count=720,
        )
    )
    radius, height = points[:, 0], points[:, 1]
    assert float(np.max(radius) + np.min(radius)) / 2.0 == pytest.approx(1.0, abs=1e-9)
    assert float(np.max(radius) - np.min(radius)) / 2.0 == pytest.approx(0.3, abs=1e-9)
    assert float(np.max(height) + np.min(height)) / 2.0 == pytest.approx(0.05, abs=1e-9)
    assert float(np.max(height) - np.min(height)) / 2.0 == pytest.approx(
        1.8 * 0.3, abs=1e-9
    )
    upper = float(radius[int(np.argmax(height))])
    assert (1.0 - upper) / 0.3 == pytest.approx(0.4, abs=1.0e-9)


def test_shape_rows_respond_to_a_circuit_as_a_central_difference_does(machine):
    """Every row's circuit response equals the difference along its flux image."""
    profile, seed = machine
    flux = jnp.asarray(seed)
    lattice = profile.lattice
    axis = np.asarray([AXIS_RADIUS, 0.0])
    points = np.asarray(
        miller_boundary_points(
            geometric_radius=AXIS_RADIUS,
            geometric_height=0.0,
            minor_radius=0.2,
            elongation=1.3,
            triangularity=0.0,
            count=CONTROL_POINTS,
        )
    )
    wall = np.asarray(profile.operator.wall.null.coordinate, dtype=float)
    origin = wall[int(np.argmax(wall[:, 0]))]
    reference = _reference_point(profile, flux)
    span = float(jnp.max(jnp.abs(flux)))
    pairs = (
        ConstraintPair(
            functional=IsofluxConstraint(
                point_count=CONTROL_POINTS, reference="reference_point"
            ),
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((CONDUCTORS, CONTROL_POINTS)).at[0].set(1.0),
                ampere_scale=jnp.full((CONTROL_POINTS,), 1.0e3),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(CONTROL_POINTS),
                tolerance=jnp.full((CONTROL_POINTS,), 1.0e-4),
                scale=jnp.full((CONTROL_POINTS,), span),
                initial_unknown=jnp.zeros(CONTROL_POINTS),
                payload=jnp.concatenate(
                    (jnp.asarray(points), jnp.asarray(reference)[None])
                ),
            ),
        ),
        ConstraintPair(
            functional=XPointConstraint(),
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((CONDUCTORS, 2)).at[1].set(1.0),
                ampere_scale=jnp.full((2,), 1.0e3),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(2),
                tolerance=jnp.full((2,), 1.0e-4),
                scale=jnp.full((2,), span / float(lattice.radial_step)),
                initial_unknown=jnp.zeros(2),
                payload=jnp.asarray(axis),
            ),
        ),
        ConstraintPair(
            functional=WallGapConstraint(reference="reference_point"),
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((CONDUCTORS, 1)).at[2].set(1.0),
                ampere_scale=jnp.full((1,), 1.0e3),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(1),
                tolerance=jnp.full((1,), 1.0e-4),
                scale=jnp.full((1,), span),
                initial_unknown=jnp.zeros(1),
                payload=WallGapTarget(
                    origin=jnp.asarray(origin),
                    direction=jnp.asarray([-1.0, 0.0]),
                    gap=jnp.asarray(0.05),
                    reference_point=jnp.asarray(reference),
                ),
            ),
        ),
    )
    matrix = np.asarray(constraint_response_matrix(profile, pairs, flux))
    response = np.asarray(profile.operator.prescribed_current_field.response)
    assert matrix.shape == (CONTROL_POINTS + 3, CONDUCTORS)

    context = ConstraintContext(flux, None, None, None)

    def observe(state):
        return np.concatenate(
            [
                np.ravel(
                    np.asarray(
                        pair.functional.observed(
                            profile, context._replace(flux=state), pair.binding.payload
                        )
                    )
                )
                for pair in pairs
            ]
        )

    step = 1.0e-3 * span / float(np.max(np.abs(response)))
    for circuit in (0, 5, 11):
        image = jnp.asarray(response[:, circuit])
        difference = (observe(flux + step * image) - observe(flux - step * image)) / (
            2.0 * step
        )
        np.testing.assert_allclose(
            matrix[:, circuit], difference, rtol=2.0e-5, atol=1.0e-9
        )


def test_isoflux_rows_steer_the_boundary_onto_a_moved_miller_target(machine, converged):
    """Moving the Miller target moves the achieved boundary onto it."""
    profile, _seed = machine
    figures = _shape(profile, converged.flux)
    moved = dict(figures, geometric_height=figures["geometric_height"] + VERTICAL_STEP)
    points = miller_boundary_points(count=CONTROL_POINTS, **moved)
    span = float(jnp.max(jnp.abs(converged.flux)))
    pair, selection = _isoflux_pair(
        profile,
        converged.flux,
        points,
        scale=span,
        tolerance=1.0e-6 * span,
    )
    assert compensator_rule_name(np.asarray([int(selection.rule)])) in (
        "dominant_authority",
        "singular_distribution",
    )
    result = profile.solve(
        converged.flux,
        route="newton_krylov",
        constraint_pairs=(pair,),
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    )
    record = result.constraints[0]
    assert bool(np.all(np.asarray(record.qualified)))
    assert np.all(
        np.abs(np.asarray(record.physical_residual)) <= np.asarray(record.tolerance)
    )
    achieved = _shape(profile, result.flux)
    assert achieved["geometric_height"] > figures["geometric_height"]
    assert np.all(np.abs(np.asarray(record.physical_unknown)) > 0.0)


def test_stationary_point_rows_command_the_flux_map_null(machine, converged):
    """Both gradient rows vanish at the commanded point when the solve closes.

    The Solov'ev machine's only stationary point is its magnetic axis, so the
    commanded null here is that one; a diverted machine's saddle is the same
    two rows read at a different point.
    """
    profile, _seed = machine
    _masks, topology = profile.operator.read(converged.flux)
    axis = np.asarray(topology.axis, dtype=float)
    commanded = axis + np.asarray([0.0, VERTICAL_STEP])
    span = float(jnp.max(jnp.abs(converged.flux)))
    scale = span / float(profile.lattice.radial_step)
    pair = ConstraintPair(
        functional=XPointConstraint(),
        unknown=CircuitCurrentUnknown(
            direction=jnp.zeros((CONDUCTORS, 2)).at[0].set(1.0),
            ampere_scale=jnp.full((2,), 1.0e3),
        ),
        binding=ConstraintBinding(
            target=jnp.zeros(2),
            tolerance=jnp.full((2,), 1.0e-6 * scale),
            scale=jnp.full((2,), scale),
            initial_unknown=jnp.zeros(2),
            payload=jnp.asarray(commanded),
        ),
    )
    (derived,), _selection = derive_circuit_compensators(
        profile, (pair,), converged.flux, circuits=range(CONDUCTORS)
    )
    result = profile.solve(
        converged.flux,
        route="newton_krylov",
        constraint_pairs=(derived,),
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    )
    record = result.constraints[0]
    assert bool(np.all(np.asarray(record.qualified)))
    _masks, moved = profile.operator.read(result.flux)
    assert float(np.asarray(moved.axis)[1]) > float(axis[1])


def test_wall_gap_row_stands_the_boundary_off_by_the_commanded_distance(
    machine, converged
):
    """The commanded clearance is achieved at the wall point it was set on."""
    profile, _seed = machine
    wall = np.asarray(profile.operator.wall.null.coordinate, dtype=float)
    origin = wall[int(np.argmax(wall[:, 0]))]
    inward = np.asarray([-1.0, 0.0])
    grid = np.asarray(converged.flux)[: profile.lattice.node_count]
    grid = grid.reshape(profile.lattice.shape)
    _masks, topology = profile.operator.read(converged.flux)
    level = float(np.asarray(topology.boundary_flux))
    polarity = float(profile.operator.polarity)
    low, high = 0.0, 0.2
    for _step in range(60):
        middle = 0.5 * (low + high)
        value = float(np.asarray(_sample(profile, grid, origin + middle * inward)))
        if polarity * (value - level) > 0.0:
            high = middle
        else:
            low = middle
    achieved_gap = 0.5 * (low + high)
    commanded = achieved_gap + VERTICAL_STEP
    span = float(jnp.max(jnp.abs(converged.flux)))
    pair = ConstraintPair(
        functional=WallGapConstraint(reference="reference_point"),
        unknown=CircuitCurrentUnknown(
            direction=jnp.zeros((CONDUCTORS, 1)).at[0].set(1.0),
            ampere_scale=jnp.full((1,), 1.0e3),
        ),
        binding=ConstraintBinding(
            target=jnp.zeros(1),
            tolerance=jnp.full((1,), 1.0e-6 * span),
            scale=jnp.full((1,), span),
            initial_unknown=jnp.zeros(1),
            payload=WallGapTarget(
                origin=jnp.asarray(origin),
                direction=jnp.asarray(inward),
                gap=jnp.asarray(commanded),
                reference_point=jnp.asarray(_reference_point(profile, converged.flux)),
            ),
        ),
    )
    (derived,), _selection = derive_circuit_compensators(
        profile, (pair,), converged.flux, circuits=range(CONDUCTORS)
    )
    result = profile.solve(
        converged.flux,
        route="newton_krylov",
        constraint_pairs=(derived,),
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    )
    record = result.constraints[0]
    assert bool(np.all(np.asarray(record.qualified)))
    assert abs(float(np.asarray(record.physical_residual)[0])) <= float(
        np.asarray(record.tolerance)[0]
    )


def test_a_warm_started_move_costs_fewer_trips_than_the_same_move_from_the_seed(
    machine, converged
):
    """The previous equilibrium is a cheaper start than the analytic seed."""
    profile, seed = machine
    figures = _shape(profile, converged.flux)
    moved = dict(figures, elongation=figures["elongation"] * (1.0 + ELONGATION_STEP))
    points = miller_boundary_points(count=CONTROL_POINTS, **moved)
    span = float(jnp.max(jnp.abs(converged.flux)))
    pair, _selection = _isoflux_pair(
        profile,
        converged.flux,
        points,
        scale=span,
        tolerance=1.0e-6 * span,
    )

    def trips(start):
        """Return the frozen-mask trips one constrained solve took."""
        result = profile.solve(
            start,
            route="newton_krylov",
            constraint_pairs=(pair,),
            newton_steps=NEWTON_STEPS,
            gmres_iterations=GMRES_ITERATIONS,
        )
        return result, int(np.asarray(result.fixed_point.active_set_iterations))

    warm_result, warm = trips(converged.flux)
    cold_result, cold = trips(seed)
    assert bool(np.all(np.asarray(warm_result.constraints[0].qualified)))
    assert warm < cold
    assert bool(np.all(np.asarray(cold_result.constraints[0].qualified)))
