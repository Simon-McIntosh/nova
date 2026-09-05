"""Bounding-box shape-control rows on the bootstrapped Solov'ev machine.

The machine is the same one the forward-solve contract uses: a ring of
external conductors fitted to hold an analytic seed, driven by an
edge-vanishing absolute source so the wall-limited branch attracts.  The
conductors are given a prescribed-current field with a zero baseline, so a
compensating circuit current is a pure actuator handle on a real Green
response rather than an algebraic identity.

The row set is the low-DOF pulse-design bounding box: boundary-flux rows at
the outer, upper, inner and lower control points, a radial-field row at the
outer and inner points (the vertical flux gradient vanishing, a radial
turning point), a vertical-field row at the upper and lower points, and the
two-gradient null row at the X-point.  The points come from
:class:`~nova.equilibrium.constraint.BoundingBoxTarget`, either from the
pulse-design parameter vocabulary through ControlPoints or from an achieved
boundary's own turning points, so the unmoved command reproduces the
boundary exactly with no curve fit.

Pinned here: the target builders return the commanded turning points; every
row's constraint-response matrix equals a central difference along the same
circuit's flux image; a constrained solve with flux and field rows at the
seed's own bounding box closes every row to its declared tolerance; the
stationary-point and wall-gap rows command their points; and a moved target
re-solved from the previous equilibrium costs fewer trips than the same
target solved from the seed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0
import xarray

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.equilibrium import (
        BoundingBoxTarget,
        CircuitCurrentUnknown,
        ConstraintBinding,
        ConstraintPair,
        FieldComponentConstraint,
        IsofluxConstraint,
        WallGapConstraint,
        WallGapTarget,
        XPointConstraint,
        compensator_rule_name,
        constraint_response_matrix,
        derive_circuit_compensators,
    )
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.constraint import ConstraintContext
    from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import PrescribedCurrentField
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.geometry.plasmapoints import ControlPoints
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
#: Commanded moves. Both are a few percent of the minor radius, the scale a
#: shape controller actually steers in, and small enough that the previous
#: equilibrium is a good warm start for the next.
VERTICAL_STEP = 0.02
ELONGATION_STEP = 0.05
#: The relative row tolerance every constrained-solve contract declares.  On
#: this machine the circuit set cannot independently carry the bounding-box
#: rows: the drivable response spectrum spans 2.2e-6 down to 1e-10 row units
#: per ampere, and the singular-distribution directions that hold competing
#: rows settle with the worst row at five percent of its scale under a few
#: kilo-ampere compensator.  The solved rows close to that floor, not to the
#: residual accuracy the map itself reaches, so a contract that demands the
#: floor must state it; the benchmark receipt on the bank row sees the same
#: limit and the compensation followup owns the improvement.
SOLVE_TOLERANCE_FRACTION = 0.1


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


def _seed_target(profile, flux):
    """Return the bounding-box target at one flux map's own turning points."""
    return BoundingBoxTarget.from_boundary(
        _boundary_polygon(profile, flux),
        reference_point=_reference_point(profile, flux),
    )


def _bounding_box_pairs(
    profile,
    flux,
    target,
    *,
    span,
    ampere_scale=1.0e3,
    flux_tolerance=None,
    field_tolerance=None,
):
    """Return the isoflux and field pairs one bounding-box target produces.

    The isoflux rows carry the four control points against the declared
    reference point; the field rows carry a radial-field component on the
    outer and inner points and a vertical-field component on the upper and
    lower points, scaled so a flux-gradient row of one grid cell reads like
    the flux rows do.  Both pairs start from a unit direction on circuit
    zero; :func:`derive_circuit_compensators` replaces it from the response
    matrix before any solve.
    """
    reference = target.reference_point
    if reference is None:
        raise ValueError("the bounding-box rows need a reference point")
    flux_points = np.asarray(target.flux_points, dtype=float)
    radial = np.asarray(target.radial_field_points, dtype=float)
    vertical = np.asarray(target.vertical_field_points, dtype=float)
    flux_rows = flux_points.shape[0]
    field_rows = radial.shape[0] + vertical.shape[0]
    if flux_tolerance is None:
        flux_tolerance = SOLVE_TOLERANCE_FRACTION * span
    if field_tolerance is None:
        field_tolerance = (
            SOLVE_TOLERANCE_FRACTION
            * span
            / (TOTAL_FLUX_FACTOR * AXIS_RADIUS * float(profile.lattice.radial_step))
        )
    flux_payload = jnp.concatenate(
        (jnp.asarray(flux_points), jnp.asarray(reference)[None])
    )
    field_payload = jnp.concatenate((jnp.asarray(radial), jnp.asarray(vertical)))
    field_scale = span / (
        TOTAL_FLUX_FACTOR * AXIS_RADIUS * float(profile.lattice.radial_step)
    )
    return (
        ConstraintPair(
            functional=IsofluxConstraint(
                point_count=flux_rows, reference="reference_point"
            ),
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((CONDUCTORS, flux_rows)).at[0].set(1.0),
                ampere_scale=jnp.full((flux_rows,), ampere_scale),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(flux_rows),
                tolerance=jnp.full((flux_rows,), flux_tolerance),
                scale=jnp.full((flux_rows,), span),
                initial_unknown=jnp.zeros(flux_rows),
                payload=flux_payload,
            ),
        ),
        ConstraintPair(
            functional=FieldComponentConstraint(
                components=("radial",) * radial.shape[0]
                + ("vertical",) * vertical.shape[0]
            ),
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((CONDUCTORS, field_rows)).at[0].set(1.0),
                ampere_scale=jnp.full((field_rows,), ampere_scale),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(field_rows),
                tolerance=jnp.full((field_rows,), field_tolerance),
                scale=jnp.full((field_rows,), field_scale),
                initial_unknown=jnp.zeros(field_rows),
                payload=field_payload,
            ),
        ),
    )


def _move_upper(target, delta):
    """Return the same target with the upper turning point moved by ``delta``."""
    flux = np.asarray(target.flux_points, dtype=float).copy()
    vertical = np.asarray(target.vertical_field_points, dtype=float).copy()
    flux[1] = flux[1] + np.asarray([0.0, delta])
    vertical[0] = vertical[0] + np.asarray([0.0, delta])
    return BoundingBoxTarget(
        flux_points=jnp.asarray(flux),
        radial_field_points=target.radial_field_points,
        vertical_field_points=jnp.asarray(vertical),
        x_point=target.x_point,
        reference_point=target.reference_point,
    )


def test_control_points_build_the_bounding_box_from_the_shape_figures():
    """The parameter vocabulary places the four turning points it is handed."""
    configure_dtypes()
    target = BoundingBoxTarget.from_control_points(
        geometric_axis=(1.0, 0.05),
        minor_radius=0.3,
        elongation=1.8,
        triangularity_upper=0.4,
        triangularity_lower=0.2,
        triangularity_outer=0.1,
        triangularity_inner=0.05,
        square=True,
        squareness=0.0,
    )
    points = np.asarray(target.flux_points, dtype=float)
    assert points.shape == (8, 2)
    # The four principal rows are the ControlPoints outer, upper, inner and
    # lower points, so the builder hands the same geometry the inverse
    # design reads its sliders through.
    data = xarray.Dataset(
        {
            "geometric_axis": ("point", [1.0, 0.05]),
            "minor_radius": 0.3,
            "elongation": 1.8,
            "triangularity_upper": 0.4,
            "triangularity_lower": 0.2,
            "elongation_upper": 0.1,
            "elongation_lower": 0.05,
            "squareness_upper_outer": 0.0,
            "squareness_upper_inner": 0.0,
            "squareness_lower_inner": 0.0,
            "squareness_lower_outer": 0.0,
        }
    )
    expected = ControlPoints(data, square=True)
    for row, attr in (
        (0, "outer"),
        (1, "upper"),
        (2, "inner"),
        (3, "lower"),
        (4, "upper_outer"),
        (5, "upper_inner"),
        (6, "lower_inner"),
        (7, "lower_outer"),
    ):
        np.testing.assert_allclose(
            points[row], np.asarray(getattr(expected, attr), dtype=float)
        )
    assert np.asarray(target.radial_field_points).shape == (2, 2)
    assert np.asarray(target.vertical_field_points).shape == (2, 2)
    # The radial rows sit on the outer and inner points, the vertical rows on
    # the upper and lower points.
    np.testing.assert_allclose(np.asarray(target.radial_field_points)[0], points[0])
    np.testing.assert_allclose(np.asarray(target.vertical_field_points)[0], points[1])


def test_boundary_turning_points_are_the_polygon_own_extrema():
    """An unmoved command reads the contour's own extremes, nothing else."""
    boundary = np.asarray(
        [[1.00, 0.00], [0.95, 0.12], [1.00, 0.20], [1.05, 0.12], [1.06, 0.02]]
    )
    target = BoundingBoxTarget.from_boundary(boundary)
    flux = np.asarray(target.flux_points, dtype=float)
    assert flux.shape == (4, 2)
    np.testing.assert_allclose(flux[0], boundary[4])  # outer, max R
    np.testing.assert_allclose(flux[1], boundary[2])  # upper, max Z
    np.testing.assert_allclose(flux[2], boundary[1])  # inner, min R
    np.testing.assert_allclose(flux[3], boundary[0])  # lower, min Z
    np.testing.assert_allclose(np.asarray(target.radial_field_points), flux[[0, 2]])
    np.testing.assert_allclose(np.asarray(target.vertical_field_points), flux[[1, 3]])
    assert len(target.rows) == 8


@pytest.mark.slow
def test_shape_rows_respond_to_a_circuit_as_a_central_difference_does(
    machine,
):
    """Every row's circuit response equals the difference along its flux image."""
    profile, seed = machine
    flux = jnp.asarray(seed)
    span = float(jnp.max(jnp.abs(flux)))
    target = _seed_target(profile, seed)
    pairs = _bounding_box_pairs(profile, flux, target, span=span)
    matrix = np.asarray(constraint_response_matrix(profile, pairs, flux))
    response = np.asarray(profile.operator.prescribed_current_field.response)
    assert matrix.shape == (8, CONDUCTORS)

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


@pytest.mark.slow
def test_bounding_box_rows_close_on_the_seed_own_turning_points(machine):
    """Flux and field rows at the achieved box close without any move.

    The unmoved command is the seed boundary's own bounding box, so no curve
    fit stands between the command and the boundary: the unconstrained
    equilibrium already satisfies every row, and the constrained solve closes
    them to their declared tolerances on each row's own physical scale.
    """
    profile, seed = machine
    span = float(jnp.max(jnp.abs(seed)))
    target = _seed_target(profile, seed)
    pairs = _bounding_box_pairs(profile, seed, target, span=span)
    (derived_a, derived_b), selection = derive_circuit_compensators(
        profile, pairs, jnp.asarray(seed), circuits=range(CONDUCTORS)
    )
    assert compensator_rule_name(np.asarray([int(selection.rule)])) in (
        "dominant_authority",
        "singular_distribution",
    )
    result = profile.solve(
        seed,
        route="newton_krylov",
        constraint_pairs=(derived_a, derived_b),
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    )
    for record in result.constraints:
        assert bool(np.all(np.asarray(record.qualified)))
        assert np.all(
            np.abs(np.asarray(record.physical_residual)) <= np.asarray(record.tolerance)
        )
        assert np.all(np.isfinite(np.asarray(record.physical_unknown)))


@pytest.mark.slow
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
            tolerance=jnp.full((2,), SOLVE_TOLERANCE_FRACTION * scale),
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


@pytest.mark.slow
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
            tolerance=jnp.full((1,), SOLVE_TOLERANCE_FRACTION * span),
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


@pytest.mark.slow
def test_a_warm_started_move_costs_fewer_trips_than_the_same_move_from_the_seed(
    machine, converged
):
    """The previous equilibrium is a cheaper start than the analytic seed."""
    profile, seed = machine
    span = float(jnp.max(jnp.abs(converged.flux)))
    target = _seed_target(profile, converged.flux)
    moved = _move_upper(target, VERTICAL_STEP)
    pairs = _bounding_box_pairs(profile, converged.flux, moved, span=span)
    (flux_derived, field_derived), _selection = derive_circuit_compensators(
        profile,
        pairs,
        jnp.asarray(converged.flux),
        circuits=range(CONDUCTORS),
    )

    def trips(start):
        """Return the frozen-mask trips one constrained solve took."""
        result = profile.solve(
            start,
            route="newton_krylov",
            constraint_pairs=(flux_derived, field_derived),
            newton_steps=NEWTON_STEPS,
            gmres_iterations=GMRES_ITERATIONS,
        )
        return result, int(np.asarray(result.fixed_point.active_set_iterations))

    warm_result, warm = trips(converged.flux)
    cold_result, cold = trips(seed)
    assert warm < cold
    for record in warm_result.constraints:
        assert bool(np.all(np.asarray(record.qualified)))
    for record in cold_result.constraints:
        assert bool(np.all(np.asarray(record.qualified)))
