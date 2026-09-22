"""Parity checks for topology labels read on the cell carrier."""

import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.connectivity_boundary import (
    _PRE_SADDLE_OFFSET_FRACTION,
    _raster_hex_partition_geometry,
)
from nova.equilibrium.domain import classify_domains
from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    hex_edge_admissibility,
    label_saddle_aware_hex_connected_components,
    polish_census_stationary_points,
)
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.topology import (
    _canonicalize_reciprocal_hex_edges,
    _points_inside_polygon,
)
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


def _zero_profile(psi_norm):
    """Return a source-free profile so only the supplied analytic flux is read."""
    return jnp.zeros_like(psi_norm)


def _raster_operator(shape: tuple[int, int]) -> ForwardFluxOperator:
    """Build a negative-polarity reader for analytic fields with minimum axes."""
    radius = np.linspace(1.05, 2.35, shape[0])
    height = np.linspace(-0.72, 0.72, shape[1])
    lattice = FluxLattice(radius, height)
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    wall_coordinate = np.c_[
        1.7 + 0.64 * np.cos(wall_angle),
        0.68 * np.sin(wall_angle),
    ]
    node_count = lattice.node_count
    return ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.zeros((node_count, 1)),
            plasma_target=jnp.zeros((node_count, 1)),
            null=Null2D.from_coordinates(
                lattice.coordinate, hex_stencil(lattice.shape), maxsize=5
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.zeros((len(wall_coordinate), 1)),
            plasma_target=jnp.zeros((len(wall_coordinate), 1)),
            null=Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64)),
        ),
        source=ForwardSource(
            core=DomainProfile(p_prime=_zero_profile, ff_prime=_zero_profile)
        ),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        use_linear_moments=False,
        polarity=-1,
    )


def _limited_flux(radius, height):
    """Return a convex minimum bounded only by the surrounding wall."""
    return (radius - 1.7) ** 2 + height**2


def _single_null_flux(radius, height):
    """Return two convex minima separated by one four-crossing saddle."""
    local = radius - 1.7
    offset = 0.31
    return (local**2 + (height - offset) ** 2) * (local**2 + (height + offset) ** 2)


def _independent_spline_wall_anchor(surface, wall_coordinate, polarity):
    """Fit the extremal wall-node bracket without the production selector.

    The spline is sampled at every wall node, the signed extremal node chooses
    its cyclic neighbours, and a quadratic in local arc length supplies the
    stationary coordinate.  The returned coordinate bound propagates one
    binary64 rounding unit in the three sampled flux values through that fit,
    then adds the binary64 interpolation floor at the measured node spacing.
    """
    coordinate = np.asarray(wall_coordinate, dtype=np.float64)
    wall_flux = np.asarray(
        surface(
            jnp.asarray(coordinate[:, 0]),
            jnp.asarray(coordinate[:, 1]),
        ),
        dtype=np.float64,
    )
    winner = int(np.argmax(float(polarity) * wall_flux))
    bracket = np.mod(winner + np.asarray([-1, 0, 1]), wall_flux.size)
    points = coordinate[bracket]
    samples = wall_flux[bracket]
    node_spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_length = np.r_[0.0, np.cumsum(node_spacing)]
    system = np.column_stack((arc_length**2, arc_length, np.ones_like(arc_length)))
    coefficients = np.linalg.solve(system, samples)
    stationary_length = -coefficients[1] / (2.0 * coefficients[0])
    radius = np.interp(stationary_length, arc_length, points[:, 0])
    height = np.interp(stationary_length, arc_length, points[:, 1])
    selected_flux = float(np.asarray(surface(radius, height)))
    kind = -np.sign(coefficients[0])

    binary64_epsilon = np.finfo(np.float64).eps
    sampled_flux_roundoff = binary64_epsilon * np.max(np.abs(samples))
    inverse_system = np.linalg.inv(system)
    coefficient_roundoff = np.abs(inverse_system) @ np.full(3, sampled_flux_roundoff)
    curvature_roundoff, slope_roundoff = coefficient_roundoff[:2]
    curvature_margin = abs(coefficients[0]) - curvature_roundoff
    assert curvature_margin > 0.0
    stationary_length_roundoff = (
        slope_roundoff * abs(coefficients[0])
        + abs(coefficients[1]) * curvature_roundoff
    ) / (2.0 * abs(coefficients[0]) * curvature_margin)
    interpolation_roundoff = binary64_epsilon * (
        np.max(np.abs(points)) + np.max(node_spacing)
    )
    coordinate_roundoff_bound = np.nextafter(
        stationary_length_roundoff + interpolation_roundoff,
        np.inf,
    )
    return (
        jnp.asarray([radius, height, selected_flux, kind], dtype=jnp.float64),
        coordinate_roundoff_bound,
    )


def _containment_stationary_read(operator, physical):
    """Select boundary state from zero-/four-crossing containment candidates.

    The wall read independently reproduces the mechanism introduced by
    ``9f0692fb``: tensor-spline samples at the wall nodes precede an arc-length
    quadratic through the extremal node and its two cyclic neighbours.
    """
    topology = operator._fixed_design_topology
    radial = np.unique(np.asarray(operator.grid.coordinate)[:, 0])
    vertical = np.unique(np.asarray(operator.grid.coordinate)[:, 1])
    shape = (radial.size, vertical.size)
    grid_flux, wall_flux = topology.split_flux_map(physical)
    nodal_flux = jnp.asarray(grid_flux).reshape(shape).T
    surface = fit_tensor_spline(jnp.asarray(radial), jnp.asarray(vertical), nodal_flux)
    census = topology.grid.candidate_table_status(grid_flux)
    retained = census["retained_candidate"]
    retained_valid = census["retained_valid"]
    polarity = jnp.asarray(operator.polarity, dtype=retained.dtype)

    extremum_valid = retained_valid[0] & (retained[0, :, 3] == polarity)
    axis_index = jnp.argmax(
        jnp.where(extremum_valid, polarity * retained[0, :, 2], -jnp.inf)
    )
    axis = jnp.where(
        jnp.any(extremum_valid), retained[0, axis_index], jnp.full(4, jnp.nan)
    )

    saddle_valid = retained_valid[1] & _points_inside_polygon(
        retained[1, :, 0],
        retained[1, :, 1],
        operator.wall.coordinate[:, 0],
        operator.wall.coordinate[:, 1],
    )
    saddle_index = jnp.argmax(
        jnp.where(
            saddle_valid,
            polarity * (retained[1, :, 2] - axis[2]),
            -jnp.inf,
        )
    )
    saddle = jnp.where(
        jnp.any(saddle_valid),
        retained[1, saddle_index],
        jnp.full(4, jnp.nan),
    )
    wall, wall_roundoff_bound = _independent_spline_wall_anchor(
        surface,
        operator.wall.coordinate,
        operator.polarity,
    )

    saddle_heights = jnp.where(saddle_valid, retained[1, :, 1], jnp.nan)
    lower_saddle = jnp.nanmin(saddle_heights)
    upper_saddle = jnp.nanmax(saddle_heights)
    lower_saddle = jnp.where(lower_saddle > axis[1], -jnp.inf, lower_saddle)
    upper_saddle = jnp.where(upper_saddle < axis[1], jnp.inf, upper_saddle)
    wall_shadowed = (wall[1] < lower_saddle) | (wall[1] > upper_saddle)
    wall_binds = jnp.where(polarity < 0.0, wall[2] < saddle[2], wall[2] > saddle[2])
    boundary_is_xpoint = jnp.any(saddle_valid) & (wall_shadowed | ~wall_binds)
    boundary = jnp.where(boundary_is_xpoint, saddle, wall)

    axis, saddle, receipt = polish_census_stationary_points(
        nodal_flux,
        jnp.asarray(radial),
        jnp.asarray(vertical),
        boundary[2],
        polarity,
        axis,
        saddle,
        surface=surface,
    )
    boundary = jnp.where(boundary_is_xpoint, saddle, wall)
    selected_x_value = receipt["selected_value"][1]
    local_x_value = receipt["local_value_evidence"][1]
    spline_authored = receipt["spline_authored"][1]
    uncertainty_resolved = (
        boundary_is_xpoint
        & spline_authored
        & jnp.isfinite(selected_x_value)
        & jnp.isfinite(local_x_value)
    )
    uncertainty = jnp.where(
        uncertainty_resolved,
        jnp.abs(selected_x_value - local_x_value),
        jnp.asarray(0.0, dtype=selected_x_value.dtype),
    )
    return {
        "axis": axis,
        "x_point": saddle,
        "boundary": boundary,
        "boundary_is_xpoint": boundary_is_xpoint,
        "boundary_interpolation_uncertainty": uncertainty,
        "candidate_count": census["retained_count"],
        "ring_crossing_count": census["ring_crossing_count"],
        "representative_mask": census["representative_mask"],
        "surface": surface,
        "wall_roundoff_bound": wall_roundoff_bound,
    }


def _raster_label_oracle(operator, physical):
    """Build labels from contained nulls and spline/deadband edge decisions."""
    radial = np.unique(np.asarray(operator.grid.coordinate)[:, 0])
    vertical = np.unique(np.asarray(operator.grid.coordinate)[:, 1])
    shape = (radial.size, vertical.size)
    stationary = _containment_stationary_read(operator, physical)
    surface = stationary["surface"]
    coordinate = (
        jnp.asarray(operator.grid.coordinate)
        .reshape(shape + (2,))
        .transpose((1, 0, 2))
        .reshape((-1, 2))
    )
    flux = surface(coordinate[:, 0], coordinate[:, 1]).reshape(
        (vertical.size, radial.size)
    )
    inside = jnp.asarray(operator.inside_material).reshape(shape).T
    rings, shared_edges = _raster_hex_partition_geometry(
        jnp.asarray(radial), jnp.asarray(vertical)
    )

    polarity = float(operator.polarity)
    axis = stationary["axis"]
    x_point = stationary["x_point"]
    boundary = stationary["boundary"]
    boundary_is_xpoint = stationary["boundary_is_xpoint"]
    uncertainty = stationary["boundary_interpolation_uncertainty"]
    comparison_boundary = boundary[2] + jnp.where(
        axis[2] >= boundary[2], uncertainty, -uncertainty
    )
    closed = (
        flux >= comparison_boundary if polarity > 0.0 else flux < comparison_boundary
    )
    inside_flux = jnp.where(closed & inside, flux, jnp.nan)
    inward = _PRE_SADDLE_OFFSET_FRACTION * (
        jnp.nanmax(inside_flux) - jnp.nanmin(inside_flux)
    )
    direction = jnp.where(axis[2] >= boundary[2], 1.0, -1.0)
    component_flux = comparison_boundary + direction * inward
    component_flux = jnp.where(boundary_is_xpoint, component_flux, comparison_boundary)
    confined = closed & inside
    exact_links = hex_edge_admissibility(
        flux,
        jnp.asarray(radial),
        jnp.asarray(vertical),
        comparison_boundary,
        axis[2],
        shared_edges,
        surface=surface,
    )
    inward_links = hex_edge_admissibility(
        flux,
        jnp.asarray(radial),
        jnp.asarray(vertical),
        component_flux,
        axis[2],
        shared_edges,
        surface=surface,
    )
    centre = coordinate[rings[:, :1]]
    neighbour = coordinate[rings]
    edge_pitch = jnp.linalg.norm(neighbour - centre, axis=-1)
    edge_midpoint = jnp.mean(shared_edges, axis=-2)
    saddle_distance = jnp.linalg.norm(edge_midpoint - x_point[:2], axis=-1)
    saddle_neighbourhood = boundary_is_xpoint & (saddle_distance <= 3.0 * edge_pitch)
    links = _canonicalize_reciprocal_hex_edges(
        rings, exact_links & (inward_links | ~saddle_neighbourhood)
    )
    labels = label_saddle_aware_hex_connected_components(
        confined, rings, links, confined.size
    )
    radius_grid, height_grid = jnp.meshgrid(jnp.asarray(radial), jnp.asarray(vertical))
    distance2 = (radius_grid - axis[0]) ** 2 + (height_grid - axis[1]) ** 2
    seed_index = jnp.argmin(jnp.where(confined, distance2, jnp.inf))
    seed_label = labels.reshape(-1)[seed_index]
    connected = (labels == seed_label) & (seed_label > 0)
    psi_norm = (flux - axis[2]) / (boundary[2] - axis[2])
    masks = classify_domains(psi_norm, closed, connected, inside)
    return masks.label.T.reshape(-1), stationary


def test_raster_fixture_labels_are_bitwise_equal_to_independent_oracle():
    """Contained nulls and spline-resolved edges reproduce every carrier label."""
    configure_dtypes()
    compared = 0
    differing_count = 0
    fixtures = ((_limited_flux, (1, 0)), (_single_null_flux, (2, 1)))
    for flux, expected_candidate_count in fixtures:
        operator = _raster_operator((17, 19))
        grid_coordinate = np.asarray(operator.grid.coordinate)
        wall_coordinate = np.asarray(operator.wall.coordinate)
        physical = jnp.asarray(
            np.r_[
                flux(grid_coordinate[:, 0], grid_coordinate[:, 1]),
                flux(wall_coordinate[:, 0], wall_coordinate[:, 1]),
            ]
        )

        actual, state, _connected, admitted = operator._fixed_design_read(physical)
        expected, stationary = _raster_label_oracle(operator, physical)

        assert bool(admitted)
        assert (
            tuple(np.asarray(stationary["candidate_count"])) == expected_candidate_count
        )
        representative_mask = np.asarray(stationary["representative_mask"])
        crossing_count = np.asarray(stationary["ring_crossing_count"])
        census = operator._fixed_design_topology.grid.candidate_table_status(
            operator.null_flux_pool(physical)
        )
        quadratic_admitted = np.asarray(census["quadratic_admitted_mask"])
        assert quadratic_admitted.shape == representative_mask.shape
        assert np.all(~representative_mask | quadratic_admitted)
        np.testing.assert_array_equal(crossing_count, census["ring_crossing_count"])
        np.testing.assert_array_equal(state.axis, stationary["axis"][:2])
        np.testing.assert_array_equal(state.x_point, stationary["x_point"][:2])
        boundary_roundoff_bound = stationary["wall_roundoff_bound"]
        np.testing.assert_allclose(
            state.boundary,
            stationary["boundary"][:2],
            rtol=0.0,
            atol=boundary_roundoff_bound,
            err_msg=(
                "9f0692fb moved the limited-wall authority to tensor-spline "
                "samples before the independently reconstructed arc-length "
                "quadratic selector; binary64 wall-flux and node-spacing "
                f"roundoff bound={boundary_roundoff_bound:.17e} m"
            ),
        )
        differing = np.flatnonzero(np.asarray(actual.label) != np.asarray(expected))
        compared += actual.label.size
        differing_count += differing.size
        assert differing.size == 0
        np.testing.assert_array_equal(actual.label, expected)
    print(f"raster_oracle_compared={compared} differing={differing_count}")
    assert compared == 646
    assert differing_count == 0


def test_carrier_shadow_operands_do_not_request_tensor_axes():
    """Wall shadowing reads carrier labels and nulls without a raster reshape."""
    configure_dtypes()
    operator = _raster_operator((13, 15))
    operator._fixed_design_topology.connectivity_radius = jnp.empty(0)
    operator._fixed_design_topology.connectivity_height = jnp.empty(0)
    coordinate = np.asarray(operator.grid.coordinate)
    wall = np.asarray(operator.wall.coordinate)
    state = jnp.asarray(
        np.r_[
            _limited_flux(coordinate[:, 0], coordinate[:, 1]),
            _limited_flux(wall[:, 0], wall[:, 1]),
        ]
    )

    flood_shadow, wall_shadow = operator.residual_shadow_components(state)
    qualification = operator._fixed_design_topology.read_qualification(
        state, operator.polarity, operator.inside_material
    )

    assert flood_shadow.shape == (operator.grid.node_number,)
    assert wall_shadow.shape == (operator.wall.node_number,)
    assert flood_shadow.dtype == jnp.bool_
    assert wall_shadow.dtype == jnp.bool_
    assert qualification.polish_receipt["fit_attempted"].tolist() == [True, True]
    assert qualification.polish_receipt["fit_iterations"].tolist() == [1, 1]
    assert np.all(np.isfinite(np.asarray(qualification.polish_receipt["fit_residual"])))
