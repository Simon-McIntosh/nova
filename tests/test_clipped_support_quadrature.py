from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.clip_quadrature import (
    _QuadratureSupport,
    _flux_selected_current_moments,
    _straight_edge_monomial_moments,
    clipped_support_current_moments,
    clipped_support_field_integrals,
    clipped_support_quadrature,
    cut_cell_bank_capacity,
)
from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments
from nova.equilibrium.source import _FluxSelectedProfile
from nova.equilibrium.stencil_mesh import (
    FluxFieldPolynomial,
    flux_field_polynomial,
)
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


pytestmark = pytest.mark.slow
MEASURED_CHAIN_RELATIVE_TOLERANCE = 1.0e-6
GIB = 2**30


@pytest.fixture(scope="module", autouse=True)
def _binary64_cpu() -> None:
    configure_dtypes()
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "cpu"


def _analytic_support(case_name: str, requested_cells: int):
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinate = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = certificate._exact_state(case_name, exact, coordinate)
    operator = fixture.forward_operator(source_case, machine)
    support = fixture._analytic_profile_support(exact, operator, state)
    physical = jnp.asarray(state[: operator.physical_node_number])
    grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
    axis_flux = jnp.asarray(fixture._analytic_axis_flux(exact), dtype=grid_flux.dtype)
    flux_span = -axis_flux
    centroid_flux = (grid_flux - axis_flux) / flux_span
    sample_flux = (
        operator.sample_node_flux(jnp.asarray(state)) - axis_flux
    ) / flux_span
    field = flux_field_polynomial(
        operator._support_moment_stencils, centroid_flux, sample_flux
    )
    ring_centres = np.concatenate(
        [stencil.ring_centre for stencil in operator._support_moment_stencils]
    )
    bank_capacity = cut_cell_bank_capacity(
        operator.moment_geometry.atomic_mesh.centroids, ring_centres
    )
    return (
        operator,
        support,
        field,
        flux_span,
        bank_capacity,
        centroid_flux,
        sample_flux,
    )


def _dense_integrals(operator, support, field, flux_span):
    selected = jnp.asarray(support.included, dtype=bool)
    points, weights = clipped_support_quadrature(support, selected)
    psi_norm, radial_gradient, vertical_gradient = field.sample(points)
    radius = points[..., 0]
    pressure = operator.source.core.pressure(
        radius, psi_norm, operator.source.boundary_pressure, flux_span
    )
    gradient_squared = flux_span**2 * (radial_gradient**2 + vertical_gradient**2)
    field_squared = gradient_squared / (2.0 * jnp.pi * radius) ** 2
    volume_weight = 2.0 * jnp.pi * radius * weights
    density_weight = operator.source.core.current_density(radius, psi_norm) * weights
    first = jnp.sum(
        density_weight[..., None] * (points - support.centroids[:, None, :]), axis=1
    )
    return (
        jnp.sum(density_weight, axis=1),
        first,
        jnp.sum(pressure * volume_weight, axis=1),
        jnp.sum(field_squared * volume_weight, axis=1),
    )


def _assert_cellwise_close(name, observed, expected, mask, *, rtol):
    """Compare reductions without mistaking scan-order roundoff for signal.

    A scan may change floating-point summation order. Entries at or below
    ``1e-12`` of the dense row's largest magnitude therefore use that quantity
    as an absolute floor; resolved entries retain the declared relative test.
    """
    cells = np.flatnonzero(mask)
    observed_values = np.asarray(observed)[cells]
    dense_values = np.asarray(expected)
    expected_values = dense_values[cells]
    difference = np.abs(observed_values - expected_values)
    row_magnitude = float(np.max(np.abs(dense_values)))
    absolute_floor = 1.0e-12 * row_magnitude
    resolved = np.abs(expected_values) > absolute_floor
    allowed = np.where(resolved, rtol * np.abs(expected_values), absolute_floor)
    violation = difference > allowed
    if np.any(violation):
        normalized = difference / np.maximum(allowed, np.finfo(np.float64).tiny)
        maximum = int(np.argmax(normalized))
        raise AssertionError(
            f"{name}: difference {difference[maximum]:.17g} at cell "
            f"{int(cells[maximum])}, allowed {allowed[maximum]:.17g}, "
            f"row magnitude {row_magnitude:.17g}"
        )


@pytest.mark.parametrize(
    "case_name", ("weak-rotation-reactor-static", "diverted-single-null")
)
def test_compact_reduction_matches_dense_quadrature(case_name: str):
    (
        operator,
        support,
        field,
        flux_span,
        bank_capacity,
        centroid_flux,
        sample_flux,
    ) = _analytic_support(case_name, -110)
    expected_current, expected_first, expected_pressure, expected_field = (
        _dense_integrals(operator, support, field, flux_span)
    )
    actual = jax.jit(
        lambda carried_support, carried_field: clipped_support_field_integrals(
            carried_support,
            carried_support.included,
            carried_field,
            operator.source.core.pressure,
            operator.source.boundary_pressure,
            flux_span,
            cut_cell_capacity=bank_capacity,
        )
    )(support, field)
    jax.block_until_ready(actual)
    actual_current = jax.jit(
        lambda carried_support, carried_field: clipped_support_current_moments(
            carried_support,
            carried_support.included,
            carried_field,
            operator.source.core,
            cut_cell_capacity=bank_capacity,
        )
    )(support, field)
    jax.block_until_ready(actual_current)
    production_current = jax.jit(
        lambda centroid, sample, carried_support: operator.support_current_moments(
            operator.source.core,
            centroid,
            sample,
            carried_support,
        )
    )(centroid_flux, sample_flux, support)
    jax.block_until_ready(production_current)

    whole = np.asarray(support.included) & ~np.asarray(support.boundary)
    cut = np.asarray(support.included) & np.asarray(support.boundary)
    assert np.all(np.isfinite(np.asarray(actual.pressure_volume)))
    _assert_cellwise_close(
        "pressure volume",
        actual.pressure_volume,
        expected_pressure,
        whole,
        rtol=1.0e-12,
    )
    _assert_cellwise_close(
        "field volume", actual.field_volume, expected_field, whole, rtol=1.0e-12
    )
    _assert_cellwise_close(
        "cut pressure volume",
        actual.pressure_volume,
        expected_pressure,
        cut,
        rtol=MEASURED_CHAIN_RELATIVE_TOLERANCE,
    )
    _assert_cellwise_close(
        "cut field volume",
        actual.field_volume,
        expected_field,
        cut,
        rtol=MEASURED_CHAIN_RELATIVE_TOLERANCE,
    )
    active = np.asarray(field.active)
    for name, observed, expected, carried in (
        (
            "compact current",
            actual_current.cell_current,
            expected_current,
            np.ones_like(active),
        ),
        (
            "compact radial moment",
            actual_current.radial_moment,
            expected_first[:, 0],
            np.ones_like(active),
        ),
        (
            "compact vertical moment",
            actual_current.vertical_moment,
            expected_first[:, 1],
            np.ones_like(active),
        ),
        (
            "production current",
            production_current.cell_current,
            expected_current,
            active,
        ),
        (
            "production radial moment",
            production_current.radial_moment,
            expected_first[:, 0],
            active,
        ),
        (
            "production vertical moment",
            production_current.vertical_moment,
            expected_first[:, 1],
            active,
        ),
    ):
        _assert_cellwise_close(name, observed, expected, whole & carried, rtol=1.0e-12)
        _assert_cellwise_close(
            f"cut {name}",
            observed,
            expected,
            cut & carried,
            rtol=MEASURED_CHAIN_RELATIVE_TOLERANCE,
        )
    for observed in production_current:
        np.testing.assert_array_equal(np.asarray(observed)[~active], 0.0)


def _tile_cell_field(value, cell_count: int):
    array = jnp.asarray(value)
    repeat = (cell_count + array.shape[0] - 1) // array.shape[0]
    return jnp.tile(array, (repeat,) + (1,) * (array.ndim - 1))[:cell_count]


def _large_support(support, cell_count: int):
    replacements = {}
    cell_fields = (
        "support_vertices",
        "vertex_count",
        "centroids",
        "included",
        "boundary",
        "area",
        "full_area",
        "first_area_moment",
        "second_area_moment",
        "branch_support_vertices",
        "branch_vertex_count",
        "branch_area",
        "branch_first_area_moment",
        "branch_second_area_moment",
        "saddle",
        "saddle_vertex",
    )
    for name in cell_fields:
        replacements[name] = _tile_cell_field(getattr(support, name), cell_count)
    return support._replace(**replacements)


def _large_field(field: FluxFieldPolynomial, cell_count: int) -> FluxFieldPolynomial:
    return FluxFieldPolynomial(
        *(_tile_cell_field(value, cell_count) for value in field)
    )


def _shape_bytes(shape, dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def test_moving_outer_polygon_contributes_to_clipped_current_tangent():
    class ConstantCurrent:
        def current_density(self, radius, psi_norm):
            return jnp.ones_like(radius)

    polygon = jnp.asarray([[[0.8, -0.2], [1.2, -0.2], [1.2, 0.2], [0.8, 0.2]]])
    field = FluxFieldPolynomial(
        coefficient=jnp.asarray([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        centre=jnp.asarray([[1.0, 0.0]]),
        scale=jnp.ones((1, 2)),
        active=jnp.asarray([True]),
    )
    profile = _FluxSelectedProfile(ConstantCurrent(), None)

    def current(displacement):
        moving = polygon + jnp.asarray([displacement, 0.0])
        support = _QuadratureSupport(
            moving, jnp.asarray([4]), jnp.asarray([[1.0, 0.0]])
        )
        return _flux_selected_current_moments(
            support, jnp.asarray([True]), field, profile, cut_cell_capacity=1
        ).cell_current[0]

    displacement = jnp.asarray(0.0)
    _, tangent = jax.jvp(current, (displacement,), (jnp.asarray(1.0),))
    step = 1.0e-5
    central = (current(step) - current(-step)) / (2.0 * step)
    assert abs(float(central)) > 0.1
    np.testing.assert_allclose(tangent, central, rtol=1.0e-5, atol=1.0e-7)


def test_compact_reduction_work_arrays_stay_below_one_gibibyte():
    operator, support, field, flux_span, bank_capacity, _centroid, _sample = (
        _analytic_support("weak-rotation-reactor-static", -110)
    )
    cell_count = 2500
    carried_support = _large_support(support, cell_count)
    carried_field = _large_field(field, cell_count)
    result_shape = jax.eval_shape(
        lambda current_support, current_field: clipped_support_field_integrals(
            current_support,
            current_support.included,
            current_field,
            operator.source.core.pressure,
            operator.source.boundary_pressure,
            flux_span,
            cut_cell_capacity=min(cell_count, bank_capacity * 5),
        ),
        carried_support,
        carried_field,
    )
    assert result_shape.pressure_volume.shape == (cell_count,)
    assert result_shape.field_volume.shape == (cell_count,)

    whole_support = carried_support._replace(
        support_vertices=carried_support.support_vertices[:, :24],
        vertex_count=jnp.minimum(carried_support.vertex_count, 24),
    )
    whole_shape = jax.eval_shape(
        clipped_support_quadrature,
        whole_support,
        whole_support.included,
    )[0]
    first_cut = int(np.flatnonzero(np.asarray(support.boundary))[0])
    cut_support = support._replace(
        **{
            name: jnp.asarray(getattr(support, name))[first_cut : first_cut + 1]
            for name in (
                "support_vertices",
                "vertex_count",
                "centroids",
                "included",
                "boundary",
                "area",
                "full_area",
                "first_area_moment",
                "second_area_moment",
                "branch_support_vertices",
                "branch_vertex_count",
                "branch_area",
                "branch_first_area_moment",
                "branch_second_area_moment",
                "saddle",
                "saddle_vertex",
            )
        }
    )
    cut_shape = jax.eval_shape(
        clipped_support_quadrature,
        cut_support,
        cut_support.included,
    )[0]
    design_columns = 6
    largest_intermediate_bytes = max(
        _shape_bytes((*whole_shape.shape[:2], design_columns), whole_shape.dtype),
        _shape_bytes((*cut_shape.shape[:2], design_columns), cut_shape.dtype),
    )
    assert largest_intermediate_bytes < GIB


# ---------------------------------------------------------------------------
# Exact moments of a quadratic density over a clipped polygon.
#
# The closed-form floor books a cut cell by contracting its fitted quadratic
# density against the owner's monomial table. The dense reference below is
# independent of that route: the polygon is fanned into signed triangles from
# its first vertex and each triangle carries a forty-node product rule, which
# resolves a degree-four integrand exactly. This is the property the parallel
# quadratic-density module's test held and this file did not, so it moves here.
# That module's other assertions are already held elsewhere: the owner table
# against exact rationals and the centred degree-two construction against those
# same moments in ``test_polygon_moment_owner.py``, and the refusal of a
# malformed vertex buffer in ``test_equilibrium_separatrix_clip.py``.
_QUADRATIC_POWERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
_OWNER_POWERS = tuple(
    (radial, total - radial) for total in range(5) for radial in range(total + 1)
)
_OWNER_SLOT = {power: index for index, power in enumerate(_OWNER_POWERS)}
#: A convex hexagon and a concave five-vertex arrow, both carrying vertices in
#: every quadrant, and a quadratic with every coefficient live.
_DENSITY_POLYGONS = (
    np.array(
        [[0.1, -0.2], [1.3, -0.35], [1.9, 0.6], [1.1, 1.4], [-0.2, 0.9], [-0.35, 0.15]]
    ),
    np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 0.6], [0.0, 2.0]]),
)
_QUADRATIC_DENSITY = np.array([1.5, -0.75, 0.4, 0.6, -0.35, 0.25])


def _dense_quadratic_moments(polygon: np.ndarray, coefficient: np.ndarray):
    """Integrate a quadratic density with a dense Duffy fan quadrature."""
    node, weight = np.polynomial.legendre.leggauss(40)
    node = 0.5 * (node + 1.0)
    weight = 0.5 * weight
    radial = np.repeat(node, 40)[:, None]
    vertical = np.tile(node, 40)[:, None]
    rule = (np.repeat(weight, 40) * np.tile(weight, 40))[:, None]
    total = np.zeros(6, dtype=np.float64)
    first = polygon[0]
    for index in range(1, len(polygon) - 1):
        second, third = polygon[index], polygon[index + 1]
        point = (
            first
            + radial * (second - first)
            + (1.0 - radial) * vertical * (third - first)
        )
        # Signed, not absolute: a concave polygon's fan has mixed orientation.
        third_from_first = third - first
        affine = float(
            (second[0] - first[0]) * third_from_first[1]
            - (second - first)[1] * third_from_first[0]
        )
        jacobian = (affine * (1.0 - radial) * rule).ravel()
        design = np.stack(
            [point[:, 0] ** p * point[:, 1] ** q for p, q in _QUADRATIC_POWERS], axis=1
        )
        weighted = (design @ coefficient) * jacobian
        for slot, (p, q) in enumerate(_QUADRATIC_POWERS):
            total[slot] += np.sum(weighted * point[:, 0] ** p * point[:, 1] ** q)
    return total


def _owner_quadratic_moments(vertices, count, coefficient):
    """Contract a quadratic against the owner's monomial table, six channels."""
    table = np.asarray(
        _straight_edge_monomial_moments(
            jnp.asarray(vertices), jnp.asarray(count), max_degree=4
        )
    )
    shifts = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    channels = []
    for radial_shift, vertical_shift in shifts:
        slots = np.asarray(
            [
                _OWNER_SLOT[(radial + radial_shift, vertical + vertical_shift)]
                for radial, vertical in _QUADRATIC_POWERS
            ]
        )
        channels.append(np.einsum("nm,nm->n", coefficient, table[:, slots]))
    return np.stack(channels, axis=1)


@pytest.mark.parametrize("polygon", _DENSITY_POLYGONS, ids=["convex", "concave"])
def test_quadratic_density_moments_match_the_dense_reference(polygon: np.ndarray):
    """The owner's quadratic contraction reproduces an independent quadrature."""
    capacity = len(polygon) + 8
    vertices = np.zeros((1, capacity, 2), dtype=np.float64)
    vertices[0, : len(polygon)] = polygon
    count = np.asarray([len(polygon)])
    coefficient = _QUADRATIC_DENSITY[None, :]
    reference = _dense_quadratic_moments(polygon, _QUADRATIC_DENSITY)
    assert np.all(reference != 0.0)
    observed = _owner_quadratic_moments(vertices, count, coefficient)[0]
    # Padding adds exact-zero edges but lengthens the reduction axis, so the
    # comparison is relative rather than bitwise.
    relative = np.abs(observed - reference) / np.abs(reference)
    np.testing.assert_array_less(relative, 1e-13)


def test_quadratic_density_moments_move_with_a_perturbed_coefficient():
    """Positive control: the dense comparison bites on a wrong integrand."""
    polygon = _DENSITY_POLYGONS[0]
    reference = _dense_quadratic_moments(polygon, _QUADRATIC_DENSITY)
    perturbed_coefficient = _QUADRATIC_DENSITY + np.array(
        [0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
    )
    perturbed = _dense_quadratic_moments(polygon, perturbed_coefficient)
    relative = np.abs(perturbed - reference) / np.abs(reference)
    assert np.max(relative) > 2e-3


def test_polynomial_layer_books_the_quadratic_density_over_a_carrier_cell():
    """The surviving polynomial layer reproduces the quadratic moments exactly.

    One cut cell of a stored carrier is posed with a quadratic density in its
    own local frame; the layer the closed-form floor books through must give the same
    area and first moments as the direct owner contraction, whose second
    moments the dense reference checks above.
    """
    _operator, support, _field, _span, _bank, _centroid, _sample = _analytic_support(
        "weak-rotation-reactor-static", -110
    )
    cut = np.flatnonzero(np.asarray(support.included) & np.asarray(support.boundary))
    assert cut.size > 0
    cell = int(cut[0])
    vertices = np.asarray(support.support_vertices, dtype=np.float64)[cell : cell + 1]
    count = np.asarray(support.vertex_count, dtype=np.intp)[cell : cell + 1]
    coefficient = _QUADRATIC_DENSITY[None, :]
    direct = _owner_quadratic_moments(vertices, count, coefficient)
    current, first = padded_polynomial_current_moments(
        jnp.asarray(vertices),
        jnp.asarray(count),
        jnp.zeros((1, 2)),
        jnp.ones((1, 2)),
        jnp.asarray(coefficient),
    )
    np.testing.assert_allclose(np.asarray(current), direct[:, 0], rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(np.asarray(first), direct[:, 1:3], rtol=1e-13, atol=0.0)
