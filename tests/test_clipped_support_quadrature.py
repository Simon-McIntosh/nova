from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.clip_quadrature import (
    clipped_support_field_integrals,
    clipped_support_quadrature,
    cut_cell_bank_capacity,
)
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
    return operator, support, field, flux_span, bank_capacity


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
    return (
        jnp.sum(pressure * volume_weight, axis=1),
        jnp.sum(field_squared * volume_weight, axis=1),
    )


@pytest.mark.parametrize(
    "case_name", ("weak-rotation-reactor-static", "diverted-single-null")
)
def test_compact_reduction_matches_dense_quadrature(case_name: str):
    operator, support, field, flux_span, bank_capacity = _analytic_support(
        case_name, -110
    )
    expected_pressure, expected_field = _dense_integrals(
        operator, support, field, flux_span
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

    whole = np.asarray(support.included) & ~np.asarray(support.boundary)
    cut = np.asarray(support.included) & np.asarray(support.boundary)
    assert np.all(np.isfinite(np.asarray(actual.pressure_volume)))
    np.testing.assert_allclose(
        np.asarray(actual.pressure_volume)[whole],
        np.asarray(expected_pressure)[whole],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual.field_volume)[whole],
        np.asarray(expected_field)[whole],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual.pressure_volume)[cut],
        np.asarray(expected_pressure)[cut],
        rtol=MEASURED_CHAIN_RELATIVE_TOLERANCE,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual.field_volume)[cut],
        np.asarray(expected_field)[cut],
        rtol=MEASURED_CHAIN_RELATIVE_TOLERANCE,
        atol=1.0e-12,
    )


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


def test_compact_reduction_work_arrays_stay_below_one_gibibyte():
    operator, support, field, flux_span, bank_capacity = _analytic_support(
        "weak-rotation-reactor-static", -110
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
