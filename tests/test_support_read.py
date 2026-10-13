"""The support read: fixed points, Bell membership and booking in one step."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, fixed_points, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy
from nova.equilibrium.stencil_mesh import CellCurrentMoments

assert jax.config.jax_enable_x64 is True


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class UnitDensity:
    def current_density(self, radius, normalized):
        return jnp.ones_like(radius)


def _case(kind):
    from tests.equilibrium import test_topology_read as fixture

    _, analytic, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, 50)
    samples = np.asarray(geometry.sample_points)
    nodes = np.unique(np.round(samples.reshape(-1, 2), 12), axis=0)
    jets = fixed_points.stack_jet(jax.vmap(analytic.evaluate)(jnp.asarray(nodes)))
    rows = jnp.zeros((len(nodes), 1, 3, 6))
    operands = support_read.SupportOperands.from_rows(geometry, rows, jets)
    cache = fixed_points.RowCache(
        operands.mesh, operands.nodes, jnp.arange(len(nodes)), rows, jets
    )
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    policy = TopologyPolicy()
    moments = CellCurrentMoments(jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))
    pitch = jnp.median(geometry.pitch)

    @jax.jit
    def step(operands, cache, seeds, previous):
        return support_read.read_and_book(
            operands,
            cache,
            moments,
            seeds,
            previous,
            convention,
            policy,
            UnitDensity(),
            quadrature,
        )

    reference = jax.jit(topology.read)(analytic, geometry, convention, policy)
    return operands, cache, roots, pitch, step, reference, axis


@pytest.fixture(scope="module", params=["limited", "diverted"])
def route(request):
    operands, cache, roots, pitch, step, reference, axis = _case(request.param)
    support, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    return (
        request.param,
        operands,
        cache,
        roots,
        pitch,
        step,
        reference,
        axis,
        (
            support,
            booked,
        ),
    )


def test_axis_class_and_membership_follow_the_closed_form(route):
    kind, operands, _cache, _roots, pitch, _step, reference, axis, result = route
    support, booked = result
    reading = support.reading
    assert bool(support.valid) and bool(booked.valid)
    assert np.all(np.asarray(support.stationary.status) == 0)
    assert int(reading.boundary_class) == int(reference.boundary_class)
    assert np.linalg.norm(np.asarray(reading.axis) - np.asarray(axis)) < 0.05 * float(
        pitch
    )
    area = np.asarray(operands.geometry.full_area)
    selected = np.asarray(reading.membership) * area
    closed = np.asarray(reference.membership) * area
    assert abs(selected.sum() - closed.sum()) < 5e-3 * closed.sum()
    assert float(np.max(np.abs(selected - closed) / area)) < 0.05


def test_booked_current_is_the_selected_area_for_unit_density(route):
    _kind, operands, _cache, _roots, _pitch, _step, _reference, _axis, result = route
    support, booked = result
    area = np.asarray(operands.geometry.full_area)
    selected = float(np.sum(np.asarray(support.reading.membership) * area))
    assert float(np.sum(np.asarray(booked.cell_current))) == pytest.approx(
        selected, rel=1e-4
    )
    private = np.asarray(support.reading.membership) == 0
    assert np.all(np.asarray(booked.cell_current)[private] == 0)


def test_a_row_window_miss_invalidates_the_read_and_the_booking(route):
    _kind, operands, cache, roots, pitch, step, _reference, _axis, _result = route
    missing = replace(cache, lookup=jnp.full_like(cache.lookup, -1))
    support, booked = step(operands, missing, roots + 0.03 * pitch, roots)
    assert not bool(support.valid)
    assert np.all(
        np.asarray(support.stationary.status)
        == fixed_points.FixedPointStatus.NEEDS_ROWS
    )
    assert not bool(booked.valid)
    assert not np.any(np.isfinite(np.asarray(booked.cell_current)))


def test_a_saddle_offered_as_the_axis_is_refused(route):
    kind, operands, cache, roots, pitch, step, _reference, _axis, _result = route
    if kind != "diverted":
        pytest.skip("a limited state has no saddle to mistake for the axis")
    swapped = roots[::-1]
    support, booked = step(operands, cache, swapped + 0.03 * pitch, swapped)
    assert not bool(support.valid)
    assert not bool(booked.valid)
