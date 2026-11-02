"""Explicit moving saddle geometry against directional reference derivatives."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import inspect
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, support_read, topology
from tests.test_support_read import _case


def test_step_geometry_has_no_directional_autodiff():
    source = inspect.getsource(support_read)
    geometry = source.split("def _bell_derivatives(", 1)[1].split(
        "class NewtonStep(", 1
    )[0]
    assert "jax.jvp(" not in geometry


def test_normal_form_rate_matches_directional_reference():
    hessian = jnp.asarray([[2.1, 0.3], [0.3, -1.7]])
    hessian_rate = jnp.asarray([[0.02, -0.03], [-0.03, 0.04]])
    third = jnp.arange(8, dtype=hessian.dtype).reshape(2, 2, 2) * 0.01
    third = sum(jnp.transpose(third, order) for order in permutations(range(3))) / 6
    third_rate = third * 0.13
    fourth = jnp.arange(16, dtype=hessian.dtype).reshape(2, 2, 2, 2) * 0.001
    fourth = sum(jnp.transpose(fourth, order) for order in permutations(range(4))) / 24
    fourth_rate = fourth * -0.17
    point = jnp.asarray([1.2, -0.4])
    point_rate = jnp.asarray([0.03, -0.02])
    form, expected = jax.jvp(
        lambda p, h, d3, d4: topology.saddle_normal_form(p, h, d3, 1e-12, d4),
        (point, hessian, third, fourth),
        (point_rate, hessian_rate, third_rate, fourth_rate),
    )
    actual = support_read._normal_form_rate(
        form, hessian, hessian_rate, third, third_rate, fourth, fourth_rate
    )
    for observed, reference in zip(actual, expected[1:4], strict=True):
        np.testing.assert_allclose(observed, reference, rtol=1e-10, atol=1e-10)


def test_sorted_cuts_and_ring_velocities_match_directional_reference():
    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    support, _ = step(operands, cache, roots + 0.03 * pitch, roots)
    reading = support.reading
    form = reading.saddle_form
    changes = (
        jnp.asarray([0.002, -0.003]),
        jnp.asarray([[0.01, -0.02], [-0.01, 0.03], [0.02, -0.01], [-0.02, 0.01]]),
        jnp.asarray([[0.002, 0.001], [-0.001, 0.003], [0.001, 0.002], [-0.002, 0.001]]),
        jnp.asarray(
            [[0.0002, -0.0001], [0.0001, 0.0002], [-0.0002, 0.0001], [0.0001, -0.0002]]
        ),
    )
    chosen = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
    active = np.nonzero(
        np.asarray(reading.normal_form_cells & (reading.membership > 0))
    )[0]
    assert len(active) > 0
    for cell in active:
        vertices = operands.geometry.vertices[cell]
        count = operands.geometry.vertex_count[cell]
        centre = operands.geometry.centre[cell]
        intervals = reading.edge_interval[cell]
        selected = reading.fragment_selected[cell]

        def edge_values(position, direction, curvature, cubic):
            changed = form._replace(
                position=position, direction=direction, curvature=curvature, cubic=cubic
            )
            return topology._normal_form_cell_fragments(
                vertices, count, changed, topology.TopologyPolicy().edge_tolerance
            ).edge_interval[chosen]

        primals = (form.position, form.direction, form.curvature, form.cubic)
        _, expected_interval = jax.jvp(edge_values, primals, changes)
        actual_interval = support_read._edge_interval_rate(
            vertices, count, form, changes, topology.TopologyPolicy().edge_tolerance
        )[chosen]
        np.testing.assert_allclose(
            actual_interval, expected_interval, rtol=1e-10, atol=1e-10
        )
        interval_rate = jnp.pad(
            actual_interval, ((0, intervals.shape[0] - 2), (0, 0), (0, 0), (0, 0))
        )

        def ring_values(position, direction, curvature, cubic, interval):
            changed = form._replace(
                position=position, direction=direction, curvature=curvature, cubic=cubic
            )
            return current._normal_rings(
                vertices, count, centre, changed, selected, interval, 8
            )[0]

        _, expected_velocity = jax.jvp(
            ring_values, (*primals, intervals), (*changes, interval_rate)
        )
        actual_velocity = support_read._normal_ring_rates(
            vertices,
            count,
            centre,
            form,
            selected,
            intervals,
            interval_rate,
            changes,
            8,
        )
        np.testing.assert_allclose(
            actual_velocity, expected_velocity, rtol=1e-10, atol=1e-10
        )
