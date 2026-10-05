"""Total-field null tangents and regular saddle support contracts."""

# Configure precision before importing modules with array-valued defaults.
# ruff: noqa: E402
from dataclasses import dataclass, replace
import json

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from nova.equilibrium.solve_request import ForwardSolvePolicy, TopologyPolicy
from nova.equilibrium.topology import (
    FieldJet,
    TopologyReason,
    saddle_cell_fragments,
    saddle_normal_form,
    stationary_read,
)

assert jax.config.jax_enable_x64 is True


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QuadraticPointField:
    """Point evaluator whose translating stationary point is known exactly."""

    centre: jax.Array
    hessian: jax.Array

    def evaluate(self, point):
        offset = point - self.centre
        return FieldJet(
            0.5 * offset @ self.hessian @ offset,
            self.hessian @ offset,
            self.hessian,
        )


def test_topology_policy_round_trip():
    policy = replace(
        ForwardSolvePolicy(),
        topology=TopologyPolicy(null_capacity=7, hessian_tolerance=1e-8),
    )
    assert (
        ForwardSolvePolicy.from_dict(json.loads(json.dumps(policy.to_dict()))) == policy
    )
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            TopologyPolicy(null_capacity=invalid)
    for invalid in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            TopologyPolicy(hessian_tolerance=invalid)


def test_stationary_position_implicit_tangent():
    field = QuadraticPointField(
        jnp.asarray((2.0, 0.1)), jnp.asarray(((-3.0, 0.2), (0.2, 2.0)))
    )
    seed = jnp.asarray((2.05, 0.06))
    policy = TopologyPolicy()

    def position(centre):
        return stationary_read(
            replace(field, centre=centre), seed, 0.1, policy
        ).position

    direction = jnp.asarray((0.2, -0.7))
    actual, tangent = jax.jvp(position, (field.centre,), (direction,))
    np.testing.assert_allclose(actual, field.centre, atol=1e-14)
    np.testing.assert_allclose(tangent, direction, atol=1e-14)
    step = 1e-5
    central = (
        position(field.centre + step * direction)
        - position(field.centre - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, central, atol=2e-11)
    singular = replace(field, hessian=jnp.asarray(((1.0, 0.0), (0.0, 0.0))))
    refused = stationary_read(singular, seed, 0.1, policy)
    assert not refused.valid
    assert not refused.tangent_valid
    assert refused.reason == TopologyReason.SINGULAR_TANGENT
    _, singular_tangent = jax.jvp(
        lambda centre: (
            stationary_read(
                replace(singular, centre=centre), seed, 0.1, policy
            ).position
        ),
        (field.centre,),
        (direction,),
    )
    assert np.all(np.isnan(singular_tangent))


def _saddle_jet():
    # psi = x*x - y*y + 0.3*x**3 + 0.2*x*x*y.
    third = np.zeros((2, 2, 2), dtype=np.float64)
    third[0, 0, 0] = 1.8
    third[0, 0, 1] = third[0, 1, 0] = third[1, 0, 0] = 0.4
    return jnp.diag(jnp.asarray((2.0, -2.0))), jnp.asarray(third)


def _hexagon(radius):
    angle = np.arange(6) * np.pi / 3.0
    return radius * np.stack((np.cos(angle), np.sin(angle)), axis=1)


def _exact_core_area(vertices):
    def height(radius):
        heights = []
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            if abs(second[0] - first[0]) < 1e-15:
                continue
            fraction = (radius - first[0]) / (second[0] - first[0])
            if 0.0 <= fraction <= 1.0:
                heights.append(first[1] + fraction * (second[1] - first[1]))
        if len(heights) < 2:
            return 0.0
        discriminant = 0.04 * radius**4 + 4 * (radius**2 + 0.3 * radius**3)
        lower = (0.2 * radius**2 - np.sqrt(discriminant)) / 2.0
        upper = (0.2 * radius**2 + np.sqrt(discriminant)) / 2.0
        return max(0.0, min(max(heights), upper) - max(min(heights), lower))

    return quad(
        height,
        0.0,
        float(vertices[:, 0].max()),
        epsabs=1e-13,
        epsrel=1e-12,
        points=[float(vertices[:, 0].max()) / 2],
        limit=200,
    )[0]


def test_saddle_curved_support_order():
    hessian, third = _saddle_jet()
    form = saddle_normal_form(jnp.zeros(2), hessian, third, 1e-10)
    assert form.valid
    np.testing.assert_allclose(
        np.einsum("ai,ij,aj->a", form.direction, hessian, form.direction),
        0.0,
        atol=1e-14,
    )
    radii = np.asarray((0.2, 0.12, 0.08, 0.05, 0.03))
    errors = []
    for radius in radii:
        vertices = _hexagon(radius)
        result = saddle_cell_fragments(jnp.asarray(vertices), 6, form, 1e-12)
        assert result.valid, result
        direction = np.asarray(form.direction)
        core = (direction[:, 0] > 0) & (np.roll(direction[:, 0], -1) > 0)
        assert np.count_nonzero(core) == 1
        area = float(np.asarray(result.area)[core][0])
        reference = _exact_core_area(vertices)
        full_area = 3 * np.sqrt(3) * radius**2 / 2
        errors.append(abs(area - reference) / full_area)
        np.testing.assert_allclose(np.sum(result.area), full_area, rtol=2e-14)
        assert np.all(np.asarray(result.area) > 0.0)
    order = float(np.polyfit(np.log(radii), np.log(errors), 1)[0])
    print(f"SADDLE_NORMAL_FORM radii={radii.tolist()} errors={errors} order={order}")
    assert order >= 2.0


def test_stationary_and_saddle_primitives_trace():
    policy = TopologyPolicy()
    field = QuadraticPointField(
        jnp.asarray((2.0, 0.1)), jnp.diag(jnp.asarray((-2.0, 3.0)))
    )
    seed = jnp.asarray((2.03, 0.08))
    eager = stationary_read(field, seed, 0.1, policy)
    traced = jax.jit(stationary_read)(field, seed, 0.1, policy)
    for left, right in zip(
        jax.tree.leaves(eager), jax.tree.leaves(traced), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-14)
    centres = jnp.stack((field.centre, field.centre + 0.01))
    batched = jax.jit(
        jax.vmap(
            lambda centre: stationary_read(
                replace(field, centre=centre), seed, 0.1, policy
            )
        )
    )(centres)
    np.testing.assert_allclose(batched.position, centres, atol=1e-14)
    hessian, third = _saddle_jet()
    form = jax.jit(saddle_normal_form)(jnp.zeros(2), hessian, third, 1e-10)
    fragments = jax.jit(saddle_cell_fragments)(
        jnp.asarray(_hexagon(0.1)), 6, form, 1e-12
    )
    assert fragments.valid
    refused = saddle_normal_form(jnp.zeros(2), jnp.eye(2), third, 1e-10)
    assert not refused.valid
    assert refused.reason == TopologyReason.SINGULAR_REPRESENTATION
