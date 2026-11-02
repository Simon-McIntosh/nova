"""A contracting bank step reuses its balanced Newton factor."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace

from nova.equilibrium import current, support_read


@pytest.fixture
def linear_step(monkeypatch):
    moments = tuple(jnp.asarray([1.0]) for _ in range(3))
    booked = current.Booking(*moments, jnp.asarray(1.0), jnp.asarray(1.0), True)
    residual = tuple(jnp.asarray([1.0]) for _ in range(3))
    monkeypatch.setattr(
        support_read,
        "step_residual",
        lambda *args: support_read.StepResidual(residual, booked, None, True),
    )
    calls = []

    def jacobian(*args, **kwargs):
        jax.debug.callback(lambda _: calls.append(1), jnp.asarray(0))
        return jnp.diag(jnp.asarray([2.0, 3.0, 4.0]))

    monkeypatch.setattr(support_read, "step_jacobian", jacobian)
    return moments, calls


def test_reuses_factor_only_while_residual_contracts(linear_step):
    moments, calls = linear_step
    first = support_read.newton_step(None, moments, None, None, None, None)
    reused = support_read.newton_step(
        None,
        moments,
        None,
        None,
        None,
        None,
        factor_cache=first.factor,
        contraction_ratio=0.5,
    )
    assert calls == [1]
    assert not bool(reused.refactored)
    for actual, expected in zip(reused.update, first.update, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    rebuilt = support_read.newton_step(
        None,
        moments,
        None,
        None,
        None,
        None,
        factor_cache=reused.factor,
        contraction_ratio=0.5001,
    )
    assert calls == [1, 1]
    assert bool(rebuilt.refactored)


def test_reuse_requires_observed_contraction(linear_step):
    moments, _ = linear_step
    first = support_read.newton_step(None, moments, None, None, None, None)
    with pytest.raises(ValueError, match="measured residual contraction"):
        support_read.newton_step(
            None, moments, None, None, None, None, factor_cache=first.factor
        )


def test_booking_block_is_assembled_from_one_batched_sensitivity_call(monkeypatch):
    moments = tuple(jnp.asarray([1.0]) for _ in range(3))
    booked = current.Booking(*moments, jnp.asarray(1.0), jnp.asarray(1.0), True)
    operands = SimpleNamespace(sample_index=jnp.asarray([[0]]), geometry=object())
    support = SimpleNamespace(reading=SimpleNamespace(axis=0.0, boundary=0.0))
    calls = []

    def local(*args):
        calls.append(1)
        return jnp.asarray([[[2.0] + [0.0] * 8, [0.0] * 9, [0.0] * 9]])

    class DirectionalField:
        def sampled_coefficient(self, geometry):
            return jnp.zeros((1, 6))

        def value(self, point):
            return jnp.asarray(0.0)

    def directional(operands, direction):
        return jnp.stack(direction, axis=-1), DirectionalField()

    monkeypatch.setattr(support_read, "_local_booking_sensitivities", local)
    monkeypatch.setattr(support_read, "_directional_field", directional)
    assembled = jax.jit(
        lambda: support_read.step_jacobian(
            operands,
            moments,
            support,
            None,
            None,
            current.Quadrature(),
            booked,
        )
    )
    matrix = assembled()
    matrix.block_until_ready()
    assert calls == [1]
    np.testing.assert_allclose(matrix, jnp.diag(jnp.asarray([-1.0, 1.0, 1.0])))
