"""A contracting bank step reuses its balanced Newton factor."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import hashlib
import numpy as np
import pytest
from types import SimpleNamespace

from nova.equilibrium import current, support_read, topology


def test_book_primal_keeps_exact_diverted_moments():
    from tests.test_support_read import _case

    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    _, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    expected = (
        "73db87c3ed35d06111216d2e94f40e707a3358056e16898da445945b3bc358a5",
        "a50bd0ec4f6c3fa86f4590d4bf4286c349fb181721887be99a9ffa9cafe81d4b",
        "648a03e71877e7279744d1fa475f9cc9580798e070343ce51fb3ef5531d048d6",
        "340bc5ca6f9b6666779b145b364f5d372e04400736edc43da3d8d4c7dd3d9621",
        "6c3c396ed6b5c36dcae172271f462051b1266b851e92df3deea8ac65478fd712",
        "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a",
    )
    sirius_cpu = (
        "41e8d3baa02177c2fee5e6ebc6d6acdbec7870ccb4ce9aa0fad6e26b9400cd14",
        "f8634779a2b4991036084487ad882e8774ce2c87951f6d17eabc2fc6e5c8dbdf",
        "7a2992ba2b91f108a2ce41adb8e1d7ecaf80dbe6525efd0800413832c7dcd0fa",
        "4f290cb1412769ccc838db46fbcd7ab8c8e4d99a3d2a7d0d1bbe3f0f9e3f5622",
        "6c3c396ed6b5c36dcae172271f462051b1266b851e92df3deea8ac65478fd712",
        "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a",
    )
    actual = tuple(
        hashlib.sha256(np.asarray(getattr(booked, name)).tobytes()).hexdigest()
        for name in booked._fields
    )
    assert actual in (expected, sirius_cpu)


def test_batched_booking_tangent_matches_single_channel_reference():
    from tests.test_support_read import UnitDensity, _case

    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    support, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    cells, samples = operands.sample_index.shape
    moments = tuple(jnp.zeros(cells) for _ in range(3))
    psi = support_read.nodal_jets(operands, moments)[operands.sample_index, 0]
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=8)

    def gradient(radius, flux):
        return jnp.ones_like(radius)

    def reference(index):
        psi_direction = jnp.broadcast_to(
            jax.nn.one_hot(jnp.minimum(index, samples - 1), samples)
            * (index < samples),
            psi.shape,
        )
        field_direction = jnp.broadcast_to(
            jax.nn.one_hot(jnp.clip(index - samples, 0, 5), 6)
            * ((index >= samples) & (index < samples + 6)),
            (cells, 6),
        )
        result = current.book_tangent(
            psi,
            psi_direction,
            support.reading,
            field_direction,
            (index == samples + 6).astype(psi.dtype),
            (index == samples + 7).astype(psi.dtype),
            operands.geometry,
            UnitDensity(),
            gradient,
            quadrature,
            booked,
        )
        return jnp.stack(tuple(result), axis=-1)

    serial = jax.jit(jax.vmap(reference))(jnp.arange(samples + 8))
    batched = jax.jit(
        lambda: support_read._local_booking_sensitivities(
            operands,
            moments,
            support,
            UnitDensity(),
            gradient,
            quadrature,
            booked,
        )
    )()
    expected = np.moveaxis(np.asarray(serial), 0, -1)
    observed = np.asarray(batched)
    assert np.all(np.isfinite(observed))
    relative = np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1.0)
    worst = np.unravel_index(np.argmax(np.abs(observed - expected)), observed.shape)
    assert relative < 1e-10, (relative, worst, observed[worst], expected[worst])


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
