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
    # Broadwell Xeon E5-2680 v4, the login-class CPU family.
    expected = (
        "233e491e1789813021391660aa9ea4b96c5295b16a632b825253f92de1a13cd3",
        "85b545de4012cd26c1ffb2c5fd724842b1fb87a8487c20bc37b5ce2ba0d372f0",
        "0e0d20611210f6a40e183859ecf9932369a2093ac8ea797e878815d003f1d1a3",
        "8f55488d671b15a882491e03214f7ce8999b9f833890164b210b5d84d61875fb",
        "6c3c396ed6b5c36dcae172271f462051b1266b851e92df3deea8ac65478fd712",
        "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a",
    )
    # Sapphire Rapids Xeon Gold 6416H, the sirius partition.
    sirius_cpu = (
        "896f07c9c4e103cf2ffec42b55ace8481f4815fe0a1d648691edb9a5a74103e4",
        "a1bbf9beb251b222b13f65c08e01ab1d9c6548844ef2f0c6c7a4f8e685e817a6",
        "3d8fca0f3a179cda7b7b1a230307da93ba63bf05f84c164c2751e95ff36ce325",
        "15ec33f7d6a711e79530de4938ee25aaf9952221dfac67a1441bfc04d2b741fe",
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
