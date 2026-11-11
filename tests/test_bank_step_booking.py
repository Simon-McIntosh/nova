"""A contracting bank step reuses its balanced Newton factor."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import hashlib
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from nova.equilibrium import current, support_read, topology

REFERENCE_PATH = Path(__file__).with_name("data") / "booking_primal_diverted.npz"

# The booked arrays are a deterministic function of the case, so agreement
# within double-precision roundoff of the sums the read performs is all that is
# required. A reordered accumulation moves the bits and stays inside this.
RELATIVE_TOLERANCE = 1e-12
# cell_current carries exact zeros and the moments cross zero, where a relative
# comparison has no meaning. Against fields peaking between 1e-3 and 1e-1 this
# floor sits well above the roundoff a reordered sum can leave on a zero entry
# and far below any entry that carries a physical contribution.
ABSOLUTE_FLOOR = 1e-15


def _booked_diverted():
    """The booked result of the diverted case's contracted step."""
    from tests.test_support_read import _case

    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    _, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    return booked


def test_book_primal_keeps_exact_diverted_moments():
    """Every booked field reproduces the stored reference within tolerance.

    ``tests/data/booking_primal_diverted.npz`` was written from the diverted
    case on the CPU lane and is reproduced bit for bit at revision
    ``936c1b95c10bc177a519d33fdc74916c07b8c912`` — the commit that introduced
    :func:`_booked_diverted` — by::

        PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -c '
        import numpy as np
        from tests.test_bank_step_booking import _booked_diverted
        b = _booked_diverted()
        np.savez("tests/data/booking_primal_diverted.npz",
                 **{n: np.asarray(v) for n, v in zip(b._fields, b)})'

    Rerunning that command at that revision leaves every stored field
    identical in dtype, shape and bytes. The digests below are printed only
    when a field disagrees, so a failure says which field moved and by how
    much without pinning the bits.
    """
    booked = _booked_diverted()
    reference = np.load(REFERENCE_PATH)
    assert sorted(reference.files) == sorted(booked._fields)
    digests = [
        "%s: reference=%s actual=%s"
        % (
            name,
            hashlib.sha256(np.asarray(reference[name]).tobytes()).hexdigest(),
            hashlib.sha256(np.asarray(getattr(booked, name)).tobytes()).hexdigest(),
        )
        for name in booked._fields
    ]
    report = "booked field digests\n" + "\n".join(digests)
    for name in booked._fields:
        actual = np.asarray(getattr(booked, name))
        expected = reference[name]
        if expected.dtype.kind == "b":
            np.testing.assert_array_equal(actual, expected, err_msg=report)
        else:
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=RELATIVE_TOLERANCE,
                atol=ABSOLUTE_FLOOR,
                err_msg=report,
            )


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
