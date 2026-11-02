"""The bank update is unchanged when exact condition diagnostics are omitted."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax.numpy as jnp
import jax
import numpy as np

from nova.equilibrium import current, support_read, topology


def test_condition_diagnostics_do_not_change_update(monkeypatch):
    matrix = jnp.asarray([[1e-6, 2e-7, 0], [0, 1, 0.2], [0, 0, 1e6]])
    moments = tuple(jnp.asarray([1.0]) for _ in range(3))
    residual = tuple(jnp.asarray([value]) for value in (1e-6, 1.0, 1e6))
    booked = current.Booking(*moments, jnp.asarray(1.0), jnp.asarray(1.0), True)
    monkeypatch.setattr(
        support_read,
        "step_residual",
        lambda *args: support_read.StepResidual(residual, booked, None, True),
    )
    monkeypatch.setattr(support_read, "step_jacobian", lambda *args, **kwargs: matrix)
    measured = support_read.newton_step(
        None, moments, None, None, None, None, condition_diagnostics=True
    )

    def refuse_condition_estimate(*args, **kwargs):
        raise AssertionError("fast Newton step evaluated a dense condition estimate")

    monkeypatch.setattr(jnp.linalg, "cond", refuse_condition_estimate)
    fast = support_read.newton_step(
        None, moments, None, None, None, None, condition_diagnostics=False
    )
    for observed, expected in zip(fast.update, measured.update, strict=True):
        np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
    assert np.isnan(float(fast.condition_before))
    assert float(measured.condition_after) < float(measured.condition_before)


def test_multicell_step_omits_condition_estimates_by_default(monkeypatch):
    moments = tuple(jnp.asarray([1.0, 1.0]) for _ in range(3))
    matrix = jnp.diag(jnp.asarray([1e-6, 2e-6, 1.0, 2.0, 1e6, 2e6]))
    residual = tuple(jnp.asarray([1.0, 1.0]) for _ in range(3))
    booked = current.Booking(*moments, jnp.asarray(1.0), jnp.asarray(1.0), True)
    monkeypatch.setattr(
        support_read,
        "step_residual",
        lambda *args: support_read.StepResidual(residual, booked, None, True),
    )
    monkeypatch.setattr(support_read, "step_jacobian", lambda *args, **kwargs: matrix)

    def refuse_condition_estimate(*args, **kwargs):
        raise AssertionError("default multicell step evaluated a condition estimate")

    monkeypatch.setattr(jnp.linalg, "cond", refuse_condition_estimate)
    step = support_read.newton_step(None, moments, None, None, None, None)
    assert np.isnan(float(step.condition_before))
    assert np.isnan(float(step.condition_after))
    np.testing.assert_allclose(
        jnp.concatenate(step.update), -1.0 / jnp.diag(matrix), rtol=1e-12
    )


def test_local_booking_sensitivities_batch_channels(monkeypatch):
    from tests.test_support_read import UnitDensity, _case

    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    support, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    moments = tuple(jnp.zeros(operands.sample_index.shape[0]) for _ in range(3))
    sigma = topology.TopologyConvention.from_cocos(17, 1.0).sigma

    def refuse_serial_map(*args, **kwargs):
        raise AssertionError("booking sensitivities mapped channels serially")

    monkeypatch.setattr(jax.lax, "map", refuse_serial_map)
    sensitivities = support_read._local_booking_sensitivities(
        operands,
        moments,
        support,
        UnitDensity(),
        lambda radius, flux: jnp.zeros_like(radius),
        current.Quadrature(sigma=sigma, segments=64),
        booked,
    )
    assert sensitivities.shape == (
        operands.sample_index.shape[0],
        3,
        operands.sample_index.shape[1] + 8,
    )
    assert np.all(np.isfinite(np.asarray(sensitivities)))
