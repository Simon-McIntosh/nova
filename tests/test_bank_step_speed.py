"""The bank update is unchanged when exact condition diagnostics are omitted."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, support_read


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
    measured = support_read.newton_step(None, moments, None, None, None, None)
    fast = support_read.newton_step(
        None, moments, None, None, None, None, condition_diagnostics=False
    )
    for observed, expected in zip(fast.update, measured.update, strict=True):
        np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
    assert np.isnan(float(fast.condition_before))
    assert float(measured.condition_after) < float(measured.condition_before)
