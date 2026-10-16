"""Shared pytest fixtures for the nova test suite."""

import jax
import numpy as np
import pytest

from nova.jax.config import configure_dtypes


def pytest_configure():
    """Enable extended precision before test modules construct arrays."""
    configure_dtypes()


@pytest.fixture(autouse=True)
def _double_precision():
    """Refuse a test body when collection has disabled extended precision."""
    assert jax.config.jax_enable_x64 is True


def _green_block(target, source, section=0.05):
    """Return the total-flux coupling [Wb/A] of one source set on one target."""
    from nova.biot.greens import hybrid_greens

    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


def _solovev(radius, height, coefficients):
    """Evaluate the analytic quartic flux from its three coefficients."""
    alpha, offset, beta = coefficients
    return alpha * radius**4 + offset * radius**2 + beta * height**2
