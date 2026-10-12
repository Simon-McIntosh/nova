"""Shared pytest fixtures for the nova test suite."""

import jax
import pytest

from nova.jax.config import configure_dtypes


@pytest.fixture(autouse=True)
def _double_precision():
    """Enable the extended precision every production forward map assumes.

    Arrays constructed before this runs are silently single precision, so it is
    autouse: no test module has to remember it, and the assertion here fails
    loudly rather than letting a test measure at the wrong precision.
    """
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
