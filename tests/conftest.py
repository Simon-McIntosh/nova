"""Shared pytest fixtures for the nova test suite."""

import jax
import pytest

from nova.jax.config import configure_dtypes


def pytest_configure():
    """Enable extended precision before test modules construct arrays."""
    configure_dtypes()


@pytest.fixture(autouse=True)
def _double_precision():
    """Refuse a test body when collection has disabled extended precision."""
    assert jax.config.jax_enable_x64 is True
