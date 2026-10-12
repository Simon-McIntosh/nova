"""Regression checks for the retained nonconforming control field."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import importlib.util
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

path = Path(os.environ["READ_ROUTE_JETS_MODULE"])
spec = importlib.util.spec_from_file_location("control_jets", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(f"MEASUREMENT_MODULE={module.__file__} CWD={Path.cwd().resolve()}", flush=True)


@pytest.mark.parametrize("point", [(0.13, 0.21), (0.4, -0.2), (0.0, 0.0)])
def test_nonconforming_control_reproduces_quadratic(point):
    theta = np.arange(6) * np.pi / 3
    nodes = np.concatenate(
        (np.zeros((1, 2)), np.stack((np.cos(theta), np.sin(theta)), axis=-1))
    )
    x, y = nodes.T
    jets = np.stack(
        (
            0.4 + x + 2 * y + x * x + 3 * x * y + 2 * y * y,
            1 + 2 * x + 3 * y,
            2 + 3 * x + 4 * y,
            np.full(7, 2.0),
            np.full(7, 3.0),
            np.full(7, 4.0),
        ),
        axis=-1,
    )
    field = module.HermiteField(
        jnp.asarray(jets),
        jnp.asarray(nodes),
        jnp.arange(7)[None, :],
        jnp.zeros((1, 2)),
        jnp.ones(1),
    )
    result = jax.jit(field.evaluate)(jnp.asarray(point))
    x, y = point
    assert np.isclose(
        result.value,
        0.4 + x + 2 * y + x * x + 3 * x * y + 2 * y * y,
        atol=1e-13,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.gradient, (1 + 2 * x + 3 * y, 2 + 3 * x + 4 * y), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        result.hessian, ((2.0, 3.0), (3.0, 4.0)), atol=1e-11, rtol=0
    )
