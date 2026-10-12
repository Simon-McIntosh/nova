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


def _quartic(point):
    x, y = point
    return 0.4 + x + 2 * y + 0.3 * x**4 - 0.2 * x * y**3 + 0.1 * y**4


def _element():
    from types import SimpleNamespace
    from nova.equilibrium.topology import FieldJet

    theta = np.arange(6) * np.pi / 3
    vertices = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    centres = np.asarray(((0.0, 0.0), (1.5, np.sqrt(3) / 2)))
    samples = np.concatenate((centres[:, None], centres[:, None] + vertices), axis=1)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    indices = inverse.reshape(2, 7)

    def evaluate(point):
        return FieldJet(
            _quartic(point), jax.grad(_quartic)(point), jax.hessian(_quartic)(point)
        )

    measured = jax.vmap(evaluate)(jnp.asarray(nodes))
    jets = jnp.stack(
        (
            measured.value,
            measured.gradient[:, 0],
            measured.gradient[:, 1],
            measured.hessian[:, 0, 0],
            measured.hessian[:, 0, 1],
            measured.hessian[:, 1, 1],
        ),
        axis=-1,
    )
    bell_path = path.with_name("bell.py")
    if bell_path.exists():
        specification = importlib.util.spec_from_file_location(
            "control_bell", bell_path
        )
        bell = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = bell
        specification.loader.exec_module(bell)
        mesh = bell.build_mesh(nodes, indices, SimpleNamespace(pitch=jnp.ones(2)))
        field = bell.field_from_jets(jets, jnp.asarray(indices), mesh)
        return field.value, lambda cell, point: field.cell_value(cell, point)
    fields = [
        module.HermiteField(
            jets,
            jnp.asarray(nodes),
            jnp.asarray(indices[cell : cell + 1]),
            jnp.asarray(centres[cell : cell + 1]),
            jnp.ones(1),
        )
        for cell in range(2)
    ]
    return fields[0].value, lambda cell, point: fields[cell].value(point)


def test_read_representation_reproduces_quartic():
    value, _ = _element()
    points = jnp.asarray(((0.2, 0.1), (-0.3, 0.15), (0.7, -0.1)))
    expected = jax.vmap(_quartic)(points)
    measured = jax.vmap(value)(points)
    np.testing.assert_allclose(measured, expected, atol=1e-11, rtol=0)


def test_read_representation_has_shared_normal_derivative():
    _, local = _element()
    first, second = np.asarray((1.0, 0.0)), np.asarray((0.5, np.sqrt(3) / 2))
    normal = jnp.asarray((np.sqrt(3) / 2, 0.5))
    for weight in (0.2, 0.5, 0.8):
        point = jnp.asarray((1 - weight) * first + weight * second)
        gradients = [jax.grad(lambda p: local(cell, p))(point) for cell in range(2)]
        np.testing.assert_allclose(
            jnp.dot(gradients[0], normal),
            jnp.dot(gradients[1], normal),
            atol=1e-10,
            rtol=0,
        )
