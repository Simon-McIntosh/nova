"""The graded map's Jacobian is the map's cosh, taken once.

Rebuilding it from the stretch as ``sqrt(1 + sinh^2)`` is algebraically the same
value but a larger traced program: a square, an add and a square root on top of
the sinh.  Counting the equations of the traced residual is deterministic, so it
holds the shape where compile seconds on a shared host cannot.
"""

import jax
import jax.numpy as jnp
import pytest

from nova.biot import gradedresidual

NODES = 8
PANELS = (
    (jnp.asarray([0.3]), jnp.asarray([0.1]), jnp.asarray([1.2])),  # offset, end, scale
    (jnp.asarray([0.3]), jnp.asarray([0.1]), jnp.asarray([1.2])),
)


def _pieces(x, y):
    return x + y, x * y + 1.0


def _program():
    panels = [
        (offset, end, scale, jnp.asarray(0.0), jnp.asarray(gradedresidual.QUARTER))
        for offset, end, scale in PANELS
    ]
    return jax.make_jaxpr(
        lambda: gradedresidual.graded_residual(panels, _pieces, NODES, jnp)
    )().jaxpr


def _count(jaxpr):
    total = 0
    for equation in jaxpr.eqns:
        total += 1
        for value in equation.params.values():
            inner = value.jaxpr if hasattr(value, "jaxpr") else None
            if inner is not None:
                total += _count(inner)
    return total


def test_graded_map_jacobian_traces_the_hyperbolic_form():
    assert _count(_program()) == 402


def test_rebuilding_the_jacobian_from_the_sinh_is_caught(monkeypatch):
    """Negative control: the counted shape reddens when the cosh is rebuilt."""
    baseline = _count(_program())
    assert baseline == 402

    def rebuilt(value):
        return jnp.sqrt(1.0 + jnp.sinh(value) ** 2)

    monkeypatch.setattr(jnp, "cosh", rebuilt)
    mutated = _count(_program())
    assert mutated == 408
    assert mutated > baseline
