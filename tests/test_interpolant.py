import importlib.util
import subprocess
import sys

import numpy as np
import pytest

JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from nova.linalg.interpolant import Bernstein, BSpline, Interp, Linear, Polynomial

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pylab as pylab


def test_linalg_and_design_matrices_do_not_import_jax():
    """The linear-algebra package and design matrices work without JAX."""
    code = """
import builtins

original_import = builtins.__import__

def reject_jax(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "jax" or name.startswith("jax."):
        raise AssertionError(f"unexpected JAX import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_jax

import nova.linalg
from nova.linalg.basis import Bernstein

assert Bernstein(4, order=2).matrix.shape == (4, 3)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_bernstein_order():
    order = 6
    basis = Bernstein(order=order)
    assert basis.order == order


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_polynomial_remains_a_registered_pytree():
    polynomial = Polynomial(model=jnp.array([1.0, 2.0, 3.0]), order=2)
    leaves, structure = jax.tree_util.tree_flatten(polynomial)
    restored = jax.tree_util.tree_unflatten(structure, leaves)

    assert isinstance(restored, Polynomial)
    assert restored.order == polynomial.order
    assert np.allclose(restored.model, polynomial.model)


@pytest.mark.skipif(
    not (JAX_AVAILABLE and MATPLOTLIB_AVAILABLE),
    reason="JAX and matplotlib are required",
)
def test_polynomial_plot():
    poly = Polynomial(model=jnp.array([1, 1, 1]), order=2)
    with pylab.ioff():
        assert len(poly.plot(basis=False).lines) == 1
        assert len(poly.plot(basis=True).lines) == 4
    coordinate = jnp.linspace(0, 1, 12)
    data = jnp.ones_like(coordinate)
    poly_data = Polynomial(coordinate, data, jnp.array([1, 1, 1]), order=2)
    with pylab.ioff():
        assert len(poly_data.plot(basis=True).lines) == 5


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_spline_fit():
    x = jnp.linspace(0, 1, 12)
    data = -(3 * x**5) + 3 * x**2 + -3 * x + 4
    poly = BSpline(x, order=5) / data
    assert np.allclose(poly(x), data)
    poly_2nd = BSpline(x, order=2) / data
    assert not np.allclose(poly_2nd(x), data)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_linear_fit():
    x = jnp.linspace(0, 1, 12)
    data = -(3 * x**5) + 3 * x**2 + -3 * x + 4
    linear = Interp(x) / data
    assert np.allclose(linear(x), data)


@pytest.mark.skipif(
    not (JAX_AVAILABLE and MATPLOTLIB_AVAILABLE),
    reason="JAX and matplotlib are required",
)
def test_linear_plot():
    x = jnp.linspace(0, 1, 12)
    data = -(3 * x**5) + 3 * x**2 + -3 * x + 4
    linear = Linear(x, data)
    with pylab.ioff():
        assert len(linear.plot().lines) == 2


if __name__ == "__main__":
    pytest.main([__file__])
