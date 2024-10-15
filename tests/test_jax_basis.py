import jax.numpy as jnp
import matplotlib.pylab
import numpy as np
import pytest

from nova.jax.basis import Bernstein, BSpline, Interp, Linear, Polynomial


def test_basis_order():
    order = 6
    basis = Bernstein(order=order)
    assert basis.order == order


def test_polynomial_plot():
    poly = Polynomial(model=jnp.array([1, 1, 1]), order=2)
    with matplotlib.pylab.ioff():
        assert len(poly.plot(basis=False).lines) == 1
        assert len(poly.plot(basis=True).lines) == 4
    coordinate = jnp.linspace(0, 1, 12)
    data = jnp.ones_like(coordinate)
    poly_data = Polynomial(coordinate, data, jnp.array([1, 1, 1]), order=2)
    with matplotlib.pylab.ioff():
        assert len(poly_data.plot(basis=True).lines) == 5


def test_spline_fit():
    x = jnp.linspace(0, 1, 12)
    data = -(3 * x**5) + 3 * x**2 + -3 * x + 4
    poly = BSpline(x, order=5) / data
    assert np.allclose(poly(x), data)
    poly_2nd = BSpline(x, order=2) / data
    assert not np.allclose(poly_2nd(x), data)


def test_linear_fit():
    x = jnp.linspace(0, 1, 12)
    data = -(3 * x**5) + 3 * x**2 + -3 * x + 4
    linear = Interp(x) / data
    assert np.allclose(linear(x), data)


def test_linear_plot():
    x = jnp.linspace(0, 1, 12)
    data = -(3 * x**5) + 3 * x**2 + -3 * x + 4
    linear = Linear(x, data)
    with matplotlib.pylab.ioff():
        assert len(linear.plot().lines) == 2


if __name__ == "__main__":
    pytest.main([__file__])
