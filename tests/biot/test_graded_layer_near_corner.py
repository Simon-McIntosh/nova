"""Resolve the logarithmic end layer of a ring corner integral."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

from nova.biot.gradedresidual import QUARTER, graded_residual  # noqa: E402


def _residual(radius, level, radial_offset):
    span = 2.0 * radius
    zero = jnp.zeros_like(radial_offset)
    limit = jnp.full_like(radial_offset, QUARTER)
    panels = (
        (jnp.abs(level), radial_offset + span, span, zero, limit),
        (jnp.abs(level), radial_offset, span, zero, limit),
    )

    def pieces(x, y):
        numerator = radial_offset[:, None] + span[:, None] * y
        denominator = jnp.sqrt(level[:, None] ** 2 + span[:, None] ** 2 * x * y)
        return numerator, denominator

    return graded_residual(panels, pieces, 128, jnp)


def _reference(radius, level, radial_offset):
    """Integrate the exact integral and its radial-offset derivative at 50 digits."""
    with mp.workdps(50):
        r, u, o = (mp.mpf(str(value)) for value in (radius, level, radial_offset))
        limit = mp.pi / 2
        near = [mp.mpf(10) ** exponent for exponent in range(-15, 0)]
        points = [mp.mpf(0), *near, limit / 2]
        points += [limit - point for point in reversed(near)]
        points += [limit]

        def terms(angle):
            sine, cosine = mp.sin(angle), mp.cos(angle)
            numerator = o + 2 * r * cosine**2
            denominator_squared = u**2 + 4 * r**2 * sine**2 * cosine**2
            return numerator, denominator_squared

        value = mp.quad(
            lambda angle: mp.asinh(terms(angle)[0] / mp.sqrt(terms(angle)[1])),
            points,
        )
        derivative = mp.quad(
            lambda angle: 1 / mp.sqrt(terms(angle)[0] ** 2 + terms(angle)[1]),
            points,
        )
    return float(value), float(derivative)


def test_near_corner_layer_value_and_derivative():
    radius, level, radial_offset = 1.2, 1e-12, 1e-12
    r = jnp.asarray([radius])
    u = jnp.asarray([level])
    o = jnp.asarray([radial_offset])
    value, derivative = jax.jvp(
        lambda offset: _residual(r, u, offset), (o,), (jnp.ones_like(o),)
    )
    exact_value, exact_derivative = _reference(radius, level, radial_offset)
    value_error = abs(float(value[0]) - exact_value) / abs(exact_value)
    derivative_error = abs(float(derivative[0]) - exact_derivative) / abs(
        exact_derivative
    )
    print(
        f"CORNER offset={radial_offset:.0e} value_relative={value_error:.3e} "
        f"derivative_relative={derivative_error:.3e}"
    )
    assert np.isfinite(value_error) and value_error <= 1e-9
    assert np.isfinite(derivative_error) and derivative_error <= 1e-9
