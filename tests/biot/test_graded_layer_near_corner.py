"""Resolve the logarithmic end layer of a ring corner integral."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

import nova.biot.gradedresidual as gradedresidual  # noqa: E402
from tangent_identity import load_base_module  # noqa: E402

BASE_GRADED = load_base_module(
    "nova/biot/gradedresidual.py", "646b68b8e9481178e41cc2059c8a9a534fa9b9b3"
)


def _residual(radius, level, radial_offset, module=gradedresidual):
    span = 2.0 * radius
    zero = jnp.zeros_like(radial_offset)
    limit = jnp.full_like(radial_offset, gradedresidual.QUARTER)
    panels = (
        (jnp.abs(level), radial_offset + span, span, zero, limit),
        (jnp.abs(level), radial_offset, span, zero, limit),
    )

    def pieces(x, y):
        numerator = radial_offset[:, None] + span[:, None] * y
        denominator = jnp.sqrt(level[:, None] ** 2 + span[:, None] ** 2 * x * y)
        return numerator, denominator

    return module.graded_residual(panels, pieces, 128, jnp)


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
    return mp.nstr(value, 50), mp.nstr(derivative, 50)


def test_near_corner_layer_value_and_derivative():
    radius, level = 1.2, 1e-12
    offsets = [0.0]
    offsets.extend(
        sign * 10.0**exponent for exponent in range(-12, -2) for sign in (-1.0, 1.0)
    )
    offsets.extend((-1e-2, 1e-2))
    r = jnp.full(len(offsets), radius)
    u = jnp.full(len(offsets), level)
    o = jnp.asarray(offsets)
    value, derivative = jax.jvp(
        lambda offset: _residual(r, u, offset), (o,), (jnp.ones_like(o),)
    )
    old = np.asarray(_residual(r, u, o, BASE_GRADED))
    value, derivative = np.asarray(value), np.asarray(derivative)
    by_decade = {}
    less_accurate = []
    outside = []
    for index, offset in enumerate(offsets):
        with mp.workdps(50):
            exact_value, exact_derivative = (
                mp.mpf(number) for number in _reference(radius, level, offset)
            )
            value_error = float(
                abs(mp.mpf(float(value[index])) - exact_value) / abs(exact_value)
            )
            derivative_error = float(
                abs(mp.mpf(float(derivative[index])) - exact_derivative)
                / abs(exact_derivative)
            )
            base_error = float(
                abs(mp.mpf(float(old[index])) - exact_value) / abs(exact_value)
            )
        decade = "exact" if offset == 0.0 else f"{abs(offset):.0e}"
        by_decade.setdefault(decade, []).append(
            (value_error, derivative_error, base_error)
        )
        if abs(offset) > 1e-3:
            outside_error = abs(value[index] - old[index]) / abs(old[index])
            outside.append((offset, outside_error))
        elif value_error > base_error:
            less_accurate.append((offset, value_error, base_error))
    maxima = {
        decade: np.max(np.asarray(errors), axis=0)
        for decade, errors in by_decade.items()
    }
    for decade, (value_error, derivative_error, base_error) in maxima.items():
        print(
            f"CORNER decade={decade} value_relative_max={value_error:.3e} "
            f"derivative_relative_max={derivative_error:.3e} "
            f"base_value_relative_max={base_error:.3e}"
        )
    print(f"CORNER less_accurate={less_accurate} outside={outside}")
    assert all(
        np.isfinite(value_error) and value_error <= 1e-9
        for value_error, _, _ in maxima.values()
    )
    assert all(
        np.isfinite(derivative_error) and derivative_error <= 1e-9
        for _, derivative_error, _ in maxima.values()
    )
    assert not less_accurate
    assert all(error <= 1e-13 for _, error in outside)
