"""Identity, mutation, and compile checks for the stable edge logarithm tangent."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

from nova.biot import polygonanalytic  # noqa: E402
from tangent_identity import (  # noqa: E402
    EXACT_TOLERANCE,
    SAMPLES,
    compile_ratio,
    finite_fraction,
    identity_row,
    load_base_module,
    relative_error,
    truncation_active,
    worst,
)

BASE = load_base_module(
    "nova/biot/polygonanalytic.py", "b57dfb24855ed7b650e25a7490c5a7d32d61d731"
)


def _cases():
    rng = np.random.default_rng(20261008)
    length = 10.0 ** rng.uniform(-2.0, 1.3, SAMPLES)
    along_a = rng.uniform(-20.0, 20.0, SAMPLES)
    along_a[3000:6000] = 10.0 ** rng.uniform(-2.0, 1.3, 3000)
    along_a[6000:8000] = -length[6000:8000] - 10.0 ** rng.uniform(-2.0, 1.3, 2000)
    along_a[8000:9000] = -rng.uniform(0.1, 0.9, 1000) * length[8000:9000]
    along_b = along_a + length
    gap = 10.0 ** rng.uniform(-10.0, 1.3, SAMPLES)
    gap[3000:8000] = np.minimum(
        np.abs(along_a[3000:8000]), np.abs(along_b[3000:8000])
    ) * 10.0 ** rng.uniform(-12.0, -8.0, 5000)
    gap[9500:] = 0.0
    active = np.ones(SAMPLES, dtype=bool)
    active[9000:9500] = False
    primals = tuple(jnp.asarray(v) for v in (along_a, along_b, gap, length))
    tangents = tuple(jnp.asarray(rng.normal(size=SAMPLES)) for _ in primals)
    return primals, tangents, jnp.asarray(active)


PRIMALS, TANGENTS, ACTIVE = _cases()
CASES = {"near_collinear": (None, None, PRIMALS, TANGENTS)}


def _primal(*values):
    return BASE._near_collinear_arsinh_difference(jnp, *values, ACTIVE)


def _tangent(primals, tangents):
    return polygonanalytic._near_collinear_arsinh_difference_tangent(
        jnp, *primals, ACTIVE, *tangents
    )


def _identity():
    got = jax.jit(_tangent)(PRIMALS, TANGENTS)
    expected = jax.jit(lambda p, t: jax.jvp(_primal, p, t))(PRIMALS, TANGENTS)
    assert np.asarray(got[0]).tobytes() == np.asarray(expected[0]).tobytes()
    return got, expected


def test_tangent_matches_base_jvp(monkeypatch):
    if truncation_active():
        monkeypatch.setattr(
            polygonanalytic,
            "_product_tangent",
            lambda left, d_left, right, d_right: d_left * right,
        )
        jax.clear_caches()
    got, expected = _identity()
    error = worst(got[1], expected[1])
    covered = float(np.mean(relative_error(got[1], expected[1]) <= EXACT_TOLERANCE))
    finite = finite_fraction(expected[1])
    identity_row(
        "near_collinear",
        0.0,
        error,
        EXACT_TOLERANCE,
        extra=f"covered_fraction={covered:.4f} "
        f"base_jvp_finite_fraction={finite:.4f} samples={SAMPLES}",
    )
    assert finite == 1.0
    assert covered == 1.0


def test_truncated_tangent_fails_identity(monkeypatch):
    if truncation_active():
        pytest.skip("the declared mutation is applied to the identity row")
    monkeypatch.setattr(
        polygonanalytic,
        "_product_tangent",
        lambda left, d_left, right, d_right: d_left * right,
    )
    jax.clear_caches()
    try:
        got, expected = _identity()
        assert worst(got[1], expected[1]) > EXACT_TOLERANCE
    finally:
        monkeypatch.undo()
        jax.clear_caches()


def test_tangent_against_extended_precision():
    """Check smooth and selected branch boundaries against a 50-digit derivative."""
    hand, base = _identity()
    indices = (
        0,
        100,
        1000,
        2999,
        3000,
        4000,
        5999,
        6000,
        7000,
        7999,
        8000,
        8500,
        8999,
        9000,
        9500,
        9999,
    )
    hand_errors = []
    base_errors = []
    value_errors = []
    with mp.workdps(50):
        for index in indices:
            values = [mp.mpf(float(value[index])) for value in PRIMALS]
            directions = [mp.mpf(float(value[index])) for value in TANGENTS]
            a, b, gap, length = values
            active = bool(ACTIVE[index])
            near = (
                active
                and gap > 0
                and a * b > 0
                and gap
                <= (
                    mp.mpf(float(polygonanalytic._ARCSINH_DIFFERENCE_SWITCH))
                    * min(abs(a), abs(b))
                )
            )

            def branch(t):
                aa, bb, gg, ll = [
                    value + t * direction
                    for value, direction in zip(values, directions, strict=True)
                ]
                if not active or gap <= 0:
                    return mp.mpf(0)
                if near:
                    divisor = bb * mp.hypot(aa, gg) + aa * mp.hypot(bb, gg)
                    return mp.asinh(ll * (aa + bb) / divisor)
                return mp.asinh(bb / gg) - mp.asinh(aa / gg)

            reference_value = float(branch(mp.mpf(0)))
            reference_tangent = float(mp.diff(branch, mp.mpf(0)))
            scale = max(abs(reference_tangent), 1e-14)
            hand_errors.append(abs(float(hand[1][index]) - reference_tangent) / scale)
            base_errors.append(abs(float(base[1][index]) - reference_tangent) / scale)
            value_errors.append(abs(float(base[0][index]) - reference_value))
    print(
        "REFERENCE near_collinear samples=16 digits=50 "
        f"primal_absolute_max={max(value_errors):.3e} "
        f"hand_relative_max={max(hand_errors):.3e} "
        f"base_jvp_relative_max={max(base_errors):.3e}"
    )
    assert max(hand_errors) <= 1e-9
    assert max(base_errors) <= 1e-9


def compile_arm(name, arm):
    assert name == "near_collinear"
    if arm == "primal":
        return lambda p, t: _primal(*p)
    if arm == "tangent":
        return _tangent
    return lambda p, t: jax.jvp(_primal, p, t)


@pytest.mark.slow
def test_tangent_compiles_within_bound():
    assert compile_ratio("test_near_collinear_tangent", "near_collinear") <= 3.0
