"""Hand-written tangents of the polygon-analytic edge terms.

Each private tangent in :mod:`nova.biot.polygonanalytic` that the Biot point jet
differentiates is checked three ways: against ``jax.jvp`` of the base revision's
own function on ten thousand inputs spanning the edge domain; with its product
or quotient rule truncated by one term, which must fail that bound; and by a
cold compile in a fresh, cache-disabled process against the primal.  The primal
half each tangent returns is asserted bit-identical to the base revision's.

``NOVA_TANGENT_TRUNCATION=1`` applies the truncation to the identity rows
themselves, which is the declared mutation those rows must fail against.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

import nova.biot.polygonanalytic as polygonanalytic  # noqa: E402
from tangent_identity import (  # noqa: E402
    EXACT_TOLERANCE,
    SAMPLES,
    compile_ratio,
    identity_row,
    load_base_module,
    truncation_active,
    worst,
)

BASE_REVISION = "72112421cb157599b3e3e97b31f3d5f6246735d4"
BASE = load_base_module("nova/biot/polygonanalytic.py", BASE_REVISION)

CONSTRUCTION_KEYS = (
    "slope",
    "squared_slope",
    "axial_slope",
    "plane_offset",
    "plane_radius_value",
    "plane_radius",
    "plane_squared",
    "edge_slope",
    "edge_slope_over_radius",
)


def _pack(part, keys):
    return tuple(getattr(part, key) for key in keys)


def _construction_primal(r, z, edge):
    return _pack(BASE._Edge(r, z, edge, None, xp=jnp), CONSTRUCTION_KEYS)


def _construction_tangent(r, z, edge, d_r, d_z, d_edge):
    _, tangent = polygonanalytic._edge_tangent(
        r, z, edge, d_r, d_z, d_edge, None, xp=jnp
    )
    return tuple(tangent[key] for key in CONSTRUCTION_KEYS)


def _tangents(rng, *arrays):
    return tuple(
        jnp.asarray(rng.normal(size=np.shape(np.asarray(array)))) for array in arrays
    )


def _construction_domain(rng):
    """Return the target radius, height and edge endpoints of the jet's domain.

    The target radius runs from a centimetre to twenty metres and the height
    over a forty metre span; the endpoints span the same radius band over that
    height, so the slope is finite and the extended line's offset stays bounded.
    A block of vertical edges is included -- the slope vanishes there -- and a
    block of targets sitting on an endpoint, where the offset vanishes.
    """
    radius = 10.0 ** rng.uniform(-2.0, 1.3, SAMPLES)
    height = rng.uniform(-20.0, 20.0, SAMPLES)
    edge_r = 10.0 ** rng.uniform(-2.0, 1.3, (2, SAMPLES))
    edge_z = rng.uniform(-20.0, 20.0, (2, SAMPLES))
    vertical = slice(5000, 5500)
    edge_r[1, vertical] = edge_r[0, vertical]
    edge_z[1, vertical] = edge_z[0, vertical] + 1.0
    return radius, height, (edge_r[0], edge_z[0], edge_r[1], edge_z[1])


def _construction_cases():
    rng = np.random.default_rng(20261008)
    radius, height, edge = _construction_domain(rng)
    primals = (
        jnp.asarray(radius),
        jnp.asarray(height),
        tuple(jnp.asarray(value) for value in edge),
    )
    tangents = (
        jnp.asarray(rng.normal(size=SAMPLES)),
        jnp.asarray(rng.normal(size=SAMPLES)),
        tuple(jnp.asarray(rng.normal(size=SAMPLES)) for _ in range(4)),
    )
    return {
        "edge_construction": (
            _construction_tangent,
            _construction_primal,
            primals,
            tangents,
        )
    }


CASES = _construction_cases()


def _truncated_product(left, d_left, right, d_right):
    """The product rule's tangent with the right factor's term dropped."""
    if d_left is None:
        return None if d_right is None else 0.0 * d_right
    return d_left * right


@contextmanager
def _truncated():
    original = polygonanalytic._product_tangent
    polygonanalytic._product_tangent = _truncated_product
    jax.clear_caches()
    try:
        yield
    finally:
        polygonanalytic._product_tangent = original
        jax.clear_caches()


def _identity(name, tangent, primal, primals, tangents):
    got = jax.jit(lambda p, t: tangent(*p, *t))(primals, tangents)
    expected = jax.jit(lambda p, t: jax.jvp(primal, p, t)[1])(primals, tangents)
    return got, expected


def _primal_identity(primal, primals):
    got = jax.jit(primal)(*primals)
    expected = jax.jit(_base_primal)(*primals)
    leaves_got = jax.tree.leaves(got)
    leaves_expected = jax.tree.leaves(expected)
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(leaves_got, leaves_expected, strict=True)
    )


def _base_primal(radius, height, edge):
    return _pack(BASE._Edge(radius, height, edge, None, xp=jnp), CONSTRUCTION_KEYS)


def test_primal_bit_identical():
    for name, (_, primal, primals, _) in CASES.items():
        assert _primal_identity(primal, primals), name


def test_tangent_matches_base_jvp():
    for name, (tangent, primal, primals, tangents) in CASES.items():
        if truncation_active():
            with _truncated():
                got, expected = _identity(name, tangent, primal, primals, tangents)
        else:
            got, expected = _identity(name, tangent, primal, primals, tangents)
        identity_row(name, 0.0, worst(got, expected), EXACT_TOLERANCE)


def test_truncated_tangent_fails_identity():
    if truncation_active():
        pytest.skip("the declared control is applied to the identity rows")
    for name, (tangent, primal, primals, tangents) in CASES.items():
        with _truncated():
            got, expected = _identity(name, tangent, primal, primals, tangents)
        error = worst(got, expected)
        assert error > EXACT_TOLERANCE, f"{name} truncated tangent matched: {error}"
        assert not jax.tree.all(
            jax.tree.map(
                lambda a, b: np.all(
                    np.abs(np.asarray(a) - np.asarray(b))
                    <= EXACT_TOLERANCE * (1.0 + np.abs(np.asarray(b)))
                ),
                got,
                expected,
            )
        )
        print(f"TRUNCATION {name} max_relative={error:.3e}")


def compile_arm(name, arm):
    tangent, primal, _, _ = CASES[name]
    if arm == "primal":
        return lambda p, t: _primal_only(p)
    if arm == "tangent":
        return lambda p, t: tangent(*p, *t)
    return lambda p, t: jax.jvp(primal, p, t)[1]


def _primal_only(primals):
    radius, height, edge = primals
    return _construction_primal(radius, height, edge)


def test_tangent_compiles_within_bound():
    ratio = compile_ratio("test_edge_term_tangents", "edge_construction", repeats=3)
    assert ratio <= 3.0, ratio
