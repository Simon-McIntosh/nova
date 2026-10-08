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
from functools import lru_cache
import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

import nova.biot.polygonanalytic as polygonanalytic  # noqa: E402
import nova.biot.momentchannel as momentchannel  # noqa: E402
from tangent_identity import (  # noqa: E402
    EXACT_TOLERANCE,
    SAMPLES,
    compile_ratio,
    identity_row,
    load_base_module,
    truncation_active,
    worst,
    finite_fraction,
)
from test_vertex_term_tangents import _normwise  # noqa: E402

BASE_REVISION = "716d859f0b501797e28849ec8878f0243de80cb1"
BASE = load_base_module("nova/biot/polygonanalytic.py", BASE_REVISION)
BASE_CHANNEL = load_base_module("nova/biot/momentchannel.py", BASE_REVISION)
NODES = 128
SCAN_TOLERANCE = 1e-9
CONSTRUCTION_KEYS = (
    "radius",
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


def _construction_domain(rng):
    """Return the target radius, height and edge endpoints of the jet's domain.

    The target radius runs from a centimetre to twenty metres and the height
    over a forty metre span; the endpoints span the same radius band over that
    height, so the slope is finite and the extended line's offset stays bounded.
    A block of vertical edges is included -- the slope vanishes there -- and a
    blocks beside each endpoint and the edge interior, down to picometres.
    """
    radius = 10.0 ** rng.uniform(-2.0, 1.3, SAMPLES)
    height = rng.uniform(-20.0, 20.0, SAMPLES)
    edge_r = 10.0 ** rng.uniform(-2.0, 1.3, (2, SAMPLES))
    edge_z = rng.uniform(-20.0, 20.0, (2, SAMPLES))
    vertical = slice(5000, 5500)
    edge_r[1, vertical] = edge_r[0, vertical]
    edge_z[1, vertical] = edge_z[0, vertical] + 1.0
    for start, fraction in ((6000, 0.0), (7000, 1.0), (8000, 0.5)):
        selected = slice(start, start + 1000)
        radius[selected] = (
            (1.0 - fraction) * edge_r[0, selected]
            + fraction * edge_r[1, selected]
            + 10.0 ** rng.uniform(-12.0, -5.0, 1000)
        )
        height[selected] = (
            (1.0 - fraction) * edge_z[0, selected]
            + fraction * edge_z[1, selected]
            + 10.0 ** rng.uniform(-12.0, -5.0, 1000)
        )
    return radius, height, (edge_r[0], edge_z[0], edge_r[1], edge_z[1])


def _cases():
    rng = np.random.default_rng(20261008)
    r, z, edge = _construction_domain(rng)
    primals = (jnp.asarray(r), jnp.asarray(z), tuple(jnp.asarray(v) for v in edge))
    tangents = jax.tree.map(lambda v: jnp.asarray(rng.normal(size=v.shape)), primals)

    def construction(p, t):
        edge, d_edge = polygonanalytic._edge_tangent(*p, *t, NODES, xp=jnp)
        return _pack(edge, CONSTRUCTION_KEYS), tuple(
            d_edge[k] for k in CONSTRUCTION_KEYS
        )

    cases = {
        "edge_construction": (
            construction,
            lambda *p: _pack(BASE._Edge(*p, NODES, xp=jnp), CONSTRUCTION_KEYS),
            primals,
            tangents,
        )
    }
    extended = (
        *primals,
        jnp.asarray(rng.uniform(0.1, 10.0, SAMPLES)),
        jnp.asarray(rng.normal(size=SAMPLES)),
    )
    d_extended = (
        *tangents,
        jnp.asarray(rng.normal(size=SAMPLES)),
        jnp.asarray(rng.normal(size=SAMPLES)),
    )

    def endpoint(p, t):
        edge, d_edge = polygonanalytic._edge_tangent(*p[:3], *t[:3], NODES, xp=jnp)
        vertex, d_vertex = polygonanalytic._vertex_tangent(
            *p[:2], *p[2][:2], *t[:2], *t[2][:2], NODES, residual=True, xp=jnp
        )
        return edge, d_edge, vertex, d_vertex

    def base_endpoint(p):
        vertex = BASE._Vertex(*p[:2], *p[2][:2], NODES, residual=True, xp=jnp)
        vertex.channel.__class__ = BASE_CHANNEL.Channel
        return BASE._Edge(*p[:3], NODES, xp=jnp), vertex

    def make_case(name):
        def tangent(p, t):
            edge, d_edge, vertex, d_vertex = endpoint(p, t)
            if name == "second_residual":
                return polygonanalytic._second_residual_tangent(
                    edge, d_edge, vertex, d_vertex
                )
            if name == "edge_integrands":
                return polygonanalytic._edge_integrands_tangent(
                    edge, d_edge, vertex, d_vertex
                )
            if name == "flux_line_moments":
                return polygonanalytic._edge_flux_line_moments_tangent(
                    edge, d_edge, vertex, d_vertex, p[3], t[3], p[4], t[4]
                )
            if name == "flux_and_moment_terms":
                return polygonanalytic._edge_flux_and_moment_terms_tangent(
                    edge, d_edge, vertex, d_vertex, p[3], t[3], p[4], t[4]
                )
            if name == "edge_terms":
                return polygonanalytic._edge_terms_tangent(
                    *p[:3], *t[:3], 0, NODES, xp=jnp
                )
            return momentchannel._channel_split_tangent(
                vertex.channel,
                edge.plane_squared,
                d_edge["plane_squared"],
                vertex.moments,
                d_vertex["moments"],
                vertex.parameter,
                d_vertex["parameter"],
                vertex.parameter_complement,
                d_vertex["parameter_complement"],
                jnp,
            )

        def primal(*p):
            edge, vertex = base_endpoint(p)
            if name == "second_residual":
                return edge._second_residual(vertex)
            if name == "edge_integrands":
                return edge.terms(vertex)
            if name == "flux_line_moments":
                return edge.flux_line_moments(vertex, p[3], p[4])
            if name == "flux_and_moment_terms":
                return edge.flux_and_moment_terms(vertex, p[3], p[4])
            if name == "edge_terms":
                return tuple(
                    a + b
                    for a, b in zip(
                        edge.terms(vertex), vertex.arsinh_terms(), strict=True
                    )
                )
            return vertex.channel.split(edge.plane_squared)

        return tangent, primal, extended, d_extended

    for name in (
        "second_residual",
        "channel_split",
        "edge_integrands",
        "flux_line_moments",
        "flux_and_moment_terms",
        "edge_terms",
    ):
        cases[name] = make_case(name)
    return cases


CASES = _cases()


def _truncated_product(left, d_left, right, d_right):
    """Drop the right factor's product-rule term."""
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


def _identity(name):
    tangent, primal, primals, tangents = CASES[name]
    got = jax.jit(lambda p, t: tangent(p, t))(primals, tangents)
    expected = jax.jit(lambda p, t: jax.jvp(primal, p, t))(primals, tangents)
    own, base = jax.jit(lambda p, t: (tangent(p, t)[0], primal(*p)))(primals, tangents)
    for left, right in zip(jax.tree.leaves(own), jax.tree.leaves(base), strict=True):
        left, right = np.asarray(left), np.asarray(right)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes(), f"{name}: primal bits differ"
    return got, expected, worst(own, base)


@pytest.mark.parametrize("name", list(CASES))
def test_tangent_matches_base_jvp(name):
    if truncation_active():
        with _truncated():
            got, expected, primal_error = _identity(name)
    else:
        got, expected, primal_error = _identity(name)
    bound = (
        EXACT_TOLERANCE
        if name in {"edge_construction", "second_residual"}
        else SCAN_TOLERANCE
    )
    errors = _normwise(got[1], expected[1])
    covered = float(np.mean(errors <= bound))
    finite = finite_fraction(expected[1])
    magnitude = float(
        np.median(
            np.abs(np.concatenate([np.ravel(v) for v in jax.tree.leaves(expected[1])]))
        )
    )
    identity_row(
        name,
        primal_error,
        float(errors.max()),
        bound,
        extra=(
            f"tangent_max_elementwise_relative={worst(got[1], expected[1]):.3e} "
            f"covered_fraction={covered:.4f} base_jvp_finite_fraction={finite:.4f} "
            f"base_tangent_median_abs={magnitude:.3e} samples={SAMPLES}"
        ),
    )
    assert finite == 1.0
    assert covered == 1.0


@pytest.mark.parametrize("name", list(CASES))
def test_truncated_tangent_fails_identity(name):
    if truncation_active():
        pytest.skip("the declared control is applied to the identity rows")
    with _truncated():
        got, expected, _ = _identity(name)
    error = float(_normwise(got[1], expected[1]).max())
    print(f"TRUNCATION {name} max_relative={error:.3e}")
    assert error > SCAN_TOLERANCE


def compile_arm(name, arm):
    tangent, primal, _, _ = CASES[name]
    if arm == "primal":
        return lambda p, t: primal(*p)
    if arm == "tangent":
        return lambda p, t: tangent(p, t)
    return lambda p, t: jax.jvp(primal, p, t)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CASES))
def test_tangent_compiles_within_bound(name):
    assert compile_ratio("test_edge_term_tangents", name) <= 3.0


_EDGE_REFERENCE_PROBE = r"""
import json, sys
import mpmath as mp
mp.mp.dps = 50

def reference(record):
    values, directions = [[mp.mpf(v) for v in row] for row in record]
    def geometry(e):
        r,z,ra,za,rb,zb = [a + e*b for a,b in zip(values,directions)]
        b = (rb-ra)/(zb-za)
        w = ((ra-r)*(zb-z)-(rb-r)*(za-z))/(zb-za)
        return r,za-z,ra-r,b,w
    r,u,o,b,w = geometry(mp.mpf(0))
    dr,du,do,db,dw = [mp.diff(lambda e: geometry(e)[k],0) for k in range(5)]
    def integrand(a, derivative, geometry_value=None):
        rr,uu,oo,bb,ww = geometry_value or (r,u,o,b,w)
        y = mp.sin(a)**2
        x = 1-y
        n = uu+bb*oo+2*bb*rr*y
        q = ww+2*rr*y
        ws = q*q+4*(1+bb*bb)*rr*rr*x*y
        if not derivative:
            return mp.asinh(n/mp.sqrt(ws))
        dn = du+db*o+b*do+2*(db*r+b*dr)*y
        dq = dw+2*dr*y
        dws = 2*q*dq+4*(2*b*db*r*r+(1+b*b)*2*r*dr)*x*y
        return (dn-n*dws/(2*ws))/mp.sqrt(ws+n*n)
    width = abs(w)/(2*mp.sqrt(1+b*b)*r)
    split = sorted(set([mp.mpf(0),mp.pi/2,mp.pi/4,mp.mpf('.01'),mp.mpf('.1'),
        *[min(mp.pi/4,width*factor) for factor in
          [mp.mpf('.01'),mp.mpf('.1'),mp.mpf(1),mp.mpf(10),mp.mpf(100),mp.mpf(1000)]]]))
    value = mp.quad(lambda a:integrand(a,False),split)
    derivative = mp.quad(lambda a:integrand(a,True),split)
    # A one-sided difference stays on the geometry's selected smooth branch.
    step = mp.mpf('1e-25')
    displaced = mp.quad(lambda a:integrand(a,False,geometry(step)),split)
    difference = (displaced-value)/step
    error = abs(difference-derivative)/max(abs(derivative),mp.mpf('1e-100'))
    return float(value),float(derivative),float(error)
print(json.dumps([reference(row) for row in json.load(sys.stdin)]))
"""


@lru_cache(maxsize=1)
def _near_edge_reference_rows():
    """Audit smooth controls and both ends against the 50-digit exact integral.

    The fixed audit includes the CPU identity's largest discrepancy as well as
    targets near both endpoints and the extended line. Every full-domain sample
    remains in the base-JVP identity row; this independent reference measures
    where reproducing the program's derivative differs from the integral.
    """
    indices = np.asarray(
        [
            0,
            1,
            2,
            6000,
            6001,
            6100,
            6204,
            6500,
            6999,
            7000,
            7001,
            7500,
            7999,
            8000,
            8500,
            8999,
        ]
    )
    got, expected, _ = _identity("second_residual")
    primals, tangents = CASES["second_residual"][-2:]
    records = []
    for index in indices:
        records.append(
            [
                [
                    float(tree[0][index]),
                    float(tree[1][index]),
                    *[float(v[index]) for v in tree[2]],
                ]
                for tree in (primals, tangents)
            ]
        )
    result = subprocess.run(
        [sys.executable, "-I", "-c", _EDGE_REFERENCE_PROBE],
        input=json.dumps(records),
        capture_output=True,
        text=True,
        check=True,
    )
    reference = np.asarray(json.loads(result.stdout))
    value, tangent = reference[:, 0], reference[:, 1]
    primal = np.abs((np.asarray(expected[0])[indices] - value) / value)
    hand = np.abs((np.asarray(got[1])[indices] - tangent) / tangent)
    base = np.abs((np.asarray(expected[1])[indices] - tangent) / tangent)
    print(
        f"NEAR_EDGE second_residual samples={indices.size} "
        f"base_primal_vs_reference_max={primal.max():.3e} "
        f"hand_vs_reference_max={hand.max():.3e} "
        f"base_jvp_vs_reference_max={base.max():.3e} "
        f"smooth_control_tangent_max={hand[:3].max():.3e} "
        f"reference_one_sided_difference_max={reference[:, 2].max():.3e}"
    )
    return primal, hand, base, reference[:, 2]


def test_near_edge_reference_is_the_exact_integral():
    primal, hand, _, difference = _near_edge_reference_rows()
    assert primal.max() <= 1e-10
    assert hand[:3].max() <= EXACT_TOLERANCE
    assert difference.max() <= 1e-15


@pytest.mark.xfail(
    strict=True,
    reason="the differentiated graded edge quadrature near endpoints leaves "
    "the exact integral's tangent (measured maximum 5.160e-7 relative); the "
    "hand rule preserves the base program's derivative",
)
def test_near_edge_tangent_meets_reference():
    _, hand, _, _ = _near_edge_reference_rows()
    assert hand.max() <= SCAN_TOLERANCE
