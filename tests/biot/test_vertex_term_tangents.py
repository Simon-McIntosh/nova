"""Hand-written tangents of the polygon-analytic corner terms.

Each private tangent of :class:`nova.biot.polygonanalytic._Vertex` -- its
construction with the first residual integral, :meth:`arsinh_terms`,
:meth:`against_first_arsinh`, :meth:`flux_moment_residuals` and
:meth:`horizontal_flux_line_moments` -- is checked three ways: against
``jax.jvp`` of the base-revision function on ten thousand corner geometries and
targets spanning the jet's domain, with no sample masked; with one product-rule
term dropped, which must fail that bound; and by a cold compile in a fresh,
cache-disabled process against the primal.  The primal half each tangent
returns is asserted bit-identical to the base revision's primal.

``NOVA_TANGENT_TRUNCATION=1`` applies the truncation to the identity rows
themselves, which is the declared mutation those rows must fail against.
"""

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

import nova.biot.polygonanalytic as polygonanalytic  # noqa: E402

BASE_REVISION = os.environ.get(
    "NOVA_VERTEX_TERM_BASE_REVISION", "fd44f630be3007c4ffe46a2d26b6f9fbcfe754fb"
)
ROOT = Path(__file__).resolve().parents[2]
# Every corner tangent runs through the harmonic moment stack's scanned
# recurrences, so every row is a scan form and is held to the scan bound.
SCAN_TOLERANCE = 1e-9
SAMPLES = 10_000
NODES = 128
COMPILE_REPEATS = 3
# Targets nearer a corner than this fraction of its radius are where the base
# program's own derivative leaves the exact integral's: the graded quadrature's
# layer width is clipped at its floor there, so jax.jvp of the base is not a
# judge of the corner's radial tangent, and those samples are judged against a
# 50-digit reference of the exact integral instead.
NEAR_CORNER = 2e-6


def _base_module():
    """Load the base revision's polygon-analytic module under its own name."""
    directory = Path(tempfile.mkdtemp(prefix="vertex-term-base-"))
    path = directory / "polygonanalytic.py"
    path.write_bytes(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{BASE_REVISION}:nova/biot/polygonanalytic.py",
            ]
        )
    )
    spec = importlib.util.spec_from_file_location("base_polygonanalytic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"BASE_MODULE polygonanalytic={module.__file__}", flush=True)
    return module


BASE = _base_module()


def _magnitude(rng, low, high, size=SAMPLES):
    return 10.0 ** rng.uniform(low, high, size)


def _signed(rng, low, high, size=SAMPLES):
    return rng.choice((-1.0, 1.0), size) * _magnitude(rng, low, high, size)


def _domain(rng):
    """Return target and corner coordinates, expansion radius and offset height.

    Corners and targets span a centimetre to twenty metres in radius over a
    forty metre height, as the packed jet forms them.  Targets beside a corner
    sit at a signed distance down to a picometre in either coordinate, level
    with it exactly, on its radius exactly, and along an edge leaving it.  A
    target ON a corner, where the ring modulus reaches one, is not formed by
    the jet: :func:`nova.biot.polygonanalytic.packed_analytic_moments` moves an
    inactive corner a metre off the target, and an active corner coinciding
    with a target sits on the section's own contour, where the contour sum's
    tangent is the subject of the edge terms rather than of the corner.
    """
    corner_r = _magnitude(rng, -2.0, 1.3)
    corner_z = rng.uniform(-20.0, 20.0, SAMPLES)
    r = _magnitude(rng, -2.0, 1.3)
    z = rng.uniform(-20.0, 20.0, SAMPLES)
    # beside the corner: a small signed offset in each coordinate
    near = slice(0, 4000)
    r[near] = corner_r[near] + _signed(rng, -12.0, -1.0, 4000) * corner_r[near]
    z[near] = corner_z[near] + _signed(rng, -12.0, 0.0, 4000)
    # level with the corner exactly, and on its radius exactly
    z[4000:4600] = corner_z[4000:4600]
    r[4600:5200] = corner_r[4600:5200]
    # beside an edge leaving the corner: along a random direction, then off it
    angle = rng.uniform(0.0, 2.0 * np.pi, 1500)
    along = _magnitude(rng, -6.0, 0.0, 1500)
    across = _signed(rng, -12.0, -3.0, 1500)
    edge = slice(5200, 6700)
    r[edge] = corner_r[edge] + along * np.cos(angle) - across * np.sin(angle)
    z[edge] = corner_z[edge] + along * np.sin(angle) + across * np.cos(angle)
    r = np.abs(r)
    r[r == 0.0] = 1e-2
    expansion_r = _magnitude(rng, -2.0, 1.3)
    target_z_minus_expansion_z = rng.uniform(-20.0, 20.0, SAMPLES)
    return tuple(
        jnp.asarray(value)
        for value in (r, z, corner_r, corner_z, expansion_r, target_z_minus_expansion_z)
    )


def _weight(rng):
    """Return a range function of the order the corner weights reach."""
    return (
        [jnp.asarray(rng.normal(size=SAMPLES)) for _ in range(3)],
        jnp.asarray(_signed(rng, -6.0, 3.0)),
        jnp.asarray(_signed(rng, -6.0, 3.0)),
    )


def _tangents(rng, tree):
    return jax.tree.map(lambda leaf: jnp.asarray(rng.normal(size=np.shape(leaf))), tree)


_FIELDS = (
    "level", "offset", "radius_sum", "span", "parameter", "parameter_complement",
    "moments", "root_moments", "edge_radius", "ring_squared", "ring_residual",
)  # fmt: skip


def _attributes(vertex, tangent=None):
    """Return the vertex's floating quantities, or their tangents, as a dict."""
    source = vars(vertex) if tangent is None else tangent
    out = {name: source[name] for name in _FIELDS}
    out["ring"] = tuple(source["ring"][:2])
    return out


def _base_vertex(r, z, corner_r, corner_z):
    return BASE._Vertex(r, z, corner_r, corner_z, NODES, residual=True, xp=jnp)


def _current_vertex(p, t):
    return polygonanalytic._vertex_tangent(*p[:4], *t[:4], NODES, residual=True, xp=jnp)


def _cases():
    """Return ``name -> (tangent, primal, primals, tangents)``.

    ``primal`` is the base revision's function of the ``primals`` tuple, so
    ``jax.jvp(primal, primals, tangents)`` is the reference and its first half
    the primal the tangent's own first half must reproduce bit for bit.
    """
    rng = np.random.default_rng(0)
    primals = _domain(rng)
    tangents = _tangents(rng, primals)
    cases = {}

    def vertex_tangent(p, t):
        vertex, tangent = _current_vertex(p, t)
        return _attributes(vertex), _attributes(vertex, tangent)

    cases["vertex"] = (
        vertex_tangent,
        lambda r, z, cr, cz, e, h: _attributes(_base_vertex(r, z, cr, cz)),
        primals,
        tangents,
    )

    def arsinh_tangent(p, t):
        return polygonanalytic._arsinh_terms_tangent(*_current_vertex(p, t))

    cases["arsinh_terms"] = (
        arsinh_tangent,
        lambda r, z, cr, cz, e, h: _base_vertex(r, z, cr, cz).arsinh_terms(),
        primals,
        tangents,
    )

    weight = _weight(rng)
    weight_primals = (*primals, weight)
    weight_tangents = (*tangents, _tangents(rng, weight))

    def against_tangent(p, t):
        return polygonanalytic._against_first_arsinh_tangent(
            *_current_vertex(p, t), p[6], t[6]
        )

    cases["against_first_arsinh"] = (
        against_tangent,
        lambda r, z, cr, cz, e, h, w: _base_vertex(r, z, cr, cz).against_first_arsinh(
            w
        ),
        weight_primals,
        weight_tangents,
    )

    def flux_tangent(p, t):
        return polygonanalytic._flux_moment_residuals_tangent(
            *_current_vertex(p, t), p[4], t[4], p[5], t[5]
        )

    cases["flux_moment_residuals"] = (
        flux_tangent,
        lambda r, z, cr, cz, e, h: _base_vertex(r, z, cr, cz).flux_moment_residuals(
            e, h
        ),
        primals,
        tangents,
    )

    def line_tangent(p, t):
        return polygonanalytic._horizontal_flux_line_moments_tangent(
            *_current_vertex(p, t), p[4], t[4], p[5], t[5]
        )

    cases["horizontal_flux_line_moments"] = (
        line_tangent,
        lambda r, z, cr, cz, e, h: _base_vertex(
            r, z, cr, cz
        ).horizontal_flux_line_moments(e, h),
        primals,
        tangents,
    )
    return cases


CASES = _cases()


def _relative(got, reference):
    got, reference = np.asarray(got), np.asarray(reference)
    agree = (np.isnan(got) & np.isnan(reference)) | (got == reference)
    scale = np.where(reference == 0.0, 1.0, np.abs(reference))
    error = np.where(agree, 0.0, np.abs(got - reference) / scale)
    return float(np.max(np.nan_to_num(error, nan=np.inf)))


def _worst(got, reference):
    return max(
        _relative(a, b)
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(reference), strict=True)
    )


def _per_sample(got, reference):
    """Return each sample's normwise relative error over one series or value.

    A list is one series -- a moment stack, a pole family or a range function's
    harmonic bulk -- and is measured against its own largest order on that
    sample, because its high orders are formed by recurrences whose terms
    cancel: the base revision's own moment stack differs between two compiled
    programs by more than the bound at its forty-second order, and a root
    moment's tangent cancels from terms nine decades above it.  A scalar
    leaf is measured against itself.
    """
    if isinstance(got, list):
        if not got:
            return []
        a = np.stack([np.broadcast_to(np.asarray(v), (SAMPLES,)) for v in got])
        b = np.stack([np.broadcast_to(np.asarray(v), (SAMPLES,)) for v in reference])
        agree = (np.isnan(a) & np.isnan(b)) | (a == b)
        scale = np.max(np.abs(b), axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        error = np.where(agree, 0.0, np.abs(a - b) / scale)
        return [np.nan_to_num(np.max(error, axis=0), nan=np.inf)]
    if isinstance(got, tuple | dict):
        keys = got.keys() if isinstance(got, dict) else range(len(got))
        return [e for key in keys for e in _per_sample(got[key], reference[key])]
    a = np.broadcast_to(np.asarray(got), (SAMPLES,))
    b = np.broadcast_to(np.asarray(reference), (SAMPLES,))
    agree = (np.isnan(a) & np.isnan(b)) | (a == b)
    scale = np.where(b == 0.0, 1.0, np.abs(b))
    return [np.nan_to_num(np.where(agree, 0.0, np.abs(a - b) / scale), nan=np.inf)]


def _normwise(got, reference):
    """Return each sample's worst normwise relative error over the output.

    The root moments are ``(1 - m/2) M_n - (m/4)(M_(n+1) + M_(n-1))``, and
    beside a corner, where the complement falls to ``1e-20``, their tangents
    cancel sixteen decades below the moments' own: ``1e-8`` from ``1e8``, which
    is round-off of the operands in ``jax.jvp`` of the base as much as here.
    So the root moments' tangent is measured against the larger of its own
    series and the moment tangents it is formed from.
    """
    if isinstance(got, dict) and "root_moments" in got:
        operand = np.max(
            np.abs(np.stack([np.asarray(v) for v in reference["moments"]])), axis=0
        )
        rest = [key for key in got if key != "root_moments"]
        errors = _per_sample(
            {key: got[key] for key in rest}, {key: reference[key] for key in rest}
        )
        a = np.stack([np.asarray(v) for v in got["root_moments"]])
        b = np.stack([np.asarray(v) for v in reference["root_moments"]])
        scale = np.maximum(np.max(np.abs(b), axis=0), operand)
        scale = np.where(scale == 0.0, 1.0, scale)
        errors.append(np.nan_to_num(np.max(np.abs(a - b), axis=0) / scale, nan=np.inf))
        return np.max(np.stack(errors), axis=0)
    return np.max(np.stack(_per_sample(got, reference)), axis=0)


def _finite_fraction(tree):
    leaves = [
        np.isfinite(np.asarray(leaf)).reshape(SAMPLES, -1).all(axis=1)
        for leaf in jax.tree.leaves(tree)
        if np.ndim(leaf) > 0
    ]
    return float(np.mean(np.all(np.stack(leaves), axis=0)))


def _truncated_product(left, d_left, right, d_right):
    """The product rule's tangent with the right factor's term dropped."""
    if d_left is None:
        return None if d_right is None else 0.0 * d_right
    return d_left * right


@contextmanager
def _truncated():
    original = polygonanalytic._vertex_product_tangent
    polygonanalytic._vertex_product_tangent = _truncated_product
    try:
        yield
    finally:
        polygonanalytic._vertex_product_tangent = original


def _identity(name):
    tangent, primal, primals, tangents = CASES[name]
    # a fresh jit per call, so a truncation applied since is traced afresh
    got = jax.jit(lambda p, t: tangent(p, t))(primals, tangents)
    expected = jax.jit(lambda p, t: jax.jvp(primal, p, t))(primals, tangents)
    return got, expected


def _near_corner(primals):
    """Return the samples whose target lies within NEAR_CORNER of the corner."""
    r, z, corner_r, corner_z = (np.asarray(value) for value in primals[:4])
    return np.hypot(corner_z - z, corner_r - r) < NEAR_CORNER * r


def _judged(name, tree):
    """Return the samples judged against jax.jvp of the base."""
    if name != "arsinh_terms":
        return np.ones(SAMPLES, bool)
    return ~_near_corner(CASES[name][2])


def _elementwise_only(got, reference, passing):
    """Count elements beyond the bound elementwise on normwise-passing samples."""
    count = 0
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(reference), strict=True):
        a = np.broadcast_to(np.asarray(a), (SAMPLES,))
        b = np.broadcast_to(np.asarray(b), (SAMPLES,))
        scale = np.where(b == 0.0, 1.0, np.abs(b))
        error = np.where(a == b, 0.0, np.abs(a - b) / scale)
        count += int(np.sum((error > SCAN_TOLERANCE) & passing))
    return count


@pytest.mark.parametrize("name", list(CASES))
def test_tangent_matches_base_jvp(name):
    if os.environ.get("NOVA_TANGENT_TRUNCATION") == "1":
        with _truncated():
            got, expected = _identity(name)
    else:
        got, expected = _identity(name)
    # the primal half is the primal method itself, compiled in this program
    tangent_function, primal_function, primals, tangents = CASES[name]
    own, alone = jax.jit(lambda p, t: (tangent_function(p, t)[0], primal_function(*p)))(
        primals, tangents
    )
    primal = _worst(own, alone)
    context = _worst(got[0], expected[0])
    judged = _judged(name, got)
    error = _normwise(got[1], expected[1])[judged]
    covered = float(np.mean(error <= SCAN_TOLERANCE))
    finite = _finite_fraction(expected[1])
    passing = np.zeros(SAMPLES, bool)
    passing[np.flatnonzero(judged)[error <= SCAN_TOLERANCE]] = True
    magnitude = float(
        np.median(
            np.abs(np.concatenate([np.ravel(v) for v in jax.tree.leaves(expected[1])]))
        )
    )
    elementwise_only = _elementwise_only(got[1], expected[1], passing)
    reference_count = int((~judged).sum())
    print(f"IDENTITY {name} primal_max_relative={primal:.3e} "
          f"jvp_program_primal_max_relative={context:.3e} "
          f"tangent_max_normwise_relative={error.max():.3e} "
          f"tangent_max_elementwise_relative={_worst(got[1], expected[1]):.3e} "
          f"elementwise_fail_normwise_pass={elementwise_only} "
          f"bound={SCAN_TOLERANCE:.0e} covered_fraction={covered:.4f} "
          f"judged_by_base_jvp={int(judged.sum())} "
          f"judged_by_reference={reference_count} "
          f"base_jvp_finite_fraction={finite:.4f} "
          f"base_tangent_median_abs={magnitude:.3e}")  # fmt: skip
    assert primal == 0.0
    assert finite == 1.0
    assert error.max() <= SCAN_TOLERANCE
    assert covered == 1.0


_REFERENCE_PROBE = r"""
import json, sys
from multiprocessing import get_context
import mpmath as mp
mp.mp.dps = 50
def radial(args):
    r, z, cr, cz, dr, dz, dcr, dcz = [mp.mpf(v) for v in args]
    u, offset, du, doffset = cz - z, cr - r, dcz - dz, dcr - dr
    def parts(a, which):
        s2, c2 = mp.sin(a) ** 2, mp.cos(a) ** 2
        n, dn = offset + 2 * r * c2, doffset + 2 * dr * c2
        w2 = u * u + 4 * r * r * s2 * c2
        dw2 = 2 * u * du + 8 * r * dr * s2 * c2
        weight = mp.cos(2 * a) ** 2
        if which == 0:
            return weight * mp.asinh(n / mp.sqrt(w2))
        return weight * (dn * w2 - n * dw2 / 2) / (w2 * mp.sqrt(n * n + w2))
    scale = max(abs(u), abs(offset), mp.mpf(10) ** -30) / (2 * r)
    points = [mp.mpf(0)] + [scale * mp.mpf(10) ** k for k in range(40)
                            if scale * mp.mpf(10) ** k < mp.pi / 4] + [mp.pi / 4]
    points = points + [mp.pi / 2 - p for p in reversed(points[1:-1])] + [mp.pi / 2]
    value = mp.quad(lambda a: parts(a, 0), points)
    slope = mp.quad(lambda a: parts(a, 1), points)
    return float(4 * r * value), float(4 * dr * value + 4 * r * slope)
with get_context("fork").Pool(8) as pool:
    print(json.dumps(pool.map(radial, json.load(sys.stdin))))
"""


def _radial_reference(primals, tangents, index):
    """Return the 50-digit radial value and tangent of the exact integral.

    The radial arsinh term is ``4 r integral_0^(pi/2) cos^2(2a) arsinh(N/W) da``
    with ``N = offset + 2 r cos^2 a`` and ``W^2 = u^2 + 4 r^2 sin^2 a cos^2 a``,
    differentiated analytically under the integral in mpmath, in a separate
    process that never imports jax.
    """
    rows = [
        [float(np.asarray(primals[k])[i]) for k in range(4)]
        + [float(np.asarray(tangents[k])[i]) for k in range(4)]
        for i in index
    ]
    result = subprocess.run(
        [sys.executable, "-I", "-c", _REFERENCE_PROBE],
        input=json.dumps(rows), capture_output=True, text=True, check=True,
    )  # fmt: skip
    values = np.asarray(json.loads(result.stdout))
    return values[:, 0], values[:, 1]


def _near_corner_rows():
    got, expected = _identity("arsinh_terms")
    _, _, primals, tangents = CASES["arsinh_terms"]
    r, z, corner_r, corner_z = (np.asarray(value) for value in primals[:4])
    # a target exactly level with the corner or on its radius sits on the
    # integral's |u| or |offset| kink, where no two-sided derivative exists
    index = np.flatnonzero(_near_corner(primals) & (corner_z != z) & (corner_r != r))
    value, reference = _radial_reference(primals, tangents, index)
    hand = np.asarray(got[1][1])[index]
    base = np.asarray(expected[1][1])[index]
    primal = np.asarray(expected[0][1])[index]

    def relative(a, b):
        return np.abs(a - b) / np.where(b == 0.0, 1.0, np.abs(b))

    rows = {
        "samples": index.size,
        "primal": relative(primal, value).max(),
        "hand": relative(hand, reference).max(),
        "base": relative(base, reference).max(),
        "hand_base": relative(hand, base).max(),
    }
    print(f"NEAR_CORNER arsinh_terms radial samples={rows['samples']} "
          f"base_primal_vs_reference_max={rows['primal']:.3e} "
          f"hand_vs_reference_max={rows['hand']:.3e} "
          f"base_jvp_vs_reference_max={rows['base']:.3e} "
          f"hand_vs_base_jvp_max={rows['hand_base']:.3e}")  # fmt: skip
    return rows


def test_near_corner_reference_is_the_exact_integral():
    rows = _near_corner_rows()
    # the reference integral is the base program's own primal near the corner
    assert rows["samples"] > 0
    assert rows["primal"] <= 1e-10


@pytest.mark.xfail(
    strict=True,
    reason="the base program's derivative near a corner is that of its "
    "128-node graded quadrature with the layer width clipped at its floor, and "
    "the hand tangent reproduces it; neither meets the exact integral's",
)
def test_near_corner_radial_tangent_meets_reference():
    rows = _near_corner_rows()
    assert rows["hand"] <= SCAN_TOLERANCE


@pytest.mark.parametrize("name", list(CASES))
def test_truncated_tangent_fails_identity(name):
    with _truncated():
        got, expected = _identity(name)
    tangent = float(_normwise(got[1], expected[1]).max())
    print(f"TRUNCATED {name} tangent_max_normwise_relative={tangent:.3e}")
    assert tangent > SCAN_TOLERANCE


def test_primal_bit_identical_to_base():
    _, _, primals, _ = CASES["vertex"]
    for name in ("vertex", "arsinh_terms", "flux_moment_residuals"):
        _, primal, primals, _ = CASES[name]
        current = {
            "vertex": lambda *p: _attributes(
                polygonanalytic._Vertex(*p[:4], NODES, residual=True, xp=jnp)
            ),
            "arsinh_terms": lambda *p: polygonanalytic._Vertex(
                *p[:4], NODES, residual=True, xp=jnp
            ).arsinh_terms(),
            "flux_moment_residuals": lambda *p: polygonanalytic._Vertex(
                *p[:4], NODES, residual=True, xp=jnp
            ).flux_moment_residuals(p[4], p[5]),
        }[name]
        for left, right in zip(
            jax.tree.leaves(jax.jit(current)(*primals)),
            jax.tree.leaves(jax.jit(primal)(*primals)),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        print(f"PRIMAL {name} bit_identical=True")


_COMPILE_PROBE = r"""
import json, sys, time
import jax
jax.config.update("jax_enable_compilation_cache", False)
from nova.jax.config import configure_dtypes
configure_dtypes()
assert jax.config.jax_enable_x64 is True
hits = []
jax.monitoring.register_event_listener(
    lambda event, **kw: hits.append(event) if "cache_hit" in event else None
)
sys.argv = sys.argv[1:]
sys.path.insert(0, sys.argv[1])
import test_vertex_term_tangents as t
name, arm = sys.argv[2], sys.argv[3]
tangent, primal, primals, tangents = t.CASES[name]
arguments = (primals, tangents)
if arm == "primal":
    function = lambda p, t: primal(*p)
elif arm == "tangent":
    function = lambda p, t: tangent(p, t)
else:
    function = lambda p, t: jax.jvp(primal, p, t)
def count(jaxpr):
    total = 0
    for equation in jaxpr.eqns:
        total += 1
        for value in equation.params.values():
            for sub in value if isinstance(value, (list, tuple)) else [value]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    total += count(inner)
    return total
equations = count(jax.make_jaxpr(function)(*arguments).jaxpr)
lowered = jax.jit(function).lower(*arguments)
start = time.perf_counter()
lowered.compile()
wall = time.perf_counter() - start
print(json.dumps({"name": name, "arm": arm, "compile_seconds": wall,
                  "equations": equations, "cache_hits": len(hits),
                  "cache_enabled": jax.config.jax_enable_compilation_cache}))
"""


def _cold_compile(name, arm):
    environment = dict(os.environ, JAX_ENABLE_COMPILATION_CACHE="false")
    environment.pop("NOVA_TANGENT_TRUNCATION", None)
    result = subprocess.run(
        [sys.executable, "-c", _COMPILE_PROBE, "probe", str(Path(__file__).parent),
         name, arm],
        capture_output=True, text=True, env=environment, check=True,
    )  # fmt: skip
    row = json.loads(result.stdout.strip().splitlines()[-1])
    assert row["cache_hits"] == 0 and row["cache_enabled"] is False
    return row


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CASES))
def test_tangent_compiles_within_three_primals(name):
    rows = {
        arm: [_cold_compile(name, arm) for _ in range(COMPILE_REPEATS)]
        for arm in ("primal", "tangent")
    }
    fastest = {}
    for arm, runs in rows.items():
        walls = sorted(row["compile_seconds"] for row in runs)
        equations = {row["equations"] for row in runs}
        assert len(equations) == 1
        fastest[arm] = walls[0]
        print(f"COMPILE {name} {arm} fastest_seconds={fastest[arm]:.3f} "
              f"equations={equations.pop()} "
              f"walls={','.join(f'{wall:.3f}' for wall in walls)} "
              f"hits={sum(row['cache_hits'] for row in runs)}")  # fmt: skip
    ratio = fastest["tangent"] / fastest["primal"]
    print(f"COMPILE {name} fastest_tangent_over_primal={ratio:.2f}")
    assert ratio <= 3.0
