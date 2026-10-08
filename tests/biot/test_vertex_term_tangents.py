"""Hand-written tangents of the polygon-analytic corner terms.

Each private tangent of :class:`nova.biot.polygonanalytic._Vertex` -- its
construction with the first residual integral, :meth:`arsinh_terms`,
:meth:`against_first_arsinh`, :meth:`flux_moment_residuals` and
:meth:`horizontal_flux_line_moments` -- is checked three ways: against
``jax.jvp`` of the current primal on ten thousand corner geometries and targets
spanning the jet's domain, with no sample masked; with one product-rule term
dropped, which must fail that bound; and by a cold compile in a fresh,
cache-disabled process against the primal. The graded residual's value is
compared with the earlier primal only by the corner-layer oracle.

``NOVA_TANGENT_TRUNCATION=1`` applies the truncation to the identity rows
themselves, which is the declared mutation those rows must fail against.
"""

from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import mpmath as mp
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

import nova.biot.polygonanalytic as polygonanalytic  # noqa: E402
from tangent_identity import (  # noqa: E402
    SAMPLES,
    compile_ratio,
    identity_row,
    load_base_module,
    relative_error,
    truncation_active,
)

# Every corner tangent runs through the harmonic moment stack's scanned
# recurrences, so every row is a scan form and is held to the scan bound.
SCAN_TOLERANCE = 1e-9
NODES = 128
# Targets nearer a corner than this fraction of its radius receive the
# independent 50-digit exact-integral check as well as the primal JVP identity.
NEAR_CORNER = 2e-6
BASE = load_base_module(
    "nova/biot/polygonanalytic.py", "646b68b8e9481178e41cc2059c8a9a534fa9b9b3"
)
BASE.graded_residual = load_base_module(
    "nova/biot/gradedresidual.py",
    "646b68b8e9481178e41cc2059c8a9a534fa9b9b3",
    module_name="corner_base_graded",
).graded_residual


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


def _primal_vertex(r, z, corner_r, corner_z):
    return polygonanalytic._Vertex(
        r, z, corner_r, corner_z, NODES, residual=True, xp=jnp
    )


def _current_vertex(p, t):
    return polygonanalytic._vertex_tangent(*p[:4], *t[:4], NODES, residual=True, xp=jnp)


def _cases():
    """Return ``name -> (tangent, primal, primals, tangents)``.

    ``primal`` is the current function of the ``primals`` tuple, so
    ``jax.jvp(primal, primals, tangents)`` judges the tangent at this revision.
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
        lambda r, z, cr, cz, e, h: _attributes(_primal_vertex(r, z, cr, cz)),
        primals,
        tangents,
    )

    def arsinh_tangent(p, t):
        return polygonanalytic._arsinh_terms_tangent(*_current_vertex(p, t))

    cases["arsinh_terms"] = (
        arsinh_tangent,
        lambda r, z, cr, cz, e, h: _primal_vertex(r, z, cr, cz).arsinh_terms(),
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
        lambda r, z, cr, cz, e, h, w: _primal_vertex(r, z, cr, cz).against_first_arsinh(
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
        lambda r, z, cr, cz, e, h: _primal_vertex(r, z, cr, cz).flux_moment_residuals(
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
        lambda r, z, cr, cz, e, h: _primal_vertex(
            r, z, cr, cz
        ).horizontal_flux_line_moments(e, h),
        primals,
        tangents,
    )
    return cases


CASES = _cases()


def _relative(got, reference):
    return float(np.max(relative_error(got, reference)))


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
    if got is None:
        assert reference is None
        return []
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
    original = polygonanalytic._product_tangent
    polygonanalytic._product_tangent = _truncated_product
    try:
        yield
    finally:
        polygonanalytic._product_tangent = original


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
def test_tangent_matches_primal_jvp(name):
    if truncation_active():
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
    judged = np.ones(SAMPLES, bool)
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
          f"judged_by_current_jvp={int(judged.sum())} "
          f"judged_by_reference={reference_count} "
          f"base_jvp_finite_fraction={finite:.4f} "
          f"base_tangent_median_abs={magnitude:.3e}")  # fmt: skip
    identity_row(name, primal, float(error.max()), SCAN_TOLERANCE)
    assert finite == 1.0
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
    return mp.nstr(4 * r * value, 50), mp.nstr(4 * dr * value + 4 * r * slope, 50)
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
    cache_path = os.environ.get("NOVA_CORNER_REFERENCE_CACHE")
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        saved = json.loads(cache.read_text())
        assert saved["inputs"] == rows and saved["digits"] == 50
        values = np.asarray(saved["reference"])
        return values[:, 0], values[:, 1]
    result = subprocess.run(
        [sys.executable, "-I", "-c", _REFERENCE_PROBE],
        input=json.dumps(rows), capture_output=True, text=True, check=True,
    )  # fmt: skip
    values = np.asarray(json.loads(result.stdout))
    if cache is not None:
        cache.write_text(
            json.dumps(
                {
                    "inputs": rows,
                    "digits": 50,
                    "reference": values.tolist(),
                }
            )
        )
    return values[:, 0], values[:, 1]


@lru_cache(maxsize=1)
def _near_corner_rows():
    from test_graded_layer_near_corner import _decade_floor, _program_floor

    got, expected = _identity("arsinh_terms")
    _, _, primals, tangents = CASES["arsinh_terms"]
    r, z, corner_r, corner_z = (np.asarray(value) for value in primals[:4])
    # a target exactly level with the corner or on its radius sits on the
    # integral's |u| or |offset| kink, where no two-sided derivative exists
    index = np.flatnonzero(_near_corner(primals) & (corner_z != z) & (corner_r != r))
    value, reference = _radial_reference(primals, tangents, index)
    hand = np.asarray(got[1][1])[index]
    primal_jvp = np.asarray(expected[1][1])[index]
    primal = np.asarray(expected[0][1])[index]
    old, _, program_difference = _program_floor(
        lambda *p: BASE._Vertex(*p[:4], NODES, residual=True, xp=jnp).arsinh_terms()[1],
        primals,
        tangents,
        select=lambda value: value[index],
    )
    decades, program_floor = _decade_floor(
        corner_r[index] - r[index], program_difference
    )

    def relative(a, b):
        with mp.workdps(50):
            return np.asarray(
                [
                    float(
                        abs(mp.mpf(float(left)) - mp.mpf(str(right)))
                        / (abs(mp.mpf(str(right))) or 1)
                    )
                    for left, right in zip(a, b, strict=True)
                ]
            )

    value_error = relative(primal, value)
    old_error = relative(old, value)
    tangent_error = relative(hand, reference)
    strict_degraded = value_error > old_error
    degraded = value_error > old_error + program_floor
    rows = {
        "samples": index.size,
        "primal": value_error.max(),
        "base_primal": old_error.max(),
        "strict_less_accurate": int(strict_degraded.sum()),
        "less_accurate": int(degraded.sum()),
        "hand": tangent_error.max(),
        "jvp": relative(primal_jvp, reference).max(),
        "hand_jvp": relative(hand, primal_jvp).max(),
    }
    receipt_path = os.environ.get("NOVA_CORNER_MEASUREMENT")
    if receipt_path:
        Path(receipt_path).write_text(
            json.dumps(
                {
                    "indices": index.tolist(),
                    "primals": [np.asarray(p)[index].tolist() for p in primals[:4]],
                    "tangents": [np.asarray(t)[index].tolist() for t in tangents[:4]],
                    "primal": primal.tolist(),
                    "base_primal": old.tolist(),
                    "hand": hand.tolist(),
                    "jvp": primal_jvp.tolist(),
                    "reference_value": value.tolist(),
                    "reference_tangent": reference.tolist(),
                    "value_error": value_error.tolist(),
                    "base_error": old_error.tolist(),
                    "program_floor": program_floor.tolist(),
                    "tangent_error": tangent_error.tolist(),
                }
            )
        )
    worst = int(np.argmax(tangent_error))
    print(
        f"CORNER_WORST sample={int(index[worst])} "
        f"primals={[float(np.asarray(p)[index[worst]]) for p in primals[:4]]} "
        f"tangents={[float(np.asarray(t)[index[worst]]) for t in tangents[:4]]}"
    )
    for decade in np.unique(decades):
        selected = decades == decade
        print(
            f"CORNER_SAMPLES decade=1e{decade:+d} samples={int(selected.sum())} "
            f"value_relative_max={value_error[selected].max():.3e} "
            f"derivative_relative_max={tangent_error[selected].max():.3e} "
            f"base_value_relative_max={old_error[selected].max():.3e} "
            f"program_floor={program_floor[selected].max():.3e} "
            f"strict_less_accurate={int(strict_degraded[selected].sum())} "
            f"less_accurate={int(degraded[selected].sum())}"
        )
    print(f"NEAR_CORNER arsinh_terms radial samples={rows['samples']} "
          f"primal_vs_reference_max={rows['primal']:.3e} "
          f"hand_vs_reference_max={rows['hand']:.3e} "
          f"primal_jvp_vs_reference_max={rows['jvp']:.3e} "
          f"hand_vs_primal_jvp_max={rows['hand_jvp']:.3e}")  # fmt: skip
    return rows


def test_near_corner_reference_is_the_exact_integral():
    rows = _near_corner_rows()
    # the independent integral checks the repaired primal near the corner
    assert rows["samples"] > 0
    assert rows["primal"] <= 1e-10
    assert rows["less_accurate"] == 0


def test_near_corner_radial_tangent_meets_reference():
    rows = _near_corner_rows()
    assert rows["hand"] <= SCAN_TOLERANCE
    assert rows["jvp"] <= SCAN_TOLERANCE
    assert rows["hand_jvp"] <= SCAN_TOLERANCE


@pytest.mark.parametrize("name", list(CASES))
def test_truncated_tangent_fails_identity(name):
    with _truncated():
        got, expected = _identity(name)
    tangent = float(_normwise(got[1], expected[1]).max())
    print(f"TRUNCATED {name} tangent_max_normwise_relative={tangent:.3e}")
    assert tangent > SCAN_TOLERANCE


def compile_arm(name, arm):
    tangent, primal, _, _ = CASES[name]
    if arm == "primal":
        return lambda p, t: primal(*p)
    if arm == "tangent":
        return lambda p, t: tangent(p, t)
    return lambda p, t: jax.jvp(primal, p, t)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CASES))
def test_tangent_compiles_within_three_primals(name):
    assert compile_ratio("test_vertex_term_tangents", name) <= 3.0
