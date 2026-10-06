"""Hand-written tangents of the moment-channel and graded-residual terms.

Each private tangent in :mod:`nova.biot.momentchannel` and
:mod:`nova.biot.gradedresidual` is checked three ways: against ``jax.jvp`` of
the base-revision function on ten thousand inputs spanning the jet's domain,
with no sample masked; with its recurrence or product rule truncated by one
term, which must fail that bound; and by a cold compile in a fresh,
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

import nova.biot.gradedresidual as gradedresidual  # noqa: E402
import nova.biot.momentchannel as momentchannel  # noqa: E402

BASE_REVISION = os.environ.get(
    "NOVA_VERTEX_EDGE_BASE_REVISION", "e74d8fce12601e372d91572aba1b5602f2d7df55"
)
ROOT = Path(__file__).resolve().parents[2]
# Every tangent here is an unrolled closed form or recurrence, so it is held to
# the exact bound; round-off between two orderings of the same sum is all that
# separates it from jax.jvp of the base.
EXACT_TOLERANCE = 1e-13
SAMPLES = 10_000
BULK = 7
MOMENTS = 12
FAMILY = 10
NODES = 128
# A program this small compiles in a tenth of a second, where scheduler noise is
# a multiple of the difference between arms, so each arm keeps the fastest of
# this many fresh, cache-disabled processes.
COMPILE_REPEATS = 3


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_modules():
    """Load the base revision's two modules under their own names."""
    directory = Path(tempfile.mkdtemp(prefix="vertex-edge-base-"))
    loaded = []
    for name in ("momentchannel", "gradedresidual"):
        path = directory / f"{name}.py"
        path.write_bytes(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{BASE_REVISION}:nova/biot/{name}.py"]
            )
        )
        module = _load(f"base_{name}", path)
        print(f"BASE_MODULE {name}={module.__file__}", flush=True)
        loaded.append(module)
    return loaded


BASE_CHANNEL, BASE_GRADED = _base_modules()


def _magnitude(rng, low, high, size=SAMPLES):
    return 10.0 ** rng.uniform(low, high, size)


def _signed(rng, low, high, size=SAMPLES):
    return rng.choice((-1.0, 1.0), size) * _magnitude(rng, low, high, size)


def _denominator_domain(rng):
    """Return denominators formed as the reduction forms them, from geometry.

    The ring denominator is ``4 r^2 x y + u^2 x + u^2 y`` (``_Vertex``) and the
    plane one ``4 b1^2 r^2 x y + w^2 x + (w + 2 r)^2 y`` (``_Edge``), with ``w``
    the target's offset from the edge's extended line taken as the edge does.
    Target and edge radii run from a centimetre to twenty metres over a forty
    metre height.  The factorisation's pivot root cancels to zero -- and the
    base primal itself returns an infinite shift -- only where a plane end
    value exceeds the other end plus the ``x y`` coefficient by sixteen decades,
    which needs the extended line at negative radius and the product of the
    target's and the edge's radii below about ``1e-8`` of the squared height;
    the ring's equal end values never cancel.  So that corner is not sampled.
    """
    radius = _magnitude(rng, -2.0, 1.3)
    height = rng.uniform(-20.0, 20.0, SAMPLES)
    level = _signed(rng, -12.0, 1.3)
    level[:200] = 0.0  # a target level with the corner: both ring ends vanish
    ring = 4000
    edge_r = _magnitude(rng, -2.0, 1.3, (2, SAMPLES))
    edge_z = rng.uniform(-20.0, 20.0, (2, SAMPLES))
    edge_r[1, ring : ring + 500] = edge_r[0, ring : ring + 500]  # a vertical edge
    # a target on the edge's end: the plane's near end value vanishes exactly
    radius[ring + 500 : ring + 800] = edge_r[0, ring + 500 : ring + 800]
    height[ring + 500 : ring + 800] = edge_z[0, ring + 500 : ring + 800]
    (ra, rb), (za, zb) = edge_r, edge_z
    slope = (rb - ra) / (zb - za)
    offset = ((ra - radius) * (zb - height) - (rb - radius) * (za - height)) / (zb - za)
    plane = slice(ring, None)
    leading = 4.0 * radius * radius
    near = level * level
    far = near.copy()
    leading[plane] = (4.0 * slope * slope * radius * radius)[plane]
    near[plane] = (offset**2)[plane]
    far[plane] = ((offset + 2.0 * radius) ** 2)[plane]
    rest = [rng.normal(size=SAMPLES) for _ in range(2)]
    return [leading, *rest], near, far


def _shift_domain(rng):
    shift = _magnitude(rng, -9.0, 12.0)
    shift[:200] = 0.0
    shift[200:400] = momentchannel.POLE_SWITCH
    shift[400:900] = momentchannel.POLE_SWITCH * (1.0 + rng.uniform(-1e-3, 1e-3, 500))
    shift[900:1000] = momentchannel.POLE_CEILING
    return shift


def _tangents(rng, tree):
    return jax.tree.map(lambda leaf: jnp.asarray(rng.normal(size=np.shape(leaf))), tree)


def _arrays(tree):
    return jax.tree.map(jnp.asarray, tree)


def _vertex_domain(rng):
    """Return the ring residual's radius, level, offset and panel bounds.

    The level and offset reach zero together -- a target on the corner -- and
    each alone -- a target level with the corner or on its radius -- because the
    graded panels' layers collapse exactly there.
    """
    radius = _magnitude(rng, -6.0, 1.0)
    level = _signed(rng, -12.0, 1.0)
    offset = _signed(rng, -12.0, 1.0)
    level[:300] = 0.0
    offset[300:600] = 0.0
    level[600:800] = offset[600:800] = 0.0
    # near the axis, which only an exact zero radius is held away from
    radius[800:900] = _magnitude(rng, -6.0, -3.0, 100)
    lower = np.zeros((2, SAMPLES))
    upper = np.full((2, SAMPLES), gradedresidual.QUARTER)
    # a finite arc: each panel stops at an interior amplitude
    upper[:, 5000:] = rng.uniform(1e-6, gradedresidual.QUARTER, (2, 5000))
    lower[:, 7500:] = upper[:, 7500:] * rng.uniform(0.0, 1.0, (2, 2500))
    return radius, level, offset, lower, upper


def _vertex_panels(radius, level, offset, lower, upper):
    span = 2.0 * jnp.where(radius > 0.0, radius, 1.0)
    level_offset = jnp.abs(level)
    return (
        (level_offset, offset + 2.0 * radius, span, lower[0], upper[0]),
        (level_offset, offset, span, lower[1], upper[1]),
    )


def _vertex_pieces(radius, level, offset):
    def pieces(x, y):
        return (
            offset[:, None] + 2.0 * radius[:, None] * y,
            jnp.sqrt(level[:, None] ** 2 + 4.0 * radius[:, None] ** 2 * x * y),
        )

    return pieces


def _graded_tangent(primals, tangents):
    radius, level, offset, lower, upper = primals
    d_radius, d_level, d_offset, d_lower, d_upper = tangents
    d_span = 2.0 * jnp.where(radius > 0.0, d_radius, 0.0)
    d_level_offset = jnp.where(level >= 0.0, d_level, -d_level)
    panels = _vertex_panels(radius, level, offset, lower, upper)
    d_panels = (
        (d_level_offset, d_offset + 2.0 * d_radius, d_span, d_lower[0], d_upper[0]),
        (d_level_offset, d_offset, d_span, d_lower[1], d_upper[1]),
    )
    r, u, o = radius[:, None], level[:, None], offset[:, None]
    dr, du, do = d_radius[:, None], d_level[:, None], d_offset[:, None]

    def pieces_tangent(x, d_x, y, d_y):
        # each product's tangent in the order the pieces form it
        twice = 2.0 * r
        numerator = o + twice * y
        d_numerator = do + (2.0 * dr * y + twice * d_y)
        quadruple = 4.0 * r**2
        d_quadruple = 4.0 * (dr * (2.0 * r))
        cross = quadruple * x
        d_cross = d_quadruple * x + quadruple * d_x
        cross, d_cross = cross * y, d_cross * y + cross * d_y
        denominator = jnp.sqrt(u**2 + cross)
        d_denominator = (du * (2.0 * u) + d_cross) * (0.5 / denominator)
        return (numerator, denominator), (d_numerator, d_denominator)

    return gradedresidual._graded_residual_tangent(
        panels, d_panels, pieces_tangent, NODES, jnp
    )


def _graded_primal(radius, level, offset, lower, upper):
    return BASE_GRADED.graded_residual(
        _vertex_panels(radius, level, offset, lower, upper),
        _vertex_pieces(radius, level, offset),
        NODES,
        jnp,
    )


def _cases():
    """Return ``name -> (tangent, primal, primals, tangents)``.

    ``primal`` is the base revision's function of the ``primals`` tuple, so
    ``jax.jvp(primal, primals, tangents)`` is the reference and its first half
    the primal the tangent's own first half must reproduce bit for bit.
    """
    rng = np.random.default_rng(0)
    cases = {}

    denominator = _arrays(_denominator_domain(rng))
    cases["factorise"] = (
        lambda p, t: momentchannel._factorise_tangent(p[0], t[0], jnp),
        lambda d: BASE_CHANNEL.factorise(d, jnp),
        (denominator,),
        (_tangents(rng, denominator),),
    )

    numerator = _arrays(
        (
            [rng.normal(size=SAMPLES) for _ in range(BULK)],
            _signed(rng, -6.0, 3.0),
            _signed(rng, -6.0, 3.0),
        )
    )
    shift = jnp.asarray(_shift_domain(rng))
    seed = jnp.asarray(_signed(rng, -3.0, 6.0))
    family = [jnp.asarray(_signed(rng, -12.0, 2.0)) for _ in range(FAMILY)]
    moments = [jnp.asarray(_signed(rng, -12.0, 2.0)) for _ in range(MOMENTS)]
    for mirrored in (False, True):
        primals = (numerator, shift, seed, family, moments)
        name = "pole_contraction" + ("_mirrored" if mirrored else "")
        cases[name] = (
            lambda p, t, mirrored=mirrored: momentchannel._pole_contraction_tangent(
                p[0],
                t[0],
                p[1],
                t[1],
                p[2],
                t[2],
                p[3],
                t[3],
                p[4],
                t[4],
                mirrored,
                xp=jnp,
            ),  # fmt: skip
            lambda n, s, c, f, m, mirrored=mirrored: BASE_CHANNEL._pole_contraction(
                n, s, c, f, m, mirrored, xp=jnp
            ),
            primals,
            _tangents(rng, primals),
        )

    factors = BASE_CHANNEL.factorise(_arrays(_denominator_domain(rng)), jnp)
    poles = (
        jnp.asarray(_signed(rng, -3.0, 6.0)),
        jnp.asarray(_signed(rng, -3.0, 6.0)),
        [jnp.asarray(_signed(rng, -12.0, 2.0)) for _ in range(FAMILY)],
        [jnp.asarray(_signed(rng, -12.0, 2.0)) for _ in range(FAMILY)],
    )
    primals = (numerator, factors, poles, moments)

    def across(numerator, factors, poles, moments):
        channel = object.__new__(BASE_CHANNEL.Channel)
        channel.xp, channel.moments = jnp, moments
        return channel.across(numerator, (factors, poles, None))

    cases["across"] = (
        lambda p, t: momentchannel._across_tangent(
            p[0], t[0], p[1], t[1], p[2], t[2], p[3], t[3], xp=jnp
        ),
        across,
        primals,
        _tangents(rng, primals),
    )

    offset = _magnitude(rng, -12.0, 1.0)
    offset[:500] = 0.0
    scale = _magnitude(rng, -9.0, 1.0)
    scale[500:1000] = 0.0
    lower = rng.uniform(0.0, 0.5, SAMPLES)
    lower[:3000] = 0.0
    upper = lower + rng.uniform(0.0, gradedresidual.QUARTER, SAMPLES)
    primals = _arrays((offset, scale, lower, upper))
    cases["model_integral"] = (
        lambda p, t: gradedresidual._model_integral_tangent(
            p[0], t[0], p[1], t[1], p[2], t[2], p[3], t[3], jnp
        ),
        lambda o, s, low, high: BASE_GRADED._model_integral(o, s, low, high, jnp),
        primals,
        _tangents(rng, primals),
    )

    numerator = _signed(rng, -12.0, 2.0)
    numerator[:500] = 0.0
    denominator = _magnitude(rng, -12.0, 2.0)
    model = _magnitude(rng, -12.0, 2.0)
    sign = jnp.asarray(rng.choice((-1.0, 0.0, 1.0), SAMPLES))
    primals = _arrays((numerator, denominator, model))
    cases["regularised"] = (
        lambda p, t: gradedresidual._regularised_tangent(
            p[0], t[0], p[1], t[1], p[2], t[2], sign, jnp
        ),
        lambda n, d, m: BASE_GRADED._regularised(n, d, m, sign, jnp),
        primals,
        _tangents(rng, primals),
    )

    primals = _arrays(_vertex_domain(rng))
    cases["graded_residual"] = (
        _graded_tangent,
        _graded_primal,
        primals,
        _tangents(rng, primals),
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


def _finite_fraction(tree):
    leaves = [np.isfinite(np.asarray(leaf)) for leaf in jax.tree.leaves(tree)]
    return float(np.mean(np.all(np.stack([np.ravel(v) for v in leaves]), axis=0)))


def _truncated_quotient(numerator, d_numerator, denominator, d_denominator):
    """The quotient rule's tangent with its denominator term dropped."""
    return d_numerator / denominator


def _truncated_deflate_step(
    coefficient, d_coefficient, root, d_root, current, d_current, upper, d_upper
):
    """The Clenshaw step's tangent with the lagged order's term dropped."""
    return (
        2.0 * coefficient + 2.0 * root * current - upper,
        2.0 * d_coefficient + 2.0 * (d_root * current + root * d_current),
    )


def _truncated_contract(numerator, d_numerator, moments, d_moments):
    """The contraction's tangent with the moments' own tangent term dropped."""
    total = 0.0
    for order in range(len(numerator)):
        total = total + d_numerator[order] * moments[order]
    return total


def _truncated_held(condition, value, d_value, fill, xp):
    """The held branch with its tangent dropped."""
    return xp.where(condition, value, fill), 0.0 * d_value


def _truncated_model_integral(*arguments):
    """The panel's model integral with its tangent dropped from the sum."""
    value, d_value = _MODEL_INTEGRAL(*arguments)
    return value, 0.0 * d_value


_MODEL_INTEGRAL = gradedresidual._model_integral_tangent

TRUNCATIONS = {
    "factorise": [(momentchannel, "_quotient_tangent", _truncated_quotient)],
    "pole_contraction": [
        (momentchannel, "_deflate_step_tangent", _truncated_deflate_step)
    ],
    "pole_contraction_mirrored": [
        (momentchannel, "_contract_tangent", _truncated_contract)
    ],
    "across": [(momentchannel, "_contract_tangent", _truncated_contract)],
    "model_integral": [(gradedresidual, "_held_tangent", _truncated_held)],
    "regularised": [(gradedresidual, "_held_tangent", _truncated_held)],
    "graded_residual": [
        (gradedresidual, "_model_integral_tangent", _truncated_model_integral)
    ],
}


@contextmanager
def _truncated(name):
    saved = []
    for module, attribute, replacement in TRUNCATIONS[name]:
        saved.append((module, attribute, getattr(module, attribute)))
        setattr(module, attribute, replacement)
    try:
        yield
    finally:
        for module, attribute, original in saved:
            setattr(module, attribute, original)


def _identity(name):
    tangent, primal, primals, tangents = CASES[name]
    # a fresh jit per call, so a truncation applied since is traced afresh
    got = jax.jit(lambda p, t: tangent(p, t))(primals, tangents)
    expected = jax.jit(lambda p, t: jax.jvp(primal, p, t))(primals, tangents)
    return got, expected


@pytest.mark.parametrize("name", list(CASES))
def test_tangent_matches_base_jvp(name):
    if os.environ.get("NOVA_TANGENT_TRUNCATION") == "1":
        with _truncated(name):
            got, expected = _identity(name)
    else:
        got, expected = _identity(name)
    primal = _worst(got[0], expected[0])
    tangent = _worst(got[1], expected[1])
    print(f"IDENTITY {name} primal_max_relative={primal:.3e} "
          f"tangent_max_relative={tangent:.3e} bound={EXACT_TOLERANCE:.0e} "
          f"covered_fraction=1.000 masked_samples=0 "
          f"base_jvp_finite_fraction={_finite_fraction(expected[1]):.4f}")  # fmt: skip
    assert primal == 0.0
    assert tangent <= EXACT_TOLERANCE
    assert _finite_fraction(expected[1]) == 1.0


@pytest.mark.parametrize("name", list(CASES))
def test_truncated_tangent_fails_identity(name):
    with _truncated(name):
        got, expected = _identity(name)
    tangent = _worst(got[1], expected[1])
    print(f"TRUNCATED {name} tangent_max_relative={tangent:.3e}")
    assert tangent > EXACT_TOLERANCE


def test_primal_bit_identical_to_base():
    for name, (_, primal, primals, _) in CASES.items():
        current = {
            "factorise": lambda d: momentchannel.factorise(d, jnp),
            "model_integral": lambda o, s, low, high: gradedresidual._model_integral(
                o, s, low, high, jnp
            ),
        }.get(name)
        if current is None:
            continue
        for left, right in zip(
            jax.tree.leaves(current(*primals)),
            jax.tree.leaves(primal(*primals)),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        print(f"PRIMAL {name} bit_identical=True")
    radius, level, offset, lower, upper = CASES["graded_residual"][2]
    current = gradedresidual.graded_residual(
        _vertex_panels(radius, level, offset, lower, upper),
        _vertex_pieces(radius, level, offset),
        NODES,
        jnp,
    )
    np.testing.assert_array_equal(
        np.asarray(current),
        np.asarray(_graded_primal(radius, level, offset, lower, upper)),
    )
    print("PRIMAL graded_residual bit_identical=True")


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
import test_vertex_edge_tangent_recurrences as t
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
        arm: min(
            (_cold_compile(name, arm) for _ in range(COMPILE_REPEATS)),
            key=lambda row: row["compile_seconds"],
        )
        for arm in ("primal", "tangent", "jvp")
    }
    for arm, row in rows.items():
        print(f"COMPILE {name} {arm} seconds={row['compile_seconds']:.3f} "
              f"equations={row['equations']} hits={row['cache_hits']}")  # fmt: skip
    ratio = rows["tangent"]["compile_seconds"] / rows["primal"]["compile_seconds"]
    print(f"COMPILE {name} tangent_over_primal={ratio:.2f}")
    assert ratio <= 3.0
