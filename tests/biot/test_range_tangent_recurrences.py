"""Hand-written tangents of the range-function operations the point jet reaches.

Each private tangent in :mod:`nova.biot.rangefunction` is checked four ways:
against ``jax.jvp`` of the base-revision function on ten thousand inputs
spanning the magnitudes the reductions carry, with no sample masked; with its
recurrence truncated by one term, which must fail that bound; by a cold compile
in a fresh, cache-disabled process against the primal; and by the primal half
of its result being bit-identical to the base revision's.

Every operation is polynomial in its coefficients, so the tangents are unrolled
and agree with ``jax.jvp`` of the base exactly; ``EXACT_TOLERANCE`` is the bound
the contract allows, and the maxima are printed beside it.

``NOVA_TANGENT_TRUNCATION=1`` applies the truncation to the identity rows
themselves, which is the declared mutation those rows must fail against.
"""

from contextlib import contextmanager
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

import nova.biot.rangefunction as rangefunction  # noqa: E402
from tangent_identity import (  # noqa: E402
    EXACT_TOLERANCE,
    SAMPLES,
    compile_ratio,
    identity_row,
    load_base_module,
    per_sample_worst,
    truncation_active,
    worst,
)

BASE_REVISION = os.environ.get(
    "NOVA_RANGE_BASE_REVISION", "e74d8fce12601e372d91572aba1b5602f2d7df55"
)

BASE = load_base_module("nova/biot/rangefunction.py", BASE_REVISION)
assert not hasattr(BASE, "_harmonic_multiply_tangent")


def _values(rng, shape):
    """Return samples spanning the magnitudes of the reductions' coefficients.

    Half are order-one normals; the rest carry a random sign and a magnitude
    spread over sixty decades, which is wider than the numerators' own range and
    reaches the large end values a pole close to the range supplies.  The first
    hundred samples are exact zeros.
    """
    normal = rng.normal(size=shape)
    spread = rng.choice([-1.0, 1.0], size=shape) * 10.0 ** rng.uniform(-30, 30, shape)
    chosen = np.where(rng.random(shape) < 0.5, normal, spread)
    chosen[..., :100] = 0.0
    return chosen


def _cases():
    """Return ``name -> (tangent, base function, primals, tangents)``.

    Every callable takes its arrays as arguments, so a compile measures the
    program rather than a constant-folded closure.
    """
    rng = np.random.default_rng(0)

    def series(length):
        return [jnp.asarray(_values(rng, SAMPLES)) for _ in range(length)]

    def tangent(length):
        return [jnp.asarray(rng.normal(size=SAMPLES)) for _ in range(length)]

    def scalar():
        return jnp.asarray(_values(rng, SAMPLES))

    def d_scalar():
        return jnp.asarray(rng.normal(size=SAMPLES))

    def function(length):
        return (series(length), scalar(), scalar())

    def d_function(length):
        return (tangent(length), d_scalar(), d_scalar())

    R = rangefunction
    cases = {}
    cases["harmonic_multiply"] = (
        lambda p, t: R._harmonic_multiply_tangent(p[0], t[0], p[1], t[1]),
        BASE.harmonic_multiply,
        (series(6), series(7)),
        (tangent(6), tangent(7)),
    )
    cases["harmonic_add"] = (
        lambda p, t: R._harmonic_sum_tangent(p, t),
        BASE.harmonic_add,
        (series(4), series(7), series(5)),
        (tangent(4), tangent(7), tangent(5)),
    )
    cases["harmonic_scale"] = (
        lambda p, t: R._harmonic_scale_tangent(p[0], t[0], p[1], t[1]),
        BASE.harmonic_scale,
        (series(6), scalar()),
        (tangent(6), d_scalar()),
    )
    cases["range_function"] = (
        lambda p, t: R._range_function_tangent(p[0], t[0], p[1], t[1], p[2], t[2]),
        BASE.range_function,
        function(5),
        d_function(5),
    )
    cases["paired_range_function"] = (
        lambda p, t: R._paired_range_function_tangent(
            p[0], t[0], p[1], t[1], p[2], t[2]
        ),
        BASE.paired_range_function,
        function(5),
        d_function(5),
    )
    cases["product"] = (
        lambda p, t: R._product_range_tangent(p[0], t[0], p[1], t[1]),
        BASE.product,
        (function(5), function(4)),
        (d_function(5), d_function(4)),
    )
    cases["total"] = (
        lambda p, t: R._total_tangent(p, t),
        BASE.total,
        (function(4), function(6), function(5)),
        (d_function(4), d_function(6), d_function(5)),
    )
    cases["scaled"] = (
        lambda p, t: R._scaled_tangent(p[0], t[0], p[1], t[1]),
        BASE.scaled,
        (function(6), scalar()),
        (d_function(6), d_scalar()),
    )
    cases["across_the_range"] = (
        lambda p, t: R._across_the_range_tangent(p[0], t[0]),
        BASE.across_the_range,
        (function(6),),
        (d_function(6),),
    )
    cases["sine_squared_times"] = (
        lambda p, t: R._sine_squared_times_tangent(p[0], t[0]),
        BASE.sine_squared_times,
        (series(6),),
        (tangent(6),),
    )
    cases["deflate"] = (
        lambda p, t: R._deflate_tangent(p[0], t[0], p[1], t[1]),
        BASE.deflate,
        (series(10), scalar()),
        (tangent(10), d_scalar()),
    )
    cases["contract"] = (
        lambda p, t: R._contract_tangent(p[0], t[0], p[1], t[1]),
        BASE.contract,
        (series(26), series(26)),
        (tangent(26), tangent(26)),
    )
    cases["rising_integral"] = (
        lambda p, t: R._rising_integral_tangent(p[0], t[0]),
        BASE.rising_integral,
        (series(8),),
        (tangent(8),),
    )
    return cases


CASES = _cases()


def compile_arm(name, arm):
    """Return the callable the cold-compile probe compiles for ``arm``."""
    tangent, base, _, _ = CASES[name]
    if arm == "primal":
        return lambda p, t: base(*p)
    if arm == "tangent":
        return lambda p, t: tangent(p, t)
    return lambda p, t: jax.jvp(base, p, t)


def _drop_last(series, like):
    """Return the series' tangent with its final coefficient's term removed."""
    return [*series[:-1], jnp.zeros_like(like)]


def _truncated_pair_step(
    d_rising, d_falling, one, d_one, other, d_other, *, coincident
):
    """The pair step with the second factor's tangent term dropped."""
    steps = _ORIGINAL["_harmonic_pair_step_tangent"](
        d_rising, d_falling, one, d_one, other, None, coincident=coincident
    )
    # a tangent left absent by the drop is a zero, so the structure is kept
    return tuple(jnp.zeros_like(one) if step is None else step for step in steps)


def _truncated_sum(series, d_series):
    """The harmonic sum's tangent with its final coefficient dropped."""
    value, tangent = _ORIGINAL["_harmonic_sum_tangent"](series, d_series)
    return value, _drop_last(tangent, value[-1])


def _truncated_scale(series, d_series, factor, d_factor):
    """The harmonic scale's tangent with its final coefficient dropped."""
    value, tangent = _ORIGINAL["_harmonic_scale_tangent"](
        series, d_series, factor, d_factor
    )
    return value, _drop_last(tangent, value[-1])


def _truncated_range_function(bulk, d_bulk, near, d_near, far, d_far):
    """The identity range-function tangent with its bulk's last term dropped."""
    return (bulk, near, far), (_drop_last(d_bulk, bulk[-1]), d_near, d_far)


def _truncated_paired_range_function(bulk, d_bulk, near, d_near, far, d_far):
    return _truncated_range_function(bulk, d_bulk, near, d_near, far, d_far)


def _truncated_deflate_step(
    coefficient, d_coefficient, root, d_root, current, d_current, upper, d_upper
):
    """The downward step with the carried-over order's tangent dropped."""
    return _ORIGINAL["_deflate_step_tangent"](
        coefficient, d_coefficient, root, d_root, current, d_current, upper, None
    )


def _truncated_product_sum_step(
    d_total_value, coefficient, d_coefficient, moment, d_moment
):
    """The accumulation with the moment's tangent term dropped."""
    return _ORIGINAL["_product_sum_step_tangent"](
        d_total_value, coefficient, d_coefficient, moment, None
    )


_SEAMS = (
    "_harmonic_pair_step_tangent",
    "_harmonic_sum_tangent",
    "_harmonic_scale_tangent",
    "_range_function_tangent",
    "_paired_range_function_tangent",
    "_deflate_step_tangent",
    "_product_sum_step_tangent",
)
_ORIGINAL = {name: getattr(rangefunction, name) for name in _SEAMS}
_PAIR_STEP = (("_harmonic_pair_step_tangent", _truncated_pair_step),)
_SUM = (("_harmonic_sum_tangent", _truncated_sum),)
_SCALE = (("_harmonic_scale_tangent", _truncated_scale),)
_DEFLATE = (("_deflate_step_tangent", _truncated_deflate_step),)
TRUNCATIONS = {
    "harmonic_multiply": _PAIR_STEP,
    "harmonic_add": _SUM,
    "harmonic_scale": _SCALE,
    "range_function": (("_range_function_tangent", _truncated_range_function),),
    "paired_range_function": (
        ("_paired_range_function_tangent", _truncated_paired_range_function),
    ),
    "product": _PAIR_STEP,
    "total": _SUM,
    "scaled": _SCALE,
    "across_the_range": _SUM,
    "sine_squared_times": _SCALE,
    "deflate": _DEFLATE,
    "contract": (("_product_sum_step_tangent", _truncated_product_sum_step),),
    "rising_integral": _DEFLATE,
}


@contextmanager
def _truncated(name):
    jax.clear_caches()
    for attribute, replacement in TRUNCATIONS[name]:
        setattr(rangefunction, attribute, replacement)
    try:
        yield
    finally:
        for attribute, original in _ORIGINAL.items():
            setattr(rangefunction, attribute, original)
        jax.clear_caches()


def _identity(name):
    tangent, base, primals, tangents = CASES[name]
    # a fresh jit per call, so a truncation applied since is traced afresh
    got = jax.jit(lambda p, t: tangent(p, t))(primals, tangents)
    expected = jax.jit(lambda p, t: jax.jvp(base, p, t))(primals, tangents)
    return got, expected


@pytest.mark.parametrize("name", list(CASES))
def test_tangent_matches_base_jvp(name):
    if truncation_active():
        with _truncated(name):
            got, expected = _identity(name)
    else:
        got, expected = _identity(name)
    per_sample = per_sample_worst(got[1], expected[1])
    finite = np.all(
        np.stack([np.isfinite(leaf) for leaf in jax.tree.leaves(expected[1])]), axis=0
    )
    covered = float(np.mean(per_sample <= EXACT_TOLERANCE))
    identity_row(
        name,
        worst(got[0], expected[0]),
        float(per_sample.max()),
        EXACT_TOLERANCE,
        extra=f"samples={per_sample.size} covered_fraction={covered:.6f} "
        f"base_jvp_nonfinite_samples={int(np.count_nonzero(~finite))}",
    )
    assert finite.all(), "the base jvp must be finite on every sample"
    assert covered == 1.0


@pytest.mark.parametrize("name", list(CASES))
def test_truncated_tangent_fails_identity(name):
    with _truncated(name):
        got, expected = _identity(name)
    tangent = worst(got[1], expected[1])
    print(f"TRUNCATED {name} tangent_max_relative={tangent:.3e}")
    assert tangent > EXACT_TOLERANCE


@pytest.mark.parametrize("name", list(CASES))
def test_primal_bit_identical_to_base(name):
    tangent, base, primals, tangents = CASES[name]
    got = jax.jit(lambda p, t: tangent(p, t)[0])(primals, tangents)
    alone = jax.jit(lambda p: base(*p))(primals)
    for left, right in zip(jax.tree.leaves(got), jax.tree.leaves(alone), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right), err_msg=name)
    print(f"PRIMAL {name} bit_identical=True leaves={len(jax.tree.leaves(got))}")


def test_package_functions_equal_base_functions():
    """The functions the reductions call are the base revision's, unchanged."""
    for name in rangefunction.__all__:
        assert hasattr(BASE, name), name
    tangent, base, primals, _ = CASES["product"]
    got = jax.jit(lambda p: rangefunction.product(*p))(primals)
    alone = jax.jit(lambda p: base(*p))(primals)
    for left, right in zip(jax.tree.leaves(got), jax.tree.leaves(alone), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


@pytest.mark.slow
# A compile row spawns 21 fresh processes; the repository's 300 s hang guard is
# for ordinary tests and cuts this one off before it asserts, so it gets its own
# bound.  Every other test keeps the default.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("name", list(CASES))
def test_tangent_compiles_within_three_primals(name):
    ratio = compile_ratio("test_range_tangent_recurrences", name)
    assert ratio <= 3.0
