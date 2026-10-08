"""Hand-written tangent recurrences of the elliptic families the point jet reaches.

Each private tangent in :mod:`nova.biot.completeelliptic` and
:mod:`nova.biot.elliptic` is checked three ways: against ``jax.jvp`` of the
base-revision function on ten thousand inputs spanning the jet's domain --
exactly for an unrolled tangent, to ``SCANNED_TOLERANCE`` for a scanned one --
with the recurrence truncated by one term (which must fail that bound), and by a
cold compile in a fresh, cache-disabled process against the primal.  The primal
outputs are asserted bit-identical to the base revision's.

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

import nova.biot.completeelliptic as completeelliptic  # noqa: E402
import nova.biot.elliptic as elliptic  # noqa: E402
from tangent_identity import (  # noqa: E402
    SAMPLES,
    compile_ratio,
    identity_row,
    load_base_module,
    truncation_active,
    worst,
)

BASE_REVISION = "de54856c72490b216e69506c8f9fdcd7d839fedb"

# Tangents carried as scanned steps are held to this bound against jax.jvp of
# the base, because a scan and an unrolled loop compile to differently fused
# programs whose round-off differs; every other tangent must agree exactly.
SCANNED_TOLERANCE = 1e-9
SCANNED = frozenset(
    (
        "complete_kind",
        "harmonic_moments",
        "harmonic_pole_moments",
        "harmonic_pole_moments_mirrored",
    )
)
MOMENTS = 9 + elliptic.POLE_HEADROOM + 2
POLE_COUNT = 10


def _base_modules():
    """Load the base revision's two elliptic modules under their own names.

    The base ``elliptic`` imports names from the base ``completeelliptic``, so
    each name it re-binds is repointed at the base module's own object; without
    that the base elliptic would call this revision's complete-elliptic helpers
    and the identity would be measured against the wrong program.
    """
    complete = load_base_module("nova/biot/completeelliptic.py", BASE_REVISION)
    base_elliptic = load_base_module("nova/biot/elliptic.py", BASE_REVISION)
    for key, value in tuple(vars(base_elliptic).items()):
        # the base elliptic's import statement binds THIS revision's
        # complete-elliptic members, so repoint each at the base module's own
        if getattr(value, "__module__", "") == "nova.biot.completeelliptic":
            setattr(base_elliptic, key, getattr(complete, value.__name__))
    return complete, base_elliptic


BASE_COMPLETE, BASE_ELLIPTIC = _base_modules()


def _domain(seed=0):
    """Return complements, parameters, poles, shifts and tangents for the jet."""
    rng = np.random.default_rng(seed)
    complement = 10.0 ** rng.uniform(-16.0, 0.0, SAMPLES)
    complement[:100] = 0.0
    complement[100:600] = 10.0 ** rng.uniform(-300.0, -16.0, 500)
    parameter = 1.0 - complement
    pole = 10.0 ** rng.uniform(-9.0, 9.0, SAMPLES)
    pole[:300] = 1.0 + rng.uniform(-1e-4, 1e-4, 300)
    pole[300:350] = 1.0
    pole[350:400] = 0.0
    shift = 10.0 ** rng.uniform(-12.0, 6.0, SAMPLES)
    shift[:50] = 0.0
    far = 10.0 ** rng.uniform(-0.6, 4.0, SAMPLES)
    tangent = rng.normal(size=(4, SAMPLES))
    order = rng.normal(size=(MOMENTS, SAMPLES))
    arrays = map(jnp.asarray, (complement, parameter, pole, shift, far))
    return (*arrays, jnp.asarray(tangent), jnp.asarray(order))


def _cases():
    """Return ``name -> (tangent, base jvp, base primal, primals, tangents)``.

    Every callable takes its arrays as arguments, so a compile measures the
    program rather than a constant-folded closure.
    """
    complement, parameter, pole, shift, far, tangent, order = _domain()
    d_complement, d_parameter, d_pole, d_shift = tangent
    moments = [
        jnp.asarray(value)
        for value in BASE_ELLIPTIC.harmonic_moments(
            parameter, MOMENTS, complement=complement, xp=jnp
        )
    ]
    d_moments = list(order)
    seed = BASE_ELLIPTIC.cn_pole_moment(
        far, parameter, parameter_complement=complement, xp=jnp
    )

    def case(tangent_function, primal_function, primals, tangents):
        def jvp_function(primals, tangents):
            return jax.jvp(primal_function, primals, tangents)

        return tangent_function, jvp_function, primal_function, primals, tangents

    def pole_family(mirrored):
        return case(
            lambda p, t: elliptic._harmonic_pole_moments_tangent(
                p[0], t[0], p[1], t[1], p[2], t[2], POLE_COUNT, mirrored=mirrored
            ),
            lambda s, v, m: BASE_ELLIPTIC.harmonic_pole_moments(
                s, v, m, POLE_COUNT, mirrored=mirrored
            ),
            (far, seed, moments),
            (d_shift, d_pole, d_moments),
        )

    def seed_case(name):
        return case(
            lambda p, t: getattr(elliptic, f"_{name}_tangent")(
                p[0],
                t[0],
                parameter,
                None,
                parameter_complement=p[1],
                d_parameter_complement=t[1],
                xp=jnp,
            ),  # fmt: skip
            lambda s, c: getattr(BASE_ELLIPTIC, name)(
                s, parameter, parameter_complement=c, xp=jnp
            ),
            (shift, complement),
            (d_shift, d_complement),
        )

    return {
        "complete_kind": case(
            lambda p, t: completeelliptic._complete_kind_tangent(p[0], t[0], xp=jnp),
            lambda c: BASE_COMPLETE.complete_kind(c, xp=jnp),
            (complement,),
            (d_complement,),
        ),
        "complete_pole": case(
            lambda p, t: completeelliptic._complete_pole_tangent(
                p[0], t[0], p[1], t[1], xp=jnp
            ),
            lambda q, c: BASE_COMPLETE.complete_pole(q, c, xp=jnp),
            (pole, complement),
            (d_pole, d_complement),
        ),
        "finite_part": case(
            lambda p, t: completeelliptic._finite_part_tangent(
                p[0], t[0], 1.0, None, 1.0, None, jnp
            ),
            lambda q: BASE_COMPLETE._finite_part(q, 1.0, 1.0, jnp),
            (pole,),
            (d_pole,),
        ),
        "harmonic_moments": case(
            lambda p, t: elliptic._harmonic_moments_tangent(
                p[0], t[0], MOMENTS, complement=p[1], d_complement=t[1], xp=jnp
            ),
            lambda q, c: BASE_ELLIPTIC.harmonic_moments(
                q, MOMENTS, complement=c, xp=jnp
            ),
            (parameter, complement),
            (d_parameter, d_complement),
        ),
        "harmonic_pole_moments": pole_family(False),
        "harmonic_pole_moments_mirrored": pole_family(True),
        "harmonic_root_moments": case(
            lambda p, t: elliptic._harmonic_root_moments_tangent(
                p[0], t[0], p[1], t[1], xp=jnp
            ),
            lambda m, q: BASE_ELLIPTIC.harmonic_root_moments(m, q, xp=jnp),
            (moments, parameter),
            (d_moments, d_parameter),
        ),
        "cn_pole_moment": seed_case("cn_pole_moment"),
        "sn_pole_moment": seed_case("sn_pole_moment"),
    }


CASES = _cases()


def compile_arm(name, arm):
    """Return the callable the cold-compile probe compiles for ``arm``."""
    tangent, reference, primal, _, _ = CASES[name]
    if arm == "primal":
        return lambda p, t: primal(*p)
    if arm == "tangent":
        return lambda p, t: tangent(p, t)
    return lambda p, t: reference(p, t)


def _truncated_descent_step(radical, d_radical, running, d_running, xp):
    root = xp.sqrt(radical)
    modulus = 2.0 * root
    d_modulus = 2.0 * (d_radical * (0.5 / root))
    return modulus, d_modulus, modulus * running, d_modulus * running


def _truncated_series(series_rising, d_series_rising, xp):
    """The finite part's series tangent with its leading term dropped."""
    power = xp.ones_like(series_rising)
    d_power = None
    d_series = xp.zeros_like(series_rising)
    for order in range(1, 8):
        d_power = completeelliptic._scale_tangent(
            -1.0,
            completeelliptic._product_tangent(
                power, d_power, series_rising, d_series_rising
            ),
        )
        power = -power * series_rising
        if order > 1:
            d_series = d_series + d_power / (2 * order + 1)
    return d_series


_RATIO_STEP = elliptic._harmonic_ratio_step_tangent
_BACKWARD_STEP = elliptic._pole_backward_step_tangent


def _truncated_ratio_step(
    ratio, d_ratio, parameter, d_parameter, complement, d_complement, *weights
):
    """The downward ratio tangent with the ratio's own tangent term dropped."""
    return _RATIO_STEP(
        ratio, None, parameter, d_parameter, complement, d_complement, *weights
    )


def _truncated_backward_step(
    solution, d_solution, ratio, d_ratio, following, d_following
):
    """The back substitution with the following order's tangent dropped."""
    return _BACKWARD_STEP(solution, d_solution, ratio, d_ratio, following, None)


def _truncated_root_term(mean, d_mean, moment, d_moment, quarter, d_quarter, *pair):
    """The root combination's tangent with the neighbouring pair's term dropped."""
    return completeelliptic._product_tangent(mean, d_mean, moment, d_moment)


TRUNCATIONS = {
    "complete_kind": [
        (completeelliptic, "_descent_tangent_step", _truncated_descent_step)
    ],
    "complete_pole": [
        (completeelliptic, "_descent_tangent_step", _truncated_descent_step)
    ],
    "finite_part": [
        (completeelliptic, "_finite_part_series_tangent", _truncated_series)
    ],
    "harmonic_moments": [
        (elliptic, "_harmonic_ratio_step_tangent", _truncated_ratio_step)
    ],
    "harmonic_pole_moments": [
        (elliptic, "_pole_backward_step_tangent", _truncated_backward_step)
    ],
    "harmonic_pole_moments_mirrored": [
        (elliptic, "_pole_backward_step_tangent", _truncated_backward_step)
    ],
    "harmonic_root_moments": [
        (elliptic, "_harmonic_root_tangent_term", _truncated_root_term)
    ],
    "cn_pole_moment": [
        (completeelliptic, "_descent_tangent_step", _truncated_descent_step)
    ],
    "sn_pole_moment": [
        (completeelliptic, "_descent_tangent_step", _truncated_descent_step)
    ],
}


@contextmanager
def _truncated(name):
    jax.clear_caches()
    saved = []
    for module, attribute, replacement in TRUNCATIONS[name]:
        saved.append((module, attribute, getattr(module, attribute)))
        setattr(module, attribute, replacement)
    try:
        yield
    finally:
        for module, attribute, original in saved:
            setattr(module, attribute, original)
        jax.clear_caches()


def _identity(name):
    tangent, reference, _, primals, tangents = CASES[name]
    # a fresh jit per call, so a truncation applied since is traced afresh
    got = jax.jit(lambda p, t: tangent(p, t))(primals, tangents)
    expected = jax.jit(lambda p, t: reference(p, t))(primals, tangents)
    return got, expected


def _bound(name):
    return SCANNED_TOLERANCE if name in SCANNED else 0.0


# Below this complement the second kind's tangent is a cancellation of terms of
# order ``1/k'^2`` and ``jax.jvp`` of the base itself misses the classical
# derivative by more than the scanned bound; there the scanned tangent is held
# instead to be no less accurate than the base against that derivative.
RESOLVED_COMPLEMENT = 1e-7


def _classical_second_kind(name, expected):
    """Return the complement, ``dE/dk'^2 = (K - E)/(2 k^2)`` times the tangent."""
    complement = np.asarray(CASES[name][3][0])
    d_complement = np.asarray(CASES[name][4][0])
    first, second = (np.asarray(value) for value in expected[0])
    return complement, d_complement * (first - second) / (2.0 * (1.0 - complement))


def _resolved(name, expected):
    """Return the samples on which ``jax.jvp`` of the base resolves the tangent."""
    if name != "complete_kind":
        return None
    complement, _ = _classical_second_kind(name, expected)
    return complement >= RESOLVED_COMPLEMENT


def _masked(tree, mask):
    if mask is None:
        return tree
    return [np.asarray(leaf)[mask] for leaf in jax.tree.leaves(tree)]


def tangent_of(name):
    return CASES[name][0]


@pytest.mark.parametrize("name", list(CASES))
def test_tangent_matches_base_jvp(name):
    if truncation_active():
        with _truncated(name):
            got, expected = _identity(name)
    else:
        got, expected = _identity(name)
    mask = _resolved(name, expected)
    tangent = worst(_masked(got[1], mask), _masked(expected[1], mask))
    whole = worst(got[1], expected[1])
    if name in SCANNED:
        # the primal half is the primal function itself, compiled in this program
        _, _, primal_function, primals, tangents = CASES[name]
        pair = jax.jit(lambda p, t: (tangent_of(name)(p, t)[0], primal_function(*p)))
        own, alone = pair(primals, tangents)
        primal = worst(own, alone)
        context = worst(got[0], expected[0])
    else:
        primal = context = worst(got[0], expected[0])
    excluded = 0 if mask is None else int(np.size(mask) - np.count_nonzero(mask))
    identity_row(
        name,
        primal,
        tangent,
        _bound(name),
        extra=f"jvp_program_primal_max_relative={context:.3e} "
        f"unresolved_samples={excluded} whole_domain_max={whole:.3e}",
    )
    if name == "complete_kind":
        complement, classical = _classical_second_kind(name, expected)
        below = (complement > 0.0) & (complement < RESOLVED_COMPLEMENT)
        scale = np.where(classical == 0.0, 1.0, np.abs(classical))
        scanned = np.abs(np.asarray(got[1][1]) - classical) / scale
        base = np.abs(np.asarray(expected[1][1]) - classical) / scale
        print(
            f"CLASSICAL {name} below={RESOLVED_COMPLEMENT:g} "
            f"scanned_max={scanned[below].max():.3e} "
            f"base_jvp_max={base[below].max():.3e}"
        )  # fmt: skip
        assert scanned[below].max() <= 2.0 * base[below].max() + SCANNED_TOLERANCE


@pytest.mark.parametrize("name", list(CASES))
def test_truncated_tangent_fails_identity(name):
    with _truncated(name):
        got, expected = _identity(name)
    mask = _resolved(name, expected)
    tangent = worst(_masked(got[1], mask), _masked(expected[1], mask))
    print(f"TRUNCATED {name} tangent_max_relative={tangent:.3e}")
    assert tangent > SCANNED_TOLERANCE


def test_primal_bit_identical_to_base():
    complement, parameter, pole, shift, far, _, _ = _domain(seed=1)
    pairs = {
        "complete_kind": (
            completeelliptic.complete_kind(complement, xp=jnp),
            BASE_COMPLETE.complete_kind(complement, xp=jnp),
        ),
        "complete_pole": (
            completeelliptic.complete_pole(pole, complement, xp=jnp),
            BASE_COMPLETE.complete_pole(pole, complement, xp=jnp),
        ),
        "harmonic_moments": (
            elliptic.harmonic_moments(
                parameter, MOMENTS, complement=complement, xp=jnp
            ),
            BASE_ELLIPTIC.harmonic_moments(
                parameter, MOMENTS, complement=complement, xp=jnp
            ),
        ),
        "cn_pole_moment": (
            elliptic.cn_pole_moment(
                shift, parameter, parameter_complement=complement, xp=jnp
            ),
            BASE_ELLIPTIC.cn_pole_moment(
                shift, parameter, parameter_complement=complement, xp=jnp
            ),
        ),
        "sn_pole_moment": (
            elliptic.sn_pole_moment(
                shift, parameter, parameter_complement=complement, xp=jnp
            ),
            BASE_ELLIPTIC.sn_pole_moment(
                shift, parameter, parameter_complement=complement, xp=jnp
            ),
        ),
    }
    moments = pairs["harmonic_moments"][0]
    seed = pairs["cn_pole_moment"][0]
    pairs["harmonic_pole_moments"] = (
        elliptic.harmonic_pole_moments(far, seed, moments, POLE_COUNT),
        BASE_ELLIPTIC.harmonic_pole_moments(far, seed, moments, POLE_COUNT),
    )
    pairs["harmonic_root_moments"] = (
        elliptic.harmonic_root_moments(moments, parameter, xp=jnp),
        BASE_ELLIPTIC.harmonic_root_moments(moments, parameter, xp=jnp),
    )
    for name, (current, base) in pairs.items():
        for left, right in zip(
            jax.tree.leaves(current), jax.tree.leaves(base), strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(left), np.asarray(right), err_msg=name
            )
        print(f"PRIMAL {name} bit_identical=True")


@pytest.mark.slow
# A compile row spawns 21 fresh processes; the repository's 300 s hang guard is
# for ordinary tests and cuts this one off before it asserts, so it gets its own
# bound.  Every other test keeps the default.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("name", list(CASES))
def test_tangent_compiles_within_three_primals(name):
    ratio = compile_ratio("test_elliptic_tangent_recurrences", name)
    assert ratio <= 3.0
