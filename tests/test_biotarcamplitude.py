"""Contract for the arc's amplitude fold and its sign bookkeeping.

The claim under test is that Urankar's three printed folding cases (eqs 25 and
26) are one formula in the half-turn count, and that the formula is right because
of what the integrands are rather than because it happens to reproduce them.  So
the fold is checked three ways, each stronger than the last:

* against the printed cases, transcribed literally, over each of their own
  ranges -- which catches a transcription error but would not catch a shared
  misreading;
* against DIRECT integration of two model integrands with the paper's two
  symmetries, at amplitudes running over four turns -- which is the property the
  formula is derived from and does not depend on the paper being right;
* against the full turn, where the arc's own bookkeeping has to collapse onto
  ``-2 X(pi/2)``, the ring the shipped reduction evaluates.

The exactness of the amplitude's pair at a target ON an arc end is the fourth
thing tested, and it is not a nicety: that configuration is what the full turn is
made of, and a cosine of 6e-17 instead of zero costs the incomplete integrals
their whole accuracy there.
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from nova.biot.arcamplitude import ArcLimit, arc_limits, fold
from nova.jax.config import configure_dtypes


def _f64(value):
    """Construct an explicitly selected double-precision JAX value."""
    configure_dtypes()
    import jax.numpy as jnp

    return jnp.asarray(value, dtype=jnp.float64)


QUARTER = 0.5 * np.pi


def odd_integrand(angle):
    """A model with the symmetry of the potential's radial row.

    Odd about zero and about a quarter turn, half-turn periodic -- so its
    antiderivative is EVEN in the amplitude and half-turn periodic outright.
    """
    return np.sin(2.0 * angle) * (3.0 + np.cos(4.0 * angle))


def even_integrand(angle):
    """A model with the symmetry of the potential's azimuthal row.

    Even about zero and about a quarter turn, half-turn periodic -- so its
    antiderivative is ODD in the amplitude and gains twice the quarter-turn value
    per half turn.
    """
    return np.cos(2.0 * angle) * (2.0 + np.sin(2.0 * angle) ** 2) + 1.0


def antiderivative(integrand, amplitude, panels=400, nodes=32):
    """Return ``integral_0^amplitude`` by a panelled rule, signed in the limit."""
    amplitude = np.asarray(amplitude, dtype=float)
    node, weight = leggauss(nodes)
    step = amplitude / panels
    edge = step[..., None] * np.arange(panels)
    angle = edge[..., None] + 0.5 * step[..., None, None] * (node + 1.0)
    return 0.5 * step * (integrand(angle) @ weight).sum(axis=-1)


def printed_case(value_at_quarter, folded, amplitude):
    """Return the odd row's fold exactly as eqs 25 and 26 print it.

    Transcribed literally, cases and all, so the one-formula claim is checked
    against the source rather than against a paraphrase of it.  ``folded`` is
    evaluated at the case's own ``|theta_i|``.
    """
    magnitude = abs(amplitude)
    if magnitude <= QUARTER:
        return np.sign(amplitude) * folded
    if magnitude <= 3.0 * QUARTER:
        theta = np.pi - magnitude
        return np.sign(amplitude) * (2.0 * value_at_quarter - np.sign(theta) * folded)
    return np.sign(amplitude) * (4.0 * value_at_quarter - folded)


def case_amplitudes():
    """Return amplitudes covering each printed case and both signs of theta."""
    return [
        0.1,
        1.0,
        QUARTER,
        1.7,
        np.pi - 0.3,
        np.pi,
        np.pi + 0.4,
        3.0 * QUARTER - 1e-6,
        3.0 * QUARTER + 1e-6,
        2.0 * np.pi - 0.2,
        -0.6,
        -2.0,
        -np.pi - 0.5,
    ]


def limit_at(amplitude):
    """Return the fold for a single amplitude, through the arc's own entry point.

    Common target turns are deliberately absent from the returned pair, so the
    amplitude is encoded in the arc sweep and read from its upper end:
    ``alpha = (pi + psi)/2`` with ``psi = target - end``.
    """
    return arc_limits(0.0, 0.0, np.pi - 2.0 * amplitude)[1]


def local_fields(limit):
    """Return the continuously valued fields of one fold as a numeric vector."""
    return np.asarray(
        [limit.amplitude, limit.sine, limit.cosine, limit.parity], dtype=float
    )


@pytest.mark.parametrize("amplitude", case_amplitudes())
def test_the_fold_reproduces_each_of_the_printed_cases(amplitude):
    """One formula in the half-turn count against three transcribed cases."""
    limit = limit_at(amplitude)
    quarter = float(antiderivative(even_integrand, QUARTER))
    folded = float(antiderivative(even_integrand, float(limit.amplitude)))
    got = 2.0 * float(limit.turns) * quarter + float(limit.parity) * folded
    expected = printed_case(quarter, folded, amplitude)
    assert got == pytest.approx(expected, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("amplitude", case_amplitudes() + [7.0, -7.0, 12.5])
def test_the_fold_reproduces_direct_integration_for_both_symmetries(amplitude):
    """The property the formula is derived from, over four turns of amplitude.

    Stronger than the printed cases because it does not use them: the model
    integrands carry the paper's two symmetries and nothing else, and the fold has
    to reproduce their antiderivatives at an amplitude the cases do not reach.
    """
    limit = limit_at(amplitude)
    folded_amplitude = float(limit.amplitude)
    for integrand, quarter in (
        (odd_integrand, None),
        (even_integrand, float(antiderivative(even_integrand, QUARTER))),
    ):
        direct = float(antiderivative(integrand, amplitude))
        folded = float(antiderivative(integrand, folded_amplitude))
        if quarter is None:
            got = folded
        else:
            got = 2.0 * float(limit.turns) * quarter + float(limit.parity) * folded
        assert got == pytest.approx(direct, rel=1e-12, abs=1e-12)


def test_the_folded_amplitude_stays_inside_the_quarter_the_reduction_covers():
    """Whatever the separation, and that is the point of folding at all."""
    separation = np.linspace(-4.0 * np.pi, 4.0 * np.pi, 2001)
    for limit in arc_limits(separation, 0.0, 1.0):
        assert np.all(limit.amplitude >= 0.0)
        assert np.all(limit.amplitude <= QUARTER + 1e-15)
        assert np.all(limit.cosine >= -1e-15)
        assert np.all(np.abs(limit.parity) == 1.0)


@pytest.mark.parametrize("azimuth", [0.0, 0.7, 2.0, -1.3, np.pi])
def test_a_full_turn_collapses_onto_twice_the_quarter_turn_value(azimuth):
    """The ring the shipped reduction evaluates, out of the arc's bookkeeping.

    Both ends of a full turn sit at the same azimuth, so their folded amplitudes
    coincide and their parities oppose; everything cancels but the half-turn
    counts, which differ by exactly one.  The odd rows are left with
    ``-2 X(pi/2)`` and the even rows with nothing at all -- which is the full-turn
    specialisation the closed form was built on, recovered rather than assumed.
    """
    limits = arc_limits(azimuth, 0.0, 2.0 * np.pi)
    quarter = float(antiderivative(even_integrand, QUARTER))
    odd_total = 0.0
    even_total = 0.0
    for limit in limits:
        folded = float(antiderivative(even_integrand, float(limit.amplitude)))
        odd_total += fold(limit, folded, quarter)
        even_total += fold(limit, float(antiderivative(odd_integrand, limit.amplitude)))
    assert odd_total == pytest.approx(-2.0 * quarter, rel=1e-13)
    assert even_total == pytest.approx(0.0, abs=1e-13)


@pytest.mark.parametrize("winding", [-3, -1, 1, 2, 3])
def test_multi_turn_arcs_retain_only_their_own_integer_winding(winding):
    """Local fields coincide while the upper relative count carries orientation."""
    lower, upper = arc_limits(0.7, 0.0, 2.0 * np.pi * winding)
    assert np.array_equal(local_fields(lower), local_fields(upper))
    assert float(upper.turns - lower.turns) == -winding

    quarter = 1.23456789012345
    folded = np.sqrt(2.0)
    total = fold(lower, folded, quarter) + fold(upper, folded, quarter)
    assert total == pytest.approx(-2.0 * winding * quarter, rel=2e-16)


def test_a_target_on_an_arc_end_gives_an_exact_quarter_turn():
    """Exactly, pair and all -- the configuration the full turn is made of."""
    limit, _ = arc_limits(0.75, 0.75, 2.0)
    assert float(limit.amplitude) == QUARTER
    assert float(limit.sine) == 1.0
    assert float(limit.cosine) == 0.0
    assert float(limit.parity) == 1.0


@pytest.mark.parametrize("turns", [1, 2, -1, -3])
def test_a_whole_turn_of_target_azimuth_leaves_the_arc_unchanged(turns):
    """Both ends shift by a half turn, so what the assembly differences does not.

    The periodicity is a property of the fold rather than a normalisation the
    caller has to apply, and this is what says so.
    """
    quarter = float(antiderivative(even_integrand, QUARTER))

    def total(azimuth):
        answer = 0.0
        for limit in arc_limits(azimuth, -0.4, 1.9):
            folded = float(antiderivative(even_integrand, float(limit.amplitude)))
            answer += fold(limit, folded, quarter)
        return answer

    assert total(0.6 + 2.0 * np.pi * turns) == pytest.approx(total(0.6), rel=1e-13)


@pytest.mark.parametrize("winding", [-3, -1, 1, 2, 5])
def test_upper_endpoint_winding_changes_only_its_relative_turn(winding):
    """The formal sign is negative because ``alpha`` contains target minus end."""
    baseline = arc_limits(0.6, -0.4, 1.9)
    shifted = arc_limits(0.6, -0.4, 1.9 + 2.0 * np.pi * winding)
    assert np.allclose(local_fields(shifted[0]), local_fields(baseline[0]), atol=2e-15)
    assert np.allclose(local_fields(shifted[1]), local_fields(baseline[1]), atol=2e-15)
    assert float(shifted[0].turns) == float(baseline[0].turns)
    assert float(shifted[1].turns - baseline[1].turns) == -winding


def test_huge_common_angles_lose_no_more_than_their_input_phase_spacing():
    """No large turn terms survive; only phase bits absent from the inputs move."""
    baseline = arc_limits(0.6, -0.4, 1.9)
    offset = 1.0e12
    shifted = arc_limits(offset + 0.6, offset - 0.4, offset + 1.9)
    bound = np.spacing(offset)
    for reference, moved in zip(baseline, shifted):
        assert np.max(np.abs(local_fields(moved) - local_fields(reference))) <= bound
        assert float(moved.turns) == float(reference.turns)


def test_a_huge_target_winding_is_removed_before_the_two_ends_are_evaluated():
    """The common count stays small and the phase error is bounded by one ulp."""
    baseline = arc_limits(0.6, -0.4, 1.9)
    winding = 159_154_943_092
    offset = 2.0 * np.pi * winding
    shifted = arc_limits(0.6 + offset, -0.4, 1.9)
    for reference, moved in zip(baseline, shifted):
        difference = np.max(np.abs(local_fields(moved) - local_fields(reference)))
        assert difference <= np.spacing(offset)
        assert abs(float(moved.turns)) <= 1.0


def test_the_two_ends_carry_opposite_assembly_weights():
    """Eq 13's ``-(-1)^(i+1)``, so an arc's value is the plain sum of its ends."""
    lower, upper = arc_limits(0.3, -1.0, 1.0)
    assert lower.weight == -1.0
    assert upper.weight == 1.0


def test_the_fold_traces_and_batches():
    """No branch on a value anywhere in it, which is why an arc can be tiled."""
    jax = pytest.importorskip("jax")
    configure_dtypes()
    jnp = jax.numpy
    azimuth = np.linspace(-3.0, 3.0, 17)

    @jax.jit
    def traced(azimuth):
        return [
            (limit.amplitude, limit.sine, limit.cosine, limit.turns, limit.parity)
            for limit in arc_limits(azimuth, -0.4, 1.9, xp=jnp)
        ]

    device = traced(_f64(azimuth))
    host = arc_limits(azimuth, -0.4, 1.9)
    for one, other in zip(host, device):
        for field, value in zip(
            (one.amplitude, one.sine, one.cosine, one.turns, one.parity), other
        ):
            assert np.max(np.abs(np.asarray(value) - field)) < 1e-15


def test_the_local_phase_has_the_same_jvp_on_host_and_device():
    """The exposed array namespace keeps derivatives through the smooth fields."""
    jax = pytest.importorskip("jax")
    configure_dtypes()
    jnp = jax.numpy

    def device_signature(azimuth):
        limits = arc_limits(azimuth, -0.7, 1.1, xp=jnp)
        return jnp.stack(
            [
                field
                for limit in limits
                for field in (limit.amplitude, limit.sine, limit.cosine)
            ]
        )

    point = _f64(0.2)
    value, tangent = jax.jvp(device_signature, (point,), (_f64(1.0),))
    step = 1.0e-6

    def host_signature(azimuth):
        return np.concatenate(
            [local_fields(limit)[:3] for limit in arc_limits(azimuth, -0.7, 1.1)]
        )

    difference = (host_signature(0.2 + step) - host_signature(0.2 - step)) / (
        2.0 * step
    )
    assert np.allclose(np.asarray(value), host_signature(0.2), rtol=1e-14, atol=1e-14)
    assert np.allclose(np.asarray(tangent), difference, rtol=1e-8, atol=1e-10)


def test_the_limit_carries_its_own_fields():
    """A shape check, so a consumer can rely on what it is handed."""
    limit, _ = arc_limits(np.zeros(3), 0.0, 1.0)
    assert isinstance(limit, ArcLimit)
    for field in (limit.amplitude, limit.sine, limit.cosine, limit.turns, limit.parity):
        assert np.shape(field) == (3,)


def test_all_three_angle_inputs_broadcast_to_each_limit():
    """A scalar target and lower end still broadcast over an upper-end vector."""
    limits = arc_limits(0.3, -0.4, np.asarray([0.2, 0.8, 1.4]))
    for limit in limits:
        for field in (
            limit.amplitude,
            limit.sine,
            limit.cosine,
            limit.turns,
            limit.parity,
        ):
            assert np.shape(field) == (3,)
