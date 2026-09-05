r"""Autodiff tangent of the wall-point boundary flux against a central difference.

The wall read returns the extremum of the poloidal flux along the material
boundary: the three wall nodes bracketing the largest sampled value are fitted
with a quadratic in arclength and the fit's stationary value is the boundary
flux of a limited plasma.  Every constraint row, moment or elimination that
differentiates a limited boundary differentiates that value, so the tangent has
to be the derivative of the quantity it names.

The selection is continuous: the discrete argmax names the winning three-node
bracket and its strongest adjacent neighbour, and the two are blended by a
smooth weight in the sampled-flux margin, so the boundary flux of a limited
plasma is smooth across a bracket flip, not piecewise.  That smoothness is what
lets the tangent be pinned against a central difference even where a step
promotes a different wall node.

Two regimes remain, and they are different questions.

Where the wall flux has a strict extremum the value is a smooth function of the
sampled flux and the tangent is its derivative; the contracts below pin that
against a central difference along one conductor's flux image at a step that
crosses the bracket flip as well as at steps that do not.

Where the wall lies exactly ON a flux surface the sampled flux is bit-exactly
constant, every node is a maximiser, and the extremum is a KINK: perturbing one
way promotes the node where the perturbation is largest, perturbing the other
way promotes the node where it is smallest, and the two one-sided slopes are
different numbers.  A central difference there returns their mean, which is not
a derivative of anything and which no tangent can reproduce.  The continuous
selection deliberately keeps the discrete winner bracket on such a flat wall -
there is no selection to smooth - so the tangent remains the slope of that
winner bracket, and the kink is pinned here explicitly, one-sided slope by
one-sided slope, rather than left to be rediscovered as an apparent tangent
error.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.biot.null import Null1D
    from nova.jax.config import configure_dtypes

#: Solov'ev seed coefficients, wall span and conductor ring of the bootstrapped
#: free-boundary machine the forward-solve contracts use.
P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
CONDUCTORS = 16
WALL_POINTS = 61

#: Central-difference steps, as a fraction of the flux span of the state being
#: differentiated.  A conductor's flux image carries about a microweber per
#: ampere, so a step counted in amperes would move the flux ten decades below
#: the span and the difference quotient would be pure cancellation noise; the
#: probe direction is scaled to unit peak flux so the step is literally the
#: stated fraction of the span.
COARSE_STEP = 1.0e-4
FINE_STEP = 1.0e-6

#: Agreement required between the tangent and the coarse-step central
#: difference.  The step is far enough above the subtraction's cancellation
#: floor that what remains is the difference quotient's own truncation.  The
#: wall read's selection is continuous, so the central difference is a valid
#: derivative estimate even across a step that promotes a different bracket,
#: and the figure is the contraction of the flip-case contract from the
#: one-sided quotient that the discrete selection forced.
COARSE_AGREEMENT = 1.0e-6

#: Agreement required at the fine step.  Differencing flux values of order one
#: weber over a step of order one microweber cancels about eleven decades, so a
#: double-precision central difference carries some 1e-5 of relative noise of
#: its own at this step and cannot certify a tangent more tightly than that.
#: The looser figure brackets that noise instead of predicting it.
FINE_AGREEMENT = 1.0e-5

#: Agreement required where a step crosses a selection boundary and the central
#: difference is not admissible because the wall read's extremum is not a local
#: bracket flip.  The kept side's slope is then measured by a second-order
#: one-sided quotient, whose truncation is the step squared rather than the
#: step, so the figure reaches 1e-5 at the fine step.
ONE_SIDED_AGREEMENT = 1.0e-5

#: Agreement required for the same second-order one-sided quotient at the
#: coarse step.  The single-null fixture used to keep the wall outside a
#: strictly-converged bracket is far steeper along its wall than the Solov'ev
#: machine, so its third derivative leaves a coarse-step quotient truncation
#: near 1e-4 that the fine-step figure cannot bracket.
COARSE_ONE_SIDED_AGREEMENT = 1.0e-3

#: Amplitude, as a fraction of the flux span, of the flux image used to lift
#: the wall off the seed surface so its extremum is strict.  Any amplitude
#: above the arithmetic floor does so; this one leaves a wall-flux spread four
#: decades above it.
STRICT_EXTREMUM_AMPLITUDE = 1.0e-2


def _terms():
    """Return the Solov'ev quartic, offset and vertical coefficients."""
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius, height):
    """Return the analytic seed flux [Wb]."""
    alpha, offset, beta = _terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _wall_loop(points=WALL_POINTS):
    """Return a material boundary lying on one seed flux surface."""
    alpha, offset, beta = _terms()
    wall_flux = _solovev(AXIS_RADIUS, 0.0) - SEED_SPAN
    inner, outer = np.sqrt(np.sort(np.roots([alpha, offset, -wall_flux])))
    centre, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
    angle = 2 * np.pi * np.arange(points) / points
    radius = centre + half * np.cos(angle)
    argument = np.clip((wall_flux - _solovev(radius, 0.0)) / beta, 0.0, None)
    return np.c_[radius, np.sign(np.sin(angle)) * np.sqrt(argument)]


def _conductor_ring():
    """Return the external conductor positions of the bootstrapped machine."""
    angle = 2 * np.pi * np.arange(CONDUCTORS) / CONDUCTORS
    return np.c_[
        AXIS_RADIUS + 0.62 * np.cos(angle),
        0.62 * np.sin(angle),
    ]


def _flux_image(target, source, section=0.05):
    """Return the total-flux coupling [Wb/A] of one source set on one target."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


@pytest.fixture(scope="module")
def wall_read():
    """Return the wall read, the seed wall flux and the conductor images."""
    configure_dtypes()
    wall = _wall_loop()
    null = Null1D(jnp.asarray(wall, dtype=jnp.float64))
    seed = jnp.asarray(_solovev(wall[:, 0], wall[:, 1]))
    image = jnp.asarray(_flux_image(wall, _conductor_ring()))
    return null, seed, image


def _flux_span(state):
    """Return the flux span of one wall state, or its scale when flat."""
    span = float(jnp.max(state) - jnp.min(state))
    return span if span > 0.0 else float(jnp.max(jnp.abs(state)))


def _unit_peak(direction):
    """Return the flux image scaled to unit peak amplitude."""
    return direction / float(jnp.max(jnp.abs(direction)))


def _tangent(null, state, direction):
    """Return the autodiff tangent of the wall-point flux along one image."""
    return float(jax.jvp(lambda psi: null(psi, 1)[2], (state,), (direction,))[1])


def _central_difference(null, state, direction, step):
    """Return the central difference of the wall-point flux along one image."""
    forward = float(null(state + step * direction, 1)[2])
    backward = float(null(state - step * direction, 1)[2])
    return (forward - backward) / (2.0 * step)


def _promoted_index(state):
    """Return the wall node whose cluster the polish fits."""
    from nova.geometry import select  # noqa: PLC0415

    index, roll = select.traced_wall_index(state)
    return int(index - roll) % int(state.shape[0])


def _one_sided(null, state, direction, step):
    """Return the forward and backward difference quotients."""
    here = float(null(state, 1)[2])
    forward = (float(null(state + step * direction, 1)[2]) - here) / step
    backward = (here - float(null(state - step * direction, 1)[2])) / step
    return forward, backward


def _second_order_quotient(null, state, direction, step, forward):
    """Return the second-order one-sided quotient on the kept side.

    The three-point formula has truncation of the order of the step squared
    times the value's third derivative, rather than of the step itself, so the
    kept side's slope is pinned more tightly than a first-order quotient could.
    """
    here = float(null(state, 1)[2])
    if forward:
        one = float(null(state + step * direction, 1)[2])
        two = float(null(state + 2 * step * direction, 1)[2])
        return (-3.0 * here + 4.0 * one - two) / (2.0 * step)
    one = float(null(state - step * direction, 1)[2])
    two = float(null(state - 2 * step * direction, 1)[2])
    return (3.0 * here - 4.0 * one + two) / (2.0 * step)


def _assert_difference_quotient(
    null, state, direction, step, tangent, agreement, one_sided_agreement
):
    """Pin the tangent against the difference quotient the fixture admits.

    The polish blends the winning three-node bracket with its strongest
    adjacent neighbour, so the wall flux is smooth across a bracket flip and
    the central difference is a valid derivative estimate wherever the step
    keeps the promoted node.  Where a step changes the promoted node by more
    than one cell the extremum is not a local bracket flip at all, and only
    the side that keeps the base bracket measures the slope the tangent
    differentiates; that side is read through a second-order one-sided
    quotient.
    """
    base = _promoted_index(state)
    ahead = _promoted_index(state + step * direction)
    behind = _promoted_index(state - step * direction)
    if ahead == base and behind == base:
        central = _central_difference(null, state, direction, step)
        assert tangent == pytest.approx(central, rel=agreement)
        return
    assert ahead == base or behind == base
    kept = _second_order_quotient(null, state, direction, step, forward=ahead == base)
    assert tangent == pytest.approx(kept, rel=one_sided_agreement)


def _strict_extremum_state(seed, image, span):
    """Return a wall flux lifted off the seed surface by one conductor image.

    The lifting image and the probed image are different conductors, so the
    state being differentiated is not built from the direction it is
    differentiated along.
    """
    lift = image[:, CONDUCTORS // 2]
    lift = lift / float(jnp.max(jnp.abs(lift)))
    return seed + STRICT_EXTREMUM_AMPLITUDE * span * lift


def test_seed_wall_flux_is_exactly_constant(wall_read):
    """The seed puts the material boundary on a flux surface, to the last bit."""
    _null, seed, _image = wall_read
    assert float(jnp.max(seed) - jnp.min(seed)) == 0.0


def test_flat_wall_extremum_is_a_kink_not_a_tangent_error(wall_read):
    """On a flat wall the one-sided slopes bracket the central difference.

    Both slopes are read off the probed image directly: promoting the node
    where the image is largest gives the forward slope and promoting the node
    where it is smallest gives the backward slope, so the central difference is
    their mean and is not the derivative of the wall flux along the image.  The
    wall lies exactly on a flux surface, so the continuous selection keeps the
    discrete winner bracket (there is no selection to smooth) and the tangent
    returns that bracket's slope, which is the derivative that exists; the
    tie-break position is at the rightmost wall node, where this image also
    peaks, so the tangent is the forward slope to the accuracy of that
    bracket's own one-sided read.
    """
    null, seed, image = wall_read
    direction = _unit_peak(image[:, 0])
    step = COARSE_STEP * _flux_span(seed)
    forward, backward = _one_sided(null, seed, direction, step)
    central = _central_difference(null, seed, direction, step)

    assert _promoted_index(seed + step * direction) == int(jnp.argmax(direction))
    assert _promoted_index(seed - step * direction) == int(jnp.argmin(direction))
    assert central == pytest.approx(0.5 * (forward + backward), rel=1e-9)
    assert abs(forward - backward) > 0.5 * abs(forward)

    tangent = _tangent(null, seed, direction)
    assert tangent == pytest.approx(forward, rel=1e-3)


@pytest.mark.parametrize("conductor", range(CONDUCTORS))
def test_strict_wall_extremum_tangent_matches_the_central_difference(
    wall_read, conductor
):
    """Every conductor's image differentiates the strict wall extremum right.

    The selection is continuous, so even where the step promotes a different
    wall node the boundary flux is smooth and the central difference is a valid
    derivative estimate; the tangent must match it through the flip.
    """
    null, seed, image = wall_read
    state = _strict_extremum_state(seed, image, _flux_span(seed))
    span = _flux_span(state)
    assert span > 1.0e-4

    direction = _unit_peak(image[:, conductor])
    tangent = _tangent(null, state, direction)
    for step, agreement in (
        (COARSE_STEP, COARSE_AGREEMENT),
        (FINE_STEP, FINE_AGREEMENT),
    ):
        central = _central_difference(null, state, direction, step * span)
        assert tangent == pytest.approx(central, rel=agreement)


@pytest.fixture(scope="module")
def diverted_wall_read():
    """Return the wall read of a manufactured field with a saddle inside.

    The vertical potential has stationary points at two heights on one side of
    the midplane, so the field carries a saddle within the wall and the map is
    diverted rather than wall-limited.  The wall read is the same functional
    either way, and here its cluster sits far from the flat case.
    """
    configure_dtypes()
    angle = 2.0 * np.pi * np.arange(64) / 64
    wall = np.c_[1.7 + 1.15 * np.cos(angle), 1.22 * np.sin(angle)]
    roots = (0.0, 0.35, 0.52, 0.70, 0.98)
    potential = np.polyint(np.poly1d(roots, r=True))
    flux = -((wall[:, 0] - 1.7) ** 2) - 100.0 * potential(wall[:, 1])
    conductor = np.c_[1.7 + 1.9 * np.cos(angle[::4]), 2.0 * np.sin(angle[::4])]
    null = Null1D(jnp.asarray(wall, dtype=jnp.float64))
    return null, jnp.asarray(flux), jnp.asarray(_flux_image(wall, conductor))


def test_diverted_wall_extremum_tangent_matches_the_central_difference(
    diverted_wall_read,
):
    """The same contract holds on a field carrying a saddle within the wall.

    The manufactured field holds two near-equal wall maxima on opposite sides
    of the wall, so a probe that tips one over the other changes the promoted
    node by more than one cell; the kept side is read through the second-order
    one-sided quotient.
    """
    null, state, image = diverted_wall_read
    span = _flux_span(state)
    assert span > 0.0

    for conductor in range(image.shape[1]):
        direction = _unit_peak(image[:, conductor])
        tangent = _tangent(null, state, direction)
        for step, agreement, one_sided in (
            (COARSE_STEP, COARSE_AGREEMENT, COARSE_ONE_SIDED_AGREEMENT),
            (FINE_STEP, FINE_AGREEMENT, ONE_SIDED_AGREEMENT),
        ):
            _assert_difference_quotient(
                null,
                state,
                direction,
                step * span,
                tangent,
                agreement,
                one_sided,
            )


def test_wall_extremum_value_is_the_largest_sampled_flux(wall_read):
    """The polished value stays inside the bracket its cluster supplies."""
    null, seed, image = wall_read
    state = _strict_extremum_state(seed, image, _flux_span(seed))
    value = float(null(state, 1)[2])
    assert value >= float(jnp.max(state))
    assert value - float(jnp.max(state)) <= float(jnp.max(state) - jnp.min(state))
