r"""The wall polish's selection is continuous across the promoted bracket.

The wall read returns the extremum of the poloidal flux along the material
boundary: a quadratic in arclength is fitted through the three wall nodes
bracketing the largest sampled value.  When the bracket is chosen by a discrete
argmax, the boundary flux of a limited plasma is piecewise-smooth in the flux
state, with a slope jump wherever a Newton step moves the promoted wall node.
The selection is now a smooth blend of the winning bracket and its strongest
adjacent neighbour, weighted by the sampled-flux margin between them, so the
boundary flux is continuous and differentiable across a bracket flip while a
well-separated extremum is reproduced bit-for-bit.

This module pins the two properties the change trades on.

The value where the selection is unambiguous is unchanged to the bit: a wall
maximum whose nearest competitor sits more than the blend window below it is
read as the pure winning-bracket fit, exactly as the discrete selection read
it.  That holds on the limited Solov'ev fixture and on the diverted fixture,
and it is what keeps a diverted row bit-identical - the X-point flux that
defines a diverted boundary does not route through the wall polish at all, and
even the wall read itself is untouched wherever its extremum is unambiguous.

The value where the selection flips is now smooth: on a conductor image that
promotes a different wall node across the step, the autodiff tangent agrees
with a central difference at the contracted precision instead of showing the
step-independent disagreement a bracket jump used to leave behind.
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
    from nova.geometry import select
    from nova.jax.config import configure_dtypes

#: Solov'ev seed coefficients, wall span and conductor ring of the bootstrapped
#: free-boundary machine the forward-solve contracts use.
P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
CONDUCTORS = 16
WALL_POINTS = 61

#: Amplitude, as a fraction of the flux span, of the flux image used to lift
#: the wall off the seed surface so its extremum is strict.
STRICT_EXTREMUM_AMPLITUDE = 1.0e-2

#: Lift amplitudes, as fractions of the flux span, that make the wall maximum
#: unambiguous on each fixture.  All sit well outside the selection's blend
#: window, so the read must reproduce the discrete winner bracket bit-for-bit.
STABLE_MARGIN_FRACTIONS = (0.05, 0.1, 0.2, 0.5)

#: The conductor images the earlier measurement found to flip the promoted
#: bracket, open at 1.6e-5 and 4.8e-5 relative to a central difference.  The
#: selection-continuous read must close both to 1e-6.
FLIP_CONDUCTORS = (2, 6)

#: Central-difference steps, as a fraction of the flux span of the state being
#: differentiated.
COARSE_STEP = 1.0e-4
FINE_STEP = 1.0e-6

#: Agreement required between the tangent and the coarse-step central
#: difference across a bracket flip.
COARSE_AGREEMENT = 1.0e-6


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


@pytest.fixture(scope="module")
def diverted_wall_read():
    """Return the wall read of a manufactured field with a saddle inside.

    The vertical potential has stationary points at two heights on one side of
    the midplane, so the field carries a saddle within the wall and the map is
    diverted rather than wall-limited.
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


def _flux_span(state):
    """Return the flux span of one wall state, or its scale when flat."""
    span = float(jnp.max(state) - jnp.min(state))
    return span if span > 0.0 else float(jnp.max(jnp.abs(state)))


def _strict_extremum_state(seed, image, span):
    """Return a wall flux lifted off the seed by one conductor's image.

    The lifting image and probed images are different conductors, so states are
    not built from the direction they are differentiated along.
    """
    lift = image[:, CONDUCTORS // 2]
    lift = lift / float(jnp.max(jnp.abs(lift)))
    return seed + STRICT_EXTREMUM_AMPLITUDE * span * lift


def _unit_peak(direction):
    """Return the flux image scaled to unit peak amplitude."""
    return direction / float(jnp.max(jnp.abs(direction)))


def _promoted_index(state):
    """Return the wall node whose bracket the polish fits."""
    index, roll = select.traced_wall_index(state)
    return int(index - roll) % int(state.shape[0])


def _reference_discrete_value(null, state):
    """Return the discrete wall read this continuous one replaces.

    The reference is the pre-continuity selection frozen here: the quadratic
    stationary value through the three nodes bracketing the argmax, with no
    blending.  It shares the fitting kernels with the live read, so where the
    continuous selection reduces to the winner bracket the two are the same to
    the bit.
    """
    index, roll = select.traced_wall_index(state)

    def rolled(value):
        return jnp.where(roll != 0, jnp.roll(value, roll), value)

    x_cluster = jax.lax.dynamic_slice(rolled(null.coordinate[:, 0]), [index - 1], 3)
    z_cluster = jax.lax.dynamic_slice(rolled(null.coordinate[:, 1]), [index - 1], 3)
    psi_cluster = jax.lax.dynamic_slice(rolled(state), [index - 1], 3)
    length_cluster = select.length_2d(x_cluster, z_cluster, array_namespace=jnp)
    coefficients = select.traced_quadratic_wall(length_cluster, psi_cluster)
    coordinate = select.wall_length(coefficients, array_namespace=jnp)
    value = coefficients[0] * coordinate**2 + coefficients[1] * coordinate
    return value + coefficients[2]


def test_selection_is_continuous_across_the_promoted_bracket(wall_read):
    """A dense sweep across the flip shows no value jump.

    The discrete selection the continuous one replaces steps from one bracket's
    fit to the other's the moment the promoted node changes; the continuous
    read must pass through that point without an outlier increment.
    """
    null, seed, image = wall_read
    state = _strict_extremum_state(seed, image, _flux_span(seed))
    span = _flux_span(state)
    direction = _unit_peak(image[:, FLIP_CONDUCTORS[0]])
    step = FINE_STEP * span
    # the base sits on the tie between the two top wall nodes, so the sweep
    # must straddle the flip across its range, not at any one sampled position
    assert _promoted_index(state - step * direction) != _promoted_index(
        state + step * direction
    )
    positions = np.linspace(-10.0, 10.0, 101)
    values = np.array(
        [float(null(state + offset * step * direction, 1)[2]) for offset in positions]
    )

    increments = np.abs(np.diff(values))
    median = np.median(increments)
    assert median > 0.0
    # no increment may stand out: a bracket jump would exceed every smooth
    # increment by the full value spread of the two fits.
    assert np.max(increments) < 20.0 * median


def test_stable_bracket_value_is_bit_identical_to_the_discrete_read(wall_read):
    """An unambiguous maximum is reproduced to the bit on the limited fixture."""
    null, seed, image = wall_read
    state = _strict_extremum_state(seed, image, _flux_span(seed))
    span = _flux_span(state)
    for fraction in STABLE_MARGIN_FRACTIONS:
        state = state.at[30].add(fraction * span)
        live = float(null(state, 1)[2])
        reference = float(_reference_discrete_value(null, state))
        assert live == reference
        assert live == pytest.approx(reference, rel=1e-12)


def test_stable_bracket_value_is_bit_identical_to_the_discrete_read_diverted(
    diverted_wall_read,
):
    """An unambiguous maximum is reproduced to the bit on the diverted fixture.

    On a diverted field the boundary flux is the X-point flux, which does not
    route through the wall polish; this pins the stronger fact that even the
    wall read itself is bit-identical wherever its extremum is unambiguous, so
    a diverted row rides on no changed number at all.
    """
    null, state, _image = diverted_wall_read
    span = _flux_span(state)
    for fraction in STABLE_MARGIN_FRACTIONS:
        state = state.at[21].add(fraction * span)
        live = float(null(state, 1)[2])
        reference = float(_reference_discrete_value(null, state))
        assert live == reference
        assert live == pytest.approx(reference, rel=1e-12)


def test_flat_seed_value_is_bit_identical_to_the_discrete_read(wall_read):
    """The flat seed stays the discrete winner read to the bit.

    Where the wall lies exactly on a flux surface the sampled flux is
    bit-exactly constant, every node is a maximiser, and there is no selection
    to smooth; the continuous blend keeps the discrete winner bracket, so the
    kink behaviour is unchanged.
    """
    null, seed, _image = wall_read
    live = float(null(seed, 1)[2])
    reference = float(_reference_discrete_value(null, seed))
    assert live == reference


@pytest.mark.parametrize("conductor", FLIP_CONDUCTORS)
def test_flip_bracket_tangent_matches_the_central_difference(wall_read, conductor):
    """Across a bracket flip the tangent now matches the central difference.

    These conductor images change the promoted wall node under a coarse step;
    under the discrete selection they left a step-independent central-difference
    disagreement of 1.6e-5 (conductor 2) and 4.8e-5 (conductor 6).  The
    continuous selection makes the boundary flux smooth there, so the central
    difference is a valid derivative estimate and the tangent must close to the
    contracted precision.
    """
    null, seed, image = wall_read
    state = _strict_extremum_state(seed, image, _flux_span(seed))
    span = _flux_span(state)
    direction = _unit_peak(image[:, conductor])
    step = COARSE_STEP * span
    # the base sits on the tie between two top nodes, so one side of the step
    # promotes a different wall node than the other
    assert _promoted_index(state - step * direction) != _promoted_index(
        state + step * direction
    )

    tangent = float(jax.jvp(lambda psi: null(psi, 1)[2], (state,), (direction,))[1])
    forward = float(null(state + step * direction, 1)[2])
    backward = float(null(state - step * direction, 1)[2])
    central = (forward - backward) / (2.0 * step)
    assert tangent == pytest.approx(central, rel=COARSE_AGREEMENT)
