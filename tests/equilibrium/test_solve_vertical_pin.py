"""Vertical pin on the converged solve: an elongated carrier held by one unknown.

The carrier is an elongated analytic equilibrium posed on the current centroid
``(R, Z)`` with elongation ``kappa``. Its vertical row reads the elongation-
driven force ``c * Z * (Z - Z_free)`` plus the external vertical-field source
``-beta * u``; the vertical stiffness falls with elongation, so the free root
is uncontrolled in height. One compensating unknown -- the current in the
declared vertical-field coil set, or unit amplitude on a uniform radial field
-- is the last element of the state; the pin row holds the centroid height.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import solve
from nova.jax.config import configure_dtypes

KAPPA = 1.8
Z_FREE = 0.5
R_TARGET = 0.0
COUPLING = 0.6
FIELD_LEVERAGE = 1.0

SEED = jnp.asarray([1.2, 0.62])
TARGETS = (0.7, 0.9)
OPERANDS = jnp.asarray([0.0])


def free_map(state, operands):
    """The elongated carrier with the vertical field held at zero."""
    del operands
    radius, height = state[0], state[1]
    radial = (radius - R_TARGET) + COUPLING * height * height
    vertical = COUPLING * height * (height - Z_FREE)
    return jnp.stack((radial, vertical))


def pinned_map(state, operands):
    """The carrier with the compensating unknown ``u = state[-1]`` as its source."""
    del operands
    radius, height, unknown = state[0], state[1], state[2]
    radial = (radius - R_TARGET) + COUPLING * height * height
    vertical = COUPLING * height * (height - Z_FREE) - FIELD_LEVERAGE * unknown
    return jnp.stack((radial, vertical))


def centroid_height(state, operands):
    """The pinned observation: the current-centroid height of the state."""
    del operands
    return state[1]


def _pin(target):
    return solve.VerticalPin(
        target=jnp.asarray(target),
        pin_scale=jnp.asarray(1.0),
        initial=jnp.asarray(0.0),
    )


@pytest.mark.slow
def test_elongated_carrier_free_solve_drifts_in_height():
    """The negative control: with no pin the centroid lands off the commanded height."""
    configure_dtypes()
    assert KAPPA >= 1.6
    result = solve.fixed_point(free_map, SEED, OPERANDS)
    assert bool(result.converged)
    free_height = float(result.state[1])
    # the free root sits at Z_free, a controlled-target-independent location
    np.testing.assert_allclose(free_height, Z_FREE, atol=1e-9)
    for target in TARGETS:
        assert abs(free_height - target) > 1e-3


@pytest.mark.slow
def test_pinned_solve_holds_the_target_height_at_two_targets():
    configure_dtypes()
    pinned = []
    for target in TARGETS:
        result = solve.pinned_fixed_point(
            pinned_map, centroid_height, SEED, OPERANDS, _pin(target)
        )
        assert bool(result.converged)
        assert float(result.residual) <= 1e-12
        height = float(centroid_height(result.state, OPERANDS))
        assert abs(height - target) <= 1e-10
        pinned.append(result)
    assert float(pinned[0].state[2]) != float(pinned[1].state[2])


@pytest.mark.slow
def test_one_compiled_program_serves_every_target():
    """A single lowered-and-compiled pin program solves both target heights."""
    configure_dtypes()
    program = jax.jit(
        lambda state, ops, pin: solve.pinned_fixed_point(
            pinned_map, centroid_height, state, ops, pin
        )
    )
    compiled = program.lower(SEED, OPERANDS, _pin(TARGETS[0])).compile()
    heights = []
    for target in TARGETS:
        result = compiled(SEED, OPERANDS, _pin(target))
        assert bool(result.converged)
        assert float(result.residual) <= 1e-12
        heights.append(float(centroid_height(result.state, OPERANDS)))
    for target, height in zip(TARGETS, heights, strict=True):
        assert abs(height - target) <= 1e-10
