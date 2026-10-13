"""gs.apply images moments booked on a support read without reading support."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import gs
from nova.equilibrium.stencil_mesh import CellCurrentMoments


class RecordingOperator:
    """A stand-in whose every support-reading entry point fails the test."""

    def __init__(self):
        self.imaged = None

    def internal(self, *arguments):
        raise AssertionError("the operator's own support read was used")

    def _internal_on_partition(self, *arguments):
        raise AssertionError("the operator's own support read was used")

    def residual_shadow_mask(self, *arguments):
        raise AssertionError("the operator derived its own residual shadow")

    def coupling_current_moments(self, moments):
        return CellCurrentMoments(
            moments.cell_current, 2 * moments.radial_moment, 3 * moments.vertical_moment
        )

    def current_moment_image(self, moments):
        return moments.cell_current + moments.radial_moment + moments.vertical_moment

    def _exclude_shadow_residual(self, psi, image, requested_class, shadow):
        return jnp.where(shadow, psi, image)


def _case():
    moments = CellCurrentMoments(jnp.arange(4.0), jnp.ones(4), jnp.full(4, 2.0))
    psi = jnp.linspace(-1.0, 1.0, 4)
    external = jnp.full(4, 0.5)
    shadow = jnp.asarray([True, False, False, False])
    return psi, external, moments, shadow


def test_booked_moments_are_imaged_through_the_operand_interface():
    psi, external, moments, shadow = _case()
    mapped = jax.jit(
        lambda *operands: gs.apply(
            operands[0],
            operands[1],
            RecordingOperator(),
            moments=operands[2],
            shadow=operands[3],
        )
    )(psi, external, moments, shadow)
    image = np.asarray(external + np.arange(4.0) + 2.0 + 6.0)
    expected = np.where(np.asarray(shadow), np.asarray(psi), image)
    np.testing.assert_allclose(mapped, expected, rtol=0, atol=0)


def test_booked_moments_without_their_shadow_are_refused():
    psi, external, moments, _shadow = _case()
    with pytest.raises(ValueError, match="residual shadow"):
        gs.apply(psi, external, RecordingOperator(), moments=moments)
