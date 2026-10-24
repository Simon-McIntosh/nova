"""A non-finite cell moment is refused by index and never zeroed."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.forward_operator import (
    ForwardFluxOperator,
    cell_moment_nonfinite,
    refuse_non_finite_cell_moments,
)
from nova.equilibrium.source import CurrentNormalisationError
from nova.equilibrium.stencil_mesh import CellCurrentMoments


CELL_COUNT = 340
PLANTED = 103


def planted_moments(*, radial: bool) -> CellCurrentMoments:
    """Return moments with one cell's radial moment, or current, set to NaN."""
    index = np.arange(CELL_COUNT, dtype=np.float64)
    current = 10.0 + index
    radial_values = 0.5 * current
    vertical = -0.25 * current
    if radial:
        radial_values[PLANTED] = np.nan
    else:
        current[PLANTED] = np.nan
    return CellCurrentMoments(
        jnp.asarray(current), jnp.asarray(radial_values), jnp.asarray(vertical)
    )


def test_a_non_finite_radial_moment_is_named_and_no_cell_is_zeroed():
    """The mask covers every moment component and the values are left alone."""
    moments = planted_moments(radial=True)
    mask, indices = cell_moment_nonfinite(moments)

    assert indices == (PLANTED,)
    assert int(np.sum(np.asarray(mask))) == 1
    assert bool(np.asarray(mask)[PLANTED])
    original = np.asarray(moments.radial_moment)
    assert not np.isfinite(original[PLANTED])
    np.testing.assert_array_equal(np.asarray(moments.radial_moment), original)
    assert np.all(np.isfinite(np.asarray(moments.cell_current)))


def test_a_planted_radial_moment_is_refused_by_index():
    """A finite cell current does not hide a non-finite radial moment."""
    with pytest.raises(CurrentNormalisationError) as caught:
        refuse_non_finite_cell_moments(planted_moments(radial=True))
    assert caught.value.indices == (PLANTED,)
    assert str(PLANTED) in str(caught.value)


def test_a_non_finite_cell_current_is_refused_by_index():
    """The same refusal fires when the total current itself is non-finite."""
    with pytest.raises(CurrentNormalisationError) as caught:
        refuse_non_finite_cell_moments(planted_moments(radial=False))
    assert caught.value.indices == (PLANTED,)


def test_a_finite_state_is_admitted():
    """A finite state sets no cell in the mask and raises nothing."""
    index = np.arange(CELL_COUNT, dtype=np.float64)
    moments = CellCurrentMoments(
        jnp.asarray(index + 1.0),
        jnp.asarray(0.5 * (index + 1.0)),
        jnp.asarray(-0.25 * (index + 1.0)),
    )
    mask, indices = cell_moment_nonfinite(moments)
    assert indices == ()
    assert not bool(np.any(np.asarray(mask)))
    np.testing.assert_array_equal(
        np.asarray(refuse_non_finite_cell_moments(moments)), np.asarray(mask)
    )


def test_a_traced_step_reports_the_mask_and_the_offending_index():
    """The production route is traced, so the mask must survive a jit boundary."""
    moments = planted_moments(radial=True)

    def step(current, radial, vertical):
        one = CellCurrentMoments(current, radial, vertical)
        mask, _indices = cell_moment_nonfinite(one)
        return mask.any(), jnp.argmax(mask), jnp.sum(~jnp.isfinite(radial))

    flagged, first, counted = jax.jit(step)(
        moments.cell_current, moments.radial_moment, moments.vertical_moment
    )
    assert bool(np.asarray(flagged))
    assert int(np.asarray(first)) == PLANTED
    assert int(np.asarray(counted)) == 1


def test_the_traced_mask_is_refused_by_the_host_boundary():
    """The host refusal names the index the traced step reported."""
    moments = planted_moments(radial=True)
    _flag, first, _count = jax.jit(
        lambda current, radial, vertical: _step_flags(current, radial, vertical)
    )(moments.cell_current, moments.radial_moment, moments.vertical_moment)
    index = int(np.asarray(first))
    with pytest.raises(CurrentNormalisationError) as caught:
        refuse_non_finite_cell_moments(planted_moments(radial=True))
    assert index in caught.value.indices


def _step_flags(current, radial, vertical):
    mask, _indices = cell_moment_nonfinite(
        CellCurrentMoments(current, radial, vertical)
    )
    return mask.any(), jnp.argmax(mask), jnp.sum(mask)


def test_an_admissible_amplitude_is_reported_admissible():
    """The amplitude carries a flag rather than a non-finite signal."""
    total = jnp.asarray(100.0)
    assert bool(
        np.asarray(
            ForwardFluxOperator.current_normalisation_admissible(
                jnp.asarray(50.0), total
            )
        )
    )
    amplitude = ForwardFluxOperator.current_normalisation_amplitude(
        jnp.asarray(50.0), total
    )
    assert float(np.asarray(amplitude)) == pytest.approx(0.5)


def test_an_inadmissible_amplitude_is_flagged_and_refused():
    """A non-finite total is inadmissible and the eager route raises."""
    total = jnp.asarray(np.nan)
    flagged = ForwardFluxOperator.current_normalisation_admissible(
        jnp.asarray(50.0), total
    )
    assert not bool(np.asarray(flagged))
    with pytest.raises(CurrentNormalisationError):
        ForwardFluxOperator.current_normalisation_amplitude(jnp.asarray(50.0), total)


def test_the_traced_amplitude_never_scales_a_cell_by_nan():
    """An inadmissible traced amplitude is zero, and its flag says why."""
    flagged, amplitude = jax.jit(
        lambda target, unscaled: (
            ForwardFluxOperator.current_normalisation_admissible(target, unscaled),
            ForwardFluxOperator.current_normalisation_amplitude(target, unscaled),
        )
    )(jnp.asarray(50.0), jnp.asarray(np.nan))
    assert not bool(np.asarray(flagged))
    assert np.isfinite(float(np.asarray(amplitude)))


def test_arc_root_tangent_is_finite_across_the_discriminant_seam():
    """The arc root keeps a finite tangent where its discriminant changes sign."""
    from nova.equilibrium.separatrix_clip import _traced_quadratic_arc

    coefficient = jnp.asarray([[-0.25, 0.0, 0.0, 1.0, 0.0, 1.0]])
    tangent = jnp.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    centre = jnp.asarray([[0.0, 0.0]])
    scale = jnp.asarray([[1.0, 1.0]])
    start = jnp.asarray([[-1.0, 0.0]])
    end = jnp.asarray([[1.0, 0.0]])

    def arc(weights):
        return _traced_quadratic_arc(start, end, weights, centre, scale)

    value, direction = jax.jvp(arc, (coefficient,), (tangent,))
    assert bool(np.all(np.isfinite(np.asarray(value))))
    assert bool(np.all(np.isfinite(np.asarray(direction))))