"""A non-finite cell moment is named and cannot poison every other cell."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.forward_operator import (
    ForwardFluxOperator,
    NonFiniteCellMomentError,
    qualified_cell_current_moments,
    qualify_cell_current_moments,
)
from nova.equilibrium.source import CurrentNormalisationError
from nova.equilibrium.stencil_mesh import CellCurrentMoments


CELL_COUNT = 340
PLANTED = 103


def planted_moments() -> CellCurrentMoments:
    """Return concrete moments with one cell's current set to NaN."""
    index = np.arange(CELL_COUNT, dtype=np.float64)
    current = 10.0 + index
    current[PLANTED] = np.nan
    radial = 0.5 * current
    vertical = -0.25 * current
    return CellCurrentMoments(
        jnp.asarray(current), jnp.asarray(radial), jnp.asarray(vertical)
    )


def test_planted_non_finite_cell_is_named_and_local():
    """The receipt names the planted cell and no other cell becomes non-finite."""
    moments = planted_moments()
    qualified, indices = qualify_cell_current_moments(moments)

    assert indices == (PLANTED,)
    assert np.all(np.isfinite(np.asarray(qualified.cell_current)))
    assert np.all(np.isfinite(np.asarray(qualified.radial_moment)))
    assert np.all(np.isfinite(np.asarray(qualified.vertical_moment)))

    original = np.asarray(moments.cell_current)
    survivor = np.arange(CELL_COUNT) != PLANTED
    np.testing.assert_array_equal(
        np.asarray(qualified.cell_current)[survivor], original[survivor]
    )
    assert float(np.asarray(qualified.cell_current)[PLANTED]) == 0.0
    assert float(np.sum(np.asarray(qualified.cell_current))) == float(
        np.sum(original[survivor])
    )


def test_qualification_is_reported_before_the_total_current():
    """A qualified total is finite where the unqualified total is not."""
    moments = planted_moments()
    assert not np.isfinite(float(jnp.sum(moments.cell_current)))
    qualified, indices = qualify_cell_current_moments(moments)
    assert indices == (PLANTED,)
    assert np.isfinite(float(jnp.sum(qualified.cell_current)))


def test_amplitude_refuses_by_name_on_a_non_finite_total():
    """An inadmissible total current raises the named normalisation error."""
    moments = planted_moments()
    total = jnp.sum(moments.cell_current)
    with pytest.raises(CurrentNormalisationError):
        ForwardFluxOperator.current_normalisation_amplitude(jnp.asarray(37632.0), total)


def test_a_concrete_non_finite_cell_refuses_by_index():
    """The solve-path qualification raises instead of erasing a broken cell."""
    with pytest.raises(NonFiniteCellMomentError) as caught:
        qualified_cell_current_moments(planted_moments())
    assert caught.value.indices == (PLANTED,)
    assert str(PLANTED) in str(caught.value)


def test_a_finite_state_passes_the_solve_path_qualification():
    """A finite state is returned unchanged and nothing is raised."""
    index = np.arange(CELL_COUNT, dtype=np.float64)
    moments = CellCurrentMoments(
        jnp.asarray(index + 1.0),
        jnp.asarray(0.5 * (index + 1.0)),
        jnp.asarray(-0.25 * (index + 1.0)),
    )
    returned = qualified_cell_current_moments(moments)
    np.testing.assert_array_equal(
        np.asarray(returned.cell_current), np.asarray(moments.cell_current)
    )


def test_a_traced_state_is_qualified_without_a_value_to_judge():
    """A traced evaluation carries the qualification forward and names nothing."""
    traced = jax.jit(
        lambda scale: qualified_cell_current_moments(
            CellCurrentMoments(
                scale * jnp.ones(CELL_COUNT),
                jnp.zeros(CELL_COUNT),
                jnp.zeros(CELL_COUNT),
            )
        ).cell_current.sum()
    )
    assert np.isfinite(float(traced(jnp.asarray(2.0))))


def test_qualified_total_normalises():
    """The same plant still yields a finite amplitude once qualified."""
    qualified, _indices = qualify_cell_current_moments(planted_moments())
    total = jnp.sum(qualified.cell_current)
    amplitude = ForwardFluxOperator.current_normalisation_amplitude(
        jnp.asarray(float(total) / 0.5), total
    )
    assert np.isfinite(float(amplitude))
    assert float(amplitude) == pytest.approx(2.0)


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
