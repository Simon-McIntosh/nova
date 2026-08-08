"""Contract for the hexagonal neighbour stencil over a structured grid."""

import numpy as np
import pytest

from nova.geometry.hexstencil import HEX_RING, hex_stencil


def test_ring_is_six_point():
    """A hex tiling touches six neighbours, not the eight of a square raster."""
    assert HEX_RING.shape == (6, 2)


def test_row_is_centre_then_ring():
    """Seven columns for a six-point stencil: the centre is prepended."""
    assert hex_stencil((6, 5)).shape == (4 * 3, 7)


@pytest.mark.parametrize("shape", [(3, 3), (5, 5), (7, 4), (12, 9)])
def test_column_zero_is_the_interior_cell_itself(shape):
    """Column 0 indexes the cell a reduction is evaluated at."""
    n_x, n_z = shape
    stencil = hex_stencil(shape)
    interior = np.ravel_multi_index(
        (np.indices((n_x - 2, n_z - 2)).reshape(2, -1) + 1), (n_x, n_z)
    )
    assert np.array_equal(stencil[:, 0], interior)


@pytest.mark.parametrize("shape", [(5, 5), (7, 4), (12, 9)])
def test_neighbour_columns_carry_the_ring_offsets(shape):
    """Columns 1..6 are the centre displaced by each ring offset, in order."""
    n_x, n_z = shape
    stencil = hex_stencil(shape)
    x, z = np.unravel_index(stencil[:, 0], (n_x, n_z))
    for column, (dx, dz) in enumerate(HEX_RING, start=1):
        expected = np.ravel_multi_index((x + dx, z + dz), (n_x, n_z))
        assert np.array_equal(stencil[:, column], expected)


@pytest.mark.parametrize("shape", [(5, 5), (7, 4), (12, 9), (20, 31)])
def test_every_index_stays_on_the_grid(shape):
    """Border cells are dropped rather than clamped, so no index is invented."""
    n_x, n_z = shape
    stencil = hex_stencil(shape)
    x, z = np.unravel_index(stencil, (n_x, n_z))
    assert x.min() >= 0 and x.max() < n_x
    assert z.min() >= 0 and z.max() < n_z
    # a clamp or wrap would repeat the centre inside its own ring
    assert (stencil[:, 1:] != stencil[:, :1]).all()
