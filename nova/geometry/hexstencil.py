"""Hexagonal neighbour stencil over a structured grid.

A hex-tiled plasma grid stores its cells in a rectangular ``(n_x, n_z)`` index
array, but the tiling is hexagonal: successive rows are offset by half a cell,
so a cell touches SIX neighbours rather than the eight a square raster would
give. In axial index coordinates those six sit at::

    (-1, 0)  (0, -1)  (1, -1)  (1, 0)  (0, 1)  (-1, 1)

which is the ring this module builds, and the same six offsets the host
categoriser walks in :meth:`nova.biot.fieldnull.DataNull.categorize_2d`.

The ring is returned with the cell itself prepended, so a row is
``[centre, n0 ... n5]`` and the array is ``(n_interior, 7)`` wide. Seven columns
for a six-point stencil: column 0 is the point a reduction is being evaluated
AT, columns 1 to 6 are what it is evaluated AGAINST. Keeping both in one row is
what lets a null search, a biquadratic fit, or a sign-change count read a cell
and its neighbourhood as a single fixed-shape gather.

Only interior cells appear. A border cell has neighbours off the grid, and the
alternative to dropping it is a clamp or a wrap, either of which invents a
neighbour value and would place spurious nulls around the rim.

The unstructured counterpart is
:meth:`nova.biot.plasmagrid.PlasmaGrid.tessellate`, which recovers the same
six-neighbour rings from a Delaunay triangulation of hex filament centroids and
packs them centre-first identically, so both grid kinds present one stencil
contract to the readers above them.
"""

import numpy as np

__all__ = ["hex_stencil", "HEX_RING"]

#: The six axial-coordinate offsets of a hexagonal neighbourhood, angle-ordered.
HEX_RING: np.ndarray = np.array([(-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)])


def hex_stencil(shape: tuple[int, int]) -> np.ndarray:
    """Return flat indices of every interior cell and its six hex neighbours.

    Parameters
    ----------
    shape
        Grid extent as ``(n_x, n_z)``.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``((n_x - 2) * (n_z - 2), 7)``. Each row holds
        indices into the grid FLATTENED in C order: column 0 the centre cell,
        columns 1 to 6 its neighbours in :data:`HEX_RING` order.
    """
    n_x, n_z = shape
    interior = np.indices((n_x - 2, n_z - 2)).reshape(2, -1, 1) + 1
    patch = np.r_[np.zeros((1, 2), int), HEX_RING]
    return np.ravel_multi_index(interior + patch.T[:, np.newaxis], (n_x, n_z))
