"""Bind a saved trip field to the lattice storage order its writer declares.

``trip_trajectory`` reshapes the solved flux onto ``profile.lattice.shape``,
and ``FluxLattice.shape`` is ``(radius, height)``: the lattice flattens its
nodes in C order over ``(radius, height)``
(``nova/equilibrium/conservation.py``).  An archived ``flux`` (and ``label``)
map is therefore radius-major.

Every reader in this package wants the plotting order ``(height, radius)``:
``fit_tensor_spline``, ``draw_flux_contours``, ``RegularGridInterpolator`` over
``(height, radius)`` and ``host_boundary_read``'s ``(nz, nr)`` field all
declare it.

The order cannot be inferred from the array shape.  A square lattice has the
same extent along both axes, so a shape test answers whichever branch it
happens to test first, and a radius-major 33x33 map read height-major is a
transposed field.  On frame 146's trip 1 that transposition moves the maximum
0.328 m away from the reported axis and invents a second lobe.
"""

from __future__ import annotations

import numpy as np

ARCHIVE_STORAGE_ORDER = ("radius", "height")
"""Order the archive writer flattens the lattice in, matching ``FluxLattice``."""

READER_ORDER = ("height", "radius")
"""Order every reader in this package declares its field in."""


def analysis_field(flux, radius, height) -> np.ndarray:
    """Return one archived flux map in the reader order ``(height, radius)``.

    The archive's declared storage order is ``ARCHIVE_STORAGE_ORDER`` and it
    decides the square case, where the two orders cannot be told apart by
    shape.  The shape is consulted only to accept a non-square map that
    already carries the reader order, or to refuse a map that is neither.
    """
    array = np.asarray(flux, dtype=float)
    radius = np.asarray(radius, dtype=float).reshape(-1)
    height = np.asarray(height, dtype=float).reshape(-1)
    if array.shape == (radius.size, height.size):
        return array.T
    if array.shape == (height.size, radius.size):
        return array
    raise ValueError(
        f"flux shape {array.shape} matches neither the archive storage order "
        f"{ARCHIVE_STORAGE_ORDER} = {(radius.size, height.size)} nor its "
        f"transpose {(height.size, radius.size)}"
    )
