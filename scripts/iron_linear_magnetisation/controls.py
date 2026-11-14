"""Shared closed-shell geometry, source and analytic shielding control."""

import numpy as np
import shapely

from nova.biot.greens import greens_bz_br

SHELL_MU = 1000.0
SHELL_PROBES = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
UNIFORM_SIZES = (0.04, 0.02, 0.01)


def shell():
    """Return the polygonal half section of the 0.9–1 m spherical shell."""
    return (
        shapely.Point(0, 0)
        .buffer(1, 512)
        .difference(shapely.Point(0, 0).buffer(0.9, 512))
        .intersection(shapely.box(0, -1, 1, 1))
    )


def loop(points):
    """Return the field of a one-ampere loop of radius 50 mm."""
    bz, br = greens_bz_br(points[:, 0], points[:, 1], 0.05, 0.0)
    return np.column_stack((br, bz))


def exact_shielding():
    """Dipole shielding factor for the linear spherical shell."""
    mu = SHELL_MU
    return ((2 * mu + 1) * (mu + 2) - 2 * (mu - 1) ** 2 * 0.9**3) / (9 * mu)
