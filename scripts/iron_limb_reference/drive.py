"""Area-averaged normal drive from the prescribed circular filaments."""

import numpy as np
from numpy.polynomial.legendre import leggauss

from nova.biot.greens import greens_psi
from scripts.prototypes.iron_limb_fill.bricks import drive


class FilamentDrive:
    """The case's pointwise field and its Stokes-integrated panel flux."""

    def __init__(self, sources):
        self.pointwise = drive(sources)
        self.radius, self.height, self.amps = (
            np.asarray(sources[k]) for k in ("r", "z", "amps")
        )
        self.last_quadrature = None

    def __call__(self, points):
        return self.pointwise(points)

    def normal_average(self, panels, order=16):
        """Integrate B.n by the vector-potential line integral around each face.

        For a circular source A_phi = Phi/(2*pi*R). Along a straight edge,
        A.dl = Phi*(-y*dx+x*dy)/(2*pi*R**2), whose geometric numerator is
        constant. Vertical and radial edges contribute exactly zero. The rule
        therefore integrates the near-filament drive without sampling B at a
        single panel centre.
        """
        node, weight = leggauss(order)
        node, weight = (node + 1) / 2, weight / 2
        start = panels.vertices.reshape(-1, 3)
        direction = panels.edges.reshape(-1, 3)
        cross = start[:, 0] * direction[:, 1] - start[:, 1] * direction[:, 0]
        live = np.flatnonzero(abs(cross) > 1e-18)
        integrals = np.zeros(len(start))
        for first in range(0, len(live), 64):
            rows = live[first : first + 64]
            points = start[rows, None] + node[None, :, None] * direction[rows, None]
            radius = np.hypot(points[..., 0], points[..., 1]).ravel()
            height = points[..., 2].ravel()
            phi = (
                greens_psi(radius[:, None], height[:, None], self.radius, self.height)
                @ self.amps
            )
            values = (phi / radius**2).reshape(len(rows), order)
            integrals[rows] = cross[rows] * (values @ weight) / (2 * np.pi)
        return integrals.reshape(-1, 3).sum(axis=1) / panels.areas
