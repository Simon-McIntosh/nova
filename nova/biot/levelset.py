"""Manage contouring algorithums solved on a rectngular plasma grid."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.grid import Grid
from nova.biot.contour import Contour
from nova.geometry.kdtree import Proximate

# pylint: disable=too-many-ancestors


@dataclass
class LevelSet(Proximate, Grid):
    """Extend Grid class with levelset contouring algorithums."""

    levels: int | np.ndarray = 50
    kd_factor: float = np.inf
    contour: Contour = field(init=False, repr=False)

    def __call__(self, psi):
        """Return flux surface."""
        return self.contour.closedlevelset(psi).points

    def solve(self, number=None, limit=0, index="plasma"):
        """Solve rectangular grid fit to first wall contour."""
        super().solve(number, limit=limit, index=index)

    def load_operators(self):
        """Extend Grid.load_operators to initalize contour instance."""
        super().load_operators()
        if self.number is not None:
            self.contour = Contour(
                self.data.x2d, self.data.z2d, self.psi_, levels=self.levels
            )
            self.kd_points = np.c_[
                self.data.x2d.data.flatten(), self.data.z2d.data.flatten()
            ]

    def check_contour(self):
        """Check contour flux operators."""
        self.check_plasma("Psi")
        self.check_source("psi")

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        if attr == "contour":
            self.check_contour()
        return super().__getattribute__(attr)

    def plot_levelset(self, psi, closed=True, **kwargs):
        """Expose contour.plot_levelset."""
        self.contour.plot_levelset(psi, closed, **kwargs)
