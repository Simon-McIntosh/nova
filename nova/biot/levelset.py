"""Manage contouring algorithums solved on a rectngular plasma grid."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nova.biot.grid import Grid
from nova.biot.contour import Contour
from nova.biot.separatrix import PlasmaShape
from nova.geometry.kdtree import KDTree

# pylint: disable=too-many-ancestors


@dataclass
class LevelSet(Grid):
    """Extend Grid class with levelset contouring algorithums."""

    levels: int | np.ndarray = 50
    contour: Contour = field(init=False, repr=False)
    tree: KDTree = field(init=False, repr=False)

    def solve(self, number=None, limit=0, index='plasma'):
        """Solve rectangular grid fit to first wall contour."""
        super().solve(number, limit=limit, index=index)

    def load_operators(self):
        """Extend Grid.load_operators to initalize contour instance."""
        super().load_operators()
        if self.number is not None:
            self.contour = Contour(self.data.x2d, self.data.z2d, self.psi_,
                                   levels=self.levels)
            self.tree = KDTree(np.c_[self.data.x2d.data.flatten(),
                                     self.data.z2d.data.flatten()])

    def check_contour(self):
        """Check contour flux operators."""
        self.check_plasma('Psi')
        self.check_source('psi')

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        if attr == 'contour':
            self.check_contour()
        return super().__getattribute__(attr)

    def lcfs(self, psi):
        """Return last closed flux surface."""
        psi_levelset = self.contour.closedlevelset(psi)
        return PlasmaShape(psi_levelset.points)

    def query(self, other: np.ndarray):
        """Return point index from kdtree."""
        return self.tree.query(other)[1]
