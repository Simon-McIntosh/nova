"""Manage contouring algorithums solved on a rectngular plasma grid."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.biotgrid import BiotGrid
from nova.biot.contour import Contour
from nova.biot.separatrix import LCFS

# pylint: disable=too-many-ancestors


@dataclass
class LevelSet(BiotGrid):
    """Extend BiotGrid class with levelset contouring algorithums."""

    nplasma: float = 1000
    levels: int | np.ndarray = 50
    contour: Contour = field(init=False, repr=False)

    def solve(self, *args, limit=0, index='plasma'):
        """Solve rectangular grid fit to first wall contour."""
        try:
            ngrid = args[0]
        except IndexError:
            ngrid = self.nplasma
        super().solve(ngrid, limit=limit, index=index)

    def load_operators(self):
        """Extend BiotGrid.load_operators to initalize contour instance."""
        super().load_operators()
        self.contour = Contour(self.data.x2d, self.data.z2d, self.psi_,
                               levels=self.levels)

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
        return LCFS(psi_levelset.points)
