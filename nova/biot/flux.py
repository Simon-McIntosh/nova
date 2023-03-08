"""Manage contouring algorithums solved on a rectngular plasma grid."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.biotgrid import BiotGrid
from nova.biot.contour import Contour


@dataclass
class Flux(BiotGrid):
    """Extend BiotGrid class with flux contouring algorithums."""

    nplasma: float = 1000
    levels: int | np.ndarray = 50
    contour: Contour = field(init=False, repr=False)

    def solve(self):
        """Solve rectangular grid fit to first wall contour."""
        super().solve(self.nplasma, limit=0, index='plasma')

    def load_operators(self):
        """Extend BiotGrid.load_operators to initalize contour instance."""
        super().load_operators()
        self.contour = Contour(self.data.x2d, self.data.z2d, self.psi_,
                               levels=self.levels)
