"""Extrapolate equilibria beyond separatrix."""
from dataclasses import dataclass, field, fields
from typing import Union

import numpy.typing as npt

from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import Machine


@dataclass
class Grid:
    """Specify interpolation grid attributes."""

    number: int
    limit: Union[float, list[float]] = 0.15
    index: Union[str, slice, npt.ArrayLike] = 'plasma'

    @property
    def grid_attrs(self) -> dict:
        """Return grid attributes."""
        return {attr: getattr(self, attr)
                for attr in [attr.name for attr in fields(Grid)]}


@dataclass
class Extrapolate(Machine, Grid):
    """Extrapolate equlibrium beyond separatrix ids."""

    filename: str = 'iter'
    dplasma: float = -200
    geometry: list[str] = field(default_factory=lambda: ['pf_active', 'wall'])

    @property
    def hash_attrs(self):
        """Extend machine hash attributes."""
        return super().hash_attrs | self.grid_attrs

    def build(self, **kwargs):
        """Build frameset and interpolation grid."""
        super().build(**kwargs)
        self.grid.solve(**self.grid_attrs)
        return self.store(self.filename)


if __name__ == '__main__':

    coilset = Extrapolate(1000, dplasma=-500)
    coilset.sloc['plasma', 'Ic'] = -15e6
    coilset.sloc['coil', 'Ic'] = 15e3

    eq = Equilibrium(114101, 41)

    #coilset.grid.solve2d(eq.data.r2d.values[::20, ::20],
    #                     eq.data.z2d.values[::20, ::20])
    coilset.plasma.separatrix = eq.data.boundary[0]


    itime = 0
    eq.plot_2d(itime, 'psi', colors='C3', levels=21)
    eq.plot_boundary(itime)


    coilset.plot('plasma')
    coilset.grid.plot()
