"""Interpolate equilibria within separatrix."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import mu_0
import xarray

from nova.imas.profile import Current, Profile

if TYPE_CHECKING:
    import pandas

from nova.imas import Equilibrium, Ids, Machine


@dataclass
class Grid:
    """
    Specify interpolation grid.

    Parameters
    ----------
    ngrid : {int, 'ids'}, optional
        Grid dimension. The default is 5000.

        - int: use input to set aproximate total node number
        - ids: aproximate total node number extracted from equilibrium ids.
    limit : {float, list[float], 'ids'}, optional
        Grid bounds. The default is 0.25.

        - float: expansion relative to coilset index. Must be greater than -1.
        - list[float]: explicit grid bounds [rmin, rmax, zmin, zmax].
        - ids: bounds extracted from from equilibrium ids.
    index : {'plasma', 'coil', slice, pandas.Index}
        Filament index from which relative grid limits are set.
    equilibrium : Equilibrium, optional
        Equilibrium ids required for equilibrium derived grid dimensions.
        The default is False

    Examples
    --------
    Manualy specify grid relitive to coilset:
    >>> Grid(100, 0, 'coil').grid_attrs
    {'ngrid': 100, 'limit': 0, 'index': 'coil'}

    Specify grid relitive to equilibrium ids.
    >>> equilibrium = Equilibrium(130506, 403)
    >>> Grid(50, 'ids', equilibrium=equilibrium).grid_attrs
    {'ngrid': 50, 'limit': [2.75, 8.9, -5.49, 5.51], 'index': 'plasma'}

    Extract exact grid from equilibrium ids.
    >>> grid = Grid('ids', 'ids', equilibrium=equilibrium)
    >>> grid.grid_attrs['ngrid']
    8385

    Raises attribute error when grid initialied with unset equilibrium ids:
    >>> Grid(1000, 'ids', 'coil')
    Traceback (most recent call last):
        ...
    AttributeError: equilibrium ids is None
    require valid ids when limit:ids or ngrid:1000 == 'ids'
    """

    ngrid: int | str = 5000
    limit: float | list[float] | str = 0.25
    index: str | slice | pandas.Index = 'plasma'
    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)

    def __post_init__(self):
        """Update grid attributes for equilibrium derived properties."""
        self.update_grid()
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    @property
    def grid_attrs(self) -> dict:
        """Return grid attributes."""
        return {attr: getattr(self, attr)
                for attr in ['ngrid', 'limit', 'index']}

    def update_grid(self):
        """Update  and update grid limits."""
        if self.limit != 'ids' and self.ngrid != 'ids':
            return
        if len(self.data) == 0:
            raise AttributeError('data is empty\n'
                                 'require valid ids data when '
                                 f'limit:{self.limit} '
                                 f'or ngrid:{self.ngrid} == \'ids\'')
        if self.limit == 'ids':  # Load grid limit from equilibrium ids.
            if self.data.grid_type != 1:
                raise TypeError('ids limits only valid for rectangular grids'
                                f'{self.data.grid_type} != 1')
            limit = [self.data.r.values, self.data.z.values]
            if self.ngrid == 'ids':
                self.limit = limit
            else:
                self.limit = [limit[0][0], limit[0][-1],
                              limit[1][0], limit[1][-1]]
        if self.ngrid == 'ids':
            self.ngrid = self.data.dims['r'] * self.data.dims['z']


@dataclass
class Operate(Machine, Current, Profile, Grid, Equilibrium):
    """
    Extend Machine with default values for Operate class.

    Extract coil and plasma currents from ids and apply to CoilSet.

    Parameters
    ----------
    pf_active: Ids | bool, optional
        pf active IDS. The default is True
    pf_passive: Ids | bool, optional
        pf passive IDS. The default is False
    wall: Ids | bool, optional
        wall IDS. The default is True
    """

    pf_active: Ids | bool = True
    pf_passive: Ids | bool = False
    wall: Ids | bool = True
    nplasma: int = 2500

    @property
    def group_attrs(self):
        """Return group attributes."""
        return super().group_attrs | self.grid_attrs

    def solve_biot(self):
        """Extend machine solve biot to include extrapolation grid."""
        super().solve_biot()
        if self.sloc['plasma'].sum() > 0:
            self.grid.solve(**self.grid_attrs)

    def update(self):
        """Extend itime update."""
        super().update()
        self.update_plasma()
        #self.sloc['coil', 'Ic'] = self['current']

    def update_plasma(self):
        """Ionize plasma filaments and set turn number."""
        self.plasma.separatrix = self.boundary
        self.sloc['plasma', 'Ic'] = self['ip']
        ionize = self.aloc['ionize']
        radius = self.aloc['x'][ionize]
        height = self.aloc['z'][ionize]
        psi = self.psi_rbs(radius, height)
        psi_norm = self.normalize(psi)
        current_density = radius * self.p_prime(psi_norm) + \
            self.ff_prime(psi_norm) / (mu_0 * radius)
        current_density *= -2*np.pi
        current = current_density * self.aloc['area'][ionize]
        self.aloc['nturn'][ionize] = current / current.sum()


if __name__ == '__main__':

    operate = Operate(105028, 1, limit=0)  # DINA
    # operate = Operate(130506, 403, limit=0)  # CORSICA

    operate.itime = 50
    operate.plot('plasma')
    operate.grid.plot()
