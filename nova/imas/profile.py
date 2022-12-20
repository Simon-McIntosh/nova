"""Extract time slices from equilibrium IDS."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

from nova.imas.equilibrium import Equilibrium

if TYPE_CHECKING:
    import xarray


@dataclass
class GetSlice:
    """Convinence method to provide access to sliced equilibrium data."""

    time_index: int
    data: xarray.Dataset

    def __getitem__(self, key: str):
        """Regulate access to equilibrium dataset."""
        match key:
            case 'dpressure_dpsi':
                return
        return self.data[self.match(key)][self.time_index]

    def match(self, key: str) -> str:
        """Return key matched to internal naming convention."""
        match key:
            case 'p_prime':
                return 'dpressure_dpsi'
            case 'ff_prime':
                return 'f_df_dpsi'
            case str():
                return key
            case _:
                raise ValueError(f'invalid key {key}')


@dataclass
class Profile(Equilibrium):
    """Interpolation of profiles from an equilibrium time slice."""

    time_index: int = field(init=False, default=0)

    def __post_init__(self):
        """Define get slice."""
        super().__post_init__()
        self.get = GetSlice(self.time_index, self.data)

    @property
    def itime(self):
        """Manage time index."""
        return self.time_index

    @itime.setter
    def itime(self, time_index: int):
        self.time_index = time_index
        self.itime_update()

    def itime_update(self):
        """Clear cache following itime update. Extend as required."""
        self._clear_cached_properties()

    def _clear_cached_properties(self):
        """Clear cached properties."""
        for attr in ['boundary', 'psi_rbs', 'p_prime', 'ff_prime']:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    @cached_property
    def boundary(self):
        """Return trimmed boundary contour."""
        return self.get['boundary'][:self.get['boundary_length'].values].values

    def normalize(self, psi):
        """Return normalized poloidal flux."""
        return (psi - self.get['psi_axis']) / \
            (self.get['psi_boundary'] - self.get['psi_axis'])

    @cached_property
    def psi_rbs(self):
        """Return cached 2D RectBivariateSpline psi2d interpolant."""
        return self._rbs('psi2d')

    def _rbs(self, attr):
        """Return 2D RectBivariateSpline interpolant."""
        return RectBivariateSpline(self.data.r, self.data.z, self.get[attr]).ev

    def _interp1d(self, x, y):
        """Return 1D interpolant."""
        return interp1d(x, y, kind='quadratic', fill_value='extrapolate')

    @cached_property
    def p_prime(self):
        """Return cached pprime 1D interpolant."""
        return self._interp1d(self.data.psi_norm, self.get['p_prime'])

    @cached_property
    def ff_prime(self):
        """Return cached pprime 1D interpolant."""
        return self._interp1d(self.data.psi_norm, self.get['f_df_dpsi'])

    def plot_profile(self, attr='p_prime', axes=None):
        """Plot flux function interpolants."""
        self.set_axes(axes, '1d')
        psi_norm = np.linspace(0, 1, 500)
        self.axes.plot(self.data.psi_norm, self.get[attr], '.')
        self.axes.plot(psi_norm, getattr(self, attr)(psi_norm), '-')
        self.axes.set_ylabel(attr)
        self.axes.set_xlabel(r'$\psi_{norm}$')


if __name__ == '__main__':

    profile = Profile(105028, 1)
    profile.plot_2d()
