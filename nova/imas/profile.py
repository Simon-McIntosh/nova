"""Extract time slices from equilibrium IDS."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

from nova.imas.database import IdsData
from nova.imas.equilibrium import Equilibrium, EquilibriumData
from nova.imas.getslice import GetSlice
from nova.imas.pf_active import PF_Active


@dataclass
class Profile(Equilibrium, GetSlice, IdsData):
    """Interpolation of profiles from an equilibrium time slice."""

    def __post_init__(self):
        """Build and merge ids datasets."""
        super().__post_init__()
        self.load_data(PF_Active)
        self.load_data(EquilibriumData)

    def update(self):
        """Clear cache following update to itime. Extend as required."""
        super().update()
        self.clear_cached_properties()

    def clear_cached_properties(self):
        """Clear cached properties."""
        for attr in ['p_prime', 'ff_prime']:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    @property
    def psi_axis(self):
        """Return on-axis poloidal flux."""
        return float(self['psi_axis'])

    @property
    def psi_boundary(self):
        """Return boundary poloidal flux."""
        return float(self['psi_boundary'])

    def normalize(self, psi):
        """Return normalized poloidal flux."""
        return (psi - self.psi_axis) / (self.psi_boundary - self.psi_axis)

    def denormalize(self, psi_norm):
        """Return poloidal flux."""
        return psi_norm * (self.psi_boundary - self.psi_axis) + self.psi_axis

    @property
    def psi_rbs(self):
        """Return cached 2D RectBivariateSpline psi2d interpolant."""
        return self._rbs('psi2d')

    @property
    def j_tor_rbs(self):
        """Return cached 2D RectBivariateSpline j_tor2d interpolant."""
        return self._rbs('j_tor2d')

    def _rbs(self, attr):
        """Return 2D RectBivariateSpline interpolant."""
        return RectBivariateSpline(self.data.r, self.data.z, self[attr]).ev

    def _interp1d(self, x, y):
        """Return 1D interpolant."""
        return interp1d(x, y, kind='quadratic', fill_value='extrapolate')

    @cached_property
    def p_prime(self):
        """Return cached p prime 1D interpolant."""
        return self._interp1d(self.data.psi_norm, self['p_prime'])

    @cached_property
    def ff_prime(self):
        """Return cached ff prime 1D interpolant."""
        return self._interp1d(self.data.psi_norm, self['ff_prime'])

    def plot_profile(self, attr='p_prime', axes=None):
        """Plot flux function interpolants."""
        self.set_axes('1d', axes=axes)
        psi_norm = np.linspace(0, 1, 500)
        self.axes.plot(self.data.psi_norm, self[attr], '.')
        self.axes.plot(psi_norm, getattr(self, attr)(psi_norm), '-')
        self.axes.set_ylabel(attr)
        self.axes.set_xlabel(r'$\psi_{norm}$')

    def plot_current(self, axes=None):
        """Plot current timeseries."""
        self.set_axes('1d', axes=axes)
        self.axes.plot(self.data.time, 1e-3*self.data.current)
        self.axes.set_xlabel('time s')
        self.axes.set_ylabel(r'$I$ kA')


if __name__ == '__main__':

    pulse, run = 135013, 2
    profile = Profile(pulse, run, 'iter', 0)
    profile.time = 300
    profile.plot_profile(attr='ff_prime')

    profile.plot_2d()
