"""Extract time slices from equilibrium IDS."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

from nova.frame.baseplot import Plot
from nova.imas.database import IdsData
from nova.imas.equilibrium import Equilibrium
from nova.imas.getslice import GetSlice
from nova.imas.pf_active import PF_Active


@dataclass
class Profile(Plot, GetSlice, IdsData):
    """Interpolation of profiles from an equilibrium time slice."""

    def build(self):
        """Merge ids datasets."""
        self.load_data(PF_Active)
        self.load_data(Equilibrium)
        super().build()

    def update(self):
        """Clear cache following update to itime. Extend as required."""
        super().update
        self._clear_cached_properties()

    def _clear_cached_properties(self):
        """Clear cached properties."""
        for attr in ['psi_axis', 'psi_boundary',
                     'psi_rbs', 'j_tor_rbs', 'p_prime', 'ff_prime']:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    @cached_property
    def psi_axis(self):
        """Return on-axis poloidal flux."""
        return float(self['psi_axis'])

    @cached_property
    def psi_boundary(self):
        """Return boundary poloidal flux."""
        return float(self['psi_boundary'])

    def normalize(self, psi):
        """Return normalized poloidal flux."""
        return (psi - self.psi_axis) / (self.psi_boundary - self.psi_axis)

    def denormalize(self, psi_norm):
        """Return poloidal flux."""
        return psi_norm * (self.psi_boundary - self.psi_axis) + self.psi_axis

    @cached_property
    def psi_rbs(self):
        """Return cached 2D RectBivariateSpline psi2d interpolant."""
        return self._rbs('psi2d')

    @cached_property
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
        self.set_axes(axes, '1d')
        psi_norm = np.linspace(0, 1, 500)
        self.axes.plot(self.data.psi_norm, self[attr], '.')
        self.axes.plot(psi_norm, getattr(self, attr)(psi_norm), '-')
        self.axes.set_ylabel(attr)
        self.axes.set_xlabel(r'$\psi_{norm}$')


if __name__ == '__main__':

    profile = Profile(105007, 10, 'iter', 1)
    profile.plot_profile(attr='p_prime')
