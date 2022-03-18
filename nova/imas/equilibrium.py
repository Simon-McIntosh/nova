"""Manage access to equilibrium data."""
from dataclasses import dataclass, field

import numpy as np

from nova.imas.scenario import Scenario
from nova.electromagnetic.biotgrid import BiotPlot
from nova.utilities.pyplot import plt


@dataclass
class Grid(Scenario):
    """Load grid from ids data instance."""

    def build(self):
        """Build grid from single timeslice and store in data."""
        super().build()
        time_slice = self.time_slice('profiles_2d', 0)
        grid_type, grid = time_slice.grid_type, time_slice.grid
        self.data.attrs['grid_type'] = grid_type.index
        if grid_type.index == 1:
            return self.rectangular_grid(grid)
        raise NotImplementedError(f'grid {grid_type} not implemented.')

    def rectangular_grid(self, grid):
        """
        Store rectangular grid.

        Cylindrical R,Z aka eqdsk (R=dim1, Z=dim2).
        In this case the position arrays should not be
        filled since they are redundant with grid/dim1 and dim2.
        """
        self.data['r'] = grid.dim1
        self.data['z'] = grid.dim2
        r2d, z2d = np.meshgrid(self.data['r'], self.data['z'], indexing='ij')
        self.data['r2d'] = ('r', 'z'), r2d
        self.data['z2d'] = ('r', 'z'), z2d
        self.data.attrs['profiles_2d'] = ('time', 'r', 'z')


@dataclass
class Parameter0D(Scenario):
    """Load 0D parameter timeseries from equilibrium ids."""

    attrs_0d: list[str] = field(
            default_factory=lambda: ['ip', 'beta_pol', 'li_3'])

    def build(self):
        """Build 0D parameter timeseries."""
        super().build()
        self.data.attrs['global_quantities'] = ('time',)
        self.time_slice.build('global_quantities', self.attrs_0d, index=None)

    def plot_0d(self, attr):
        """Plot 0D parameter timeseries."""
        plt.plot(self.data.time, self.data[attr], label=attr)
        plt.despine()


@dataclass
class Profile1D(Scenario):
    """Manage extraction of 1d profile data from imas ids."""

    attrs_1d: list[str] = field(
            default_factory=lambda: ['psi', 'dpressure_dpsi', 'f_df_dpsi'])

    def build(self):
        """Build 1d profile data."""
        super().build()
        time_slice = self.time_slice('profiles_1d', 0, index=None)
        self.data['psi_norm'] = np.linspace(0, 1, len(time_slice.psi))
        self.data.attrs['profiles_1d'] = ('time', 'psi_norm')
        self.time_slice.build('profiles_1d', self.attrs_1d, index=None)
        for itime in range(self.data.dims['time']):  # normalize 1D profiles
            psi = self.data.psi[itime]
            if np.isclose(psi[-1] - psi[0], 0):
                continue
            psi_norm = (psi - psi[0]) / (psi[-1] - psi[0])
            for attr in self.attrs_1d:
                self.data[attr][itime] = np.interp(
                    self.data.psi_norm, psi_norm, self.data[attr][itime])

    def plot_1d(self, itime=-1, attr='psi'):
        """Plot 1d profile."""
        plt.plot(self.data.psi_norm, self.data[attr][itime])


@dataclass
class Profile2D(Scenario, BiotPlot):
    """Manage extraction of 2d profile data from imas ids."""

    attrs_2d: list[str] = field(
        default_factory=lambda: ['psi', 'phi', 'j_tor', 'j_parallel',
                                 'b_r', 'b_z', 'b_tor'])

    def build(self):
        """Build profile 2d data and store to xarray data structure."""
        super().build()
        self.time_slice.build('profiles_2d', self.attrs_2d, postfix='2d')

    def plot_2d(self, itime=-1, attr='psi', axes=None, **kwargs):
        """Plot 2d profile."""
        super().plot(axes)
        kwargs = self.contour_kwargs(**kwargs)
        QuadContourSet = plt.contour(self.data.r, self.data.z,
                                     self.data[f'{attr}2d'][itime].T,
                                     **kwargs)
        plt.axis('equal')
        plt.axis('off')
        return QuadContourSet.levels


@dataclass
class Equilibrium(Profile2D, Profile1D, Parameter0D, Grid):
    """Manage active poloidal loop ids, pf_passive."""

    shot: int = 135011
    run: int = 7
    ids_name: str = 'equilibrium'

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            self.data['time'] = self.ids_data.time
            super().build()


if __name__ == '__main__':
    shot, run = 135011, 7
    shot, run = 135013, 2
    shot, run = 130506, 403
    eq = Equilibrium(shot, run)
    # eq.build()

    itime = 30
    #eq.plot_0d('ip')
    #eq.plot_2d(itime, 'psi', colors='C3', levels=21)
    #eq.plot_2d(500, 'j_tor')
    #eq.plot_1d(itime, 'dpressure_dpsi')
    eq.plot_1d(itime, 'f_df_dpsi')
