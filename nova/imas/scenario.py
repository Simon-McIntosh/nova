"""Manage access to scenario data."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray

from nova.imas.database import Database
from nova.utilities.pyplot import plt
from nova.utilities.time import clock


@dataclass
class Scenario(Database):
    """Manage access to scenario data."""

    tokamak: str = 'iter'


@dataclass
class ISDdata:
    """IDS data baseclass."""

    ids_data: Any
    data: xarray.Dataset

    def time_slice(self, itime: int, attr: str, index=0):
        """Return time slice data array."""
        if index is None:
            return getattr(self.ids_data.time_slice[itime], attr)
        return getattr(self.ids_data.time_slice[itime], attr).array[index]

    @staticmethod
    def name(attr: str, label: str):
        """Return variable name."""
        if label == '2d':
            return f'{attr}{label}'
        return attr

    def initialize_profile(self, label: str, index=None):
        """Create xarray profile data entries."""
        time_slice = self.time_slice(0, f'profiles_{label}', index)
        coords = self.data.attrs[f'profile_{label}']
        shape = tuple(self.data.dims[coordinate] for coordinate in coords)
        attrs = getattr(self, f'attrs_{label}')
        for attr in list(attrs):
            if len(getattr(time_slice, attr)) > 0:
                attr_name = self.name(attr, label)
                self.data[attr_name] = coords, np.zeros(shape, float)
            else:
                attrs.remove(attr)

    def build_profile(self, label: str, index=0):
        """Populate xarray dataset with profile data."""
        if label == '1d':
            index = None
        self.initialize_profile(label, index)
        attrs = getattr(self, f'attrs_{label}')
        for itime in range(self.data.dims['time']):
            time_slice = self.time_slice(itime, f'profiles_{label}', index)
            for attr in attrs:
                attr_name = self.name(attr, label)
                self.data[attr_name][itime] = getattr(time_slice, attr)


@dataclass
class Grid(ISDdata):
    """Load grid from ids data instance."""

    def build(self):
        """Build grid from single timeslice and store in data."""
        array = self.time_slice(0, 'profiles_2d', 0)
        grid_type, grid = array.grid_type, array.grid
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
        self.data.attrs['profile_2d'] = ('time', 'r', 'z')


@dataclass
class Profile1D(ISDdata):
    """Manage extraction of 1d profile data from imas ids."""

    attrs_1d: list[str] = field(
            default_factory=lambda: ['psi', 'dpressure_dpsi', 'f_df_dpsi'])

    def build(self):
        """Build 1d profile data."""
        super().build()  # build grid
        time_slice = self.time_slice(0, 'profiles_1d', index=None)
        self.data['flux_index'] = range(len(time_slice.psi))
        self.data.attrs['profile_1d'] = ('time', 'flux_index')
        self.build_profile('1d')


@dataclass
class Profile2D(ISDdata):
    """Manage extraction of 2d profile data from imas ids."""

    attrs_2d: list[str] = field(
        default_factory=lambda: ['psi', 'phi', 'j_tor', 'j_parallel',
                                 'b_r', 'b_z', 'b_tor'])

    def build(self):
        """Build profile 2d data and store to xarray data structure."""
        super().build()  # build grid
        self.build_profile('2d', index=0)
        '''
        for itime in range(self.data.dims['time']):
            time_slice = self.time_slice(itime, 'profiles_2d')
            for attr in self.attrs_2d:
                self.data[f'{attr}2d'][itime] = getattr(time_slice, attr)
        '''


@dataclass
class Equilibrium(Scenario, Profile2D, Profile1D, Grid):
    """Manage active poloidal loop ids, pf_passive."""

    shot: int = 135011
    run: int = 7
    ids_name: str = 'equilibrium'
    ids_data: Any = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    def __post_init__(self):
        """Load data."""
        self.build()

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        self.ids_data = self.load_ids_data()
        self.data['time'] = self.ids_data.time
        super().build()

        print(self.data)

    def plot_slice(self, itime=-1, attr='psi'):
        """Plot time slice from profiles_2d."""
        plt.contour(self.data.r, self.data.z,
                    self.data[f'{attr}2d'][itime].T)
        plt.axis('equal')
        plt.axis('off')

    def plot_profile(self, itime=-1, attr='psi'):
        """Plot 1d profile."""
        plt.plot(self.data.flux_index, self.data.psi[itime])


if __name__ == '__main__':

    #ids = IMASdb(tokamak='iter_md').ids(111001, 1, 'pf_passive')
    #ids = IMASdb(tokamak='iter').ids(135011, 7, 'equilibrium')

    eq = Equilibrium()
    #eq.plot_slice(500, 'j_tor')
    eq.plot_profile()
