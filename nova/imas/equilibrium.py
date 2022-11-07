"""Manage access to equilibrium data."""
from dataclasses import dataclass, field

import numpy as np

from nova.imas.scenario import Scenario
from nova.biot.biotgrid import BiotPlot
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
        if self.data.grid_type == -999999999:  # unset
            self.data.attrs['grid_type'] = 1
        if self.data.grid_type == 1:
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
class Boundary(Scenario):
    """Load boundary timeseries from equilibrium ids."""

    def outline(self, itime):
        """Return r,z."""
        outline = self.time_slice('boundary', itime, index=None).outline
        return np.c_[outline.r, outline.z]

    def build(self):
        """Build 0D parameter timeseries."""
        super().build()
        outline = self.outline(0)
        self.data['boundary_index'] = range(len(outline))
        self.data['coordinate'] = ['radial', 'vertical']
        self.data['boundary'] = ('time', 'boundary_index', 'coordinate'), \
            np.zeros((self.data.dims['time'],
                      self.data.dims['boundary_index'],
                      self.data.dims['coordinate']))

        for itime in range(self.data.dims['time']):
            self.data['boundary'][itime] = self.outline(itime)

    def plot_boundary(self, itime: int):
        """Plot 2D boundary at itime."""
        plt.plot(self.data.boundary[itime, :, 0],
                 self.data.boundary[itime, :, 1], 'gray', alpha=0.5)
        plt.axis('equal')


@dataclass
class Parameter0D(Scenario):
    """Load 0D parameter timeseries from equilibrium ids."""

    attrs_0d: list[str] = field(
            default_factory=lambda: ['ip', 'beta_pol', 'li_3',
                                     'psi_axis', 'psi_boundary'])

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
        plt.despine()


@dataclass
class Profile2D(Scenario, BiotPlot):
    """Manage extraction of 2d profile data from imas ids."""

    attrs_2d: list[str] = field(
        default_factory=lambda: ['psi', 'phi', 'j_tor', 'j_parallel',
                                 'b_field_r', 'b_field_z', 'b_field_tor'])

    def build(self):
        """Build profile 2d data and store to xarray data structure."""
        super().build()
        self.time_slice.build('profiles_2d', self.attrs_2d, postfix='2d')

    def plot_2d(self, itime=-1, attr='psi', axes=None, **kwargs):
        """Plot 2d profile."""
        self.axes = axes
        kwargs = self.contour_kwargs(**kwargs)
        QuadContourSet = plt.contour(self.data.r, self.data.z,
                                     self.data[f'{attr}2d'][itime].T,
                                     **kwargs)
        plt.axis('equal')
        plt.axis('off')
        return QuadContourSet.levels


@dataclass
class Equilibrium(Profile2D, Profile1D, Parameter0D, Boundary, Grid):
    """
    Manage active equilibrium ids.

    Load, cache and plot equilibrium ids data taking database identifiers to
    load from file or operating directly on an open ids.

    Parameters
    ----------
    pulse: int, optional (required when ids not set)
        Pulse number. The default is 0.
    run: int, optional (required when ids not set)
        Run number. The default is 0.
    name: str, optional
        Ids name. The default is 'equilibrium'.
    user: str, optional
        User name. The default is public.
    machine: str, optional
        Machine name. The default is iter.
    backend: int, optional (required when ids not set)
        Access layer backend. The default is 13 (HDF5).
    ids: ImasIds, optional
        When set the ids parameter takes prefrence. The default is None.

    Attributes
    ----------
    attrs_0d: list[str]
        Avalible 0D attribute list.
    attrs_1d: list[str]
        Avalible 1D attribute list.
    attrs_2d: list[str]
        Avalible 2D attribute list.
    filepath: str
        Location of cached netCDF datafile.

    See Also
    --------
    nova.imas.Database

    Examples
    --------
    Load equilibrium data from pulse and run indicies
    asuming defaults for others:

    >>> equilibrium = Equilibrium(130506, 403)
    >>> equilibrium.name, equilibrium.user, equilibrium.machine
    ('equilibrium', 'public', 'iter')

    >>> equilibrium.filename
    'iter_130506_403'
    >>> equilibrium.group
    'equilibrium'

    (re)build equilibrium ids reading data from imas database:

    >>> equilibrium_reload = equilibrium.build()
    >>> equilibrium_reload == equilibrium
    True

    Plot poloidal flux at itime=10:

    >>> itime = 20
    >>> fig = plt.figure()
    >>> levels = equilibrium.plot_2d(itime, 'psi', colors='C3', levels=31)
    >>> equilibrium.plot_boundary(itime)

    Plot contour map of toroidal current density at itime=10:

    >>> fig = plt.figure()
    >>> levels = equilibrium.plot_2d(itime, 'j_tor')

    Plot 1D dpressure_dpsi profile.

    >>> fig = plt.figure()
    >>> equilibrium.plot_1d(itime, 'dpressure_dpsi')

    Plot plasma current waveform.

    >>> fig = plt.figure()
    >>> equilibrium.plot_0d('ip')

    """

    name: str = 'equilibrium'

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            self.data['time'] = self.ids.time
            super().build()
        return self


if __name__ == '__main__':

    import doctest
    doctest.testmod()
