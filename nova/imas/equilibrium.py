"""Manage access to equilibrium data."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nova.frame.baseplot import Plot
from nova.geometry.pointloop import PointLoop
from nova.imas.scenario import Scenario
from nova.plot.biotplot import BiotPlot


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

    @cached_property
    def mask_2d(self):
        """Return pointloop instance, used to check loop membership."""
        points = np.array([self.data.r2d.data.flatten(),
                           self.data.z2d.data.flatten()]).T
        return PointLoop(points)

    @property
    def shape(self):
        """Return grid shape."""
        return self.data.dims['r'], self.data.dims['z']

    def mask(self, boundary: np.ndarray):
        """Return boundary mask."""
        return self.mask_2d.update(boundary).reshape(self.shape)


@dataclass
class Boundary(Plot, Scenario):
    """Load boundary timeseries from equilibrium ids."""

    def outline(self, itime):
        """Return r,z."""
        outline = self.time_slice('boundary', itime, index=None).outline
        return np.c_[outline.r, outline.z]

    def build(self):
        """Build outline timeseries."""
        super().build()
        outline_length = max(len(self.outline(itime))
                             for itime in range(self.data.dims['time']))
        self.data['boundary_index'] = range(outline_length)
        self.data['coordinate'] = ['radial', 'vertical']
        self.data['boundary'] = ('time', 'boundary_index', 'coordinate'), \
            np.zeros((self.data.dims['time'],
                      self.data.dims['boundary_index'],
                      self.data.dims['coordinate']))
        self.data['boundary_length'] = 'time', \
            np.zeros(self.data.dims['time'], dtype=int)
        for itime in range(self.data.dims['time']):
            outline = self.outline(itime)
            self.data['boundary_length'][itime] = len(outline)
            self.data['boundary'][itime, :len(outline)] = outline

    @property
    def boundary(self):
        """Return trimmed boundary contour."""
        return self['boundary'][:self['boundary_length'].values].values

    def plot_boundary(self, axes=None):
        """Plot 2D boundary at itime."""
        self.get_axes(axes, '2d')
        self.axes.plot(self.boundary[:, 0], self.boundary[:, 1],
                       'gray', alpha=0.5)


@dataclass
class Parameter0D(Plot, Scenario):
    """Load 0D parameter timeseries from equilibrium ids."""

    attrs_0d: list[str] = field(
            default_factory=lambda: ['ip', 'beta_pol', 'li_3',
                                     'psi_axis', 'psi_boundary'])

    def build(self):
        """Build 0D parameter timeseries."""
        super().build()
        self.data.attrs['global_quantities'] = ('time',)
        self.time_slice.build('global_quantities', self.attrs_0d, index=None)

    def plot_0d(self, attr, axes=None):
        """Plot 0D parameter timeseries."""
        self.set_axes(axes, '1d')
        self.axes.plot(self.data.time, self.data[attr], label=attr)
        self.axes.despine()


@dataclass
class Profile1D(Plot, Scenario):
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

    def plot_1d(self, itime=-1, attr='psi', axes=None, **kwargs):
        """Plot 1d profile."""
        self.set_axes(axes, '1d')
        self.axes.plot(self.data.psi_norm, self.data[attr][itime], **kwargs)


@dataclass
class Profile2D(BiotPlot, Scenario):
    """Manage extraction of 2d profile data from imas ids."""

    attrs_2d: list[str] = field(
        default_factory=lambda: ['psi', 'phi', 'j_tor', 'j_parallel',
                                 'b_field_r', 'b_field_z', 'b_field_tor'])

    def build(self):
        """Build profile 2d data and store to xarray data structure."""
        super().build()
        self.time_slice.build('profiles_2d', self.attrs_2d, postfix='2d')

    def data_2d(self, attr: str, mask=0):
        """Return data array."""
        return self[f'{attr}2d']

    def plot_2d(self, attr='psi', mask=0, axes=None, **kwargs):
        """Plot 2d profile."""
        self.set_axes(axes, '2d')
        kwargs = self.contour_kwargs(**kwargs)
        QuadContourSet = self.axes.contour(
            self.data.r, self.data.z, self.data_2d(attr, mask).T, **kwargs)
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

    name: str | None = 'equilibrium'

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            self.data['time'] = self.ids_data.time
            self.data['itime'] = 'time', range(len(self.data['time']))
            super().build()
        return self

    def data_2d(self, attr: str, mask=0):
        """Extend to return masked data array."""
        data = super().data_2d(attr)
        if mask == 0:
            return data
        if mask == -1:
            return np.ma.masked_array(data, ~self.mask(self.boundary))
        if mask == 1:
            return np.ma.masked_array(data, self.mask(self.boundary))


if __name__ == '__main__':

    #import doctest
    #doctest.testmod()

    equilibrium = Equilibrium(105007, 10, 1)
    equilibrium.itime = 6
    equilibrium.plot_2d('psi', mask=0)
    equilibrium.plot_boundary()
