"""Manage access to equilibrium data."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString

from nova.biot.contour import Contour
from nova.frame.baseplot import Plot
from nova.geometry.pointloop import PointLoop
from nova.geometry.separatrix import LCFS
from nova.imas.machine import Wall
from nova.imas.scenario import Scenario
from nova.plot.biotplot import BiotPlot


@dataclass
class Grid(Scenario):
    """Load grid from ids data instance."""

    def build(self):
        """Build grid from single timeslice and store in data."""
        super().build()
        if self.ids_index.empty('profiles_2d.grid_type.index'):
            return
        index = self.ids_index.get_slice(0, 'profiles_2d.grid_type.index')
        grid = self.ids_index.get_slice(0, 'profiles_2d.grid')
        grid_type = index
        if grid_type == -999999999:  # unset
            grid_type = 1
        if grid_type == 1:
            return self.rectangular_grid(grid)
        raise NotImplementedError(f'grid type {grid_type} not implemented.')

    def rectangular_grid(self, grid):
        """
        Store rectangular grid.

        Cylindrical R,Z aka eqdsk (R=dim1, Z=dim2).
        In this case the position arrays should not be
        filled since they are redundant with grid/dim1 and dim2.
        """
        self.data['r'], self.data['z'] = grid.dim1, grid.dim2
        r2d, z2d = np.meshgrid(self.data['r'], self.data['z'], indexing='ij')
        self.data['r2d'] = ('r', 'z'), r2d
        self.data['z2d'] = ('r', 'z'), z2d

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
class Parameter0D(Plot, Scenario):
    """Load 0D parameter timeseries from equilibrium ids."""

    attrs_0d: list[str] = field(
            default_factory=lambda: ['ip', 'beta_pol', 'li_3',
                                     'psi_axis', 'psi_boundary',
                                     'volume', 'area', 'surface', 'length_pol',
                                     'q_axis', 'q_95',
                                     'psi_external_average', 'v_external',
                                     'plasma_inductance', 'plasma_resistance'])
    attrs_boundary: ClassVar[list[str]] = [
        'geometric_radius', 'geometric_height',
        'minor_radius', 'elongation',
        'triangularity', 'triangularity_upper', 'triangularity_lower',
        'squareness_upper_inner', 'squareness_upper_outer',
        'squareness_lower_inner', 'squareness_lower_outer']

    def build_axis(self, path: str):
        """Build axis from global quantities."""
        attr = path.split('.')[-1]
        if any(self.ids_index.empty(f'{path}.{label}') for label in 'rz'):
            return
        self.data[attr] = ('time', 'point'), \
            np.c_[self.ids_index.array(f'{path}.r'),
                  self.ids_index.array(f'{path}.z')]

    def build(self):
        """Build 0D parameter timeseries."""
        super().build()
        self.append('time', self.attrs_0d, 'global_quantities')
        self.build_axis('global_quantities.magnetic_axis')
        self.build_axis('global_quantities.current_centre')
        self.build_x_points()
        self.build_boundary_outline()
        self.build_boundary_shape()

    def outline(self, itime):
        """Return boundary outline."""
        for node in ['boundary_separatrix', 'boundary']:
            outline = self.ids_index.get_slice(itime, f'{node}.outline')
            boundary = np.c_[outline.r, outline.z]
            if len(boundary) > 0:
                return boundary
        return boundary

    def _point_array(self, itime: int, path: str) -> np.ndarray:
        """Return point array."""
        points = []
        for point in self.ids_index.get_slice(itime, path):
            points.append([point.r, point.z])
        return np.array(points, float)

    def x_point_array(self, itime: int):
        """Return x-point array at itime."""
        x_points = self._point_array(itime, 'boundary_separatrix.x_point')
        boundary = self.outline(itime)
        if len(boundary) > 0:
            delta = np.max(np.linalg.norm(
                boundary[1:] - boundary[:-1], axis=1))
            index = np.array([np.min(np.linalg.norm(
                boundary - x_point, axis=1)) for x_point in x_points]) < delta
            x_points = x_points[index]
        if len(x_points) > 0:
            return np.array(x_points, float)
        return np.zeros((1, 2), float)

    def build_x_points(self):
        """Build x-point locations."""
        x_point = np.stack([self.x_point_array(itime)[0]
                            for itime in self.data.itime.data])
        self.data['x_point'] = ('time', 'point'), x_point

    @cached_property
    def divertor(self):
        """Return divertor outline."""
        wall = Wall()
        return wall.segment(1)

    def strike_point_array(self, itime: int):
        """Return strike-point array at itime."""
        strike_points = self._point_array(
            itime, 'boundary_separatrix.strike_point')
        if len(strike_points) == 0:
            boundary = self.outline(itime)
            contour = Contour(self.data.r2d, self.data.z2d,
                              self.data.psi2d[itime])
            levelset = contour.levelset(self.data.psi_boundary[itime])
            separatrix = MultiLineString(
                [surface.points for surface in levelset])

            points = shapely.intersection(separatrix, LineString(self.divertor))
            print(points)

            self.set_axes('2d')
            for surface in levelset:
                surface.plot()
            self.axes.plot(*self.divertor.T)
            #self.axes.plot(*boundary.T)


    def build_strike_points(self):
        """Build strike-point locations when not present in IDS."""

    def x_mask(self, itime: int, outline_z: np.ndarray, eps=0):
        """Return boundary x-point mask."""
        x_point = self.data.x_point[itime].data
        o_point = self.data.magnetic_axis[itime].data
        if np.allclose(x_point, (0, 0)):
            return np.ones(len(outline_z), bool)
        if x_point[1] < o_point[1]:
            return outline_z > x_point[1] - eps
        return outline_z < x_point[1] + eps

    def boundary_outline(self, itime: int) -> np.ndarray:
        """Return masked r, z boundary outline."""
        boundary = self.outline(itime)
        if len(boundary) == 0:
            return boundary
        segment = np.linalg.norm(boundary[1:] - boundary[:-1], axis=1)
        x_point = self.data.x_point[itime].data
        o_point = self.data.magnetic_axis[itime].data
        limiter = np.allclose(x_point, (0, 0))
        if limiter:  # limiter
            step = np.mean(segment) + 3*np.std(segment)
            index = np.arange(len(segment))[segment > step]
            if len(index) > 0:
                loops = np.split(boundary, index)
                loop_index = np.argmin([np.linalg.norm(loop[-1] - loop[0]) for
                                       loop in loops])
                return np.append(loops[loop_index], loops[loop_index][:1],
                                 axis=0)
        mask = self.x_mask(itime, boundary[:, 1])
        boundary = boundary[mask]
        if sum(mask) == 0:
            return boundary
        if not limiter:
            boundary = np.append(boundary, x_point[np.newaxis, :], axis=0)
            boundary = np.unique(boundary, axis=0)
            theta = np.arctan2(boundary[:, 1]-o_point[1],
                               boundary[:, 0]-o_point[0])
            boundary = boundary[np.argsort(theta)]
        if not np.allclose(boundary[0], boundary[-1]):
            return np.append(boundary, boundary[:1], axis=0)
        return boundary

    def build_boundary_outline(self):
        """Build outline timeseries."""
        length = max(len(self.boundary_outline(itime))
                     for itime in self.data.itime.data)
        if length == 0:
            return
        self.data['boundary_index'] = range(length)
        self.data['boundary'] = ('time', 'boundary_index', 'point'), \
            np.zeros((self.data.dims['time'],
                      self.data.dims['boundary_index'],
                      self.data.dims['point']))
        self.data['boundary_length'] = 'time', \
            np.zeros(self.data.dims['time'], dtype=int)
        for itime in self.data.itime.data:
            outline = self.boundary_outline(itime)
            length = len(outline)
            self.data['boundary_length'][itime] = length
            self.data['boundary'][itime, :length] = outline

    def extract_shape_parameters(self):
        """Return shape parameters calculated from lcfs."""
        lcfs_data = {attr: np.zeros(self.data.dims['time'], float)
                     for attr in self.attrs_boundary if hasattr(LCFS, attr)}
        for itime in self.data.itime.data:
            boundary = self.boundary_outline(itime)
            if len(boundary) == 0:
                continue
            lcfs = LCFS(boundary)
            try:
                for attr in lcfs_data:
                    lcfs_data[attr][itime] = getattr(lcfs, attr)
            except ValueError:
                continue
        return lcfs_data

    def build_boundary_shape(self):
        """Build plasma shape parameters."""
        self.append('time', 'psi', 'boundary_separatrix', postfix='_boundary')
        self.append('time', 'type', 'boundary_separatrix', prefix='boundary_')
        self.append('time', self.attrs_boundary, 'boundary_separatrix')
        lcfs_data = self.extract_shape_parameters()
        for attr in self.attrs_boundary:
            path = f'boundary_separatrix.{attr}'
            if attr not in self.data and attr in lcfs_data:
                self.data[attr] = 'time', lcfs_data[attr]

        path = 'boundary_separatrix.geometric_axis'
        if any(self.ids_index.empty(f'{path}.{label}') for label in 'rz'):
            geometric_axis = np.c_[lcfs_data['geometric_radius'],
                                   lcfs_data['geometric_height']]
        else:
            geometric_axis = np.c_[self.ids_index.array(f'{path}.r'),
                                   self.ids_index.array(f'{path}.z')]
        self.data['geometric_axis'] = ('time', 'point'), geometric_axis

    @property
    def boundary(self):
        """Return trimmed boundary contour."""
        return self['boundary'][:int(self['boundary_length'])]

    def plot_0d(self, attr, axes=None):
        """Plot 0D parameter timeseries."""
        self.set_axes('1d', axes=axes)
        self.axes.plot(self.data.time, self.data[attr], label=attr)

    def plot_boundary(self, axes=None):
        """Plot 2D boundary at itime."""
        boundary = self.boundary
        self.get_axes('2d', axes=axes)
        self.axes.plot(boundary[:, 0], boundary[:, 1], 'gray', alpha=0.5)
        if not np.allclose(self['x_point'], (0, 0)):
            self.axes.plot(*self['x_point'], 'x', ms=6, mec='C3', mew=1)

    def plot_shape(self, axes=None):
        """Plot separatrix shape parameter waveforms."""
        self.set_axes('1d', axes=axes)
        for attr in ['elongation', 'triangularity',
                     'triangularity_upper', 'triangularity_lower']:
            self.axes.plot(self.data.time, self.data[attr].data,
                           label=attr)
        self.axes.legend(ncol=4)


@dataclass
class Profile1D(Plot, Scenario):
    """Manage extraction of 1d profile data from imas ids."""

    attrs_1d: list[str] = field(
            default_factory=lambda: ['dpressure_dpsi', 'f_df_dpsi'])

    def build(self):
        """Build 1d profile data."""
        super().build()
        if self.ids_index.empty('profiles_1d.psi'):
            return
        length = self.ids_index['profiles_1d.psi'][0]
        self.data['psi_norm'] = np.linspace(0, 1, length)
        self.data['psi1d'] = ('time', 'psi_norm'), \
            self.ids_index.array('profiles_1d.psi')
        self.append(('time', 'psi_norm'), self.attrs_1d, 'profiles_1d')
        for itime in self.data.itime.data:  # normalize 1D profiles
            psi = self.data.psi1d[itime]
            if np.isclose(psi[-1] - psi[0], 0):
                continue
            psi_norm = (psi - psi[0]) / (psi[-1] - psi[0])
            for attr in self.attrs_1d:
                try:
                    self.data[attr][itime] = np.interp(
                        self.data.psi_norm, psi_norm, self.data[attr][itime])
                except KeyError:
                    pass

    def plot_1d(self, attr='psi', axes=None, **kwargs):
        """Plot 1d profile."""
        self.set_axes('1d', axes=axes)
        self.axes.plot(self.data.psi_norm, self[attr], **kwargs)


@dataclass
class Profile2D(BiotPlot, Scenario):
    """Manage extraction of 2d profile data from imas ids."""

    attrs_2d: list[str] = field(
        default_factory=lambda: ['psi', 'phi', 'j_tor', 'j_parallel',
                                 'b_field_r', 'b_field_z', 'b_field_tor'])

    def build(self):
        """Build profile 2d data and store to xarray data structure."""
        super().build()
        self.append(('time', 'r', 'z'), self.attrs_2d, 'profiles_2d',
                    postfix='2d')

    def data_2d(self, attr: str, mask=0):
        """Return data array."""
        return self[f'{attr}2d']

    def plot_2d(self, attr='psi', mask=0, axes=None, **kwargs):
        """Plot 2d profile."""
        self.set_axes('2d', axes=axes)
        kwargs = self.contour_kwargs(**kwargs)
        QuadContourSet = self.axes.contour(
            self.data.r, self.data.z, self.data_2d(attr, mask).T, **kwargs)
        return QuadContourSet.levels


@dataclass
class Equilibrium(Profile2D, Profile1D, Parameter0D, Grid):
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
    Skip doctest if IMAS instalation or requisite IDS(s) not found.

    >>> import pytest
    >>> from nova.imas.database import Database
    >>> try:
    ...     _ = Database(130506, 403).get_ids('equilibrium')
    ... except:
    ...     pytest.skip('IMAS not found or 130506/403 unavailable')

    Load equilibrium data from pulse and run indicies
    asuming defaults for others:

    >>> equilibrium = Equilibrium(130506, 403)
    >>> equilibrium.name, equilibrium.user, equilibrium.machine
    ('equilibrium', 'public', 'iter')

    >>> equilibrium.filename
    'equilibrium_iter_130506_403'
    >>> equilibrium.group
    'equilibrium'

    (re)build equilibrium ids reading data from imas database:

    >>> equilibrium_reload = equilibrium.build()
    >>> equilibrium_reload == equilibrium
    True

    Plot poloidal flux at itime=10:

    >>> equilibrium.itime = 20
    >>> levels = equilibrium.plot_2d('psi', colors='C3', levels=31)
    >>> equilibrium.plot_boundary()

    Plot contour map of toroidal current density at itime=10:

    >>> levels = equilibrium.plot_2d('j_tor')

    Plot 1D dpressure_dpsi profile.

    >>> equilibrium.plot_1d('dpressure_dpsi')

    Plot plasma current waveform.

    >>> equilibrium.plot_0d('ip')

    """

    def __post_init__(self):
        """Set instance name."""
        self.name = 'equilibrium'
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            self.data.coords['point'] = ['r', 'z']
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

    pulse, run = 105028, 1  # DINA -10MA divertor PCS
    pulse, run = 135011, 7  # DINA
    pulse, run = 105011, 9
    pulse, run = 135003, 5
    # pulse, run = 135007, 4
    pulse, run = 105028, 1
    pulse, run = 135013, 2

    pulse, run = 130506, 403

    pulse, run = 135014, 1

    pulse, run = 135013, 2

    Equilibrium(pulse, run)._clear()
    equilibrium = Equilibrium(pulse, run)

    equilibrium.time = 100
    equilibrium.plot_2d('psi', mask=0)
    equilibrium.plot_boundary()
