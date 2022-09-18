"""Extrapolate equilibria beyond separatrix."""
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import ClassVar, Union

import imas
import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline, interp1d
import xarray

from nova.biot.biotgrid import BiotPlot
from nova.imas.code import Code
from nova.imas.database import Database, IDS
from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import Machine
from nova.imas.properties import Properties
from nova.utilities.pyplot import plt


@dataclass
class Grid:
    """Specify interpolation grid attributes."""

    number: int = 2500
    limit: float | list[float] | str = 0.25
    index: Union[str, slice, npt.ArrayLike] = 'plasma'

    @property
    def grid_attrs(self) -> dict:
        """Return grid attributes."""
        return {attr: getattr(self, attr)
                for attr in [attr.name for attr in fields(Grid)]}


@dataclass
class TimeSlice:
    """Generate interpolants from single time slice of equilibrium ids."""

    data: xarray.Dataset

    def __post_init__(self):
        """Load dataset."""
        self.data['psi2d_norm'] = self.normalize(self.data.psi2d)

    def __getattr__(self, attr: str):
        """Return attribute from xarray dataset."""
        return getattr(self.data, attr).values

    def normalize(self, psi):
        """Return normalized poloidal flux."""
        return (psi - self.psi_axis) / (self.psi_boundary - self.psi_axis)

    @cached_property
    def psi_rbs(self):
        """Return cached 2D RectBivariateSpline psi interpolant."""
        return RectBivariateSpline(self.r, self.z, self.data.psi2d).ev

    def _interp1d(self, x, y):
        """Return 1D interpolant."""
        return interp1d(x, y, kind='quadratic', fill_value='extrapolate')

    @cached_property
    def p_prime(self):
        """Return cached pprime 1D interpolant."""
        return self._interp1d(self.psi_norm, self.dpressure_dpsi)

    @cached_property
    def ff_prime(self):
        """Return cached pprime 1D interpolant."""
        return self._interp1d(self.psi_norm, self.f_df_dpsi)

    def plot(self):
        """Plot flux function interpolants."""
        psi_norm = np.linspace(0, 1, 500)
        axes = plt.subplots(2, 1, sharex=True)[1]
        axes[0].plot(self.psi_norm, self.dpressure_dpsi, '.')
        axes[0].plot(psi_norm, self.p_prime(psi_norm), '-')
        axes[1].plot(self.psi_norm, self.f_df_dpsi, '.')
        axes[1].plot(psi_norm, self.ff_prime(psi_norm), '-')
        axes[0].set_ylabel(r'$P^\prime$')
        axes[1].set_ylabel(r'$FF^\prime$')
        axes[1].set_xlabel(r'$\psi_{norm}$')
        plt.despine()


@dataclass
class Extrapolate(BiotPlot, Machine, Grid, IDS):
    """Extrapolate equlibrium beyond separatrix ids."""

    limit: float | list[float] | str = 'ids'
    ids_name: str = 'equilibrium'
    filename: str = 'extrapolate'
    geometry: list[str] = field(default_factory=lambda: ['pf_active', 'wall'])
    itime: int = field(init=False, default=0)

    mu_o: ClassVar[float] = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

    def __post_init__(self):
        """Load equilibrium and coilset."""
        self.load_ids()
        super().__post_init__()

    def __call__(self):
        """Return extrapolated equilibrium ids."""
        ids_data = imas.equilibrium()
        Properties('Equilibrium extrapolation',
                   provider='Simon McIntosh',
                   provenance_ids=self.ids_data)(ids_data.ids_properties)
        Code(self.machine_attrs)(ids_data.code)
        ids_data.vacuum_toroidal_field = self.ids_data.vacuum_toroidal_field
        return ids_data

    def load_ids(self):
        """Load equilibrium data and grid limits."""
        equilibrium = Equilibrium(self.pulse, self.run, ids_data=self.ids_data)
        for attr in ['ids_data', 'data']:
            setattr(self, attr, getattr(equilibrium, attr))
        if self.limit == 'ids':  # Load grid limit from ids.
            if equilibrium.data.grid_type != 1:
                raise TypeError('ids limits only valid for rectangular grids'
                                f'{equilibrium.data.grid_type} != 1')
            limit = [equilibrium.data.r.values, equilibrium.data.z.values]
            if self.number == 'ids':
                self.limit = limit
            else:
                self.limit = [limit[0][0], limit[0][-1],
                              limit[1][0], limit[1][-1]]
        if self.number == 'ids':
            self.number = equilibrium.data.dims['r'] * \
                equilibrium.data.dims['z']

    @property
    def machine_attrs(self):
        """Extend machine hash attributes."""
        return super().machine_attrs | self.grid_attrs

    def build(self, **kwargs):
        """Build frameset and interpolation grid."""
        super().build(**kwargs)
        self.grid.solve(**self.grid_attrs)
        return self.store(self.filename)

    def ionize(self, itime: int):
        """Update plasma current."""
        self.itime = itime
        time_slice = TimeSlice(self.data.isel(time=self.itime))

        self.plasma.separatrix = time_slice.boundary
        plasma = self.aloc['plasma']
        ionize = self.aloc['ionize']
        radius = self.aloc['x'][ionize]
        height = self.aloc['z'][ionize]
        area = self.aloc['area'][ionize]
        psi = time_slice.psi_rbs(radius, height)
        psi_norm = time_slice.normalize(psi)

        current_density = radius * time_slice.p_prime(psi_norm) + \
            time_slice.ff_prime(psi_norm) / (self.mu_o * radius)
        current_density *= -2*np.pi
        current = current_density * area

        nturn = self.aloc['nturn']
        nturn[ionize] = current / current.sum()
        self.sloc['plasma', 'Ic'] = time_slice.ip
        self.plasmagrid.update_turns('Psi')

        Psi = self.plasmagrid.data.Psi.values[ionize[plasma]]

        self.saloc['Ic'][:-2] = np.linalg.lstsq(
            Psi[:, :-2], -psi - Psi[:, -1]*time_slice.ip)[0]

    def plot_2d(self, itime=-1, attr='psi', axes=None, **kwargs):
        """Expose plot_2d ."""
        return Equilibrium.plot_2d(
            self, itime=itime, attr=attr, axes=None, **kwargs)

    def plot_boundary(self, itime: int):
        """Expose Equilibrium plot boundary."""
        return Equilibrium.plot_boundary(self, itime)

    def plot(self, attr='psi'):
        """Plot plasma filements and polidal flux."""
        plt.figure()
        super().plot('plasma')
        levels = self.grid.plot(attr, levels=51, colors='C0', nulls=False)
        try:
            self.plot_2d(self.itime, attr, colors='C3', levels=-levels[::-1],
                         linestyles='dashdot')
        except KeyError:
            pass
        self.plot_boundary(self.itime)


if __name__ == '__main__':

    #  pulse, run = 114101, 41  # JINTRAC
    pulse, run = 130506, 403  # CORSICA

    database = Database(pulse, run, 'equilibrium', machine='iter')
    coilset = Extrapolate(ids_data=database.ids_data,
                          dplasma=-1000, number=1000)
    # coilset.build()
    coilset.ionize(25)
    coilset.plot('br')
