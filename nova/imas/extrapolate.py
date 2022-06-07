"""Extrapolate equilibria beyond separatrix."""
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import ClassVar, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline, interp1d
import xarray

from nova.electromagnetic.biotgrid import BiotPlot
from nova.imas.database import IDS
from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import Machine
from nova.utilities.pyplot import plt


@dataclass
class Grid:
    """Specify interpolation grid attributes."""

    number: int = 1000
    limit: Union[float, list[float]] = 0.15
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
        self.data['psi2d_norm'] = self._psi_norm(self.data.psi2d)

    def __getattr__(self, attr: str):
        """Return attribute from xarray dataset."""
        return getattr(self.data, attr).values

    def _psi_norm(self, psi):
        """Return normalized poloidal flux."""
        return (psi - self.psi_axis) / (self.psi_boundary - self.psi_axis)

    @cached_property
    def psi2d_norm(self):
        """Return cached 2D RectBivariateSpline psi interpolant."""
        return RectBivariateSpline(self.r, self.z, self.data.psi2d_norm).ev

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

    itime: int = 0
    ids_name: str = 'equilibrium'
    filename: str = 'iter'
    dplasma: float = -200
    geometry: list[str] = field(default_factory=lambda: ['pf_active', 'wall'])

    mu_o: ClassVar[float] = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

    def __post_init__(self):
        """Load equilibrium data."""
        super().__post_init__()
        self.data = Equilibrium(self.pulse, self.run).data

    @property
    def hash_attrs(self):
        """Extend machine hash attributes."""
        return super().hash_attrs | self.grid_attrs

    def build(self, **kwargs):
        """Build frameset and interpolation grid."""
        super().build(**kwargs)
        self.grid.solve(**self.grid_attrs)
        return self.store(self.filename)

    def ionize(self, itime: int):
        """Update plasma current."""
        self.itime = itime
        time_slice = TimeSlice(self.data.isel(time=self.itime))

        coilset.plasma.separatrix = time_slice.boundary
        ionize = self.aloc['ionize']
        plasma = self.aloc['plasma']
        radius = self.aloc['x'][ionize]
        height = self.aloc['z'][ionize]
        area = self.aloc['area'][ionize]
        psi_norm = time_slice.psi2d_norm(radius, height)

        current_density = radius * time_slice.p_prime(psi_norm) + \
            time_slice.ff_prime(psi_norm) / (self.mu_o * radius)
        current_density *= -2*np.pi
        current = current_density * area

        nturn = self.aloc['nturn']
        nturn[ionize] = current / current.sum()
        self.sloc['plasma', 'Ic'] = time_slice.ip

        print(current.sum(), time_slice.ip)
        self.sloc['free'] = self.sloc['coil']

        #self.sl
        #print(self.sloc(), current.sum())

        time_slice.plot()

    def plot_2d(self, itime=-1, attr='psi', axes=None, **kwargs):
        """Expose plot_2d ."""
        return Equilibrium.plot_2d(
            self, itime=itime, attr='psi', axes=None, **kwargs)

    def plot_boundary(self, itime: int):
        """Expose Equilibrium plot boundary."""
        return Equilibrium.plot_boundary(self, itime)

    def plot(self):
        """Plot plasma filements and polidal flux."""
        plt.figure()
        super().plot('plasma')
        self.plot_2d(self.itime, 'psi', colors='C3', levels=21)
        self.plot_boundary(self.itime)

        self.grid.plot()


if __name__ == '__main__':

    pulse, run = 114101, 41  # JINTRAC
    pulse, run = 130506, 403  # CORSICA
    coilset = Extrapolate(pulse, run, number=1000, dplasma=-500)

    coilset.sloc['coil', 'Ic'] = 7.5e3

    coilset.ionize(10)

    coilset.plot()


    #interp.plot()
    #eq = Equilibrium(114101, 41)

    #coilset.plasma.separatrix = eq.data.boundary[0]
    '''
    #coilset.grid.solve2d(eq.data.r2d.values[::20, ::20],
    #                     eq.data.z2d.values[::20, ::20])

    from nova.electromagnetic.fieldnull import FieldNull

    null = FieldNull(xarray.Dataset(dict(x=eq.data.r, z=eq.data.z)))
    null.update_null(eq.data.psi2d[0].values)
    null.plot()

    '''

    '''

    self.Ip = self.dA * self.bp[index] /
    '''

    '''
    coilset.plasmagrid.psi = rbs.ev(coilset.loc['plasma', 'x'],
                                    coilset.loc['plasma', 'z'])
    coilset.plasmagrid.plot()
    '''
