"""Extrapolate equilibria beyond separatrix."""
from dataclasses import dataclass, field, fields
from functools import cached_property

import numpy as np
import pandas
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.constants import mu_0
import xarray

from nova.imas import (Database, Equilibrium, Ids, Machine, PF_Active)

from nova.linalg.regression import MoorePenrose


# pylint: disable=too-many-ancestors


@dataclass
class ExtrapolationGrid:
    """
    Specify extrapolation grid.

    Parameters
    ----------
    ngrid : {int, 'ids'}, optional
        ExtrapolationGrid dimension. The default is 5000.

        - int: use input to set aproximate total node number
        - ids: aproximate total node number extracted from equilibrium ids.
    limit : {float, list[float], 'ids'}, optional
        ExtrapolationGrid bounds. The default is 0.25.

        - float: expansion relative to coilset index. Must be greater than -1.
        - list[float]: explicit grid bounds [rmin, rmax, zmin, zmax].
        - ids: bounds extracted from from equilibrium ids.
    index : {'plasma', 'coil', slice, pandas.Index}
        Filament index from which relative grid limits are set.
    equilibrium : Equilibrium, optional
        Equilibrium ids required for equilibrium derived grid dimensions.
        The default is False

    Examples
    --------
    Manualy specify grid relitive to coilset:
    >>> ExtrapolationGrid(100, 0, 'coil').grid_attrs
    {'ngrid': 100, 'limit': 0, 'index': 'coil'}

    Specify grid relitive to equilibrium ids.
    >>> equilibrium = Equilibrium(130506, 403)
    >>> ExtrapolationGrid(50, 'ids', equilibrium=equilibrium).grid_attrs
    {'ngrid': 50, 'limit': [2.75, 8.9, -5.49, 5.51], 'index': 'plasma'}

    Extract exact grid from equilibrium ids.
    >>> grid = ExtrapolationGrid('ids', 'ids', equilibrium=equilibrium)
    >>> grid.grid_attrs['ngrid']
    8385

    Raises attribute error when grid initialied with unset equilibrium ids:
    >>> ExtrapolationGrid(1000, 'ids', 'coil')
    Traceback (most recent call last):
        ...
    AttributeError: equilibrium ids is None
    require valid ids when limit:ids or ngrid:1000 == 'ids'
    """

    ngrid: int | str = 5000
    limit: float | list[float] | str = 0.25
    index: str | slice | pandas.Index = 'plasma'
    equilibrium: Equilibrium | None = None

    def __post_init__(self):
        """Update grid attributes for equilibrium derived properties."""
        self.update_grid()
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    @property
    def grid_attrs(self) -> dict:
        """Return grid attributes."""
        return {attr: getattr(self, attr)
                for attr in [attr.name for attr in fields(ExtrapolationGrid)]
                if attr != 'equilibrium'}

    def update_grid(self):
        """Update  and update grid limits."""
        if self.limit != 'ids' and self.ngrid != 'ids':
            return
        if self.equilibrium is None:
            raise AttributeError('equilibrium ids is None\n'
                                 f'require valid ids when limit:{self.limit} '
                                 f'or ngrid:{self.ngrid} == \'ids\'')
        if self.limit == 'ids':  # Load grid limit from equilibrium ids.
            if self.equilibrium.data.grid_type != 1:
                raise TypeError('ids limits only valid for rectangular grids'
                                f'{self.equilibrium.data.grid_type} != 1')
            limit = [self.equilibrium.data.r.values,
                     self.equilibrium.data.z.values]
            if self.ngrid == 'ids':
                self.limit = limit
            else:
                self.limit = [limit[0][0], limit[0][-1],
                              limit[1][0], limit[1][-1]]
        if self.ngrid == 'ids':
            self.ngrid = self.equilibrium.data.dims['r'] * \
                self.equilibrium.data.dims['z']


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
        """Return cached 2D RectBivariateSpline psi2d interpolant."""
        return self._rbs('psi2d')

    def _rbs(self, attr):
        """Return 2D RectBivariateSpline interpolant."""
        return RectBivariateSpline(self.r, self.z, self.data[attr]).ev

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
        from nova.plot import plt, sns
        psi_norm = np.linspace(0, 1, 500)
        axes = plt.subplots(2, 1, sharex=True)[1]
        axes[0].plot(self.psi_norm, self.dpressure_dpsi, '.')
        axes[0].plot(psi_norm, self.p_prime(psi_norm), '-')
        axes[1].plot(self.psi_norm, self.f_df_dpsi, '.')
        axes[1].plot(psi_norm, self.ff_prime(psi_norm), '-')
        axes[0].set_ylabel(r'$P^\prime$')
        axes[1].set_ylabel(r'$FF^\prime$')
        axes[1].set_xlabel(r'$\psi_{norm}$')
        sns.despine()


@dataclass
class ExtrapolateMachine(Machine):
    """
    Extend Machine with default values for Extrapolate class.

    Parameters
    ----------
    pf_active: Ids | bool, optional
        pf active IDS. The default is True
    pf_passive: Ids | bool, optional
        pf passive IDS. The default is False
    wall: Ids | bool, optional
        wall IDS. The default is True
    """

    pf_active: Ids | bool = True
    pf_passive: Ids | bool = False
    wall: Ids | bool = True
    nplasma: int = 2500


@dataclass
class Extrapolate(ExtrapolateMachine, ExtrapolationGrid, Database):
    r"""
    An interface class for the extrapolation of an equilibrium IDS.

    Solves external coil currents in a least squares sense to match
    internal flux values provided by a source equilibrium containting:

        - values of :math:`\psi` internal to a boundary contour
        - flux functions :math:`p^\prime` and :math:`f f^\prime`

    The class may be run in one of three modes:

        - As an python IMAS **actor**, accepts and returns IDS(s)
        - As an python IMAS **code**, reads and writes IDS(s)
        - As a command line **script** see `extrapolate --help` for details

    Parameters
    ----------
    pulse: int, optional (required when ids not set)
        Pulse number. The default is 0.
    run: int, optional (required when ids not set)
        Run number. The default is 0.
    name: str, optional (required when ids not set)
        Ids name. The default is ''.
    user: str, optional (required when ids not set)
        User name. The default is public.
    machine: str, optional (required when ids not set)
        Machine name. The default is iter.
    backend: int, optional (required when ids not set)
        Access layer backend. The default is 13 (HDF5).
    ids: ImasIds, optional
        When set the ids parameter takes prefrence. The default is None.
    pf_active : Ids | bool, optional
        pf active IDS. The default is True
    pf_passive : Ids | bool, optional
        pf passive IDS. The default is False
    wall : Ids | bool, optional
        wall IDS. The default is True
    ngrid : {int, 'ids'}, optional
        Grid dimension. The default is 5000.

        - int: use input to set aproximate total node number
        - ids: aproximate total node number extracted from equilibrium ids.
    limit : {float, list[float], 'ids'}, optional
        Grid bounds. The default is 0.25.

        - float: expansion relative to coilset index. Must be greater than -1.
        - list[float]: explicit grid bounds [rmin, rmax, zmin, zmax].
        - ids: bounds extracted from from equilibrium ids.
    index : {'plasma', 'coil', slice, pandas.Index}
        Filament index from which relative grid limits are set.

    Attributes
    ----------
    group_attributes: dict
        Instance metadata.

    Raises
    ------
    LinAlgError
        Least squares fit does not converge.


    Notes
    -----
    The plasama and coils are modeled as finite area filliments with peicewise
    constant current distributions. Interactions between filiments are solved
    via the Biot Savart equation.

    Currents for each plasma filament :math:`I_i` are solved at the
    center of each filament as follows,

    .. math::
        I_i = -2 \pi A [r p\prime (\psi\prime) +
                        f f\prime(\psi\prime) / (\mu_0 r)]

    With a total plasma current :math:`I_p` condition enforced such that,

    .. math::
        I_p = \sum_i I_i

    Once the coil and plasma filament currents are known, the
    original solution may be mapped to a new grid with a boundary and a
    resolution diffrent to that given by the source equilibrium solution.

    Examples
    --------
    Pass a pulse and run number to initiate as an **IMAS code**:

    >>> from nova.imas.extrapolate import Extrapolate
    >>> pulse, run = 130506, 403  # CORSICA equilibrium solution
    >>> extrapolate = Extrapolate(pulse, run, ngrid=10, nplasma=10)
    >>> extrapolate.pulse, extrapolate.run
    (130506, 403)

    The equilibrium ids is read from file and stored as an ids attribute:

    >>> extrapolate.ids.code.name
    'CORSICA'

    To run code as an actor, first load an apropriate equilibrium IDS,

    >>> from nova.imas.database import Database
    >>> pulse, run = 130506, 403  # CORSICA equilibrium solution
    >>> equilibrium = Database(130506, 403, 'equilibrium', machine='iter')
    >>> equilibrium.pulse, equilibrium.run
    (130506, 403)

    then pass this ids to the Extrapolate class:

    >>> extrapolate = Extrapolate(ids=equilibrium.ids, limit='ids', ngrid=500, nplasma=100)
    >>> extrapolate.ionize(20)
    >>> extrapolate.itime
    20

    >>> extrapolate.plot('psi')

    """

    alpha: float = 1.2e-6
    nturn: int = 10
    filename: str = field(init=False, default='extrapolate')
    equilibrium: Equilibrium = field(init=False, repr=False)
    time_slice: TimeSlice = field(init=False, repr=False)

    def __post_init__(self):
        """Load equilibrium and coilset."""
        self.load_equilibrium()
        super().__post_init__()
        self.set_free()

    def load_equilibrium(self):
        """Load equilibrium dataset."""
        self.name = 'equilibrium'
        self.equilibrium = Equilibrium(**self.ids_attrs, ids=self.ids)

    def set_free(self):
        """Set free coils."""
        self.saloc['free'] = [self.loc[name, 'nturn'] >= self.nturn
                              for name in self.sloc.frame.index]

    #def update_metadata(self):
    #    """Return extrapolated equilibrium ids."""
    #    ids = imas.equilibrium()
    #    Properties('Equilibrium extrapolation',
    #               provider='Simon McIntosh')(ids.ids_properties)
    #    Code(self.group_attrs)(ids.code)
    #    ids.vacuum_toroidal_field = self.ids.vacuum_toroidal_field
    #    return ids

    @property
    def group_attrs(self):
        """Return group attributes."""
        return super().group_attrs | self.grid_attrs

    def build(self, **kwargs):
        """Build frameset and interpolation grid."""
        super().build(**kwargs)
        self.grid.solve(**self.grid_attrs)
        return self.store(self.filename)

    @property
    def itime(self) -> int:
        """Return time slice."""
        return int(self.time_slice.data.itime)

    def _update_time_slice(self, itime: int):
        """Update time slice instance."""
        self.time_slice = TimeSlice(self.equilibrium.data.isel(time=itime))

    def _update_turns(self):
        """Update plasma current distribution via filament nturns."""
        self.plasma.separatrix = self.time_slice.boundary
        self.sloc['plasma', 'Ic'] = self.time_slice.ip
        ionize = self.aloc['ionize']
        radius = self.aloc['x'][ionize]
        height = self.aloc['z'][ionize]
        psi = self.time_slice.psi_rbs(radius, height)
        psi_norm = self.time_slice.normalize(psi)
        current_density = radius * self.time_slice.p_prime(psi_norm) + \
            self.time_slice.ff_prime(psi_norm) / (mu_0 * radius)
        current_density *= -2*np.pi
        current = current_density * self.aloc['area'][ionize]
        self.aloc['nturn'][ionize] = current / current.sum()

    def ionize(self, itime: int):
        """Solve pf_active currents to fit internal flux."""
        self._update_time_slice(itime)
        self._update_turns()
        ionize = self.aloc['ionize']
        plasma = self.aloc['plasma']
        radius = self.aloc['x'][ionize]
        height = self.aloc['z'][ionize]
        plasma_index = self.plasmagrid.data.plasma_index
        matrix = self.plasmagrid['Psi'][ionize[plasma]]
        internal = -self.time_slice._rbs('psi2d')(radius, height)  # COCOS11
        target = internal - matrix[:, plasma_index]*self.time_slice.ip
        moore_penrose = MoorePenrose(matrix=matrix[:, self.saloc['free']],
                                     alpha=self.alpha)
        self.saloc['Ic'][self.saloc['free']] = moore_penrose / target

    def plot_boundary(self, itime: int):
        """Expose self._equilibrium plot boundary."""
        return self.equilibrium.plot_boundary(itime)

    def plot_2d(self, attr='psi'):
        """Plot plasma filements and polidal flux."""
        super().plot('plasma')
        levels = self.grid.plot(attr, levels=51, colors='C0', nulls=False)
        try:
            self.equilibrium.plot_2d(self.itime, attr, colors='C3',
                                     levels=-levels[::-1])
        except KeyError:
            pass
        self.plot_boundary(self.itime)

    def plot_bar(self):
        """Plot coil currents for single time-slice."""
        from nova.plot import plt, sns
        pf_active = PF_Active(**self.ids_attrs | dict(name='pf_active'))

        index = [name for name in self.subframe.subspace.index
                 if name in pf_active.data.coil_name.data]

        #print(self.sloc[index, ['Ic']].squeeze().values)

        plt.figure()
        plt.bar(index, 1e-3*self.sloc[index, ['Ic']].squeeze().values)
        plt.bar(index,
                1e-3 * pf_active.data.current.isel(time=self.itime).loc[index].data,
                width=0.5)

        print(np.linalg.norm(1e-3*self.sloc[index, ['Ic']].squeeze().values -
                             1e-3 * pf_active.data.current.isel(time=self.itime).loc[index].data))
        sns.despine()


        #pf_active.data.isel(time=20).current.data
        #plt.bar()


    def plot_waveform(self):
        """ """

if __name__ == '__main__':

    # import doctest
    # doctest.testmod()
    """
    # pulse, run = 114101, 41  # JINTRAC
    pulse, run = 130506, 403  # CORSICA

    extrapolate = Extrapolate(pulse, run)

    extrapolate.ionize(5)
    extrapolate.plot_2d('psi')
    # extrapolate.plasmagrid.plot()

    extrapolate.plot_bar()
    """
