"""Extrapolate equilibria beyond separatrix."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import mu_0

if TYPE_CHECKING:
    import pandas
    import xarray

from nova.imas.operate import Operate
from nova.linalg.regression import MoorePenrose


# pylint: disable=too-many-ancestors


@dataclass
class Extrapolate(Operate):
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
    via the Biot Savart law.

    Currents for each plasma filament :math:`I_i` are solved at the
    center of each filament as follows,

    .. math::
        I_i = -2 \pi A [r p^\prime (\psi) +
                        f f^\prime(\psi) / (\mu_0 r)]

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

    To run code as an **IMAS actor**,
    first load an apropriate equilibrium IDS,

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

    def __post_init__(self):
        """Load equilibrium and coilset."""
        super().__post_init__()
        self.set_free()

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

    def update(self):
        """Solve pf_active currents to fit internal flux."""
        super().update()
        ionize = self.aloc['ionize']
        plasma = self.aloc['plasma']
        radius = self.aloc['x'][ionize]
        height = self.aloc['z'][ionize]
        plasma_index = self.plasmagrid.data.plasma_index
        matrix = self.plasmagrid['Psi'][ionize[plasma]]
        internal = -self.psi_rbs(radius, height)  # COCOS11
        target = internal - matrix[:, plasma_index]*float(self['ip'])
        moore_penrose = MoorePenrose(matrix=matrix[:, self.saloc['free']],
                                     alpha=self.alpha)
        self.saloc['Ic'][self.saloc['free']] = moore_penrose / target

    def plot_2d(self, attr='psi', axes=None):
        """Plot plasma filements and polidal flux."""
        self.set_axes(axes, '2d')
        super().plot('plasma')
        levels = self.grid.plot(attr, levels=51, colors='C0', nulls=False)
        try:
            super().plot_2d(self.itime, attr, colors='C3',
                            levels=-levels[::-1], axes=self.axes)
        except KeyError:
            print('key error', attr)
            pass
        self.plot_boundary(self.itime)

    '''
    def plot_bar(self):
        """Plot coil currents for single time-slice."""
        pf_active = PF_Active(**self.ids_attrs | dict(name='pf_active'))

        index = [name for name in self.subframe.subspace.index
                 if name in pf_active.data.coil_name.data]

        #print(self.sloc[index, ['Ic']].squeeze().values)
        self.mpl_axes.generate('1d')
        self.axes.bar(index, 1e-3*self.sloc[index, ['Ic']].squeeze().values)
        self.axes.bar(index,
                1e-3 * pf_active.data.current.isel(time=self.itime).loc[index].data,
                width=0.5)

        print(np.linalg.norm(1e-3*self.sloc[index, ['Ic']].squeeze().values -
                             1e-3 * pf_active.data.current.isel(time=self.itime).loc[index].data))

        #pf_active.data.isel(time=20).current.data
        #plt.bar()


    def plot_waveform(self):
        """ """
    '''

if __name__ == '__main__':

    # import doctest
    # doctest.testmod()

    # pulse, run = 114101, 41  # JINTRAC
    pulse, run = 130506, 403  # CORSICA
    #pulse, run = 105028, 1  # DINA

    extrapolate = Extrapolate(pulse, run)

    extrapolate.itime = -1
    extrapolate.plot_2d('psi')
    #extrapolate.plasmagrid.plot()

    '''
    try:
        extrapolate.plot_bar()
    except IndexError:
        pass
    '''
