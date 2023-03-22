"""Extrapolate equilibria beyond separatrix."""
import bisect
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

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
    The plasma and coils are modelled as finite area filaments with piecewise
    constant current distributions. Interactions between filaments are solved
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
    resolution different to that given by the source equilibrium solution.

    Examples
    --------
    Skip doctest if IMAS instalation or requisite IDS(s) not found.

    >>> import pytest
    >>> from nova.imas.database import Database
    >>> try:
    ...     _ = Database(130506, 403).get_ids('equilibrium')
    ...     _ = Database(111001, 202, 'iter_md').get_ids('pf_active')
    ... except:
    ...     pytest.skip('IMAS not found or 130506/403, 111001/202 unavailable')

    Pass a pulse and run number to initiate as an **IMAS code**:

    >>> from nova.imas.extrapolate import Extrapolate
    >>> pulse, run = 130506, 403  # CORSICA equilibrium solution
    >>> extrapolate = Extrapolate(pulse, run, pf_active='iter_md',
    ...                           ngrid=10, nplasma=10)
    >>> extrapolate.pulse, extrapolate.run
    (130506, 403)

    The equilibrium ids is read from file and stored as an ids attribute:

    >>> extrapolate.get_ids().code.name
    'CORSICA'

    To run code as an **IMAS actor**,
    first load an apropriate equilibrium IDS,

    >>> from nova.imas.database import Database
    >>> pulse, run = 130506, 403  # CORSICA equilibrium solution
    >>> equilibrium = Database(130506, 403, name='equilibrium')
    >>> equilibrium.pulse, equilibrium.run
    (130506, 403)

    then pass this ids to the Extrapolate class:

    >>> extrapolate = Extrapolate(ids=equilibrium.ids_data, limit='ids',
    ...                           ngrid=500, nplasma=100)
    >>> extrapolate.itime = 20
    >>> extrapolate.itime
    20

    >>> extrapolate.plot_2d('psi')

    """

    gamma: float = 9e-6
    nturn: int = 0

    def __post_init__(self):
        """Load equilibrium and coilset."""
        super().__post_init__()
        self.select_free_coils()

    def select_free_coils(self):
        """Select free coils."""
        index = [self.loc[name, 'subref'] for name in self.sloc.frame.index]
        self.saloc['free'] = self.frame.iloc[index].nturn >= self.nturn
        self.saloc['free'] = self.saloc['free'] & ~self.saloc['plasma']

    #def update_metadata(self):
    #    """Return extrapolated equilibrium ids."""
    #    ids = imas.equilibrium()
    #    Properties('Equilibrium extrapolation',
    #               provider='Simon McIntosh')(ids.ids_properties)
    #    Code(self.group_attrs)(ids.code)
    #    ids.vacuum_toroidal_field = self.ids.vacuum_toroidal_field
    #    return ids

    def update_current(self):
        """Only update plasma current (ignore coil currents if present."""
        self.sloc['plasma', 'Ic'] = self['ip']

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
                                     gamma=self.gamma)
        self.saloc['Ic'][self.saloc['free']] = moore_penrose / target

    def boundary_mask(self, mask):
        """Return boundary mask."""
        match mask:
            case int() if mask == -1:
                return ~self.grid.mask(self.boundary)
            case int() if mask == 1:
                return self.grid.mask(self.boundary)
            case _:
                raise IndexError(f'mask {mask} not in [-1, 1]')

    def masked_data(self, attr: str, mask=0):
        """Return masked data."""
        data = getattr(self.grid, f'{attr}_')
        if mask == 0:
            return data
        return np.ma.masked_array(data, self.boundary_mask(mask), copy=True)

    def get_masks(self, mask: str | None):
        """Return mask array."""
        match mask:
            case None:
                return -1, None, None
            case 'plasma':
                return -1, -1, None
            case 'ids':
                return 0, -1, None
            case 'nova':
                return 0, -1, 1
            case 'map':
                return 0, -1, 1

    def plot_2d_masked(self, attr='psi', mask=None, levels=51, axes=None):
        """Plot masked 2d data."""
        masks = self.get_masks(mask)
        if masks[0] is not None:
            try:
                if attr == 'psi':
                    _levels = -levels[::-1]
                else:
                    _levels = levels
                super().plot_2d(attr, mask=masks[0], levels=_levels,
                                colors='gray', axes=self.axes)
            except KeyError:
                pass
        if masks[1] is not None:
            self.grid.plot(self.masked_data(attr, masks[1]), levels=levels,
                           colors='C0', nulls=False)
        if masks[2] is not None:
            self.grid.plot(self.masked_data(attr, masks[2]), levels=levels,
                           colors='C2', nulls=False)

    def plot_2d(self, attr='psi', mask=None, levels=51, axes=None):
        """Plot plasma filements and polidal flux."""
        self.get_axes('2d', axes=axes)
        super().plot(axes=self.axes)#'plasma')
        self.plasma.wall.plot()
        vector = getattr(self.grid, attr)
        levels = np.linspace(vector.min(), vector.max(), levels)
        self.plot_2d_masked(attr, mask, levels, self.axes)
        self.plot_boundary()

    def _make_frame(self, time):
        """Make frame for annimation."""
        self.axes.clear()
        max_time = np.min([self.data.time[-1], self.max_time])
        itime = bisect.bisect_left(
            self.data.time, max_time * time / self.duration)
        try:
            self.itime = itime
        except ValueError:
            pass
        # self.plot_bar()
        self.plot_2d('psi', mask='map', axes=self.axes)
        self.mpl['pyplot'].tight_layout()
        return self.mpy.mplfig_to_npimage(self.fig)

    def annimate(self, duration: float, filename='extrapolate'):
        """Generate annimiation."""
        self.duration = duration
        self.max_time = 150
        animation = self.mpy.editor.VideoClip(
            self._make_frame, duration=duration)
        animation.write_gif(f'{filename}.gif', fps=10)

    def plot_bar(self):
        """Plot coil currents for single time-slice."""
        index = [name for name in self.subframe.subspace.index
                 if name in self.data.coil_name.data]
        self.get_axes(None, '1d')
        self.axes.barh(index, 1e-3*self['current'].loc[index].data,
                       color='gray', label='ids')
        self.axes.barh(index, 1e-3*self.sloc[index, ['Ic']].squeeze().values,
                       color='C0', label='nova', height=0.5)
        self.axes.invert_yaxis()
        self.axes.set_xlim([-40, 40])
        self.axes.set_xlabel('current kA')
        self.axes.legend(ncol=2, loc='upper center',
                         bbox_to_anchor=(0.5, 1.05))

    def plot_waveform(self, ip_index=0.05):
        """Plot coil current waveform."""
        time_index = abs(self.data.ip) > ip_index*abs(self.data.ip).max()
        name_map = dict(CS1='CS1U', VS3='VS3U')
        coil_name = [name_map.get(name, name)
                     for name in self.data.coil_name.data]
        self.data = self.data.assign_coords(dict(coil_name=coil_name))

        index = [name for name in self.subframe.subspace.index
                 if name in self.data.coil_name.data]
        data_index = [np.where(self.data.coil_name.data == name)[0][0]
                      for name in index]
        self.data['_current'] = self.data.current.copy()
        for itime in tqdm(range(self.data.dims['time'])):
            self.itime = itime
            self['_current'][data_index] = \
                self.sloc[index, ['Ic']].squeeze().values

        # switch reference sign for vs3 loop (Upper to Lower)
        self.data._current[:, -1] *= -1

        self.get_axes('1d')
        self.axes.plot(self.data.time[time_index],
                       1e-3*self.data.current[time_index, data_index],
                       color='gray')
        self.axes.plot(self.data.time[time_index],
                       1e-3*self.data._current[time_index, data_index],
                       color='C0')
        self.axes.set_xlabel('time s')
        self.axes.set_ylabel('current kA')
        self.axes.set_ylim([-45, 45])
        Line2D = self.mpl['lines'].Line2D
        self.axes.legend(handles=[Line2D([0], [0], label='ids', color='gray'),
                                  Line2D([0], [0], label='nova', color='C0')],
                         ncol=2, loc='upper center',
                         bbox_to_anchor=(0.5, 1.1))


if __name__ == '__main__':

    # pulse, run = 114101, 41  # JINTRAC
    pulse, run = 130506, 403  # CORSICA
    pulse, run = 105028, 1  # DINA
    # pulse, run = 135011, 7  # DINA

    extrapolate = Extrapolate(pulse, run, pf_passive=False,
                              pf_active='iter_md')

    import matplotlib.pylab as plt
    extrapolate.mpl_axes.fig = plt.figure(figsize=(6, 9))

    # extrapolate.plot_waveform()

    extrapolate.itime = 36
    extrapolate.plot_2d('psi', mask='map')
    plt.tight_layout()

    from nova.imas.pf_passive import PF_Passive
    pf_passive = PF_Passive(pulse, run)

    #plt.savefig('build.png')

    #extrapolate.plot_bar()
    #extrapolate.plasmagrid.plot()

    #extrapolate.annimate(5, filename=filename)
