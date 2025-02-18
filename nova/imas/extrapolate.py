"""Extrapolate equilibria beyond separatrix."""

import bisect
from dataclasses import dataclass

import numpy as np
import pandas

from tqdm import tqdm

from nova.biot.biot import Nbiot
from nova.imas.dataset import Ids, ImasIds
from nova.imas.ids_index import IdsIndex
from nova.imas.operate import Operate
from nova.linalg.regression import MoorePenrose


# pylint: disable=too-many-ancestors


@dataclass
class Grid:
    """
    Specify interpolation grid.

    Parameters
    ----------
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
    ids: ImasIds, optional
        IMAS IDS required for equilibrium derived grid dimensions.
        The default is None

    Examples
    --------
    Skip doctest if IMAS instalation or requisite IDS(s) not found.

    >>> import pytest
    >>> from nova.imas.database import Database
    >>> from nova.imas.equilibrium import EquilibriumData
    >>> try:
    ...     _ = Database(130506, 403, 'equilibrium').get()
    ... except:
    ...     pytest.skip('IMAS not found or 130506/403 unavailable')

    Manualy specify grid relitive to coilset:
    >>> Grid(100, 0, 'coil').grid_attrs
    {'number': 100, 'limit': 0, 'index': 'coil'}

    Specify grid relitive to equilibrium ids.
    >>> equilibrium = EquilibriumData(130506, 403)
    >>> Grid(50, 'ids', ids=equilibrium.ids).grid_attrs
    {'number': 50, 'limit': [2.75, 8.9, -5.49, 5.51], 'index': 'plasma'}

    Extract exact grid from equilibrium ids.
    >>> grid = Grid('ids', 'ids', ids=equilibrium.ids)
    >>> grid.grid_attrs['number']
    8385

    Raises attribute error when grid initialied with unset data attribute:
    >>> Grid(1000, 'ids', 'coil')
    Traceback (most recent call last):
        ...
    AttributeError: Require IMAS ids when limit:ids or ngrid:1000 == 'ids'

    """

    ngrid: Nbiot = 5000
    limit: float | list[float] | str = 0.25
    index: str | slice | pandas.Index = "plasma"
    ids: ImasIds | None = None

    def __post_init__(self):
        """Update grid attributes for equilibrium derived properties."""
        self.update_grid()
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    @property
    def grid_attrs(self) -> dict:
        """Return grid attributes."""
        return {"number": self.ngrid} | {
            attr: getattr(self, attr) for attr in ["limit", "index"]
        }

    def update_grid(self):
        """Update grid limits."""
        if self.limit != "ids" and self.ngrid != "ids":
            return
        if self.ids is None:
            raise AttributeError(
                f"Require IMAS ids when limit:{self.limit} "
                f"or ngrid:{self.ngrid} == 'ids'"
            )
        ids_index = IdsIndex(self.ids, "time_slice")
        index = ids_index.get_slice(0, "profiles_2d.grid_type.index")
        grid = ids_index.get_slice(0, "profiles_2d.grid")
        if self.limit == "ids":  # Load grid limit from equilibrium ids.
            if index != 1:
                raise TypeError(
                    "ids limits only valid for rectangular grids" f"{index} != 1"
                )
            limit = [grid.dim1, grid.dim2]
            if self.ngrid == "ids":
                self.limit = limit
            else:
                self.limit = [limit[0][0], limit[0][-1], limit[1][0], limit[1][-1]]
        if self.ngrid == "ids":
            self.ngrid = len(grid.dim1) * len(grid.dim2)


@dataclass
class Extrapolate(Grid, Operate):
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
    ...     _ = Database(130506, 403, 'equilibrium')
    ...     _ = Database(111001, 202, 'pf_active', machine='iter_md')
    ... except:
    ...     pytest.skip('IMAS not found or 130506/403, 111001/202 unavailable')

    Pass a pulse and run number to initiate as an **IMAS code**:

    >>> from nova.imas.extrapolate import Extrapolate
    >>> pulse, run = 130506, 403  # CORSICA equilibrium solution
    >>> extrapolate = Extrapolate(pulse, run, pf_active='iter_md', wall='iter_md',
    ...                           ngrid=10, dplasma=-10, tplasma='hex')
    >>> extrapolate.pulse, extrapolate.run
    (130506, 403)

    The equilibrium ids is read from file and stored as an ids attribute:

    >>> extrapolate.get('equilibrium').code.name.value
    'CORSICA'

    To run code as an **IMAS actor**,
    first load an apropriate equilibrium IDS,

    >>> pulse, run = 130506, 403  # CORSICA equilibrium solution
    >>> equilibrium = Database(130506, 403, name='equilibrium')
    >>> equilibrium.pulse, equilibrium.run
    (130506, 403)

    then pass this ids to the Extrapolate class:

    >>> extrapolate = Extrapolate(ids=equilibrium.ids, limit='ids',
    ...                           ngrid=500, dplasma=-100, tplasma='hex')
    >>> extrapolate.itime = 20
    >>> extrapolate.itime
    20

    """

    name: str = "equilibrium"
    pf_active: Ids | bool | str = "iter_md"
    mode: str = "r"
    gamma: float = 9e-6
    nturn: int = 0

    def __post_init__(self):
        """Load equilibrium and coilset."""
        super().__post_init__()
        self.select_free_coils()

    @property
    def group_attrs(self):
        """Return group attributes."""
        return super().group_attrs | self.grid_attrs

    def solve_biot(self):
        """Extend machine solve biot to include extrapolation grid."""
        super().solve_biot()
        self.grid.solve(**self.grid_attrs)

    def select_free_coils(self):
        """Assign free coils."""
        index = [self.loc[name, "subref"] for name in self.sloc.frame.index]
        self.saloc["free"] = self.frame.iloc[index].nturn >= self.nturn
        self.saloc["free"] = self.saloc["free"] & ~self.saloc["plasma"]

    # def update_metadata(self):
    #    """Return extrapolated equilibrium ids."""
    #    ids = imas.equilibrium()
    #    Properties('EquilibriumData extrapolation',
    #               provider='Simon McIntosh')(ids.ids_properties)
    #    Code(self.group_attrs)(ids.code)
    #    ids.vacuum_toroidal_field = self.ids.vacuum_toroidal_field
    #    return ids

    def update_current(self):
        """Only update plasma current (ignore coil currents if present."""
        self.sloc["plasma", "Ic"] = self["ip"]

    def update(self):
        """Solve pf_active currents to fit internal flux."""
        super().update()
        ionize = self.aloc["ionize"]
        plasma = self.aloc["plasma"]
        radius = self.aloc["x"][ionize]
        height = self.aloc["z"][ionize]
        plasma_index = self.plasmagrid.data.source_plasma_index
        matrix = self.plasmagrid["Psi"][ionize[plasma]]
        internal = -self.psi_rbs(radius, height)  # COCOS11
        target = internal - matrix[:, plasma_index] * float(self["ip"])
        moore_penrose = MoorePenrose(
            matrix=matrix[:, self.saloc["free"]], gamma=self.gamma
        )
        self.saloc["Ic"][self.saloc["free"]] = moore_penrose / target

    def boundary_mask(self, mask):
        """Return boundary mask."""
        match mask:
            case int() if mask == -1:
                return ~self.grid.mask(self.boundary)
            case int() if mask == 1:
                return self.grid.mask(self.boundary)
            case _:
                raise IndexError(f"mask {mask} not in [-1, 1]")

    def masked_data(self, attr: str, mask=0):
        """Return masked data."""
        data = getattr(self.grid, f"{attr}_")
        if mask == 0:
            return data
        return np.ma.masked_array(data, self.boundary_mask(mask), copy=True)

    def get_masks(self, mask: str | None):
        """Return mask array."""
        match mask:
            case None:
                return -1, None, None
            case "plasma":
                return -1, -1, None
            case "ids":
                return 0, -1, None
            case "nova":
                return 0, -1, 1
            case "map":
                return 0, -1, 1

    def plot_2d_masked(self, attr="psi", mask=None, levels=51, axes=None):
        """Plot masked 2d data."""
        masks = self.get_masks(mask)
        if masks[0] is not None:
            try:
                if attr == "psi":
                    _levels = -levels[::-1]  # COCOS
                else:
                    _levels = levels
                super().plot_2d(
                    attr, mask=masks[0], levels=_levels, colors="gray", axes=self.axes
                )
            except KeyError:
                pass
        if masks[1] is not None:
            self.grid.plot(
                self.masked_data(attr, masks[1]),
                levels=levels,
                colors="C0",
                nulls=False,
            )
        if masks[2] is not None:
            self.grid.plot(
                self.masked_data(attr, masks[2]),
                levels=levels,
                colors="C2",
                nulls=False,
            )

    def plot_2d(self, attr="psi", mask=None, levels=51, axes=None):
        """Plot plasma filements and polidal flux.

        Examples
        --------
        Skip doctest if IMAS instalation or requisite IDS(s) not found.

        >>> import pytest
        >>> from nova.imas.database import Database
        >>> try:
        ...     _ = Database(130506, 403, 'equilibrium')
        ... except:
        ...     pytest.skip('IMAS not found or 130506/403 unavailable')

        >>> equilibrium = Database(130506, 403, 'equilibrium')
        >>> extrapolate = Extrapolate(ids=equilibrium.ids, limit='ids',
        ...                           ngrid=500, dplasma=-100, tplasma='hex')

        Skip doctest if graphics dependencies are not available.

        >>> try:
        ...     _ = extrapolate.set_axes('2d')
        ... except:
        ...     pytest.skip('graphics dependencies not available')

        >>> extrapolate.itime = 20
        >>> extrapolate.plot_2d('psi')
        """
        self.get_axes("2d", axes=axes)
        super().plot("plasma", axes=self.axes)
        self.plasma.wall.plot()
        vector = getattr(self.grid, attr)
        levels = np.linspace(vector.min(), vector.max(), levels)
        self.plot_2d_masked(attr, mask, levels, self.axes)
        self.plot_boundary()

    def _make_frame(self, time):
        """Make frame for annimation."""
        self.axes.clear()
        max_time = np.min([self.data.time[-1], self.max_time])
        itime = bisect.bisect_left(self.data.time, max_time * time / self.duration)
        try:
            self.itime = itime
        except ValueError:
            pass
        # self.plot_bar()
        self.plot_2d("psi", mask="map", axes=self.axes)
        self.mpl["pyplot"].tight_layout()
        return self.mpy.mplfig_to_npimage(self.fig)

    def annimate(self, duration: float, filename="extrapolate"):
        """Generate annimiation."""
        self.duration = duration
        self.max_time = 750
        animation = self.mpy.editor.VideoClip(self._make_frame, duration=duration)
        animation.write_gif(f"{filename}.gif", fps=10)

    def plot_bar(self):
        """Plot coil currents for single time-slice."""
        index = [
            name
            for name in self.subframe.subspace.index
            if name in self.data.coil_name.data
        ]
        self.get_axes(None, "1d")
        self.axes.barh(
            index, 1e-3 * self["current"].loc[index].data, color="gray", label="ids"
        )
        self.axes.barh(
            index,
            1e-3 * self.sloc[index, ["Ic"]].squeeze().values,
            color="C0",
            label="nova",
            height=0.5,
        )
        self.axes.invert_yaxis()
        self.axes.set_xlim([-40, 40])
        self.axes.set_xlabel("current kA")
        self.axes.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.05))

    def plot_waveform(self, ip_index=0.05):
        """Plot coil current waveform."""
        time_index = abs(self.data.ip) > ip_index * abs(self.data.ip).max()
        name_map = dict(CS1="CS1U", VS3="VS3U")
        coil_name = [name_map.get(name, name) for name in self.data.coil_name.data]
        self.data = self.data.assign_coords(dict(coil_name=coil_name))

        index = [
            name
            for name in self.subframe.subspace.index
            if name in self.data.coil_name.data
        ]
        data_index = [
            np.where(self.data.coil_name.data == name)[0][0] for name in index
        ]
        self.data["_current"] = self.data.current.copy()
        for itime in tqdm(range(self.data.sizes["time"])):
            self.itime = itime
            self["_current"][data_index] = self.sloc[index, ["Ic"]].squeeze().values

        # switch reference sign for vs3 loop (Upper to Lower)
        self.data._current[:, -1] *= -1  # TODO fix this

        self.get_axes("1d")
        self.axes.plot(
            self.data.time[time_index],
            1e-3 * self.data.current[time_index, data_index],
            color="gray",
        )
        self.axes.plot(
            self.data.time[time_index],
            1e-3 * self.data._current[time_index, data_index],
            color="C0",
        )
        self.axes.set_xlabel("time s")
        self.axes.set_ylabel("current kA")
        self.axes.set_ylim([-45, 45])
        Line2D = self.mpl["lines"].Line2D
        self.axes.legend(
            handles=[
                Line2D([0], [0], label="ids", color="gray"),
                Line2D([0], [0], label="nova", color="C0"),
            ],
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
        )


if __name__ == "__main__":

    # import doctest

    # doctest.testmod(verbose=False)

    pulse, run = 114101, 41  # JINTRAC
    # pulse, run = 130506, 403  # CORSICA
    # pulse, run = 105028, 1  # DINA
    # pulse, run = 135011, 7  # DINA
    # pulse, run = 135013, 2
    # pulse, run = 134173, 106  # DINA-JINTRAC

    extrapolate = Extrapolate(
        pulse,
        run,
        pf_active="iter_md",
        pf_passive=False,
        wall="iter_md",
        tplasma="h",
    )

    import matplotlib.pylab as plt

    extrapolate.mpl_axes.fig = plt.figure(figsize=(6, 9))

    # extrapolate.plot_waveform()

    extrapolate.time = 80
    extrapolate.plot_2d("psi", mask="map")
    plt.tight_layout()

    # plt.savefig('build.png')

    # extrapolate.plot_bar()
    # extrapolate.plasmagrid.plot()

    # extrapolate.annimate(5, filename=filename)
