"""Interpolate equilibria within separatrix."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import mu_0

from nova.biot.biot import Nbiot
from nova.imas.dataset import Ids, ImasIds
from nova.imas.ids_index import IdsIndex
from nova.imas.machine import Machine
from nova.imas.profiles import Profile

if TYPE_CHECKING:
    import pandas


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
        ids_index = IdsIndex(self.ids)
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
class Operate(Grid, Profile, Machine):
    """
    Extend Machine with default values for Operate class.

    Extract coil and plasma currents from ids and apply to CoilSet.

    Parameters
    ----------
    pf_active: Ids | bool, optional
        pf active IDS. The default is True
    pf_passive: Ids | bool, optional
        pf passive IDS. The default is False
    wall: Ids | bool, optional
        wall IDS. The default is True
    """

    pf_active: Ids | bool | str = True
    pf_passive: Ids | bool | str = False
    wall: Ids | bool | str = "iter_md"
    dplasma: int | float = -2500

    @property
    def group_attrs(self):
        """Return group attributes."""
        return super().group_attrs | self.grid_attrs

    def solve_biot(self):
        """Extend machine solve biot to include extrapolation grid."""
        super().solve_biot()
        self.grid.solve(**self.grid_attrs)

    def update(self):
        """Extend itime update."""
        super().update()
        self.update_plasma_shape()
        self.update_current()

    def update_current(self):
        """Update coil currents from pf_active."""
        try:
            self.sloc["coil", "Ic"] = self["current"]
            self.sloc["plasma", "Ic"] = self["ip"]
        except KeyError:  # data unavailable
            return

    def update_plasma_shape(self):
        """Ionize plasma filaments and set turn number."""
        if "boundary" not in self.data:
            return
        self.plasma.separatrix = self.boundary
        ionize = self.aloc["ionize"]
        radius = self.aloc["x"][ionize]
        height = self.aloc["z"][ionize]
        psi = self.psi_rbs(radius, height)
        psi_norm = self.normalize(psi)
        current_density = radius * self.p_prime(psi_norm) + self.ff_prime(psi_norm) / (
            mu_0 * radius
        )
        current_density *= -2 * np.pi
        current = current_density * self.aloc["area"][ionize]
        self.aloc["nturn"][ionize] = current / current.sum()


if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=False)

    # pulse, run = 105007, 9
    # pulse, run = 135007, 4
    # pulse, run = 105028, 1
    args = 135013, 2

    # args = 130506, 403  # CORSICA

    # args = 45272, 1, "mast_u"  # MastU

    operate = Operate(
        *args,
        pf_active="iter_md",
        elm=True,
        dplasma=-2000,
        ngrid=2000,
        tplasma="h",
        limit=0.25,
        nlevelset=None,
        nwall=10,
        nforce=0,
        force_index="plasma",
    )

    operate.itime = 50

    """
    import pyvista as pv
    import vedo
    
    psi = operate.grid.psi.reshape(-1)
    points = np.stack(
        [
            operate.grid.data.x2d,
            np.zeros_like(operate.grid.data.x2d),
            operate.grid.data.z2d,
        ],
        axis=-1,
    ).reshape(-1, 3)

    mesh = pv.PolyData(points).delaunay_2d()
    contours = mesh.contour(isosurfaces=71, scalars=operate.grid.psi.reshape(-1))

    vedo.Mesh(contours, c="black").show()
    operate.frame.vtkplot()
    """

    # grid.plot(show_edges=True)

    # operate.force.solve(0, index="plasma")

    # operate.itime = 50

    # operate.sloc["VS3U", "Ic"] = 0
    # operate.sloc["mELM", "Ic"] = 0

    def plot_force(mode: str, time=4, Ivs3=60e3, Ielm=10.5e3):
        """Return plasma vertical force."""
        operate.time = time
        operate.sloc["VS3U", "Ic"] = 0
        operate.sloc["passive", "Ic"] = 0
        fz_ref = operate.force.fz[0]
        match mode.lower():
            case "vs3":
                operate.sloc["VS3U", "Ic"] = -Ivs3
                current = Ivs3
            case "mid elm":
                operate.sloc["mELM", "Ic"] = Ielm
                current = Ielm
            case "all elm":
                operate.sloc["lELM", "Ic"] = Ielm
                operate.sloc["mELM", "Ic"] = Ielm
                operate.sloc["uELM", "Ic"] = Ielm
                current = Ielm
            case _:
                raise NotImplementedError(f"mode {mode} not implemented")
        delta_fz = operate.force.fz[0] - fz_ref
        operate.sloc[:-5, "Ic"] = 0
        operate.sloc["plasma", "Ic"] = 0

        operate.set_axes("2d")
        operate.plasmawall.plot(limitflux=False)
        operate.plasmagrid.plot(attr="br", clabel=True, colors="k")
        operate.plasmagrid.plot(attr="psi")
        operate.plot("plasma")
        operate.plot("elm")
        operate.plot("vs3")

        operate.axes.set_title(
            f"{mode}: {1e-3*current:1.1f}kA \n"
            + "contours: $B_r$ mT \n"
            + rf" $\Delta f_z$: {1e-6*delta_fz:1.2f}MN"
        )
        return

    # plot_force("vs3")
    # plot_force("vs3", Ivs3=5e3)
    # plot_force("mid elm")
    # plot_force("all elm")
    # operate.loc["plasma", "nturn"] = 0

    # operate.force.plot()

    """
    from nova.imas.coils_non_axisymmetric import Coils_Non_Axisymmetyric

    coils_3d = Coils_Non_Axisymmetyric(115001, 1)
    coils_3d.frame.part = "_elm"
    operate += coils_3d
    operate.frame.vtkplot()
    """

    """
    operate.grid.solve(1000)

    operate.sloc["Ic"][-1] = 20

    operate.plot("plasma")
    operate.plasma.plot()
    operate.plot_boundary()
    operate.plasma.lcfs.plot()

    operate.plot_2d()

    operate.plasmagrid.plot()
    """

    """
    index = abs(operate.data.ip.data) > 1e3

    li_3 = np.zeros(operate.data.sizes["time"])
    for i in np.arange(operate.data.sizes["time"])[index]:
        operate.itime = i
        if operate["li_3"] == 0:
            continue
        li_3[i] = operate.plasma.li_3

    operate.set_axes("1d")
    operate.axes.plot(operate.data.time[index], operate.data.li_3[index])
    operate.axes.plot(operate.data.time[index], li_3[index])
    """
