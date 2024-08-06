"""Interpolate equilibria within separatrix."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.constants import mu_0

from nova.imas.dataset import Ids
from nova.imas.machine import Machine
from nova.imas.profiles import Profile


@dataclass
class Operate(Profile, Machine):
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

    equilibrium: Ids | bool | str = True
    pf_active: Ids | bool | str = True
    pf_passive: Ids | bool | str = False
    wall: Ids | bool | str = True
    dplasma: int | float = -2500

    def update(self):
        """Extend itime update."""
        super().update()
        self.plasma.update()
        self.update_plasma_shape()
        self.update_current()

    def update_current(self):
        """Update coil currents from pf_active."""
        try:
            self.sloc["coil", "Ic"] = self["current"]
            self.sloc["plasma", "Ic"] = self["ip"]
            if "passive_current" in self.data:
                self.sloc["passive", "Ic"] = self["passive_current"]
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

    # import doctest

    # doctest.testmod(verbose=False)

    # pulse, run = 105007, 9
    # pulse, run = 135007, 4
    # pulse, run = 105028, 1
    args = 135013, 2

    # args = 130506, 403  # CORSICA

    # args = 45272, 1, "mast_u"  # MastU

    kwargs = {
        "pulse": 57410,
        "run": 0,
        "machine": "west",
        "pf_passive": {"occurrence": 0},
        "pf_active": {"occurrence": 0},
    }  # WEST
    """
    kwargs = {
        "pulse": 17151,
        "run": 3,
        "machine": "aug",
        "pf_passive": False,
        "pf_active": True,
    }  # AUG
    """
    operate = Operate(
        **kwargs,
        wall=True,
        dplasma=-2000,
        tplasma="h",
        nwall=3,
        ngrid=2e4,
    )
    """
    operate = Operate(
        **kwargs,
        pf_active=True,
        wall=True,
        elm=False,
        dplasma=-2000,
        ngrid=2000,
        tplasma="h",
        limit=0.25,
        nlevelset=None,
        nwall=10,
        nforce=0,
        force_index="plasma",
    )
    """

    operate.time = 33

    # attr = "j_tor"
    attr = "psi"
    # operate.plot()
    operate.plot_2d(attr)

    levels = operate.plot_2d(attr, colors="C1", label="NICE")

    levels = -levels[::-1] + 2.3
    # operate.sloc[0, "Ic"] *= 0
    operate.plot()
    # operate.plasma.plot(attr)
    levels = operate.grid.plot(attr, colors="C2", label="NOVA")

    """
    j_tor = (
        operate.aloc["plasma", "Ic"]
        * operate.aloc["plasma", "nturn"]
        / operate.aloc["plasma", "area"]
    )
    operate.axes.tricontour(
        operate.plasmagrid.data.x.data,
        operate.plasmagrid.data.z.data,
        j_tor,
        levels=levels,
        colors="C0",
    )
    """

    # operate.grid.plot(attr, levels=levels, colors="C0")

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
