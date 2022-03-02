"""Perform equilibrium reconstrucitons from IMAS data."""
from dataclasses import dataclass, field

from functools import cached_property
import xarray

from nova.electromagnetic.biotgrid import BiotPlot
from nova.electromagnetic.framesetloc import LocIndexer
from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import Machine
from nova.imas.pf_active import PF_Active
from nova.utilities.pyplot import plt


@dataclass
class Current:

    sloc: LocIndexer
    index: str
    ids_data: xarray.Dataset
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """ """


@dataclass
class Fit(BiotPlot, Machine):
    """Manage equilibrium reconstruciton fits."""

    shot: int = 135011
    run: int = 7
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Load ids dataset."""
        super().__post_init__()
        self.data = xarray.merge([PF_Active(self.shot, self.run).data,
                                  Equilibrium(self.shot, self.run).data],
                                 combine_attrs='drop_conflicts')

    @cached_property
    def current(self):
        """Return current array."""
        return self.data.current.data

    def plot(self):
        """Plot fit."""
        plt.set_aspect(0.8)
        super().plot('plasma')
        self.plasma.plot()
        self.plasmagrid.plot(levels=21, colors='C0')
        self.plasma.boundary.plot()


if __name__ == '__main__':

    fit = Fit(135011, 7)
    #fit.build(dcoil=0.25, dshell=0.5, dplasma=-1000, tcoil='hex')

    fit.sloc['Ic'] = 1
    fit.sloc['plasma', 'Ic'] = -700
    fit.plot()

    fit.plasmagrid.plot()



'''
pf_active = PFactive(135011, 7)
eq = Equilibrium(135011, 7)

itime = 500
# eq.build()
# eq.plot_0d('ip')
levels = eq.plot_2d(itime, 'psi', colors='C3', levels=21)
# eq.plot_2d(500, 'j_tor')
# eq.plot_1d(500)

coilset.sloc['coil', 'Ic'] = pf_active.data.current[itime]
coilset.sloc['plasma', 'Ic'] = eq.data.ip[itime]

vtk = PlasmaVTK(*coilset.frames, data=coilset.plasmagrid.data)
vtk.plot()
'''
