
from dataclasses import dataclass

from nova.imas.machine import Machine
from nova.utilities.pyplot import plt

@dataclass
class DataIDS:

    shot: int
    run: int

    def load()


@dataclass
class Fit(Machine):

    shot: int = 135011
    run: int = 7
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    def update(self):
        """Update ids datastreams."""
        super().update()  # update coilset

    def plot(self):
        """Plot fit."""
        plt.set_aspect(0.8)
        super().plot('plasma')
        self.plasma.plot()
        self.plasmagrid.plot(levels=21, colors='C0')


if __name__ == '__main__':

    fit = Fit(135011, 7, dcoil=0.5, dshell=0.5, dplasma=-1000, tcoil='hex')

    fit.itime = 500
    #fit.build()
    fit.plot()

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
