"""Perform equilibrium reconstrucitons from IMAS data."""
from dataclasses import dataclass, field

from functools import cached_property
import scipy.optimize
import xarray

from nova.biot.biotgrid import BiotPlot
from nova.frame.framesetloc import LocIndexer
from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import Machine
from nova.imas.pf_active import PF_Active
from nova.plot import plt


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
        self.axes.set_aspect(0.8)
        self.plasma.plot()


if __name__ == '__main__':

    from scipy.optimize._nonlin import BroydenFirst, KrylovJacobian

    fit = Fit(135011, 7)
    #fit.build(dcoil=0.5, dshell=0.5, nplasma=500, tcoil='hex')

    itime = 700
    fit.sloc['coil', 'Ic'] = fit.data.current[itime]
    fit.sloc['plasma', 'Ic'] = fit.data.ip[itime]

    # fit.plot()

    fit.plasma.separatrix = dict(e=(6, 0, 1.5, 1.5))
    fit.plasma.grid.update_turns('Psi')
    plasma_index = fit.plasma.grid.data.attrs['plasma_index']
    Psi_o = fit.plasma.grid.operator['Psi'].matrix[:, plasma_index].copy()

    def solve():
        #fit.plasma.separatrix = dict(e=(6, 0, 1.5, 1.5))
        #x0 = fit.plasma.boundary.psi.min()
        #x1 = fit.plasma.boundary.psi.max()
        #return scipy.optimize.root_scalar(psi_root, method='toms748',
        #                                  bracket=(x0, x1))

        return scipy.optimize.root(
            fit.plasma.residual, Psi_o,
            method='krylov', tol=0.05,
            options=dict(disp=False, jac_options=dict(method='gmres')))



    import time
    start = time.time()
    #scipy.optimize.newton_krylov(psi_root, fit.plasma.boundary.psi.min(),
    #                             iter=100, verbose=False)
    sol = solve()

    #print(sol)
    #psi_root(sol.root)

    #print('sol', sol.root, fit.plasma.boundary.psi.min())

    end = time.time()
    print(end-start)
    print(sol.success, sol.message)

    fit.plot()

    #import scipy.optimize


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
