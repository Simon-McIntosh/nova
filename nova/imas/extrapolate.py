"""Extrapolate equilibria beyond separatrix."""
from dataclasses import dataclass, field

from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import Machine


@dataclass
class Extrapolate(Machine):
    """Extrapolate equlibrium beyond separatrix ids."""

    filename: str = 'iter'
    dplasma: float = -200
    geometry: list[str] = field(default_factory=lambda: ['pf_active', 'wall'])

    #@property
    #def coilset_attrs(self):
    #    """Return coilset attributes."""
    #    return super().coilset_attrs #| dict()

    #def build(self, **kwargs):
    #    """Build frameset and grid."""
    #    super().build(**kwargs)
    #    self.grid.solve(300, 0.1, index='plasma')
    #    return self.store(self.filename)


if __name__ == '__main__':

    coilset = Extrapolate(dcoil=-1, dplasma=-1500)
    coilset.sloc['plasma', 'Ic'] = -15e6
    coilset.sloc['coil', 'Ic'] = -15e6
    coilset.sloc()
    coilset.grid.solve(1500, 0.5, index='plasma')

    eq = Equilibrium(114101, 41)

    itime = 0
    eq.plot_2d(itime, 'psi', colors='C3', levels=21)
    eq.plot_boundary(itime)

    print(coilset.plasma.version)
    coilset.plasma.separatrix = eq.data.boundary[0]

    coilset.grid.update_turns('Psi')
    print(coilset.plasma.version)

    coilset.plot()
    coilset.grid.plot()
