"""Extrapolate equilibria beyond separatrix."""
from dataclasses import dataclass, field

from nova.imas.machine import Machine


@dataclass
class Extrapolate(Machine):
    """Extrapolate equlibrium beyond separatrix ids."""

    filename: str = 'iter'
    dplasma: float = -200
    geometry: list[str] = field(default_factory=lambda: ['pf_active', 'wall'])

    @property
    def coilset_attrs(self):
        """Return coilset attributes."""
        return super().coilset_attrs #| dict()

    def build(self, **kwargs):
        """Build frameset and grid."""
        super().build(**kwargs)
        self.grid.solve(300, 0.1, index='plasma')
        return self.store(self.filename)


if __name__ == '__main__':

    coilset = Extrapolate(dcoil=-1, dplasma=-150)
    coilset.sloc['plasma', 'Ic'] = -15e6
    #coilset.grid.solve(30, index='plasma')

    coilset.plot()
    coilset.grid.plot()
