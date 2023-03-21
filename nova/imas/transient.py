"""Manage transient electromagnetic solutions."""
from dataclasses import dataclass
from functools import cached_property

from nova.frame.framedata import FrameData
from nova.frame.framesetloc import ArrayLocIndexer, LocIndexer
from nova.frame.framespace import FrameSpace
from nova.imas.database import Ids
from nova.imas.machine import Machine


@dataclass
class SupplyLoc(FrameData):
    """
    Supply Loc indexer.

        - vLoc: Access supply attributes.

    """

    #@cached_property
    #def aloc_hash(self):
    #    """Return interger hash computed on aloc array attribute."""
    #    return HashLoc('array_hash', self.aloc, self.saloc)

    @cached_property
    def vloc(self):
        """Return fast frame array attributes."""
        return LocIndexer('loc', self.source)

    @cached_property
    def valoc(self):
        """Return fast frame array attributes."""
        return ArrayLocIndexer('array', self.source)


@dataclass
class Transient(SupplyLoc, Machine):
    """Implementation of transient machine class."""

    pf_active: Ids | bool | str = 'iter_md'
    pf_passive: Ids | bool | str = 'iter_md'
    wall: Ids | bool | str = 'iter_md'
    tplasma: str = 'hex'

    def __post_init__(self):
        """Create voltage source frame."""
        super().__post_init__()
        self.supply = FrameSpace(
            base=[], required=['R'], additional=['V', 'Is', 'R'],
            available=[], subspace=['V'],
            array=['V', 'Is', 'R'], delim='_', version=[])

    def sole_biot(self):
        """Extend solve biot to include mutual-inductance."""
        super().solve_biot()
        self.inductance.solve()


if __name__ == '__main__':

    pulse, run = 105028, 1

    transient = Transient(pulse, run)

    transient.plot()

    '''
    M = machine.inductance.Psi[machine.Loc['coil']][:, machine.Loc['coil']]
    coil_inductance = machine.circuit.coil_matrix() @ M

    coil_inductance[-machine.circuit.link_number:] = \
        machine.circuit.link_matrix()
    _inductance = np.linalg.inv(coil_inductance)

    supply_matrix = machine.circuit.supply_matrix()
    voltage = np.zeros(len(supply_matrix))
    voltage[2] = 5

    def fun(time, current):
        voltage = np.zeros(len(supply_matrix))
        if time > 1 and time < 2:
            voltage[2] = time - 1.5
        return _inductance @ (supply_matrix @ voltage)

    import scipy

    sol = scipy.integrate.solve_ivp(fun, (0, 5),
                                    np.zeros(len(coil_inductance)),
                                    dense_output=True)

    import matplotlib.pyplot as plt

    plt.plot(time := np.linspace(sol.t[0], sol.t[-1], 1000),
             sol.sol(time).T)
    '''
