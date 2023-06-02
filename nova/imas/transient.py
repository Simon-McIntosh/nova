"""Manage transient electromagnetic solutions."""
from dataclasses import dataclass

from nova.imas.database import Ids
from nova.imas.machine import Machine


@dataclass
class Transient(Machine):
    """Implementation of transient machine class."""

    pf_active: Ids | bool | str = "iter_md"
    pf_passive: Ids | bool | str = "iter_md"
    wall: Ids | bool | str = "iter_md"
    tplasma: str = "hex"

    def solve_biot(self):
        """Extend solve biot to include mutual-inductance."""
        super().solve_biot()
        self.inductance.solve()


if __name__ == "__main__":
    pulse, run = 105028, 1

    transient = Transient(pulse, run)

    transient.plot()

    """
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
    """
