"""Generate coil voltage and current waveforms to suport pulse design."""
from dataclasses import dataclass

import numpy as np

from nova.imas.database import Ids
from nova.imas.machine import Machine
from nova.imas.pulse_schedule import PulseSchedule


@dataclass
class Waveform(Machine, PulseSchedule):
    """Generated coilset voltage and current waveforms."""

    name: str = 'pulse_schedule'
    pf_active: Ids | bool | str = 'iter_md'
    pf_passive: Ids | bool | str = 'iter_md'
    wall: Ids | bool | str = 'iter_md'

    def solve_biot(self):
        """Extend Machine.solve_biot."""
        super().solve_biot()
        self.inductance.solve()

    def update(self):
        """Extend itime update."""
        self.sloc['plasma', 'Ic'] = self['i_plasma']
        self.update_loop_psi()

    def update_loop_psi(self):
        """Update loop psi."""
        Psi = self.inductance.Psi[self.plasma_index, :][np.newaxis, :]
        loop_psi = np.atleast_1d(self['loop_psi'])
        plasma_psi = Psi[:, self.plasma_index] * self.sloc['plasma', 'Ic']
        self.sloc['coil', 'Ic'] = np.linalg.lstsq(
            Psi[:, self.sloc['coil']], loop_psi - plasma_psi)[0]

    def plot(self):
        """Plot machine and constraints."""
        super().plot()

    def solve(self):
        """solve waveform."""


if __name__ == '__main__':

    pulse, run = 135003, 5

    waveform = Waveform(pulse, run)

    waveform.time = 250

    waveform.plasma.separatrix = dict(e=[6, 0.5, 3, 6])

    #waveform.plasma.plot()


    #waveform.plasma.separatrix = waveform.plasmawall.w_psi
    #waveform.plasma.plot()

    waveform.saloc['Ic'][:5] *= 0.01


    def fun(nturn):
        """Return psi grid residual."""

        nturn /= np.sum(nturn)

        waveform.aloc['nturn'][waveform.aloc['plasma']] = nturn
        waveform.update_aloc_hash('nturn')

        print('axis', waveform.plasma.psi_axis)
        waveform.plasma.separatrix = -62. #waveform.plasma.psi_axis - 1.

        residual = waveform.aloc['nturn'][waveform.aloc['plasma']] - nturn
        print(np.linalg.norm(residual), np.sum(nturn))

        return residual

    from scipy import optimize

    nturn = waveform.aloc['plasma'][waveform.aloc['plasma']]

    sol = optimize.newton_krylov(fun, nturn)
    #print(sol)

    waveform.plasma.plot()



    '''
    import matplotlib.pyplot as plt

    plt.figure()
    _index = slice(index-5, index+5)
    plt.plot(np.arange(len(waveform.plasmawall.psi)),
             waveform.plasmawall.psi, '-')
    plt.plot(np.arange(len(waveform.plasmawall.psi))[_index],
             waveform.plasmawall.psi[_index], 'o-')
    '''
    #

    #waveform.plot()
    #waveform.firstwall.plot()

    #waveform.loc
