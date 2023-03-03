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

    def plot(self):
        """Plot machine and constraints."""
        super().plot()


if __name__ == '__main__':

    pulse, run = 135003, 5

    waveform = Waveform(pulse, run)

    waveform.time = 250

    waveform.plasma.separatrix = dict(e=[6, 0.5, 3, 6])
    _ = waveform.inductance.psi
    Psi = waveform.inductance.Psi[waveform.inductance.plasma_index,
                                  :][np.newaxis, :]
    loop_psi = np.atleast_1d(waveform['loop_psi'])

    #loop_psi -= waveform.inductance.psi
    waveform.sloc['coil', 'Ic'] = np.linalg.lstsq(
        Psi[:, waveform.sloc['coil']],
        loop_psi - Psi[:, waveform.inductance.plasma_index] * \
        waveform.sloc['plasma', 'Ic'])[0]

    # waveform.plot()
    waveform.plasma.plot()

    #

    #waveform.plot()
    #waveform.firstwall.plot()

    #waveform.loc
