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
    tplasma: str = 'hex'
    nplasma: int = 5000
    ngap: int | float = 150

    def solve_biot(self):
        """Extend Machine.solve_biot."""
        super().solve_biot()
        self.inductance.solve()
        self.wallgap.solve(np.c_[self.data.gap_r.data, self.data.gap_z.data],
                           self.data.gap_angle.data, self.data.gap_id.data)

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

    def update_gap(self):
        Psi = self.wallgap.matrix(self['gap'].data)
        plasma_psi = Psi[:, self.plasma_index] * self.sloc['plasma', 'Ic']
        self.sloc['coil', 'Ic'] = np.linalg.lstsq(
            Psi[:, self.sloc['coil']], -40.*np.ones(24) - plasma_psi)[0]
        self.plasma.separatrix = -40.

    def plot(self):
        """Plot machine and constraints."""
        super().plot()

    def solve(self):
        """solve waveform."""


if __name__ == '__main__':

    pulse, run = 135003, 5

    waveform = Waveform(pulse, run)

    def fun(nturn):
        """Return psi grid residual."""
        nturn /= np.sum(nturn)
        waveform.plasma.nturn = nturn
        waveform.update_gap()
        residual = waveform.aloc['plasma', 'nturn'] - nturn
        return residual

    from scipy import optimize

    waveform.time = 500
    nturn = waveform.aloc['plasma', 'nturn']
    optimize.newton_krylov(fun, nturn, x_tol=5e-2, f_tol=1e-3)

    waveform.plasma.plot()
    waveform.plot_gaps()

    #waveform.plasmaflux.contour.plot_contour(-40, color='C3')






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
