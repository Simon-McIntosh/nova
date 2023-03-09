"""Generate coil voltage and current waveforms to suport pulse design."""
import bisect
from dataclasses import dataclass

import nlopt
import numpy as np
from scipy import optimize
from tqdm import tqdm

from nova.imas.database import Ids
from nova.imas.machine import Machine
from nova.imas.pulse_schedule import PulseSchedule
from nova.linalg.regression import MoorePenrose


@dataclass
class MachineDescription(Machine):
    """Machine description default class."""

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


@dataclass
class Waveform(MachineDescription, PulseSchedule):
    """Generate coilset voltage and current waveforms."""

    name: str = 'pulse_schedule'

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

    def gap_psi(self):
        """Return gap psi matrix and data."""
        Psi = self.wallgap.matrix(self['gap'].data)
        #psi = self.plasma.psi_boundary*np.ones(len(Psi))
        psi = float(self['loop_psi'].data)*np.ones(len(Psi))
        plasma = Psi[:, self.plasma_index] * self.saloc['plasma', 'Ic']
        return Psi[:, self.saloc['coil']], psi-plasma

    def loop_psi(self):
        """Return loop psi matrix and data."""
        Psi = self.inductance.Psi[self.plasma_index, :][np.newaxis, :]
        psi = float(self['loop_psi'].data)
        plasma = Psi[:, self.plasma_index] * self.saloc['plasma', 'Ic']
        return Psi[:, self.saloc['coil']], psi-plasma

    def append(self, *args):
        """Append coupling matricies and data."""
        matrix = np.vstack([arg[0] for arg in args])
        data = np.hstack([arg[1] for arg in args])
        return matrix, data

    def update_gap(self):
        """Solve gap wall flux."""
        psi_boundary = float(self['loop_psi'].data)
        Psi, psi = self.append(self.gap_psi())
        matrix = MoorePenrose(Psi, gamma=2.5e-4)
        self.sloc['coil', 'Ic'] = matrix / psi
        self.plasma.separatrix = psi_boundary

    def plot(self):
        """Plot machine and constraints."""
        super().plot()


    def residual(self, nturn):
        """Return psi grid residual."""
        nturn /= np.sum(nturn)
        self.plasma.nturn = nturn
        self.update_gap()
        #sol = optimize.root(plasma_shape, self.saloc['coil', 'Ic'])
        #self.saloc['coil', 'Ic'] = sol.x
        #self.plasma.separatrix = self.plasma.psi_boundary
        residual = self.aloc['plasma', 'nturn'] - nturn
        return residual

    def solve(self):
        """Solve waveform."""
        optimize.newton_krylov(self.residual, self.aloc['plasma', 'nturn'],
                               x_tol=5e-2, f_tol=1e-3)

    def _make_frame(self, time):
        """Make frame for annimation."""
        self.axes.clear()
        max_time = np.min([self.data.time[-1], self.max_time])
        try:
            self.itime = bisect.bisect_left(
                self.data.time, max_time * time / self.duration) + 1902
        except ValueError:
            pass
        try:
            self.solve()
        except ValueError:
            pass
        self.plasma.plot()
        self.plot_gaps()
        return self.mpy.mplfig_to_npimage(self.fig)

    def annimate(self, duration: float, filename='newton_krylov'):
        """Generate annimiation."""
        self.duration = duration
        self.max_time = 80 #685
        self.set_axes('2d')
        animation = self.mpy.editor.VideoClip(
            self._make_frame, duration=duration)
        animation.write_gif(f'{filename}.gif', fps=10)


if __name__ == '__main__':

    pulse, run = 135003, 5

    waveform = Waveform(pulse, run)

    #waveform.annimate(2.5, 'newton_krylov_ramp_up')

    #waveform.time = 500

    '''
    def plasma_shape(coil_current):
        waveform.saloc['coil', 'Ic'] = coil_current
        #waveform.plasma.separatrix = waveform.plasma.psi_boundary
        waveform.plasma.update_lcfs()
        shape = waveform.plasma.lcfs(['major_radius', 'minor_radius',
                                      'elongation', 'triangularity'])
        result = np.zeros(12)
        result[:4] = shape - [5.0, 2, 1.76623817, 1.5]
        return result

    #waveform.plasma.plot()
    #waveform.plasma.lcfs.plot()
    '''

    '''
    currents = np.zeros((250, waveform.saloc['coil'].sum()))
    times = np.linspace(4, 685, len(currents))

    for i, time in enumerate(tqdm(times, 'calculating current waveform')):
        waveform.time = time
        try:
            waveform.solve()
        except:
            pass
        currents[i] = waveform.saloc['coil', 'Ic']

    waveform.set_axes('1d')
    waveform.axes.plot(times, 1e-3*currents,
                       label=waveform.sloc['coil', :].index)
    waveform.axes.legend(ncol=4)
    waveform.axes.set_xlabel('time, s')
    waveform.axes.set_ylabel(r'$I_c$, kA')
    waveform.savefig('waveform')
    '''

    '''
    waveform.time = 500
    nturn = waveform.aloc['plasma', 'nturn']
    nturn = optimize.newton_krylov(fun, nturn, x_tol=5e-2, f_tol=1e-3)

    waveform.plasma.plot()
    waveform.plot_gaps()
    #waveform.plasma.lcfs.plot()
    waveform.plasma.lcfs(['major_radius', 'minor_radius', 'elongation'])
    '''
