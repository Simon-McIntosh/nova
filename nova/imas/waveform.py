"""Generate coil voltage and current waveforms to suport pulse design."""
import bisect
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from tqdm import tqdm

from nova.biot.biot import Nbiot
from nova.imas.database import Ids
from nova.imas.machine import Machine
from nova.imas.pulsedesign import PulseDesign
from nova.linalg.regression import MoorePenrose


@dataclass
class MachineDescription(Machine):
    """Machine description default class."""

    pf_active: Ids | bool | str = 'iter_md'
    pf_passive: Ids | bool | str = False
    wall: Ids | bool | str = 'iter_md'
    tplasma: str = 'hex'
    dplasma: int | float = -2500


@dataclass
class Waveform(MachineDescription, PulseDesign):
    """Generate coilset voltage and current waveforms."""

    name: str = 'pulse_schedule'
    ngap: Nbiot = 1000
    ninductance: Nbiot = 0
    nlevelset: Nbiot = 2500
    nselect: Nbiot = None
    point_number: int = 5000

    def solve_biot(self):
        """Extend Machine.solve_biot."""
        super().solve_biot()
        self.inductance.solve()
        self.plasmagap.solve(self.data.gap_tail.data, self.data.gap_angle.data,
                             self.data.gap_id.data)

    def update(self):
        """Extend itime update."""
        super().update()
        self.sloc['plasma', 'Ic'] = self['i_plasma']
        self.update_loop_psi()

    def update_loop_psi(self):
        """Update loop psi."""
        Psi = self.inductance.Psi[self.plasma_index, :][np.newaxis, :]
        loop_psi = np.atleast_1d(self['loop_psi'])
        plasma_psi = Psi[:, self.plasma_index] * self.sloc['plasma', 'Ic']
        self.sloc['coil', 'Ic'] = np.linalg.lstsq(
            Psi[:, self.sloc['coil']], loop_psi - plasma_psi, rcond=None)[0]

    def gap_psi(self):
        """Return gap psi matrix and data."""
        #index = self.plasmagap.query(self.points)
        #Psi = self.plasmagap.Psi[index]
        Psi = self.plasmagap.matrix(self['gap'].data)
        psi = float(self['loop_psi'])*np.ones(len(Psi))
        plasma = Psi[:, self.plasma_index] * self.saloc['plasma', 'Ic']
        return Psi[:, self.saloc['coil']], psi-plasma

    def loop_psi(self):
        """Return loop psi matrix and data."""
        Psi = self.inductance.Psi[self.plasma_index, :][np.newaxis, :]
        psi = float(self['loop_psi'].data)
        plasma = Psi[:, self.plasma_index] * self.saloc['plasma', 'Ic']
        return Psi[:, self.saloc['coil']], psi-plasma

    def lcfs_psi(self):
        """Return separatrix psi matrix and data."""
        index = self.levelset.query(self.points)
        Psi = self.levelset.Psi[index]
        psi = float(self['loop_psi'])*np.ones(len(Psi))
        plasma = Psi[:, self.plasma_index] * self.saloc['plasma', 'Ic']
        return Psi[:, self.saloc['coil']], psi-plasma

    def append(self, *args):
        """Append coupling matricies and data."""
        matrix = np.vstack([arg[0] for arg in args])
        data = np.hstack([arg[1] for arg in args])
        return matrix, data

    def update_gap(self):
        """Solve gap wall flux."""
        Psi, psi = self.append(self.gap_psi())
        matrix = MoorePenrose(Psi, gamma=1e-5)
        self.saloc['coil', 'Ic'] = matrix / psi
        self.plasma.separatrix = float(self['loop_psi'])

    def update_lcfs(self):
        """Fit Lasc Closed Separatrix."""
        psi_boundary = float(self['loop_psi'])
        Psi, psi = self.append(self.lcfs_psi())
        matrix = MoorePenrose(Psi, gamma=0)
        self.saloc['coil', 'Ic'] = matrix / psi
        self.plasma.separatrix = psi_boundary

    def plot(self, index='plasma', axes=None, **kwargs):
        """Plot machine and constraints."""
        super().plot(index=index, axes=axes, **kwargs)
        self.plot_gaps()
        self.plasma.plot()

    def residual(self, nturn):
        """Return psi grid residual."""
        nturn /= np.sum(nturn)
        self.plasma.nturn = nturn
        #self.update_gap()
        self.update_lcfs()
        #sol = optimize.root(plasma_shape, self.saloc['coil', 'Ic'])
        #self.saloc['coil', 'Ic'] = sol.x
        #self.plasma.separatrix = self.plasma.psi_boundary
        residual = self.aloc['plasma', 'nturn'] - nturn
        return residual

    def solve(self):
        """Solve waveform."""
        self.fit()

        nturn = optimize.newton_krylov(
            self.residual, self.aloc['plasma', 'nturn'],
            x_tol=1e-3, f_tol=1e-3, maxiter=10)
        self.residual(nturn)
        #self.aloc['plasma', 'nturn'] = nturn
        #self.update_gap()

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
        self.plot()
        return self.mpy.mplfig_to_npimage(self.fig)

    def annimate(self, duration: float, filename='newton_krylov'):
        """Generate annimiation."""
        self.duration = duration
        self.max_time = 12
        self.set_axes('2d')
        animation = self.mpy.editor.VideoClip(
            self._make_frame, duration=duration)
        animation.write_gif(f'{filename}.gif', fps=10)


if __name__ == '__main__':

    pulse, run = 135003, 5

    waveform = Waveform(pulse, run)

    #waveform.annimate(5, 'newton_krylov_ramp_up')

    waveform.time = 12
    waveform.fit()
    waveform.solve()
    waveform.plot()



    # waveform.levelset.tree.plot(waveform.points)
    # waveform.axes.plot(*waveform.points.T, 'C3')

    '''
    from nova.biot.separatrix import Separatrix
    waveform.levelset.set_axes('2d')
    separatrix = Separatrix().single_null(6.5, 0.5, 1.5, 1.8, -0.3)
    waveform.levelset.plot_query(separatrix.points)
    '''


    '''
    separatrix = Separatrix(waveform['geometric_axis'][0], 0.5).single_null(
        waveform['minor_radius'],  waveform['elongation'],
        waveform['triangularity'], x_point=waveform['x_point'][0])
    separatrix.plot()
    separatrix.axes.plot(*waveform['x_point'][0], 'C0o')
    '''

    '''
    currents = np.zeros((150, waveform.saloc['coil'].sum()))
    times = np.linspace(10, 685, len(currents))

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
