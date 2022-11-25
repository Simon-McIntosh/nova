"""Analysis of TF ground loops."""
from dataclasses import dataclass

from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import numpy.typing as npt
import scipy
import xarray

from nova.frame.biotfilament import Biot
from nova.frame.biotframe import BiotFrame
from nova.frame.vde import VDE
from nova.utilities.time import clock
from nova.plot import plt
from nova.plot.addtext import linelabel


@dataclass
class LoopTransient:

    time: npt.ArrayLike
    emf: npt.ArrayLike
    resistance: float
    inductance: float



@dataclass
class GroundLoop(VDE):
    """Calculate TF ground loop response."""

    inner: tuple[float, float] = None
    outer: tuple[float, float] = None
    nturn: float = None  # ground loop turn number (segment fraction)
    resistance: float = 0.1

    def __post_init__(self):
        """Construct ground loop targets."""
        super().__post_init__()

    def update(self, index, solve_grid=True):
        """Extend VDE update."""
        super().update(index, solve_grid)
        self.loop.update_turns()

    def plot(self, axes=None):
        """Extend VDE plot, include loop markers."""
        super().plot(axes=axes)
        self.loop.plot(marker='X', color='C3')

    def build_loop(self):
        """Build flux loop."""
        target = BiotFrame(additional=['link'],
                           default=dict(nturn=self.nturn, dl=0.1, dt=0.1,
                                        segment='circle'))
        target.insert(*self.outer, name='outer')
        target.insert(*self.inner, name='inner')
        target.multipoint.link(target.index, -1)
        self.loop.solve(target)
        self.loop.data.attrs['Lo'] = Biot(
            target, target, reduce=[True, True], turns=[True, True],
            columns=['Psi']).data.Psi.values[0]
        self.loop.data.attrs['R'] = self.resistance

    def extract_waveform(self, index=slice(None), cutoff=25):
        """Extract groundloop waveform."""
        self.build_loop()
        self.loop.data.attrs['cutoff'] = cutoff
        self.loop.data['_time'] = self.data.time[index].values
        self.loop.data['_flux'] = xarray.DataArray(
                0., dims=['_time'], coords=[self.loop.data._time])
        tick = clock(self.loop.data.dims['_time'],
                     header='Extracting loop waveform')
        for i in range(self.loop.data.dims['_time']):
            self.update(i, solve_grid=False)
            self.loop.data['_flux'][i] = \
                (self.loop.data.Psi.values @ self.sloc['Ic'])[0]
            tick.tock()
        self.filter_waveform(cutoff)
        self.store(self.file)  # save data to file

    def filter_waveform(self, cutoff):
        """Apply SavGol filter to flux waveform."""
        step_length = np.min(np.diff(self.loop.data._time))
        time_delta = (self.loop.data._time[-1] - self.loop.data._time[0])
        step_number = int(time_delta / step_length)
        step_length / (step_number - 1)
        time = np.linspace(self.loop.data._time[0],
                           self.loop.data._time[-1], step_number)
        flux = scipy.interpolate.interp1d(
            self.loop.data._time, self.loop.data._flux)(time)
        window = int(1 / (step_length * cutoff))
        if window % 2 == 0:
            window += 1
        self.loop.data['time'] = time
        self.loop.data['flux'] = xarray.DataArray(
                0., dims=['time'], coords=[self.loop.data.time])
        self.loop.data['emf'] = xarray.DataArray(
                0., dims=['time'], coords=[self.loop.data.time])
        self.loop.data['flux'] = ('time',
                                  scipy.signal.savgol_filter(flux, window, 1))
        #self.loop.data['flux'] = ('time', flux)
        self.loop.data['emf'] = ('time',
                                 -np.gradient(self.loop.data['flux'], time))
        self.loop.data['power'] = \
            self.loop.data['emf']**2 / self.loop.data.attrs['R']

    def plot_waveform(self, axes=None):
        """Plot ground loop waveforms."""
        if axes is None:
            axes = plt.subplots(3, 1, sharex=True)[1]
        energy = np.trapz(self.loop.data['power'],
                          self.loop.data['time'])
        label = self.folder.replace('_', ' ') + f' {energy:1.1f}J'
        axes[0].plot(self.loop.data['time'], self.loop.data['flux'],
                     label=label)
        axes[0].set_ylabel(r'$\phi$ Tm$^2$')
        axes[1].plot(self.loop.data['time'], self.loop.data['emf'])
        axes[1].set_ylabel(r'$\dot{\phi}$ V')
        axes[2].plot(self.loop.data['time'], self.loop.data['power'])
        axes[2].set_ylabel(r'$\dot{Q}$ W')
        axes[-1].set_xlabel(r'$t$ s')
        plt.despine()
        axes[0].legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.05))

    def make_frame(self, position, axes=None):
        """Make frame for make_movie."""
        super().make_frame(position, axes)
        self.loop.plot(axes=self.axes, marker='X', color='C3')
        return mplfig_to_npimage(plt.gcf())


if __name__ == '__main__':

    folder = 'VDE_DW_slow'

    ISS = np.linalg.norm([3.2581, 0.57449]), -5.8899  # inner ISS pin
    cryo = np.linalg.norm([9.5302, 2.7783]), -7.3301  # cryo ring
    nturn = np.arcsin(np.linalg.norm([9.5302-9.9062, 2.7783-0.6483]) /
                      (2 * np.linalg.norm([9.905, 0.645]))) / np.pi

    gl = GroundLoop(folder=folder, inner=ISS, outer=cryo, nturn=nturn,
                    read_txt=False)

    #gl.make_movie('ground_loop')

    #gl.extract_waveform()
    #gl.update(10)
    #gl.plot()
    #gl.plot_waveform()

    plt.set_aspect(0.6)
    axes = plt.subplots(3, 1, sharex=True)[1]
    for folder in ['VDE_DW_slow', 'VDE_DW_slow_fast', 'MD_DW_exp16',
                   'VDE_UP_slow', 'VDE_UP_slow_fast', 'MD_UP_exp16']:
        gl = GroundLoop(folder=folder, inner=ISS, outer=cryo, nturn=nturn)
        gl.extract_waveform()
        gl.plot_waveform(axes=axes)








'''
vde = VDE(folder=folder, read_txt=False)

vde.update(-3)
vde.loc['plasma', 'nturn'] = 0

vde.plot()




'''
