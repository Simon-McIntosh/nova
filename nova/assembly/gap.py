"""Acess gap datasets."""
from dataclasses import dataclass

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas
import xarray

from nova.assembly.model import ModelData
from nova.utilities.pyplot import plt


@dataclass
class GapData(ModelData):
    """Manage gap input."""

    name: str = 'constant_adaptive_fourier'

    def read_gapfile(self):
        """Return gapfile as pandas.DataFrame."""
        gapfile = self.file(self.name, extension='.txt')
        if self.name in ['Gap_Size_18_Coils', 'F4E_vgap']:
            gapdata = pandas.read_csv(gapfile, skiprows=1,
                                      delim_whitespace=True)
            gapdata = gapdata.iloc[1:]
            columns = {column: column.replace('_', '').lower()
                       for column in gapdata}
            gapdata.rename(columns=columns, inplace=True)
            return gapdata.drop(columns=['cid', 'rid'])
        gapdata = pandas.read_csv(gapfile, skiprows=1, delim_whitespace=True)
        return gapdata.drop(columns=['rid'])

    def build(self):
        """Load input gap waveforms."""
        gapdata = self.read_gapfile()
        self.data = xarray.Dataset()
        self.data['index'] = np.arange(1, len(gapdata)+1)
        self.data['signal'] = ['gap', 'tangential']
        self.data['simulation'] = gapdata.columns
        self.data['gap'] = ('simulation', 'index'), gapdata.values.T
        gapsum = self.data.gap.sum('index')
        self.data['delta'] = ('simulation', 'index', 'signal'), \
            np.zeros(tuple(self.data.dims[dim]
                     for dim in ['simulation', 'index', 'signal']))
        self.data['delta'][..., 0] = self.data.gap - \
            gapsum / self.data.dims['index']
        self.data['delta'][..., 1] = \
            self.data.gap.cumsum('index').data - \
            gapsum * (self.data['index'] + 1) / self.data.dims['index']
        self.data['delta'][..., 1] -= self.data['delta'][..., 1].mean('index')
        self.fft(self.data)
        return self.store()

    def plot(self, simulation: str):
        """Plot gap waveforms."""
        plt.figure()
        plt.bar(self.data.index,
                self.data.delta.sel(simulation=simulation,
                                    signal='tangential'), color='C0',
                label='tangential')
        plt.bar(self.data.index,
                self.data.delta.sel(simulation=simulation,
                                    signal='gap'), width=0.5, color='C1',
                label='gap')
        self.plot_waveform('tangential', simulation, 'C0')
        self.plot_waveform('gap', simulation, 'C1')
        axis = plt.gca()
        axis.xaxis.set_major_locator(MultipleLocator(2))
        axis.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axis.xaxis.set_minor_locator(MultipleLocator(1))
        plt.despine()
        plt.legend()
        plt.xlabel('index')
        plt.ylabel('length, mm')
        plt.title(f'simulation: {simulation}')

    def plot_waveform(self, signal: str, simulation: str, color: str):
        """Plot fourier waveform."""
        phi = np.linspace(0, 2*np.pi - np.pi/self.data.nyquist,
                          20*self.data.nyquist)
        waveform = 0
        amplitude = self.data.fft.sel(
            simulation=simulation, signal=signal, coefficient='amplitude').data
        phase = self.data.fft.sel(
            simulation=simulation, signal=signal, coefficient='phase').data
        for i in np.arange(1, self.data.nyquist+1):
            waveform += amplitude[i]*np.cos(i*phi + phase[i])
        plt.plot(phi * self.data.dims['index'] / (2*np.pi) + 1, waveform,
                 color=color)
        plt.despine()


@dataclass
class Gap:
    """Manage access to gap data files."""

    simulation: str

    def __post_init__(self):
        """Load gap data."""
        self.data = GapData(self.gapfile).data.sel(simulation=self.simulation)

    @property
    def gapfile(self):
        """Return gap listing filename."""
        if self.simulation[0] == 'v':
            return 'Gap_Size_18_Coils'
        return 'constant_adaptive_fourier'

    @property
    def gap(self):
        """Return gap waveform."""
        return self.data.gap.values


if __name__ == '__main__':

    gap = GapData('Gap_Size_18_Coils')
    gap.plot('v3')
