"""Acess gap datasets."""
from dataclasses import dataclass, field
from typing import ClassVar

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import numpy.typing as npt
import pandas
import xarray

from nova.assembly.model import ModelData
from nova.plot import plt


@dataclass
class GapData:
    """Manage gap attributes."""

    simulations: list[str]
    gap: npt.ArrayLike = None
    roll: npt.ArrayLike = None
    yaw: npt.ArrayLike = None
    data: xarray.Dataset = field(init=False, repr=False)

    ncoil: ClassVar[int] = 18

    def __post_init__(self):
        """Build input gap waveforms."""
        self.data = xarray.Dataset()
        self.data['simulation'] = self.simulations
        self.data['index'] = np.arange(1, self.ncoil+1)
        self.data['signal'] = ['gap', 'roll', 'yaw']
        for signal in self.data.signal.values:
            value = getattr(self, signal)
            if value is None:
                value = np.zeros((len(self.simulations), self.ncoil))
            self.data[signal] = ('simulation', 'index'), value
        self.data['delta'] = ('simulation', 'index', 'signal'), \
            np.zeros(tuple(self.data.dims[dim]
                     for dim in ['simulation', 'index', 'signal']))
        for i, signal in enumerate(self.data.signal.values):
            self.data.delta[..., i] = self.data[signal]
        self.data.delta[:] -= self.data.delta.mean('index')
        ModelData.fft(self.data)

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
class UniformGap(ModelData):
    """Manage uniform (parallel) gap input data."""

    name: str

    @property
    def gapfile(self):
        """Return gap listing filename."""
        if self.name[0] == 'v':
            return 'Gap_Size_18_Coils'
        return 'constant_adaptive_fourier'

    def read_gapfile(self):
        """Return gapfile as pandas.DataFrame."""
        gapfile = self.file(self.gapfile, extension='.txt')
        if self.gapfile in ['Gap_Size_18_Coils', 'F4E_vgap']:
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
        """Build uniform gap dataset."""
        gapdata = self.read_gapfile()
        self.data = GapData(gapdata.columns, gapdata.values.T).data
        return self.store()


@dataclass
class WedgeGap(ModelData):
    """Manage access to wedge gap data files."""

    simulations: ClassVar[list[str]] = ['w1', 'w2', 'w3', 'w4', 'w5']
    length: ClassVar[dict[str, float]] = dict(pitch=8, roll=9.5, yaw=8)

    def build(self):
        """Build gaps and calculate coil transformations."""
        self.build_dataset()
        self.build_transforms()
        print(self.length)
        self.data = self.data.merge(GapData(
            self.simulations, self.data.gap.data,
            self.data.rotate[..., 1].data * 1e3*self.length['roll'],
            self.data.rotate[..., 2].data * 1e3*self.length['yaw']).data)
        return self.store()

    def build_dataset(self):
        """Build gap dataset."""
        self.data = xarray.Dataset()
        self.data['simulation'] = self.simulations
        self.data['index'] = range(1, self.ncoil+1)
        self.data['point'] = ['A1', 'C1', 'B2']
        self.data['point_gap'] = xarray.DataArray(0., self.data.coords)
        for i, simulation in enumerate(self.simulations):
            filename = self.file(simulation, extension='.json')
            self.data.point_gap[i] = pandas.read_json(filename).values

    def build_transforms(self):
        """Calculate coil transforms to produce wedge gaps."""
        nominal_gap = self.data.point_gap.sum(axis=1) / self.ncoil
        self.data['gap'] = ('simulation', 'index'), \
            (self.data.point_gap[..., :2].sum(axis=-1) +
             2*self.data.point_gap[..., 2]).data / 4
        delta = self.data.point_gap - nominal_gap.data[:, np.newaxis, :]
        self.data['point_offset'] = xarray.zeros_like(self.data.point_gap)
        self.data.point_offset[:, 1:] = np.cumsum(delta[:, :-1], axis=1).data
        self.data['point_offset'] += self.data.point_offset[:, -1] + \
            delta[:, -1]
        self.data['mean_offset'] = ('simulation', 'index'), \
            (self.data.point_offset[..., :2].sum(axis=-1) +
             2*self.data.point_offset[..., 2]).data / 4
        self.data['coordinate'] = ['x', 'y', 'z']
        self.data['reference'] = ('point', 'coordinate'), \
            1e3*np.array([[2.31300792, 0, 4.74626384],
                          [2.31044582, 0, -4.67900054],
                          [3.20624393, 0, 0]])
        self.data['fiducial'] = ('simulation', 'index',
                                 'point', 'coordinate'), \
            np.einsum('kl,ij->ijkl', self.data.reference,
                      np.ones((self.data.dims['simulation'], self.ncoil)))
        self.data.fiducial[..., 1] += self.data.point_offset
        self.data.fiducial[..., 1] -= self.data.mean_offset
        self.data['normal'] = ('simulation', 'index', 'coordinate'), \
            np.cross(
                self.data.fiducial[..., 2, :]-self.data.fiducial[..., 0, :],
                self.data.fiducial[..., 1, :]-self.data.fiducial[..., 0, :],
                axis=-1)
        self.data['normal'] /= np.linalg.norm(self.data['normal'],
                                              axis=-1)[..., np.newaxis]
        self.data.attrs['radius'] = self.data.reference[1:, 0].data.mean()
        self.data['angle'] = ['phi', 'roll', 'yaw']
        self.data['rotate'] = ('simulation', 'index', 'angle'), \
            np.zeros((self.data.dims['simulation'], self.ncoil,
                      self.data.dims['angle']))
        self.data['rotate'][..., 0] = self.data.mean_offset / self.data.radius
        self.data['rotate'][..., 1] = np.arctan2(
            np.cross(np.array([1, 0]), self.data['normal'][..., 1:]),
            np.dot(self.data['normal'][..., 1:], np.array([1, 0])))
        self.data['rotate'][..., 2] = np.arctan2(
            np.cross(np.array([0, 1]), self.data['normal'][..., :2]),
            np.dot(self.data['normal'][..., :2], np.array([0, 1])))
        self.data.rotate[:] -= self.data.rotate.mean(axis=1)

    def write_gapfile(self):
        """Write mean gap data to file."""
        dataframe = self.data.mean_gap.T.to_pandas()
        dataframe.insert(0, 'rid', range(2001, 2000+self.ncoil+1))
        filename = self.file('wedge', extension='.txt')
        with open(filename, 'w') as f:
            f.write('Non-parallel gap data.\n\n')
            dataframe.to_csv(f, sep='\t')

    def plot(self, simulation: str):
        """Plot wedge gap distributions."""
        rotate = self.data.rotate.sel(simulation=simulation)
        for i in range(3):
            plt.bar(range(18), rotate[:, i]*180/np.pi, width=0.8-(i*0.2))


@dataclass
class Gap:
    """Manage access to gap data files."""

    simulation: str

    def __post_init__(self):
        """Load gap data."""
        self.data = self.load_data().sel(simulation=self.simulation)

    def load_data(self):
        """Return gap data."""
        if self.simulation[0] == 'w':  # wedge gap
            return WedgeGap().data
        return UniformGap(self.simulation).data

    @property
    def gap(self):
        """Return gap waveform."""
        return self.data.gap.values


if __name__ == '__main__':

    gap = Gap('w4')
