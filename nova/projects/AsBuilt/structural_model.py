"""Perform post-processing analysis on Fourier perterbed TFC dataset."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas
import scipy
import xarray

from nova.database.filepath import FilePath
from nova.structural.TFC18 import TFC18
from nova.utilities.pyplot import plt


@dataclass
class DataAttrs:
    """Manage simulation and group dataset labels."""

    simulation: str
    filename: str = 'vault'
    datapath: str = 'data/Assembly'
    group: str = field(init=False, default=None)

    def __post_init__(self):
        """Set dataset group for netCDF file load/store."""
        self.group = f'{self.__class__.__name__.lower()}/{self.simulation}'
        self.set_path(self.datapath)


@dataclass
class Data(ABC, FilePath, DataAttrs):
    """Perform Fourier analysis on TFC deformations."""

    folder: str = 'TFCgapsG10'
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    def __post_init__(self):
        """Load / build dataset."""
        super().__post_init__()
        try:
            self.load()
        except (FileNotFoundError, OSError, KeyError):
            self.build()
            self.store()

    @abstractmethod
    def build(self):
        """Build dataset."""


@dataclass
class Points(Data):
    """Extract midplane mesh and initalize dataset."""

    origin: tuple[int] = (0, 0, 0)
    ncoil: ClassVar[int] = 18
    cluster: ClassVar[int] = 1

    def slice_mesh(self):
        """Return midplane mesh."""
        mesh = TFC18(self.folder, self.simulation, cluster=self.cluster).mesh
        return mesh.slice('z', self.origin)

    @property
    def attrs(self):
        """Return instance attributes."""
        return {attr: getattr(self, attr)
                for attr in ['filename', 'folder',
                             'origin', 'ncoil', 'cluster']}

    def initialize_dataset(self):
        """Initialize empty dataset from referance scenario."""
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data['scenario'] = self.mesh['scenario']
        self.data['index'] = range(1, self.ncoil+1)
        self.data['dimensions'] = ['x', 'y', 'z', 'radial', 'toroidal']
        self.data['points'] = xarray.DataArray(0., self.data.coords)

    def build(self):
        """Build single mesh dataset."""
        self.mesh = self.slice_mesh()
        self.initialize_dataset()
        for scenario in self.data.scenario.data:  # store ansys data [mm]
            self.data['points'].loc[scenario][:, :3] = 1e3*self.mesh[scenario]
        self.data.points[..., -2] = np.linalg.norm(self.data.points[..., :2],
                                                   axis=-1)
        self.data.points[..., -1] = np.arctan2(self.data.points[..., 1],
                                               self.data.points[..., 0])


@dataclass
class Fourier(Points):
    """Perform Fourier deomposition on single TFC simulation."""

    def initialize_dataset(self):
        """Extend points initialize dataset."""
        super().initialize_dataset()
        self.data['mode'] = range(self.ncoil//2 + 1)
        self.data['response'] = ['radial', 'tangent']

    def build(self):
        """Extend points build."""
        super().build()
        reference = Points('k0').data
        self.data['delta'] = self.data['points'] - reference['points']
        r_norm = reference['points'][..., :2].data.copy()
        r_norm /= np.linalg.norm(r_norm, axis=-1)[..., np.newaxis]
        t_norm = r_norm @ np.array([[0, -1], [1, 0]])
        self.data['delta'][..., -2] = np.einsum(
            'ijk,ijk->ij', self.data['delta'][..., :2].data, r_norm)
        self.data['delta'][..., -1] = np.einsum(
            'ijk,ijk->ij', self.data['delta'][..., :2].data, t_norm)
        self.fft()

    def fft(self):
        """Apply fft to dataset."""
        nyquist = self.ncoil // 2
        coefficient = scipy.fft.fft(self.data['delta'][..., 3:].data, axis=-2)
        coefficient = coefficient[:, :nyquist+1]
        self.data['amplitude'] = ('scenario', 'mode', 'response'), \
            np.abs(coefficient) / nyquist
        self.data['amplitude'][:, 0] /= 2
        if self.ncoil % 2 == 0:
            self.data['amplitude'][:, nyquist] /= 2
        self.data['phase'] = xarray.zeros_like(self.data['amplitude'])
        self.data['phase'][:] = np.angle(coefficient)

    def plot(self, scenario='TFonly', fft='radial', width=0.9):
        """Plot fourier components."""
        plt.figure()
        plt.bar(self.data.mode,
                self.data.phase.sel(scenario=scenario, response=fft),
                label='ANSYS', width=width)


@dataclass
class Gap(Data):
    """Manage gap input."""

    simulation: str = 'constant_adaptive_fourier'

    def build(self):
        """Load input gap waveforms."""
        gapfile = self.file(self.simulation, extension='.txt')
        gapdata = pandas.read_csv(gapfile, skiprows=1, delim_whitespace=True)
        gapdata.drop(columns=['rid'], inplace=True)
        self.data = xarray.Dataset()
        self.data['index'] = range(len(gapdata))
        self.data['signal'] = ['gap', 'tangent']
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
        self.fft()
        self.store()

    def fft(self):
        """Extract amplitude and phase information of fourier components."""
        self.data.attrs['nyquist'] = self.data.dims['index'] // 2
        self.data['mode'] = np.arange(self.data.nyquist+1)
        coefficient = scipy.fft.fft(self.data.delta.data, axis=-2)
        coefficient = coefficient[:, :self.data.nyquist+1]
        self.data['amplitude'] = ('simulation', 'mode', 'signal'), \
            np.abs(coefficient)
        self.data['amplitude'] /= self.data.nyquist
        self.data['amplitude'][:, 0] /= 2
        if self.data.dims['index'] % 2 == 0:
            self.data['amplitude'][:, self.data.nyquist] /= 2
        self.data['phase'] = ('simulation', 'mode', 'signal'), \
            np.angle(coefficient)

    def plot(self, simulation: str):
        """Plot gap waveforms."""
        plt.bar(self.data.index,
                self.data.delta.sel(simulation=simulation, coord='tangent'))
        plt.bar(self.data.index,
                self.data.delta.sel(simulation=simulation, coord='gap'),
                width=0.5)
        self.plot_waveform('tangent', simulation)
        self.plot_waveform('gap', simulation)

    def plot_waveform(self, coord: str, simulation: str):
        """Plot fourier waveform."""
        phi = np.linspace(0, 2*np.pi - np.pi/self.data.nyquist,
                          20*self.data.nyquist)
        waveform = 0
        amplitude = self.data['amplitude'].sel(
            simulation=simulation, coord=coord).data
        phase = self.data['phase'].sel(
            simulation=simulation, coord=coord).data
        for i in np.arange(1, self.data.nyquist+1):
            waveform += amplitude[i]*np.cos(i*phi + phase[i])
        plt.plot(phi * self.data.dims['index'] / (2*np.pi), waveform)
        plt.despine()


@dataclass
class Compose(Data):
    """Perform Fourier analysis on TFC deformations."""

    prefix: str = 'k'
    group: str = None
    origin: tuple[int] = (0, 0, 0)
    wavenumber: list[int] = field(default_factory=lambda: range(10))

    ncoil: ClassVar[int] = 18
    cluster: ClassVar[int] = 1

    def build(self):
        """Build ensemble of ansys fourier component simulations."""
        mode = []
        for wavenumber in self.wavenumber:
            fourier = Fourier(f'{self.prefix}{wavenumber}')
            mode.append(fourier.data.copy(deep=True))
        self.data = xarray.concat(mode, 'wavenumber',
                                  combine_attrs='drop_conflicts')
        self.data['wavenumber'] = self.data.mode.data
        self.store()

    def plot_wave(self, wavenumber: int, scenario='TFonly'):
        """Plot fft components."""
        index = dict(wavenumber=wavenumber, scenario=scenario)
        plt.figure()
        plt.bar(self.data.coil_index.data,
                np.cos(wavenumber*self.data.points.sel(**index,
                                                       coord='tangent')),
                width=0.7, label='toroidal placment error 1.00', color='C3')

        radial_amplitude = self.data.amplitude.sel(
            **index, mode=wavenumber, fft='radial').data
        radial_phase = self.data.phase.sel(
            **index, mode=wavenumber, fft='radial').data
        plt.bar(self.data.coil_index.data,
                self.data.delta.sel(**index, coord='radial'),
                width=0.5, color='C0',
                label=f'radial misalignment {radial_amplitude:1.2f}')
        phi = np.linspace(0, 2*np.pi, 150)
        plt.plot(phi * 9/np.pi + 1,
                 radial_amplitude*np.cos(wavenumber*phi + radial_phase),
                 color='gray')
        plt.despine()
        plt.xlabel('coil index')
        plt.ylabel('misalignment, mm')
        plt.xticks(range(1, self.ncoil+1))
        plt.legend(ncol=2, loc='upper center', bbox_to_anchor=[0.5, 1.14])
        plt.title(f'wavenumber={wavenumber}\n\n')

    def plot_amplitude(self):
        """Plot radial amplitude magnification."""
        amplitude = self.diag('amplitude')
        plt.figure()
        plt.bar(self.data.wavenumber, amplitude, label='ANSYS', width=0.75)
        plt.despine()
        plt.xticks(self.data.wavenumber.values)
        plt.xlabel('wavenumber')
        plt.ylabel('radial amplification factor '
                   r'$\frac{\Delta r}{r \Delta \phi}$')
        plt.legend(frameon=False)
        plt.yscale('log')

    def diag(self, attr: str, scenario='TFonly', fft='radial'):
        """Return attribute's diagonal components."""
        return np.diag(self.data[attr].sel(
                        scenario=scenario, fft=fft).data)

    def plot_argand(self, scenario='TFonly', fft='radial'):
        """Plot complex transform."""
        plt.figure()
        amplitude = self.diag('amplitude', scenario, fft)
        phase = self.diag('phase', scenario, fft).copy()
        coef = amplitude * np.exp(1j * phase)

        for i, mode in enumerate(coef):
            plt.plot(mode.real, mode.imag, 'o')
            plt.text(mode.real, mode.imag, i, va='bottom', ha='center')
        phi = np.linspace(0, 2*np.pi)
        plt.plot(np.cos(phi), np.sin(phi), '--', color='gray')
        plt.plot([-1, 1], [0, 0], '-.', color='gray')
        plt.plot([0, 0], [-1, 1], '-.', color='gray')
        plt.axis('equal')
        plt.axis('off')

    def plot_fit(self):
        """Fit model to observations."""
        H = np.array([0.396, 0.759, 1.6871, 0.934, 0.6451, 0.5373,
                      0.4835, 0.4085, 0.6682, 0.4012])
        weights = np.ones(10)
        weights[1] = 0  # exclude n=1 mode
        amplitude = self.diag('amplitude')
        matrix = np.array([np.ones(10), amplitude]).T

        coef = np.linalg.lstsq(matrix*weights[:, np.newaxis],
                               H*weights)[0]

        _amplitude = np.linspace(amplitude.min(), amplitude.max(), 51)
        _matrix = np.array([np.ones_like(_amplitude),
                            _amplitude]).T

        plt.figure()
        plt.plot(amplitude, H, 'o')
        plt.plot(_amplitude, _matrix @ coef, '-', color='gray')
        plt.despine()
        plt.xlabel(r'amplitude of radial mode, $\Delta r$ mm')
        plt.ylabel(r'peak to peak misalignment, $H$ mm')
        plt.title(rf'$H={{{coef[0]:1.2f}}}+{{{coef[1]:1.2f}}}\Delta r$')


@dataclass
class TransferFunction(Data):
    """Extract factor and phase transform from single point simulations."""

    simulation: str
    scenario: str = 'TFonly'
    eps: float = 1e-3

    def build(self):
        """Lanuch gap and fourier instances."""
        _input = Gap().data.sel(simulation=self.simulation)
        _output = Fourier(self.simulation).data.sel(
            scenario='TFonly', response='radial')
        self.data = xarray.Dataset()
        self.data['amplitude_factor'] = \
            _output.amplitude / _input.amplitude
        self.data['phase_shift'] = _output.phase - _input.phase
        for i, coord in enumerate(self.data.coord):
            max_factor = _input.amplitude[:, i].max()
            index = _input.amplitude[:, i] < self.eps * max_factor
            self.data['amplitude_factor'][index, i] = -1
            self.data['phase_shift'][index, i] = 0


@dataclass
class Model(Data):
    """Construct structural model."""

    simulation: str = 'fourier'

    ncoil: ClassVar[int] = 18

    def build(self):
        """Build fourier component model."""
        self.data = xarray.Dataset()
        self.data.attrs['nyquist'] = self.ncoil // 2
        self.data['mode'] = np.arange(self.data.nyquist+1)
        self.data['signal'] = ['gap', 'tangent']
        self.data['amplitude_factor'] = xarray.DataArray(0., self.data.coords)
        self.data['phase_shift'] = xarray.DataArray(0., self.data.coords)
        for mode in self.data.mode.values[1:]:
            for attr in ['amplitude_factor', 'phase_shift']:
                component = TransferFunction(f'k{mode}')
                self.data[attr][mode] = component.data[attr][mode]
        self.store()

    #def predict(self, signal: xarray.DataArray):





if __name__ == '__main__':

    #compose = Compose()

    #compose.plot_wave(8)
    #compose.plot_argand()
    #compose.plot_amplitude()

    #compose.plot_fit()

    model = Model()

    gap = Gap()

    tf = TransferFunction('c1')



'''
    k = TFC18('TFCgapsG10', f'k{wavenumber}',
              cluster=1).mesh.slice('z', (0, 0, 0))


    radius = 0.021
    delta_r = np.linalg.norm(k['TFonly'][:, :2], axis=1) - \
        np.linalg.norm(k0['TFonly'][:, :2], axis=1)

    delta_t = radius * (np.arctan2(*k['TFonly'][:, :2].T) -
                        np.arctan2(*k0['TFonly'][:, :2].T))


    import scipy

    def get_wave(coef, wavenumber, nfft=18):
        """Return wavenumber amplitude."""

        return amplitude, phase

    delta = delta_t

    coef = scipy.fft.fft(1e3*delta)
    amplitude, phase = get_wave(coef, wavenumber)


    plt.bar(range(18), 1e3*delta, width=0.8)
    phi = np.linspace(0, 2*np.pi, 150)
    plt.plot(phi * 9/np.pi, amplitude*np.cos(wavenumber*phi + phase))


    print(wavenumber, amplitude)
'''
