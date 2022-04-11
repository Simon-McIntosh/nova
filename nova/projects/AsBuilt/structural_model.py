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
class FourierData(FilePath, ABC):
    """Perform Fourier analysis on TFC deformations."""

    filename: str = 'fourier'
    folder: str = 'TFCgapsG10'
    group: str = None
    datapath: str = 'data/Assembly'
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    def __post_init__(self):
        """Set dataset filepath."""
        self.set_path(self.datapath)
        try:
            self.load()
        except (FileNotFoundError, OSError, KeyError):
            self.build()
            self.store()

    @abstractmethod
    def build(self):
        """Build dataset."""


@dataclass
class PointsAttrs:
    """Non-default attributes for Points class."""

    filename: str


@dataclass
class Points(FourierData, PointsAttrs):
    """Extract midplane mesh and initalize dataset."""

    origin: tuple[int] = (0, 0, 0)
    ncoil: ClassVar[int] = 18
    cluster: ClassVar[int] = 1

    def slice_mesh(self):
        """Return midplane mesh."""
        mesh = TFC18(self.folder, self.filename, cluster=self.cluster).mesh
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
        self.data['coil_index'] = range(1, self.ncoil+1)
        self.data['coord'] = ['x', 'y', 'z', 'radial', 'tangent']
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
        self.data['fft'] = ['radial', 'tangent']

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
        self.data['amplitude'] = ('scenario', 'mode', 'fft'), \
            np.abs(coefficient) / nyquist
        self.data['amplitude'][:, 0] /= 2
        if self.ncoil % 2 == 0:
            self.data['amplitude'][:, nyquist] /= 2
        self.data['phase'] = xarray.zeros_like(self.data['amplitude'])
        self.data['phase'][:] = np.angle(coefficient)
        # correct amplitude and phase to match unit gap variation
        # self.data['amplitude'][:, 1:, :] /= self.data.mode[1:] * np.pi/9
        # self.data['phase'][:, 1:, :] += self.data.mode[1:] * np.pi/36

    def plot(self, scenario='TFonly', fft='radial', width=0.9):
        """Plot fourier components."""
        #plt.figure()
        plt.bar(self.data.mode,
                self.data.phase.sel(scenario=scenario, fft=fft),
                label='ANSYS', width=width)


@dataclass
class Gap(FourierData):
    """Manage gap input."""

    filename: str = 'constant_adaptive_fourier'

    def build(self):
        """Load input gap waveforms."""
        filename = self.file(self.filename, extension='.txt')
        gapdata = pandas.read_csv(filename, skiprows=1, delim_whitespace=True)
        gapdata.drop(columns=['rid'], inplace=True)
        self.data['gap_index'] = range(len(gapdata))
        self.data['coil_index'] = range(len(gapdata))
        self.data['coord'] = ['radial', 'tangent']
        self.data['simulation'] = gapdata.columns
        self.data['gap'] = ('simulation', 'gap_index'), gapdata.values.T
        self.data['gap_sum'] = self.data.gap.sum('gap_index')
        self.data['gap_error'] = self.data.gap - \
            self.data['gap_sum'] / (self.data.dims['coil_index'])

        self.data['points'] = ('simulation', 'coil_index', 'coord'), \
            np.zeros(tuple(self.data.dims[dim]
                     for dim in ['simulation', 'coil_index', 'coord']))

        self.data.points[..., 0] = \
            self.data.gap_sum.data[:, np.newaxis] / (2*np.pi)
        self.data.points[..., 1] = self.data.points[..., 0] * 2*np.pi * \
            self.data['coil_index'] / (self.data['coil_index'][-1] + 1)

        self.data['placement_error'] = ('simulation', 'coil_index'), \
            self.data.gap.cumsum('gap_index').data - \
                self.data.points[..., 1].data
        self.data['placement_error'] -= \
            self.data['placement_error'].mean('coil_index')
        self.fft()

    def fft(self):
        """Extract amplitude and phase information of fourier components."""
        self.data.attrs['nyquist'] = self.data.dims['coil_index'] // 2
        self.data['mode'] = np.arange(self.data.nyquist+1)
        self._fft('placement_error')
        self._fft('gap_error')

    def _fft(self, attr: str):
        coefficient = scipy.fft.fft(self.data[attr].data, axis=-1)
        coefficient = coefficient[:, :self.data.nyquist+1]
        self.data[f'{attr}_amplitude'] = ('simulation', 'mode'), \
            np.abs(coefficient)
        self.data[f'{attr}_amplitude'] /= self.data.nyquist
        self.data[f'{attr}_amplitude'][:, 0] /= 2
        if self.data.dims['coil_index'] % 2 == 0:
            self.data[f'{attr}_amplitude'][:, self.data.nyquist] /= 2
        self.data[f'{attr}_phase'] = ('simulation', 'mode'), \
            np.angle(coefficient)

    def plot(self, simulation: str):
        """Plot gap waveforms."""
        plt.bar(self.data.coil_index,
                self.data.delta.sel(simulation=simulation)[:, 1])

        self.plot_waveform('placement_error', simulation)
        self.plot_waveform('gap_error', simulation)

        plt.bar(self.data.coil_index,
                self.data.gap_error.sel(simulation=simulation), width=0.5)

    def plot_waveform(self, attr: str, simulation: str):
        """Plot fourier waveform."""
        phi = np.linspace(0, 2*np.pi - np.pi/self.data.nyquist,
                          20*self.data.nyquist)
        waveform = 0
        amplitude = self.data[f'{attr}_amplitude']
        amplitude = amplitude.sel(simulation=simulation).data
        phase = self.data[f'{attr}_phase']
        phase = phase.sel(simulation=simulation).data
        for i in np.arange(1, self.data.nyquist+1):
            waveform += amplitude[i]*np.cos(i*phi + phase[i])
        plt.plot(phi * self.data.dims['coil_index'] / (2*np.pi), waveform)
        plt.despine()


@dataclass
class Compose(FourierData):
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


if __name__ == '__main__':

    #compose = Compose()

    #compose.plot_wave(8)
    #compose.plot_argand()
    #compose.plot_amplitude()

    #compose.plot_fit()

    gap = Gap()
    gap.build()
    gap.plot('k4')


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
