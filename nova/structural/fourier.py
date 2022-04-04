"""Perform post-processing analysis on Fourier perterbed TFC dataset."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import scipy
import xarray

from nova.structural.TFC18 import TFC18
from nova.utilities.pyplot import plt


@dataclass
class Fourier:
    """Perform Fourier analysis on TFC deformations."""

    prefix: str = 'k'
    folder: str = 'TFCgapsG10'
    origin: tuple[int] = (0, 0, 0)
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    ncoil: ClassVar[int] = 18
    cluster: ClassVar[int] = 1

    def __post_init__(self):
        """Initialize dataset."""

        self.build()

    @property
    def attrs(self):
        """Return instance attributes."""
        return {attr: getattr(self, attr)
                for attr in ['prefix', 'folder', 'origin', 'ncoil', 'cluster']}

    def load_midplane(self, wavenumber: int):
        """Return midplane mesh."""
        file = f'{self.prefix}{wavenumber}'
        mesh = TFC18(self.folder, file, cluster=self.cluster).mesh
        return mesh.slice('z', self.origin)

    def initialize_dataset(self, wavenumber=0):
        """Initialize empty dataset from referance scenario."""
        self.data = xarray.Dataset(attrs=self.attrs)
        mesh = self.load_midplane(wavenumber)
        self.data['wavenumber'] = range(self.ncoil//2 + 1)
        self.data['scenario'] = mesh['scenario']
        self.data['coil_index'] = range(1, self.ncoil+1)
        self.data['coord'] = ['x', 'y', 'z', 'radial', 'tangent']
        self.data['points'] = xarray.DataArray(0., self.data.coords)
        self.data['mode'] = self.data['wavenumber'].data
        self.data['fft'] = ['radial', 'tangent']

    def build(self):
        """Build dataset from ansys simulation data."""
        self.initialize_dataset()
        for wavenumber in self.data.wavenumber.data:
            mesh = self.load_midplane(wavenumber)
            for scenario in self.data.scenario.data:  # store ansys data [mm]
                self.data['points'].loc[wavenumber, scenario][:, :3] = \
                    1e3*mesh[scenario]
        self.data.points[..., -2] = np.linalg.norm(self.data.points[..., :2],
                                                   axis=-1)
        self.data.points[..., -1] = np.arctan2(self.data.points[..., 1],
                                               self.data.points[..., 0])
        self.data['delta'] = self.data['points'] - self.data['points'][0]

        r_norm = self.data['points'][0, ..., :2].data.copy()
        r_norm /= np.linalg.norm(r_norm, axis=-1)[..., np.newaxis]
        t_norm = r_norm @ np.array([[0, -1], [1, 0]])

        self.data['delta'][..., -2] = np.einsum(
            'ijkl,jkl->ijk', self.data['delta'][..., :2].data, r_norm)
        self.data['delta'][..., -1] = np.einsum(
            'ijkl,jkl->ijk', self.data['delta'][..., :2].data, t_norm)
        self.fft()

    def fft(self):
        """Apply fft to dataset."""
        nyquist = self.ncoil // 2
        coef = scipy.fft.fft(self.data['delta'][:, :, :, 3:].data, axis=-2)
        self.data['coefficient'] = ('wavenumber', 'scenario', 'mode', 'fft'), \
            coef[:, :, :nyquist+1]
        self.data['amplitude'] = np.abs(self.data['coefficient']) / nyquist
        self.data['amplitude'][:, :, 0] /= 2
        if self.ncoil % 2 == 0:
            self.data['amplitude'][nyquist] /= 2
        self.data['phase'] = xarray.zeros_like(self.data['amplitude'])
        self.data['phase'][:] = np.angle(self.data['coefficient'])

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
        amplitude = np.diag(self.data.amplitude[:, 2, :, 0])
        plt.figure()
        plt.plot(self.data.wavenumber.values[1:], amplitude[1:], 'o')
        plt.despine()
        plt.xticks(self.data.wavenumber.values[1:])
        plt.xlabel('wavenumber')
        plt.ylabel('radial amplification factor')

    def plot_fit(self):
        """Fit model to observations."""
        H = np.array([0.396, 0.759, 1.6871, 0.934, 0.6451, 0.5373,
                      0.4835, 0.4085, 0.6682, 0.4012])
        weights = np.ones(10)
        weights[1] = 0  # exclude n=1 mode
        #weights[8] = 0

        amplitude = np.diag(fourier.data.amplitude[:, 2, :, 0])
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

    fourier = Fourier()
    fourier.plot_wave(2)

    fourier.plot_amplitude()
    fourier.plot_fit()





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
