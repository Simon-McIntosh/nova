"""Manage fft EM proxy."""
from dataclasses import dataclass
import os
from typing import ClassVar

import numpy as np
import pandas
import scipy.interpolate
import xarray

from nova.definitions import root_dir
from nova.projects.AsBuilt.model_data import ModelData, ModelBase
from nova.projects.AsBuilt.structural_model import StructuralTransform
from nova.utilities.pyplot import plt


@dataclass
class BaseDataSet:
    """Manage field deviation datasets."""

    simulations: ClassVar[list[str]] = []
    ncoil: ClassVar[int] = 18

    def build(self, ndiv=360):
        """Build base dataset structure."""
        self.data = xarray.Dataset()
        self.data['simulation'] = self.simulations
        self.data['index'] = np.arange(1, self.ncoil+1)
        self.data['response'] = ['radial', 'tangential']
        self.data['delta'] = xarray.DataArray(0., self.data.coords)
        self.data['phi'] = np.linspace(0, 2*np.pi, ndiv)
        self.data['deviation'] = ('simulation', 'phi'), \
            np.zeros((self.data.dims['simulation'], ndiv))
        for i, simulation in enumerate(self.simulations):
            self.data.deviation[i] = self._load_data(simulation, ndiv)
        self.data['peaktopeak'] = self.data.deviation.max(axis=-1) - \
            self.data.deviation.min(axis=-1)
        return self.store()

    def _load_data(self, simulation: str, ndiv, ripple=True):
        """Return interpolated field line deviation dataset."""
        path = os.path.join(root_dir, 'input/Assembly/')
        filename = os.path.join(path, f'{simulation}.csv')
        data = pandas.read_csv(filename, header=None)
        data.iloc[:, 0] *= np.pi/180
        phi_sample = np.linspace(0, 2*np.pi, 3*len(data), endpoint=False)
        data = data.loc[np.unique(data.iloc[:, 0], return_index=True)[1], :]
        h_sample = scipy.interpolate.interp1d(
            data.iloc[:, 0], data.iloc[:, 1],
            fill_value='extrapolate')(phi_sample)
        h_hat = np.fft.rfft(h_sample)
        if ripple:
            h_hat[self.ncoil // 2:self.ncoil] = 0
            h_hat[self.ncoil+1:] = 0
        else:
            h_hat[self.ncoil // 2:] = 0
        return np.fft.irfft(h_hat[:self.ncoil+1], n=ndiv) * ndiv/len(h_sample)

    def plot(self, simulation: str):
        """Plot benchmark data."""
        axes = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[1, 2],
                                                   hspace=0.3))[1]
        axes[0].bar(self.data.index,
                    self.data.delta.sel(simulation=simulation,
                                        response='radial'), label='radial')
        axes[0].bar(self.data.index,
                    self.data.delta.sel(simulation=simulation,
                                        response='tangential'),
                    label='tangential', width=0.4)
        axes[0].get_xaxis().set_visible(False)
        axes[0].set_ylabel(r'displacement')
        axes[0].legend(fontsize='xx-small', ncol=2, loc='upper right',
                       bbox_to_anchor=(1, 1.3))

        axes[1].plot(self.data.phi,
                     self.data.deviation.sel(simulation=simulation))
        axes[1].set_xlabel(r'$\phi$')
        axes[1].set_ylabel(r'deviation, $h$')
        plt.despine()


@dataclass
class RadialDataSet(BaseDataSet, ModelData):
    """Manage radial dataset."""

    simulations: ClassVar[list[str]] = ['case1', 'case2', 'case3']

    def build(self):
        """Build radial pertibation dataset."""
        super().build()
        self.data.delta[..., 0] = \
            np.array([[4.911188, -4.046573, -1.700848],
                      [3.720625, 0.684551, -0.772447],
                      [4.503144, -4.844046, -3.677593],
                      [3.109321, -4.185436, -4.728377],
                      [-1.781115, -0.273149, -4.963377],
                      [1.395654, 2.234939, -3.250931],
                      [-4.430507, 1.728926, -3.246353],
                      [-4.286455, 3.105658, -4.289813],
                      [-4.567845, 0.44772, -1.126167],
                      [-4.41067, 1.413355, -4.54465],
                      [-3.732528, -2.271867, -4.153696],
                      [3.300678, 4.616676, 3.136483],
                      [3.650736, 4.525728, 4.327046],
                      [1.563206, 4.12104, 4.560215],
                      [-4.555027, 4.891656, 3.943722],
                      [4.775682, -4.252579, 2.248062],
                      [2.731185, -4.310566, 2.761704],
                      [4.198865, -4.017274, -0.864311]]).T
        return self.store()


@dataclass
class StructuralDataSet(BaseDataSet, ModelData):
    """Manage structural dataset."""

    simulations: ClassVar[list[str]] = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6',
                                        'k7', 'k8', 'k9',
                                        'a1', 'a2', 'c1', 'c2']

    def build(self):
        """Build structural pertibation dataset."""
        super().build()
        for i, simulation in enumerate(self.simulations):
            self.data.delta[i] = \
                StructuralTransform(simulation).data.response_delta
        return self.store()


@dataclass
class ElectromagneticBase(ModelBase):
    """Electromagnetic baseclass."""

    def _filter(self, signal: str):
        """Return complex filter."""
        return self.data.filter[:, 0].sel(signal=signal) + \
            1j * self.data.filter[:, 1].sel(signal=signal)

    def _predict(self, delta, signal: str, ndiv: int):
        """Return fft prediction."""
        delta_hat = np.fft.rfft(delta)
        return np.fft.irfft(delta_hat * self.filter[signal], n=ndiv) * \
            ndiv / self.ncoil

    def predict(self, radial, tangential, ndiv=360):
        """Predict field line deviation from radial and tangential signals."""
        deviation = self._predict(radial, 'radial', ndiv) + \
            self._predict(tangential, 'tangential', ndiv)
        deviation -= deviation[0]
        return deviation


@dataclass
class ElectromagneticTransform(ElectromagneticBase):
    """Build component EM transform filters."""

    def build(self):
        """Build electromagnetic model."""
        self.data = xarray.Dataset(attrs=dict(ncoil=self.ncoil,
                                              nyquist=self.ncoil // 2))
        self.data['mode'] = np.arange(self.data.nyquist + 1)
        self.data['coefficient'] = ['real', 'imag', 'radial', 'tangential']
        self.data['signal'] = ['radial', 'tangential']
        self.data['filter'] = xarray.DataArray(0., self.data.coords)

        self.build_radial()
        self.build_tangential()
        return self.store()

    def build_radial(self, simulation='case1'):
        """Build radial filter."""
        dataset = RadialDataSet().data.sel(simulation=simulation)
        r_hat = np.fft.rfft(dataset.delta[:, 0])
        h_hat = np.fft.rfft(dataset.deviation)[:self.data.nyquist+1] * \
            self.ncoil / dataset.dims['phi']
        _filter = h_hat / r_hat
        self.data['filter'][1:, 0, 0] = _filter.real[1:]
        self.data['filter'][1:, 1, 0] = _filter.imag[1:]

    def build_tangential(self, simulation='c2'):
        """Build tangential filter."""
        dataset = StructuralDataSet().data.sel(simulation=simulation)
        r_hat = np.fft.rfft(dataset.delta[:, 0])
        t_hat = np.fft.rfft(dataset.delta[:, 1])
        # subtract radial component
        radial_deviation = np.fft.irfft(
            r_hat * self._filter('radial'), n=dataset.dims['phi']) * \
            dataset.dims['phi'] / self.ncoil

        tangential_deviation = dataset.deviation - radial_deviation
        h_hat = np.fft.rfft(tangential_deviation)[:self.data.nyquist+1] * \
            self.ncoil / dataset.dims['phi']
        _filter = h_hat / t_hat
        self.data['filter'][1:, 0, 1] = _filter.real[1:]
        self.data['filter'][1:, 1, 1] = _filter.imag[1:]


@dataclass
class ElectromagneticModel(ModelBase):
    """Calculate HFS field line deviation using fft EM proxy."""


if __name__ == '__main__':

    # radial = RadialDataSet()
    # structural = StructuralDataSet()
    # structural.plot('k2')

    model = ElectromagneticTransform().build()

    #model = ElectromagneticModel()

    structural = StructuralDataSet().data.sel(simulation='a1')

    radial = structural.delta[:, 0].data
    tangential = structural.delta[:, 1].data
    h = model.predict(radial, tangential)

    h += 0.11*np.cos(18*structural.phi)
    h -= h[0]

    plt.plot(structural.phi,
             structural.deviation - structural.deviation.mean(), 'C1',
             label='Energopul')
    plt.plot(np.linspace(0, 2*np.pi, len(h), endpoint=False),
             h - h.mean(), 'C2-', label='FFT proxy')

    plt.despine()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$h$, mm')
    plt.legend()


'''



@dataclass
class ElectromagneticModel(ModelData):
    """Calculate HFS field line deviation using fft EM proxy."""

    coef: ClassVar[list[float]] = [0, 0.61]


    def fit(self, index: int):
        """Compute FIR filter."""
        r_hat = np.fft.rfft(self.data.benchmark_r[index])
        h_hat = np.fft.rfft(self._benchmark_data(
            index, np.linspace(0, 2*np.pi, self.data.dims['index'],
                               endpoint=False)))
        self.response = h_hat / r_hat

    def predict(self, radius, plot=False):
        """Predict field line deviation from radius vector."""
        coefficient = np.fft.rfft(radius, axis=-1)

        coefficient *= self.response
        deviation = np.fft.irfft(coefficient, n=self.data.ndiv) * \
            self.data.ndiv / self.data.ncoil
        deviation += 0.1*np.cos(18*self.data.phi+np.pi)
        deviation -= deviation[..., 0]
        peaktopeak = deviation.max(axis=0) - deviation.min(axis=0)

        if plot:
            plt.figure()
            plt.plot(self.data.phi, deviation)
        return peaktopeak, deviation

    def plot(self, case: int):
        """Plot benchmark comparison."""
        peaktopeak, deviation = self.predict(self.data.benchmark_r[case])
        print(peaktopeak)
        plt.plot(self.data.phi, self.data.benchmark_h[case],
                 '-', label='benchmark')
        plt.plot(self.data.phi, deviation, '--', label='FFT proxy')
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'field line deviation, $h$ mm')
        plt.despine()
        plt.legend()
        plt.title(f'case {case+1}')


if __name__ == '__main__':

    electromagnetic = ElectromagneticModel()
    electromagnetic.fit(0)
    electromagnetic.plot(1)
'''


'''

        self.data['index'] = np.arange(1, ncoil+1)



amplitude = abs(coefficient) / self.data.nyquist
amplitude[..., 0] /= 2
if self.data.ncoil % 2 == 0:
    amplitude[..., self.data.nyquist] /= 2
phase = np.angle(coefficient)
radial_field = self.coef[0] * np.ones_like(amplitude[..., 0]) * \
    np.sin(self.data.ncoil*self.data.phi)
for wavenumber in range(1, self.data.nyquist + 1):
    radial_field += self.coef[1]*amplitude[..., wavenumber] * \
        np.sin(wavenumber*self.data.phi + phase[..., wavenumber])
deviation = self.data.dphi * np.cumsum(radial_field, axis=0)
'''
