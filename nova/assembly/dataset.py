"""Manage field line deviation datasets."""
from dataclasses import dataclass
import os
from typing import ClassVar

import numpy as np
import pandas
import scipy.interpolate
import scipy.optimize
import xarray

from nova.definitions import root_dir
from nova.assembly.model import ModelData
from nova.assembly import structural
from nova.utilities.pyplot import plt


@dataclass
class BaseDataSet:
    """Manage field deviation datasets."""

    simulations: ClassVar[list[str]] = []
    ncoil: ClassVar[int] = 18

    def build(self, ndiv=360):
        """Build base dataset structure."""
        self.data = xarray.Dataset(attrs=dict(nyquist=self.ncoil // 2))
        self.data['simulation'] = self.simulations
        self.data['index'] = np.arange(1, self.ncoil+1)
        self.data['response'] = ['radial', 'tangential']
        self.data['delta'] = xarray.DataArray(0., self.data.coords)
        self.data['sample_phi'] = np.linspace(0, 2*np.pi, 2*ndiv,
                                              endpoint=False)
        self.data['sample_deviation'] = ('simulation', 'sample_phi'), \
            np.zeros((self.data.dims['simulation'], 2*ndiv))
        self.data['phi'] = np.linspace(0, 2*np.pi, ndiv, endpoint=False)
        self.data['deviation'] = ('simulation', 'phi'), \
            np.zeros((self.data.dims['simulation'], ndiv))
        self.data['ripple_deviation'] = ('simulation', 'phi'), \
            np.zeros((self.data.dims['simulation'], ndiv))
        for i, simulation in enumerate(self.simulations):
            self.data.sample_deviation[i] = self._load_data(simulation)
            self.data.deviation[i] = self._filter_data(simulation, False)
            self.data.ripple_deviation[i] = self._filter_data(simulation, True)
        self.data['peaktopeak'] = self.data.deviation.max(axis=-1) - \
            self.data.deviation.min(axis=-1)
        return self.store()

    def _load_data(self, simulation: str):
        """Return interpolated field line deviation waveform."""
        path = os.path.join(root_dir, 'input/Assembly/')
        filename = os.path.join(path, f'{simulation}.csv')
        data = pandas.read_csv(filename, header=None)
        data.iloc[:, 0] *= np.pi/180  # to radians
        data = data.loc[np.unique(data.iloc[:, 0], return_index=True)[1], :]
        return scipy.interpolate.interp1d(
            data.iloc[:, 0], data.iloc[:, 1],
            fill_value='extrapolate')(self.data.sample_phi)

    def _filter_data(self, simulation: str, ripple=True):
        """Return filtered deviation waveform."""
        h_hat = np.fft.rfft(self.data.sample_deviation.sel(
            simulation=simulation))
        if ripple:
            h_hat = h_hat[:self.ncoil+1] * \
                2*self.ncoil / self.data.dims['sample_phi']
            h_hat[self.ncoil // 2:self.ncoil] = 0
            return np.fft.irfft(h_hat, n=self.data.dims['phi']) * \
                self.data.dims['phi'] / (2*self.ncoil)
        h_hat = h_hat[:self.ncoil // 2 + 1] * \
            self.ncoil / self.data.dims['sample_phi']
        return np.fft.irfft(h_hat, n=self.data.dims['phi']) * \
            self.data.dims['phi'] / self.ncoil

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

        axes[1].plot(self.data.sample_phi,
                     self.data.sample_deviation.sel(simulation=simulation),
                     label='raw')
        axes[1].plot(self.data.phi,
                     self.data.deviation.sel(simulation=simulation), '--',
                     label='filtered')
        axes[1].plot(self.data.phi,
                     self.data.ripple_deviation.sel(simulation=simulation),
                     '--', label='ripple')
        axes[1].set_xlabel(r'$\phi$')
        axes[1].set_ylabel(r'deviation, $h$')
        plt.despine()
        plt.legend()
        plt.suptitle(f'{simulation}')


@dataclass
class MonteCarlo(BaseDataSet, ModelData):
    """Manage radial displacment Monte Carlo dataset."""

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
class Ansys(BaseDataSet, ModelData):
    """Manage ansys dataset."""

    simulations: ClassVar[list[str]] = ['k0', 'k1', 'k2', 'k3', 'k4',
                                        'k5', 'k6', 'k7', 'k8', 'k9',
                                        'a1', 'a2', 'c1', 'c2', 'v3']

    def build(self):
        """Build structural pertibation dataset."""
        super().build()
        for i, simulation in enumerate(self.simulations):
            self.data.delta[i] = \
                structural.Transform(simulation).data.response_delta
        return self.store()


if __name__ == '__main__':

    MonteCarlo().plot('case2')
    #ansys = Ansys().build()
    #ansys.plot('a2')
