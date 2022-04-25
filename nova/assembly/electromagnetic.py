"""Manage fft EM proxy."""
from dataclasses import dataclass

import numpy as np
import scipy.interpolate
import scipy.optimize
import xarray

from nova.assembly.dataset import Ansys, MonteCarlo
from nova.assembly.model import ModelBase
from nova.utilities.pyplot import plt


@dataclass
class Base(ModelBase):
    """Electromagnetic baseclass."""

    def build(self):
        """Initialize electromagnetic dataset."""
        self.data = xarray.Dataset(attrs=dict(ncoil=self.ncoil,
                                              nyquist=self.ncoil // 2))
        self.data['mode'] = np.arange(self.data.nyquist + 1)
        self.data['coefficient'] = ['real', 'imag', 'amplitude', 'phase']
        self.data['signal'] = ['radial', 'tangential']
        self.data['filter'] = xarray.DataArray(0., self.data.coords)

    def _filter(self, label: str):
        """Return complex filter."""
        return self.data.filter[:, 0].sel(signal=label) + \
            1j * self.data.filter[:, 1].sel(signal=label)

    def _predict(self, delta, signal: str, ndiv: int):
        """Return fft prediction."""
        delta_hat = np.fft.rfft(delta)
        return np.fft.irfft(delta_hat * self.filter[signal], n=ndiv) * \
            ndiv / self.ncoil

    def predict(self, radial=None, tangential=None, ndiv=360):
        """Predict field line deviation from radial and tangential signals."""
        deviation = 0
        if radial is not None:
            deviation += self._predict(radial, 'radial', ndiv)
        if tangential is not None:
            deviation += self._predict(tangential, 'tangential', ndiv)
        return deviation

    @staticmethod
    def _peaktopeak(waveform):
        """Return peaktopeak delta."""
        return waveform.max(axis=-1) - waveform.min(axis=-1)

    def peaktopeak(self, radial=None, tangential=None, ndiv=180):
        """Return peak to peak prediction."""
        deviation = self.predict(radial, tangential)
        return self._peaktopeak(deviation)

    def load_dataset(self, simulation: str):
        """Return benchmark dataset."""
        if 'case' in simulation:
            return MonteCarlo().data.sel(simulation=simulation)
        dataset = Ansys().data.sel(simulation=simulation)
        if simulation[0] == 'k' and simulation != 'k0':
            dataset['deviation'] -= Ansys().data.deviation.sel(simulation='k0')
        return dataset

    @staticmethod
    def _midpoint(waveform):
        """Return waveform midpoint."""
        bound = [waveform.min(axis=-1), waveform.max(axis=-1)]
        return np.mean(bound, axis=0)

    def _offset(self, data, model):
        """Offset model waveform to match data midpoint."""
        offset = self._midpoint(model) - self._midpoint(data)
        return model - offset

    def plot_deviation(self, axes, phi, benchmark, model, legend=True):
        """Plot field line deviation benchmark."""
        benchmark -= benchmark[..., 0]
        model = self._offset(benchmark, model)
        axes.plot(phi, benchmark, 'C0',
                  label=f'ground truth H={self._peaktopeak(benchmark):1.2f}')
        axes.plot(phi, model, 'C3-.',
                  label=f'inference H={self._peaktopeak(model):1.2f}')
        axes.set_ylabel('field line deviation')
        axes.set_xlabel(r'$\phi$')
        plt.despine()
        if legend:
            axes.legend(fontsize='x-small')

    def plot_benchmark(self, simulation: str):
        """Plot Monte Carlo benchmark."""
        dataset = self.load_dataset(simulation)
        radial = dataset.delta[:, 0]
        tangential = dataset.delta[:, 1]
        model_deviation = self.predict(radial, tangential, dataset.dims['phi'])
        axes = plt.subplots(2, 1, sharex=False, sharey=False,
                            gridspec_kw=dict(height_ratios=[1, 2]))[1]
        for i, label in enumerate([r'$\Delta r$', r'$r\Delta \phi$']):
            axes[0].bar(dataset.index, dataset.delta[:, i], width=0.8 - i*0.3,
                        color=f'C{1+i}', label=label)
        axes[0].set_xticks([])
        self.plot_deviation(axes[-1], dataset.phi,
                            dataset.deviation.data, model_deviation)
        plt.despine()
        axes[0].legend(fontsize='x-small', ncol=2)
        axes[0].set_ylabel(r'vault')
        axes[0].set_title(f'EM benchmark: {simulation}')


@dataclass
class Model(Base):
    """Build component EM transform filters."""

    name: str = 'electromagnetic'

    def build(self):
        """Build electromagnetic model."""
        super().build()
        self.build_radial()
        self.build_tangential()
        self.load_filter()
        return self.store()

    def build_radial(self, simulation='case1'):
        """Build radial filter."""
        dataset = self.load_dataset(simulation)
        r_hat = np.fft.rfft(dataset.delta[:, 0])
        h_hat = np.fft.rfft(dataset.deviation)[:self.data.nyquist+1] * \
            self.ncoil / dataset.dims['phi']
        _filter = h_hat / r_hat
        self.data['filter'][1:, 0, 0] = _filter.real[1:]
        self.data['filter'][1:, 1, 0] = _filter.imag[1:]

    def _build_tangential(self, simulation: str):
        """Build tangential filter."""
        dataset = self.load_dataset(simulation)
        r_hat = np.fft.rfft(dataset.delta[:, 0])
        t_hat = np.fft.rfft(dataset.delta[:, 1])
        radial_deviation = np.fft.irfft(
            r_hat * self._filter('radial'), n=dataset.dims['phi']) * \
            dataset.dims['phi'] / self.ncoil
        tangential_deviation = dataset.deviation - radial_deviation
        h_hat = np.fft.rfft(tangential_deviation)[:self.data.nyquist+1] * \
            self.ncoil / dataset.dims['phi']
        return h_hat / t_hat

    def build_tangential(self, simulation='k'):
        """Build tangential filter."""
        if simulation == 'k':
            for i in range(1, self.data.nyquist+1):
                _filter = self._build_tangential(f'k{i}')
                self.data['filter'][i, 0, 1] = _filter.real[i]
                self.data['filter'][i, 1, 1] = _filter.imag[i]
            return
        _filter = self._build_tangential(simulation)
        self.data['filter'][1:, 0, 1] = _filter.real[1:]
        self.data['filter'][1:, 1, 1] = _filter.imag[1:]


@dataclass
class WaveModel(Base):
    """Calculate HFS field line deviation using fft EM proxy."""

    name: str = 'electromagnetic'

    @staticmethod
    def fit(coef, *args):
        """Return absolute error between deviation target and fft filters."""
        r_hat, t_hat, h_hat = args
        if len(coef) == 1:
            return np.abs(h_hat - coef[0] * r_hat)**2
        return np.abs(h_hat - (coef[0] * r_hat + coef[1] * t_hat))**2

    @staticmethod
    def sead(r_hat, t_hat, ncoef: int):
        """Return sead vector for filter fit."""
        if ncoef == 1:
            return [1]
        factor = np.abs(t_hat) / np.abs(r_hat)
        attenuation = [1, factor]
        attenuation /= np.max(attenuation)
        return attenuation

    def build(self, phase_shift=False):
        """Build electromagnetic model."""
        super().build()
        self.data.attrs['phase_shift'] = int(phase_shift)
        reference = Ansys().data.sel(simulation='k0')
        for mode in self.data.mode.values[1:]:
            dataset = Ansys().data.sel(simulation=f'k{mode}')
            dataset['deviation'] -= reference.deviation
            r_hat = np.fft.rfft(dataset.delta[:, 0])
            t_hat = np.fft.rfft(dataset.delta[:, 1])
            h_hat = np.fft.rfft(dataset.deviation)[:self.data.nyquist+1] * \
                self.ncoil / dataset.dims['phi']
            result = scipy.optimize.minimize(
                self.fit, self.sead(r_hat[mode], t_hat[mode], phase_shift),
                args=(r_hat[mode], t_hat[mode], h_hat[mode]),
                method='Nelder-Mead')
            if phase_shift:
                self.data.filter[mode, 0] = result.x[::2]
                self.data.filter[mode, 1] = result.x[1::2]
            else:
                self.data.filter[mode, 0] = result.x
        coefficient = self.data.filter[:, 0] + self.data.filter[:, 1] * 1j
        self.data.filter[:, 2] = np.abs(coefficient) / self.data.nyquist
        self.data.filter[0, 2] /= 2
        if self.data.ncoil % 2 == 0:
            self.data.filter[self.data.nyquist, 2] /= 2
        self.data.filter[:, 3] = np.angle(coefficient)
        self.load_filter()
        return self.store()


if __name__ == '__main__':

    model = Model()
    # model.build(False)
    model.plot_benchmark('v3')

# amplitude = abs(coefficient) / self.data.nyquist
# amplitude[..., 0] /= 2
# if self.data.ncoil % 2 == 0:
#    amplitude[..., self.data.nyquist] /= 2
# phase = np.angle(coefficient)
