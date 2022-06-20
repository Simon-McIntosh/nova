"""Manage fft EM proxy."""
from dataclasses import dataclass, field

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

    fieldline: xarray.DataArray = field(init=False, repr=None, default=None)

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
        return np.fft.irfft(np.fft.rfft(delta) *
                            self.filter[signal], n=ndiv) * ndiv / self.ncoil

    def predict(self, radial=None, tangential=None, ndiv=360):
        """Predict field line deviation from radial and tangential signals."""
        if radial is None and tangential is None:
            raise ValueError('predict requires radial or tangential waveform')
        fieldline = 0
        if radial is not None:
            fieldline += self._predict(radial, 'radial', ndiv)
        if tangential is not None:
            fieldline += self._predict(tangential, 'tangential', ndiv)
        if radial.ndim == 1:
            fieldline.shape = -1, ndiv
        shape = fieldline.shape
        self.fieldline = xarray.DataArray(
            fieldline, coords=[('sample', range(shape[0])),
                               ('phi', np.linspace(0, 2*np.pi, shape[1]))])
        return self.fieldline

    def peaktopeak(self, fieldline=None, modes=3, axis_offset=False):
        """Return peaktopeak delta, H."""
        if fieldline is None:
            fieldline = self.fieldline.data
        if modes is None:
            modes = self.ncoil // 2
        fieldline_hat = np.fft.rfft(fieldline)
        if axis_offset:
            fieldline_hat[..., 1] = 0
        fieldline = np.fft.irfft(fieldline_hat[..., :modes+1],
                                 fieldline.shape[-1])
        return fieldline.max(axis=-1) - fieldline.min(axis=-1)

    @property
    def axis_offset(self):
        """Return prediction for magnetic axis offset."""
        return np.fft.rfft(self.fieldline.data)[..., 1] / \
            (self.fieldline.shape[1] // 2)

    def load_dataset(self, simulation: str):
        """Return benchmark dataset."""
        if 'case' in simulation:
            return MonteCarlo().data.sel(simulation=simulation)
        dataset = Ansys().data.sel(simulation=simulation)
        if simulation[0] == 'k' and simulation != 'k0':
            dataset['deviation'] -= Ansys().data.deviation.sel(simulation='k0')
        return dataset

    @staticmethod
    def _midpoint(fieldline):
        """Return fieldline midpoint."""
        bound = [fieldline.min(axis=-1), fieldline.max(axis=-1)]
        return np.mean(bound, axis=0)

    def _offset(self, data):
        """Offset model fieldline waveform to match data midpoint."""
        offset = self._midpoint(self.fieldline.data) - self._midpoint(data)
        return self.fieldline.data - offset

    def plot_deviation(self, axes, phi, benchmark, legend=True, sample=0):
        """Plot field line deviation benchmark."""
        benchmark -= benchmark[..., 0]
        model = self._offset(benchmark)[sample]
        axes.plot(phi, benchmark, 'C0',
                  label=f'ground truth H={self.peaktopeak(benchmark):1.2f}')
        axes.plot(phi, model, 'C3-.',
                  label=f'inference H={self.peaktopeak(model):1.2f}')
        model_delta = model.max() - model.min()
        axes.set_ylim(model.min() - 0.05*model_delta,
                      model.max() + 0.05*model_delta)
        axes.set_ylabel('field line')
        axes.set_xlabel(r'$\phi$')
        plt.despine()
        if legend:
            axes.legend(fontsize='x-small')

        print((model-benchmark).std()/(benchmark.max() - benchmark.min()))

    def plot_benchmark(self, simulation: str, title=True):
        """Plot Monte Carlo benchmark."""
        dataset = self.load_dataset(simulation)
        radial = dataset.delta[:, 0]
        tangential = dataset.delta[:, 1]
        self.predict(radial, tangential, dataset.dims['phi'])
        axes = plt.subplots(3, 1, sharex=False, sharey=False,
                            gridspec_kw=dict(height_ratios=[1, 1, 3]))[1]
        for i, label in enumerate([r'$\Delta r$', r'$r\Delta \phi$']):
            axes[i].bar(dataset.index, dataset.delta[:, i], width=0.8,
                        color='C1')
            axes[i].set_xticks([])
            axes[i].set_ylabel(label)
        axes[0].legend(['ANSYS'], fontsize='xx-small', loc='lower left')
        self.plot_deviation(axes[-1], dataset.phi, dataset.deviation.data)

        plt.despine()
        if title:
            axes[0].set_title(f'EM benchmark: {simulation}')

        plt.savefig('tmp.png', bbox_inches='tight')


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
    model.plot_benchmark('v3', title=False)


# amplitude = abs(coefficient) / self.data.nyquist
# amplitude[..., 0] /= 2
# if self.data.ncoil % 2 == 0:
#    amplitude[..., self.data.nyquist] /= 2
# phase = np.angle(coefficient)
