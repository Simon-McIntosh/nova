"""Methods to aid computational assembly studies for the TF cage."""
from abc import ABC, abstractmethod
from collections import Iterable
from dataclasses import dataclass, field
import os
from warnings import warn

import numpy as np
import scipy.fft
import scipy.optimize
import scipy.special
import xarray

from nova.definitions import root_dir
from nova.utilities.pyplot import plt
from nova.utilities.time import clock


@dataclass
class PDF(ABC):
    """Probability density function abstract base class."""

    mean: float = 2.
    variance: float = 1.
    sead: int = 2025
    name: str = 'ABC'
    parameters: list[float] = field(init=False, default=None)

    def __post_init__(self):
        """Initialize random sead and fit PDF parameters."""
        self.rng = np.random.default_rng(self.sead)
        self.fit_parameters()

    @property
    def std(self):
        """Return standard deviation."""
        return self.variance**0.5

    def update_mean(self, mean):
        """Update distribution mean."""
        self.mean = mean
        self.fit_parameters()

    def update_variance(self, variance):
        """Update distribution variance."""
        self.variance = variance
        self.fit_parameters()

    def update_std(self, std):
        """Update distribution standard deviation."""
        self.update_variance(std**2)

    def update(self, mean, variance):
        """Update distribution mean and variance."""
        self.mean = mean
        self.variance = variance
        self.fit_parameters()

    @abstractmethod
    def _mean(self, *args):
        """Return PDF mean."""

    @abstractmethod
    def _variance(self, *args):
        """Return PDF variance."""

    def _target_parameters(self, *args):
        """Return numpy array containing pdf parameters [mean, variance]."""
        return np.array([self._mean(*args), self._variance(*args)])

    def fit_error(self, x):
        """Return L2 norm of diffrence between target / calculated moments."""
        input_parameters = [self.mean, self.variance]
        vector = self._target_parameters(*x) - input_parameters
        vector /= input_parameters  # norm
        return np.linalg.norm(vector)

    @property
    @abstractmethod
    def parameter_sead(self):
        """Return PDF sead parameters."""

    @property
    def parameter_bounds(self):
        """Return parameter bounds used by minimizer."""

    def fit_parameters(self):
        """Solve PDF parameters based on input mean and variance."""
        opp = scipy.optimize.minimize(
            self.fit_error, self.parameter_sead, bounds=self.parameter_bounds,
            options=dict(ftol=1e-4))
        if not opp.success:
            warn(f'{opp.message}\n'
                 f'mean {self.mean:1.4f}, std {self.std:1.4f}\n'
                 f'_mean {self._mean(*opp.x):1.4f}, '
                 f'_std {self._variance(*opp.x)**0.5:1.4f}\n')
        self.parameters = opp.x

    @abstractmethod
    def distribution(self, sample):
        """Return sample PDF."""

    @abstractmethod
    def sample(self, size=1):
        """Return sample from Weibull distribution."""

    def plot(self):
        """Plot PDF."""
        sample = np.linspace(0, self.mean + 4*self.variance**0.5, 1000)[1:]
        plt.plot(sample, self.distribution(sample))
        plt.despine()
        plt.xlabel('gap $U$ mm')
        plt.ylabel('P($U$)')
        plt.title(f'{self.name}\n'
                  f'$\mu={self.mean:1.1f}$, $\sigma={self.std:1.2f}$')

    def plot_array(self, mean, variance, sample=None):
        """Plot distribution array."""
        checkpoint = self.mean, self.variance
        if not isinstance(variance, Iterable):
            variance = [variance]
        if not isinstance(mean, Iterable):
            mean = mean * np.ones(len(variance))
        if sample is None:
            sample = np.linspace(0, np.max(mean) +
                                 4*np.max(variance)**0.5, 500)[1:]
        for parameters in zip(mean, variance):
            self.update(*parameters)
            _mean = self._mean(*self.parameters)
            _variance = self._variance(*self.parameters)
            _std = np.sqrt(_variance)
            label = rf'$\mu=${_mean:1.1f}, $2\sigma=${2*_std:1.2f}'
            plt.plot(sample, self.distribution(sample), label=label)
        self.update(*checkpoint)  # reset distribution
        plt.legend()
        plt.despine()
        plt.xlabel('gap $U$ mm')
        plt.ylabel('P($U$)')
        plt.title(self.name)


@dataclass
class Weibull(PDF):
    """Manage Weibull PDF - construct from input mean and variance."""

    name: str = 'Weibull'

    @property
    def scale(self):
        """Return Weibull scale parameter."""
        return self.parameters[0]

    @property
    def shape(self):
        """Return shape parameter."""
        return self.parameters[1]

    def distribution(self, sample):
        """Return sample PDF."""
        return self.shape/self.scale * \
            (sample / self.scale)**(self.shape - 1) * \
            np.exp(-(sample / self.scale)**self.shape)

    def _mean(self, scale, shape):
        """Return PDF mean."""
        return scale * scipy.special.gamma(1 + 1/shape)

    def _variance(self, scale, shape):
        """Return PDF variance."""
        return scale**2 * (scipy.special.gamma(1 + 2/shape) -
                           scipy.special.gamma(1 + 1/shape)**2)

    @property
    def parameter_sead(self):
        """Return Weibull sead parameters."""
        return self.mean, 5

    @property
    def parameter_bounds(self):
        """Return parameter bounds used by minimizer."""
        return [[0, None], [0.05, None]]

    def sample(self, size=1):
        """Return sample from Weibull distribution."""
        return self.scale * self.rng.weibull(self.shape, size=size)


@dataclass
class PDFs:
    """Manage pair of PDFs to sample SSAT and pit placments."""

    sead: int = 2025
    pdf: PDF = Weibull

    def __post_init__(self):
        """Init PDF pair."""
        self.ssat = self.pdf(sead=self.sead)
        self.pit = self.pdf(sead=self.sead)

    def update_mean(self, ssat, pit):
        """Update pdf means."""
        self.ssat.update_mean(ssat)
        self.pit.update_mean(pit)

    def update_std(self, ssat, pit):
        """Update pdf standard deviations."""
        self.ssat.update_std(ssat)
        self.pit.update_std(pit)


@dataclass
class Assembly:
    """Generate single assembly trial."""

    ssat_std: float = 1.5
    pit_std: float = 1.5
    total_gap: float = 36
    ncoil: int = 18
    sead: int = 2025
    pdf: PDFs = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Init gaps."""
        self.uniform_gap = self.total_gap/self.ncoil
        self.pdf = PDFs(sead=self.sead)
        self.pdf.update_std(self.ssat_std, self.pit_std)

    def solve(self):
        """Solve TF assembly trial - pair, pair, pit, pair, pit,...."""
        self.initialize_dataset()
        self.data.gap[0] = self.sample(self.uniform_gap, 'ssat')
        for i in range(1, 9):
            self.set_target(2*i)
            self.data.gap[2*i] = self.sample(
                self.data.target[2*i].values, 'ssat')
            self.set_target(2*i-1)
            self.data.gap[2*i-1] = self.sample(
                self.data.target[2*i-1].values, 'pit')
        self.close_cage()
        self.update_error()

    def initialize_dataset(self):
        """Init xarray dataset."""
        self.data = xarray.Dataset(
            coords={'gap_index': range(self.ncoil),
                    'wavenumber': range(self.ncoil//2+1)})
        uniform_gap = np.full(self.ncoil, self.uniform_gap, dtype=float)
        self.data['target'] = ('gap_index', uniform_gap.copy())
        self.data['gap'] = ('gap_index', uniform_gap.copy())
        self.data['error'] = ('gap_index', np.zeros(self.ncoil))
        self.data['fft_real'] = ('wavenumber', np.zeros(self.ncoil//2+1))
        self.data['fft_imag'] = ('wavenumber', np.zeros(self.ncoil//2+1))
        self.data['error_modes'] = ('wavenumber', np.zeros(self.ncoil//2+1))
        self.data.attrs = self.attrs

    @property
    def attrs(self):
        """Manage assembly attributes."""
        return dict(ssat_std=self.pdf.ssat.std, pit_std=self.pdf.pit.std,
                    total_gap=self.total_gap, sead=self.sead)

    @attrs.setter
    def attrs(self, attrs):
        self.pdf.update_std(attrs['ssat_std'], attrs['pit_std'])
        self.total_gap = attrs['total_gap']
        self.sead = attrs['sead']

    def sample(self, mean, stage: str):
        """Return sample from PDF."""
        if mean < 0.01:
            mean = 0.01
        pdf = getattr(self.pdf, stage)
        pdf.update_mean(mean)
        return pdf.sample(size=1)[0]

    def get_error(self, pair_index: int):
        """Return running assembly error."""
        return np.sum(self.data.gap[:2*pair_index+1]) - \
            (2*pair_index + 1) * self.uniform_gap

    def set_target(self, index):
        """Set taget gap."""
        pair_index = (index+1) // 2
        error = self.get_error(pair_index)
        self.data.target[index] = self.uniform_gap - error
        if self.data.target[index] < 0:
            self.data.target[index] = 0

    def close_cage(self):
        """Perform TF cage closure."""
        self.data.gap[-1] = self.total_gap - np.sum(self.data.gap[:-1])
        if self.data.gap[-1] < 0:
            self.data.gap[-3] += self.data.gap[-1]
            self.data.gap[-1] = 0

    def update_error(self):
        """Update error profile and calculate fft."""
        self.data.error[:] = np.cumsum(self.data.gap) - \
            np.cumsum(np.full(self.ncoil, self.total_gap/self.ncoil))
        coef = scipy.fft.rfft(self.data.error.values)
        self.data.fft_real[:] = coef.real
        self.data.fft_imag[:] = coef.imag
        self.data.error_modes[0] = coef[0].real / self.ncoil
        self.data.error_modes[1:] = abs(coef[1:]) / (self.ncoil//2)

    @property
    def fft_coefficents(self):
        """Return fft coefficents."""
        return self.data.fft_real.values + self.data.fft_imag.values*1j

    def plot(self):
        """Plot gap distribution."""
        ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                          sharex=True)[1]
        ax[0].plot([0, self.ncoil], self.uniform_gap * np.ones(2), '-.',
                   color='lightgray', zorder=-10)
        ax[0].bar(range(0, self.ncoil, 2), self.data.gap[::2],
                  label=rf'ssat $2\sigma = {2*self.pdf.ssat.std:1.2f}$')
        ax[0].bar(range(1, self.ncoil, 2), self.data.gap[1::2],
                  label=rf'pit $2\sigma = {2*self.pdf.pit.std:1.2f}$')
        ax[0].bar(range(self.ncoil), self.data.target, color='gray',
                  width=0.2, label='target')

        longwave = scipy.fft.irfft(self.fft_coefficents[:3], n=self.ncoil).real
        label = f'$N_1$={abs(self.fft_coefficents[1])/9:1.2f}\n'
        label += f'$N_2$={abs(self.fft_coefficents[2])/9:1.2f}'
        ax[1].bar(range(self.ncoil), self.data.error, color='lightgray')
        ax[1].plot(range(self.ncoil), self.data.error, '.-', color='C7')
        ax[1].plot(range(self.ncoil), longwave, '-', color='C6')
        ax[1].text(self.ncoil-0.5, longwave[-1], label, va='bottom', color='C6',
                   fontsize='medium')
        ax[-1].xaxis.set_major_locator(
            plt.matplotlib.ticker.MaxNLocator(integer=True))
        ax[-1].set_xticks(range(0, self.ncoil, 2))

        plt.despine()
        ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
        ax[-1].set_xlabel('gap index')
        ax[0].set_ylabel('gap mm')
        ax[1].set_ylabel('error mm')


@dataclass
class Ensemble:
    """Manage ensemble dataset."""

    assembly: Assembly
    samples: int = 20
    label: str = 'TFCgaps_w0'
    data: xarray.DataArray = field(init=False, repr=False)

    def __post_init__(self):
        """Load dataset - solve if not found."""
        try:
            self.load_dataset()
        except FileNotFoundError:
            self.solve()

    @property
    def filename(self):
        """Return filename."""
        return os.path.join(root_dir, f'data/Assembly/{self.label}.nc')

    def check_filename(self):
        """Raise error if file exsists."""
        if os.path.isfile(self.filename):
            raise FileExistsError(f'{self.filename}\n'
                                  'change filename or delete exsisting.')

    def load_dataset(self):
        """Load dataset."""
        self.data = xarray.open_dataset(self.filename)
        self.assembly.attrs = self.data.attrs

    def load_sample(self, sample: int):
        """Load assembly sample."""
        self.assembly.target = self.data.target[sample]
        self.assembly.gap = self.data.gap[sample]
        self.assembly.error = self.data.error[sample]

    def plot_sample(self, sample: int):
        """Plot sample."""
        self.assembly.data = self.data[dict(sample=sample)]
        self.assembly.plot()

    def solve(self):
        """Solve ensemble dataset."""
        self.check_filename()
        self.initialize_dataset()
        tick = clock(self.samples, header='calculating assembly ensemble')
        for i in range(self.samples):
            self.assembly.solve()
            self.data[dict(sample=i)] = self.assembly.data
            tick.tock()
        self.data.to_netcdf(self.filename)

    def initialize_dataset(self):
        """Init DataSet."""
        self.assembly.initialize_dataset()
        self.data = self.assembly.data.expand_dims(
            dim={'sample': range(self.samples)}, axis=0).copy(deep=True)

    def plot_mode(self, wavenumber: int, axes=None):
        if axes is None:
            axes= plt.subplots(1, 1)[1]
        axes.hist(self.data.error_modes[:, wavenumber], rwidth=0.9, bins=21)



if __name__ == '__main__':


    assembly = Assembly(1.33/2, 0.88/2, 36, sead=57)

    ensemble = Ensemble(assembly, 20000)

    index = ensemble.data.error_modes[:, 1].argmax().values
    ensemble.plot_sample(index)

    ensemble.plot_mode(2)
    #tfcage = Assembly()

    #tfcage.plot()
    '''


    pdf = Weibull()
    pdf.update_mean(1)
    pdf.update_std(0.2)

    pdf.plot()
    #pdf.plot()
    #plt.hist(pdf.sample(5000), bins=50, density=True, rwidth=0.8)
    #pdf.plot_array(2, np.array([0.88/2, 1.33/2])**2)
    '''
