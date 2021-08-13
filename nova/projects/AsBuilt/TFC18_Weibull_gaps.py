"""Methods to aid computational assembly studies for the TF cage."""
from abc import ABC, abstractmethod
from collections import Iterable
import datetime
from dataclasses import dataclass, field
import os
from typing import Union
from warnings import warn

import fortranformat
import numpy as np
import pandas
import scipy.fft
import scipy.optimize
import scipy.special
import scipy.stats
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
                  rf'$\mu={self.mean:1.1f}$, $\sigma={self.std:1.2f}$')

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
    """Manage Weibull PDF - construct PDF from input mean and variance."""

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
        self.inpit = self.pdf(sead=self.sead)

    def update_mean(self, ssat, inpit):
        """Update pdf means."""
        self.ssat.update_mean(ssat)
        self.inpit.update_mean(inpit)

    def update_std(self, ssat, inpit):
        """Update pdf standard deviations."""
        self.ssat.update_std(ssat)
        self.inpit.update_std(inpit)


@dataclass
class Assembly:
    """Generate single assembly trial."""

    ssat_std: float = 1.5
    pit_std: float = 1.5
    total_gap: float = 36
    update_target: dict[str, bool] = field(
        default=lambda: dict(ssat=True, inpit=True))
    update_ssat: bool = False
    ncoil: int = 18
    sead: int = 2025
    pdf: PDFs = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Init gaps."""
        self.uniform_gap = self.total_gap/self.ncoil
        self.pdf = PDFs(sead=self.sead)
        self.pdf.update_std(self.ssat_std, self.pit_std)
        self.pdf.update_mean(self.uniform_gap, self.uniform_gap)

    def solve(self):
        """Solve TF assembly trial - pair, pair, pit, pair, pit,...."""
        self.initialize_dataset()
        self.data.gap[0] = self.sample(self.uniform_gap, 'ssat')
        for i in range(1, 9):
            self.set_target(2*i, self.update_target['ssat'])
            self.data.gap[2*i] = self.sample(
                self.data.target[2*i].values, 'ssat',
                self.update_target['ssat'])
            self.set_target(2*i-1, self.update_target['inpit'])
            self.data.gap[2*i-1] = self.sample(
                self.data.target[2*i-1].values, 'inpit',
                self.update_target['inpit'])
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
        return dict(ssat_std=self.pdf.ssat.std, pit_std=self.pdf.inpit.std,
                    total_gap=self.total_gap,
                    update_ssat=int(self.update_target['ssat']),
                    update_inpit=int(self.update_target['inpit']),
                    sead=self.sead)

    @attrs.setter
    def attrs(self, attrs):
        self.pdf.update_std(attrs['ssat_std'], attrs['pit_std'])
        self.total_gap = attrs['total_gap']
        self.sead = attrs['sead']
        if 'update_target' in attrs:
            update = bool(attrs['update_target'])
            self.update_target = dict(ssat=update, inpit=update)
        else:
            self.update_target['ssat'] = bool(attrs.get('update_ssat', 1))
            self.update_traget['inpit'] = bool(attrs.get('update_inpit', 1))

    def sample(self, mean, stage: str, update_target=True):
        """Return sample from PDF."""
        if mean < 0.01:
            mean = 0.01
        pdf = getattr(self.pdf, stage)
        if update_target:
            pdf.update_mean(mean)
        return pdf.sample(size=1)[0]

    def get_error(self, pair_index: int):
        """Return running assembly error."""
        return np.sum(self.data.gap[:2*pair_index+1]) - \
            (2*pair_index + 1) * self.uniform_gap

    def set_target(self, index: int, update_target: bool):
        """Set taget gap."""
        if not update_target:
            return
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

    def plot_gaps(self, plot_error, axes=None):
        """Plot gap histogram."""
        if axes is None:
            axes = plt.subplots(1, 1)[1]
        axes.plot([0, self.ncoil], self.uniform_gap * np.ones(2), '-.',
                  color='lightgray', zorder=-10)
        axes.bar(range(0, self.ncoil, 2), self.data.gap[::2],
                 label=rf'ssat $2\sigma = {2*self.pdf.ssat.std:1.2f}$')
        axes.bar(range(1, self.ncoil, 2), self.data.gap[1::2],
                 label=rf'in-pit $2\sigma = {2*self.pdf.inpit.std:1.2f}$')
        axes.bar(range(self.ncoil), self.data.target, color='gray',
                 width=0.2, label='target')
        self.set_gap_ticks(axes)
        anchor = (0.5, 1.25) if plot_error else (0.5, 1.18)
        axes.legend(loc='upper center', bbox_to_anchor=anchor, ncol=3)
        axes.set_ylabel('gap mm')
        if not plot_error:
            axes.set_xlabel('gap index')
        plt.despine()

    def plot_error(self, wavenumber, axes=None):
        """Plot gap error waveform."""
        if axes is None:
            axes = plt.subplots(1, 1)[1]
        coef = np.zeros(self.data.dims['wavenumber'], dtype=complex)
        coef[0] = self.fft_coefficents[0]
        if not isinstance(wavenumber, Iterable):
            wavenumber = [wavenumber]
        label = ''
        for wn in wavenumber:
            coef[wn] = self.fft_coefficents[wn]
            if label:
                label += '\n'
            label += f'$k_{wn}$='
            label += f'{self.data.error_modes[wn].values:1.2f}'

        nifft = 18
        ifft = scipy.fft.irfft(coef, n=nifft, norm='backward').real
        axes.bar(range(self.ncoil), self.data.error, color='lightgray')
        axes.plot(range(self.ncoil), self.data.error, '.-', color='C7')
        axes.plot(np.linspace(0, self.ncoil-1, nifft), ifft, '-', color='C6')
        axes.text(self.ncoil-0.5, ifft[-1], label, va='center',
                  color='C6', fontsize='medium')
        self.set_gap_ticks(axes)
        axes.set_xlabel('gap index')
        axes.set_ylabel('error mm')
        plt.despine()

    def set_gap_ticks(self, axes):
        """Set interger gap ticks."""
        axes.xaxis.set_major_locator(
            plt.matplotlib.ticker.MaxNLocator(integer=True))
        axes.set_xticks(range(0, self.ncoil, 2))

    def get_axes(self, plot_error):
        """Return plot axes."""
        if plot_error:  # include error subplot
            return plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)[1]
        return plt.subplots(1, 1)[1:2]

    def plot(self, wavenumber=1, plot_error=True):
        """Plot gap distribution."""
        axes = self.get_axes(plot_error)
        self.plot_gaps(plot_error, axes[0])
        if plot_error:
            self.plot_error(wavenumber, axes[-1])


@dataclass
class Ensemble:
    """Manage ensemble dataset."""

    label: str = 'TFCgaps_w1'
    assembly: Assembly = field(default_factory=Assembly)
    samples: int = 20
    data: xarray.DataArray = field(init=False, repr=False)

    def __post_init__(self):
        """Load dataset - solve if not found."""
        try:
            self.load_dataset()
        except FileNotFoundError:
            self.solve()

    @property
    def filepath(self):
        """Return filepath."""
        return os.path.join(root_dir, 'data/Assembly')

    @property
    def filename(self):
        """Return filename."""
        return os.path.join(self.filepath, f'{self.label}.nc')

    def check_filename(self):
        """Raise error if file exsists."""
        if os.path.isfile(self.filename):
            raise FileExistsError(f'{self.filename}\n'
                                  'change filename or delete exsisting.')

    def load_dataset(self):
        """Load dataset."""
        self.data = xarray.open_dataset(self.filename)
        self.assembly.attrs = self.data.attrs
        self.samples = self.data.dims['sample']
        self.check_closure(drop=True)

    def locate(self, amplitude, wavenumber):
        """Return sample indexed by amplitude of Forier mode."""
        error_mode = self.data.error_modes[:, wavenumber].values
        return np.argmin(abs(error_mode - amplitude))

    def get_sample(self, sample: Union[int, str], factor=None, wavenumber=1):
        """Return sample index mean, mode, sigma."""
        if not isinstance(sample, str):
            return sample
        if sample == 'sample':
            return np.where(self.data.sample == factor)[0][0]
        if sample == 'max':
            return self.data.error_modes[:, wavenumber].argmax().values
        crv = self.get_crv(wavenumber)
        if sample in ['mean', 'median']:
            return self.locate(getattr(crv, sample)(), wavenumber)
        if sample == 'mode':
            mode = self.get_mode(crv)
            return self.locate(mode, wavenumber)
        if sample == 'sigma':
            mean = crv.mean()
            std = crv.std()
            return self.locate(mean + factor*std, wavenumber)
        if sample == 'quartile':
            quartile = crv.ppf(factor)
            return self.locate(quartile, wavenumber)
        raise TypeError(f'sample {sample} not type(int) or in '
                        '[max, mean, mode, median, sigma, quartile]')

    def load_sample_data(self, sample, factor=None, wavenumber=1):
        """Load assembly sample data."""
        sample = self.get_sample(sample, factor, wavenumber)
        self.assembly.data = self.data[dict(sample=sample)]

    def plot_sample(self, sample, factor=None, wavenumber=1, plot_error=True):
        """Plot sample."""
        self.load_sample_data(sample, factor, wavenumber)
        self.assembly.plot(wavenumber=wavenumber, plot_error=plot_error)

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

    def get_crv(self, wavenumber):
        """Return continious random variable."""
        error_mode = self.data.error_modes[:, wavenumber]
        coef = scipy.stats.weibull_min.fit(error_mode, floc=0)
        return scipy.stats.weibull_min(*coef)

    def get_mode(self, crv, mean=None):
        """Return distribution mode."""
        if mean is None:
            mean = crv.mean()
        return scipy.optimize.minimize(lambda x: -crv.pdf(x), mean).x[0]

    def plot_wavenumber(self, wavenumber: int, axes=None, labels=True,
                        legend=True):
        """Plot fft mode PDF."""
        if axes is None:
            axes = plt.subplots(1, 1)[1]
        wavenumber = self.data.wavenumber[wavenumber].values
        error_mode = self.data.error_modes[:, wavenumber]
        crv = self.get_crv(wavenumber)
        mean, std = crv.mean(), crv.std()
        quartile = mean + 3*std
        mode = self.get_mode(crv, mean)
        gap = np.linspace(np.min(error_mode), np.max(error_mode))
        axes.hist(self.data.error_modes[:, wavenumber], rwidth=1, bins=51,
                  density=True, label=f'$k={wavenumber}$')
        axes.plot(gap, crv.pdf(gap), color='C3', lw=2)
        axes.plot(mode, crv.pdf(mode), 'ko', ms=6)
        axes.text(mode, crv.pdf(mode), f'{mode:1.2f}',
                  va='bottom', ha='left')

        axes.plot(quartile, crv.pdf(quartile), 'ko', ms=6)
        axes.text(quartile, crv.pdf(quartile), f'{quartile:1.2f}',
                  va='bottom', ha='left')
        axes.axis('off')

        if labels:
            axes.set_xlabel('Mode amplitude $A$ mm')
            axes.set_ylabel('P$(A)$')
        if legend:
            axes.legend(loc=1)
        else:
            axes.text(0.8, 0.8, f'$k_{wavenumber}$',
                      transform=axes.transAxes, va='center', ha='center',
                      bbox=dict(boxstyle="round", fc='0.9'), fontsize='small')

    def plot_wavenumber_array(self):
        """Plot array of fft mode pdfs."""
        axes = plt.subplots(3, 3,
                            sharex=False, sharey=False)[1].reshape(1, -1)[0]
        for wavenumber in self.data.wavenumber[1:]:
            self.plot_wavenumber(wavenumber, axes[wavenumber-1],
                                 labels=False, legend=False)

    def check_closure(self, drop=True):
        """Check TF closure - if drop, remove negitive gaps."""
        negative_gap = self.data.gap.min(axis=1).values < 0
        nfailures = sum(negative_gap)
        if nfailures > 0:
            warn(f'closure failures detected {nfailures}/{self.samples}, '
                 f'{100*nfailures / self.samples}%')
            if drop:
                self.data = \
                    self.data.drop_sel(sample=np.where(negative_gap)[0])


@dataclass
class Scenario(Ensemble):
    """Extract gap distributions from ensemble database(s)."""

    index: int = 0
    database: list[str] = field(
        default_factory=lambda: ['TFCgaps_c', 'TFCgaps_a'])
    gaps: pandas.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        """Load database."""
        self.load_dataset()
        self.initialize_gaps()

    def load_dataset(self):
        """Extend ensemble load dataset."""
        self.label = self.name  # set ensemble label
        super().load_dataset()

    @property
    def name(self):
        """Return database name."""
        return self.database[self.index]

    @name.setter
    def name(self, name):
        """Set database name - reload database if name != self.name."""
        if isinstance(name, int):
            name = self.database[name]
        if name != self.name:  # reload
            self.index = self.database.index(name)
            self.load_dataset()

    def initialize_gaps(self):
        """Initialize gap table."""
        self.gaps = pandas.DataFrame(index=self.data.gap_index.values)

    def append_trial(self, columns, wavenumber=1):
        """Append trail column to gap table."""
        if not isinstance(wavenumber, Iterable):
            wavenumber = np.full(len(columns), wavenumber)
        for i, scenario in enumerate(columns):
            self.load_sample_data(*columns[scenario], wavenumber=wavenumber[i])
            self.gaps[scenario] = self.assembly.data.gap.values

    def append_mode(self, wavenumber, amplitude=1, phase=0, label='k'):
        """Append Fourier mode to gaps table."""
        if not isinstance(wavenumber, Iterable):
            wavenumber = [wavenumber]
        if phase != 0:
            wavenumber = wavenumber[1:]  # skip zero mode
            label += 'p'
        for wn in wavenumber:
            error = self.generate_error(wn, amplitude, phase)
            self.gaps[f'{label}{wn}'] = self.generate_gaps(error, wn)

    def generate_error(self, wavenumber, amplitude=1, phase=0):
        """Return Fourier mode error waveform."""
        nfft = self.data.dims['gap_index']
        if wavenumber == 0 or (wavenumber == nfft//2 and nfft % 2 == 0):
            amplitude *= 2  # repeated Nyquist coefficent for even waveforms

        coef = np.full(nfft//2 + 1, 0, dtype=complex)
        coef[wavenumber] = amplitude * (nfft//2) * np.exp(1j*phase)
        error = scipy.fft.irfft(coef, n=nfft)
        return error

    def generate_gaps(self, error, wavenumber):
        """Return Fourier mode gap waveform calculated from error waveform."""
        ncoil = len(error)
        uniform_gap = self.data.total_gap / ncoil
        gaps = error + np.cumsum(np.full(ncoil, uniform_gap))
        gaps[1:] -= gaps[:-1]
        gaps[0] += self.data.total_gap - np.sum(gaps)
        adjust_error = np.cumsum(gaps) - np.cumsum(np.full(ncoil, uniform_gap))
        coef = scipy.fft.rfft(adjust_error)
        coef[0] = 0
        coef[wavenumber] = 0
        assert np.isclose(np.sum(gaps), self.data.total_gap)  # total gap
        assert np.isclose(np.sum(abs(coef)), 0)  # single mode
        return gaps

    def plot_gap_array(self, modes=range(1, 5), phase=0):
        """Plot array of error waveforms for all Fourier modes."""
        nfft = self.data.dims['gap_index']
        axes = plt.subplots(len(modes), 2, sharex=True, sharey='col')[1]
        for i, k in enumerate(modes):
            error = self.generate_error(k, phase=i*phase)
            gap = self.generate_gaps(error, k)
            axes[i, 0].bar(range(nfft), error, color=f'C{k%10}')
            axes[i, 1].bar(range(nfft), gap, color=f'C{k%10}')
            axes[i, 0].set_ylabel(f'$k_{k}$ mm')
        axes[0, 0].set_title('placement error')
        axes[0, 1].set_title('gap waveform')
        for j in range(2):
            axes[-1, j].set_xlabel('gap index')
        plt.despine()

    def build(self, target: list[str] = '', phase_shift=False):
        """Build gap table."""
        self.initialize_gaps()
        # append mode and 3sigma assembly trials
        for stragergy in target:  # constant target, adaptive target
            self.name = f'TFCgaps_{stragergy}'
            self.append_trial({f'{stragergy}1': ('mode',),
                               f'{stragergy}2': ('sigma', 3)})
        # append Fourier modes
        phase = np.pi/18 if phase_shift else 0
        self.append_mode(self.data.wavenumber.values, phase=phase)
        self.to_clipboard()

    def to_clipboard(self):
        """Copy gaps table to clipboard."""
        self.gaps.to_clipboard(float_format='{0:1.3f}'.format, index=False)

    def file_header(self, filename):
        """Return output file header."""
        title = ('Candidate TFC assembly gap waveforms proposed and '
                 'analyized by SCOD.\t')
        scenario = dict(
            constant_adaptive_fourier='c*: constant target sampled at '
                                      'distribution mode and 3-sigma.\t'
                                      'a*: adaptive target (ssat & inpit) '
                                      'sampled at '
                                      'distribution mode and 3-sigma.\t'
                                      'k*: unit amplitude Fourier modes.\t',
            constant_ssat_fourier_shift='cs*: ssat constant target & '
                                        'inpit adaptive target '
                                        ' sampled at '
                                        'distribution mode and 3-sigma.\t'
                                        'kp*: unit amplitude Fourier modes '
                                        'phase shifted by pi/18.\t')
        identity = (f'Created on {datetime.datetime.today():%d/%m/%Y}\t'
                    '@author: Simon McIntosh\t @email: mcintos@iter.org')
        return title + scenario[filename] + identity + '\n\n'

    def to_ansys(self, filename: str):
        """Write gap data to ansys txt file."""
        filepath = os.path.join(self.filepath, f'{filename}.txt')
        gaps = pandas.DataFrame(index=self.data.gap_index.values)
        gaps['rid'] = 2001+self.data.gap_index
        gaps = pandas.concat([gaps, self.gaps], axis=1)

        ngaps = len(self.gaps)
        header_format = fortranformat.FortranRecordWriter(
            f'(2A8, {ngaps}A10)')
        gap_format = fortranformat.FortranRecordWriter(
            f'(2F8.1, {ngaps}F10.3)')

        with open(filepath, 'w') as file:
            file.write(self.file_header(filename))
            gaps_txt = gaps.to_csv(
                    sep='\t', line_terminator='\n',
                    float_format='{0:1.3f}'.format).split('\n')
            file.write(header_format.write(gaps_txt[0].split('\t')) + '\n')
            for line in gaps_txt[1:-1]:
                file.write(gap_format.write(
                    [float(gap) for gap in line.split('\t')]) + '\n')
            # gaps.to_csv(file, sep='\t', line_terminator='\n',
            #             float_format='{0:1.3f}'.format)

    def rebuild(self, sample, factor=None, wavenumber=1):
        """Reconstruct error from zero-phase Fourier modes."""
        self.load_sample_data(sample, factor, wavenumber)
        data = self.assembly.data

        coef = scipy.fft.rfft(data.error.values)

        ncoil = len(data.error)

        modes = range(1, 10)

        error = np.full(ncoil, coef[0].real / 18, dtype=float)  # k0
        for wn in modes:
            phase = np.angle(coef[wn])
            wavelength = ncoil / wn
            shift = wavelength*phase / (2*np.pi) + wavelength*np.arange(wn)

            index = np.argmin(shift % 1)
            shift = -int(np.round(shift[index]))
            amplitude = np.abs(coef[wn]) / (ncoil//2)
            if wn == 9:
                amplitude /= 2

            component = self.generate_error(wn, amplitude)
            error += np.roll(component, shift)

        reduced_coef = np.full(ncoil//2 + 1, 0, dtype=complex)
        reduced_coef[0] = coef[0]
        reduced_coef[modes] = coef[modes]
        reduced_error = scipy.fft.irfft(reduced_coef)

        plt.bar(range(ncoil), data.error)
        plt.bar(range(ncoil), reduced_error, width=0.7)
        plt.bar(range(ncoil), error, width=0.5)

        print(np.linalg.norm(error-reduced_error))
        print(np.max(error))
        print(100*np.linalg.norm(error-reduced_error) / np.max(error))


if __name__ == '__main__':

    scn = Scenario()

    #scn.rebuild('sample', 37122)  # three sigma
    #scn.rebuild('sample', 71348)  # mode


    scn.build(target='ca', phase_shift=False)
    scn.to_ansys('constant_adaptive_fourier')

    #plt.set_aspect(1.1)
    #scn.plot_gap_array(range(5, 10), phase=np.pi/18)

    #wavenumber = 1
    #error = scn.generate_error(wavenumber)
    #scn.generate_gaps(error, wavenumber)

    '''
    assembly = Assembly(1.33/2, 0.88/2, 36, sead=2025,
                        update_target=dict(ssat=False, inpit=True))
    #assembly.solve()
    #assembly.plot()

    ensemble = Ensemble('TFCgaps_ap', assembly, samples=100000)
    ensemble.plot_sample('sigma', 3, wavenumber=1, plot_error=True)

    ensemble.load_sample_data('mean')
    ensemble.assembly.plot_error(range(4))
    ensemble.plot_wavenumber_array()
    '''

    #pdf = Weibull()
    #pdf.plot()
    #plt.hist(pdf.sample(5000), bins=50, density=True, rwidth=0.8)
    #pdf.plot_array(2, np.array([0.88/2, 1.33/2])**2)
