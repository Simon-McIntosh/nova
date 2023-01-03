"""Generate synthetic magnetic signals."""
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

import json
import numpy as np
import scipy.signal
from tqdm import tqdm
import xarray

import nova
from nova.database.netcdf import netCDF
from nova.frame.baseplot import Plot
from nova.imas.magnetics import Magnetics


@dataclass
class Generator:
    """Provide random number generator."""

    rng: None | int | np.random.Generator = None

    def __post_init__(self):
        """Initialize random number generator."""
        self.rng = np.random.default_rng(self.rng)
        if hasattr(super(), '__post_init__'):
            super().__post_init__()


@dataclass
class Waveform(Plot, Generator):
    """Provide random number generator.

    Parameters
    ----------
    sample_number: int
        Number of samples to generate

    """

    sample_number: int = 1
    _sample: np.ndarray | None = field(init=False, repr=False, default=None)

    def _generate(self):
        """Return sample."""
        return np.zeros(self.sample_number)

    def generate(self):
        """Return sample."""
        self._sample = self._generate()
        return self._sample

    @property
    def sample(self):
        """Return signal."""
        if self._sample is None:
            return self.generate()
        return self._sample

    def plot_psd(self, axes=None, **kwargs):
        """Plot signals power spectral densty."""
        self.set_axes(axes, '1d')
        frequency, Pxx = scipy.signal.periodogram(self.sample)
        self.axes.semilogy(frequency[1:], Pxx[1:], **kwargs)
        self.axes.set_xlabel('normalized frequency')
        self.axes.set_ylabel('PSD')


@dataclass
class WhiteNoise(Waveform):
    """
    Generate white noise waveforms.

    Parameters
    ----------
    scale: float
        Width of Gaussian (standard deviation).
    """

    scale: float = 0

    def _generate(self):
        return self.rng.normal(0, self.scale, self.sample_number)


@dataclass
class FractalNoise(WhiteNoise):
    r"""
    Generate pink noise via convolution with a :math:`1/f^\alpha` transform.

    Parameters
    ----------
    scale: float
        Width of Gaussian (standard deviation).
    frequency: float
        Scale frequency. Normalized between 0 and 1,
        where 1 is the Nyquist frequency. The default is 0.
    alpha: float
        Slope of :math:`1/f^\alpha` noise. The default is 1.

    """

    normalized_frequency: float = 0
    alpha: float = 1
    _filter: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Build 1/f filter."""
        super().__post_init__()
        sample_frequency = self.fftfreq()
        self._filter = np.ones_like(sample_frequency)
        frequency = self.normalized_frequency * self.sample_number / 2
        self._filter[1:] = \
            (frequency / sample_frequency[1:])**self.alpha

    @property
    def fftlength(self):
        """Return one-sided fft length."""
        if self.sample_number % 2 == 0:  # even
            return int(self.sample_number / 2 + 1)
        return int((self.sample_number + 1) / 2)

    def fftfreq(self):
        """Return one-sided fft frequencies."""
        return abs(scipy.fft.fftfreq(
            self.sample_number, d=1/self.sample_number))[:self.fftlength]

    def _generate(self):
        """Return sample."""
        sample = super()._generate()
        rfft = scipy.fft.rfft(sample)
        return scipy.fft.irfft(self._filter * rfft)


@dataclass
class SignalParameters:
    """Manage base waveform parameters."""

    duration: float
    sample_rate: float
    offset: float = 0
    scale: float = 0
    alpha: float = 1
    frequency: float = 1
    limit: tuple[int, int] = (-5000, 5000)

    def __post_init__(self):
        """Update sample number."""
        self.sample_number = int(self.duration * self.sample_rate)
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    @cached_property
    def time(self) -> np.ndarray:
        """Return waveform time array."""
        return np.linspace(0, self.duration, self.sample_number)

    @property
    def normalized_frequency(self):
        """Return normalized frequency."""
        return self.frequency / (self.sample_rate / 2)

    @property
    def noise_attrs(self):
        """Return noise attributes."""
        return dict(scale=self.scale,
                    normalized_frequency=self.normalized_frequency,
                    alpha=self.alpha)

    @property
    def signal_attrs(self):
        """Return signal attributes."""
        return dict(offset=self.offset, scale=self.scale,
                    frequency=self.frequency, alpha=self.alpha)


@dataclass
class TypeCast:
    """Perform type casting between voltage signals and DAQ targets."""

    dtype: str | np.dtype
    signal_limit: tuple[str]

    def __post_init__(self):
        """Calculate offset and multiplier."""
        self.dtype_limit = (np.iinfo(self.dtype).min, np.iinfo(self.dtype).max)
        signal = self.measure(self.signal_limit)
        target = self.measure(self.dtype_limit)
        self.multiplier = signal['width'] / target['width']
        self.offset = self.multiplier * (signal['center'] - target['center'])

    def measure(self, limit):
        """Return limit width and center."""
        width = limit[1] - limit[0]
        half_width = width / 2
        if isinstance(limit[0], int):
            half_width = np.ceil(half_width)
        center = limit[0] + half_width
        return dict(width=width, center=center)

    def to_dtype(self, signal: np.ndarray):
        """Return signal cast to dtype target."""
        return (signal - self.offset) / self.multiplier

    def from_dtype(self, target: np.ndarray):
        """Return dtype target cast to signal."""
        return self.offset + self.multiplier * target


@dataclass
class Signal(netCDF, Waveform, SignalParameters):
    """Manage base waveform parameters."""

    dirname: str = '.magnetics_6MHz'
    magnetics: Magnetics = field(default_factory=Magnetics)
    dtype: str | np.dtype = 'int32'
    _intergal: np.ndarray | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize component waveforms."""
        super().__post_init__()
        self.cast = TypeCast(self.dtype, self.limit)
        self.noise = FractalNoise(self.rng, self.sample_number,
                                  **self.noise_attrs)

    @cached_property
    def metadata(self):
        """Return instance metadata."""
        return dict(description='Synthetic magnetics waveform generated with '
                                'a fractal noise model.',
                    creation_date=datetime.today().strftime('%d-%m-%Y'),
                    duration=self.duration,
                    sample_rate=self.sample_rate,
                    signal_minimum=-self.limit[0],
                    signal_maximum=self.limit[1],
                    units=json.dumps(dict(time='s', frequency='Hz',
                                          proportional='V',
                                          intergral='Vs')),
                    code=json.dumps(self.code_attrs),
                    magnetics_ids=json.dumps(self.magnetics.ids_attrs),
                    noise=json.dumps(self.signal_attrs))

    @property
    def code_attrs(self):
        """Return code attributes."""
        return dict(name='nova', version=nova.__version__)

    def _generate(self):
        """Return sample."""
        return self.noise.generate() + self.rng.normal(0, self.offset)

    def generate(self):
        """Return sample."""
        self._sample = self._generate()
        self._intergral = np.cumsum(self._sample) / self.sample_rate
        return self._sample

    @property
    def intergral(self):
        """Return signal."""
        if self._intergral is None:
            self.generate()
        return self._intergral

    def initialize_dataarray(self):
        """Initialize DataArray for single diagnostic."""
        self.data = xarray.DataArray(
            np.zeros((2, self.sample_number), dtype=self.dtype),
            coords=dict(
                time=self.time.astype(np.float32),
                signal=['proportional', 'intergral'],
                signal_offset=self.cast.offset,
                signal_multiplier=self.cast.multiplier),
            dims=('signal', 'time'),
            attrs=self.metadata)

    def update_dataarray(self, name):
        """Update DataArray with sample / intergral data."""
        self.data.name = name
        self.data[0] = self.cast.to_dtype(self.sample)
        self.data[1] = self.cast.to_dtype(self.intergral)
        self.data.attrs['diagnostic'] = \
            self.magnetics['frame'].loc[name, :].to_json()

    def build(self):
        """Store samples to netCDF file."""
        self.initialize_dataarray()
        for name in tqdm(self.magnetics['frame'].index):
            self.generate()
            self.update_dataarray(name)
            self.filename = name + '.nc'
            self.store('w')

    def plot(self, axes=None):
        """Plot waveform."""
        import matplotlib.pyplot as plt
        axes = plt.subplots(2, 1, sharex=True)[1]
        for _axes in axes:
            self.set_axes(_axes, '1d')
        axes[0].plot(self.time, self.sample)
        axes[1].plot(self.time, self.intergral, 'C1')
        axes[0].set_ylabel(r'$V$')
        axes[1].set_xlabel('time s')
        axes[1].set_ylabel(r'$Vs$')


if __name__ == '__main__':

    '''
    number = 5000
    rng = Generator(2025).rng
    white = WhiteNoise(rng, number, scale=0.1)
    white.plot_psd(label='white noise')

    fractal = FractalNoise(rng, number, scale=0.1,
                           normalized_frequency=0.05, alpha=1)
    fractal.plot_psd(white.axes, label='fractal noise')
    white.axes.legend()

    '''
    dirname = '/mnt/ITER/mcintos/magnetics/data'
    # hostname = 'sdcc-login01.iter.org'
    hostname = 'access-xpoz.codac.iter.org'

    signal = Signal(5, 2e5, offset=0.005, scale=0.1, frequency=10,
                    alpha=1, rng=2025, dirname=dirname, hostname=hostname)
    signal.build()
    #print(signal.build_array(signal.magnetics['frame'].index[0]))


'''


@dataclass
class Signal(Generator):

    bit_depth: int = 16
    signal_width: float = 5
    cutoff: float | None = 1e6

    def sample(self, waveform):
        """Return sampled waveform."""
        if cutoff is not None:
            pass


if __name__ == '__main__':

    signal = Signal(11, 2e6, 0.3, 0.05)

    print(signal().std())
'''
