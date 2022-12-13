"""Generate synthetic magnetic signals."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np
import scipy.signal

from nova.frame.baseplot import Plot


@dataclass
class Noise(Plot):
    """
    Generate noise waveforms.

    Parameters
    ----------
    sample_number: int
        Number of samples to generate
    offest: float
        Location of Gaussian.
    scale: float
        Width of Gaussian (standard deviation).
    """

    sample_number: int
    offset: float = 0
    scale: float = 0
    rng: Optional[np.random.Generator] = field(
        default_factory=np.random.default_rng)
    _signal: np.ndarray | None = field(init=False, repr=False, default=None)

    def generate(self):
        """Return noise waveform."""
        self._signal = self.rng.normal(self.offset, self.scale,
                                       self.sample_number)
        return self._signal

    @property
    def signal(self):
        """Return signal."""
        if self._signal is None:
            self.generate()
        return self._signal

    def plot_psd(self, axes=None, **kwargs):
        """Plot signals power spectral densty."""
        self.set_axes(axes, '1d')
        self.axes.semilogy(*scipy.signal.periodogram(self.signal), **kwargs)
        self.axes.set_ylim([1e-8, 1e0])
        self.axes.set_xlabel('normalized frequency')
        self.axes.set_ylabel('PSD')


@dataclass
class Lowpass(Noise):
    """
    Manage low-pass filters.

    Parameters
    ----------
    normalized_cutoff: float
        Critical frequency at which gain drops below -3dB of passband.
        Normalized between 0 and 1, where 1 is the Nyquist frequency.

    """

    normalized_cutoff: Optional[float] = None
    _filter: dict[str, float | np.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        """Build lowpass filter."""
        super().__post_init__()
        if self.normalized_cutoff is not None:
            self._filter = dict(
                zip('ab', scipy.signal.butter(1, self.normalized_cutoff)))
            self._filter['inital_state'] = \
                scipy.signal.lfilter_zi(self._filter['a'], self._filter['b'])

    def generate(self):
        """Return signal with lowpass filter applied."""
        super().generate()
        if self.normalized_cutoff is None:
            return self.signal
        self._signal, _ = scipy.signal.lfilter(
            self._filter['a'], self._filter['b'], self.signal,
            zi=self._filter['inital_state']*self.signal[0])
        return self._signal


if __name__ == '__main__':

    noise = Noise(5000, 1, 0.1)
    noise.plot_psd(label='broardband')

    lowpass = Lowpass(5000, 1, 0.1, normalized_cutoff=0.25)
    lowpass.plot_psd(noise.axes, label='lowpass')

    noise.axes.legend()

'''

@dataclass
class Waveform:
    """Manage waveform attributes."""

    duration: float
    sample_rate: float
    low_frequency_noise: float = 0
    low_frequency_cutoff: float = 1
    broardband_noise: float = 0
    broardband_cutoff: float = 1
    offset_std: float = 0

    @property
    def sample_number(self) -> int:
        """Return sample number."""
        return int(self.duration * self.sample_rate)

    @cached_property
    def time(self) -> np.ndarray:
        """Return waveform time array."""
        return np.linspace(0, self.duration, self.sample_number)


@dataclass
class Generator(Waveform):
    """Generate waveform with sampled white noise and offset."""

    sead: int = 2025
    rng: np.random.Generator = field(init=False, repr=False,
                                     default_factory=np.random.default_rng)

    def __post_init__(self):
        """Initialize random number generator."""
        self.rng = np.random.default_rng(self.sead)

    def sample(self):
        """Return sampled waveform."""
        noise = self.rng.normal(0, self.noise_std, self.sample_number)
        offset = self.rng.normal(0, self.offset_std, 1)
        return noise + offset


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
