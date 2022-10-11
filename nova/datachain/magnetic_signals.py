"""Generate synthetic magnetic signals."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


@dataclass
class Noise:
    """Generate low-pass noise waveforms."""

    sample_number: int
    noise_std: float = 0
    noise_cufoff: float = 0


@dataclass
class Waveform:
    """Manage waveform attributes."""

    duration: float
    sample_rate: float
    low_frequency_noise: float = 0
    low_frequency_cutoff: float = 1
    broardband_noise: float = 0
    broardband_cutoff: float = 0
    offset_std: float = 0

    @property
    def sample_number(self) -> int:
        """Return sample number."""
        return int(self.duration * self.sample_rate)

    @cached_property
    def time(self) -> np.ndarray:
        """Return waveform time array."""
        return np.linspace(0, self.duration, self.samples)


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



if __name__ == '__main__':

    signal = Signal(11, 2e6, 0.3, 0.05)

    print(signal().std())
