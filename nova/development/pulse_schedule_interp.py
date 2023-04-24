"""Short script to demonstrate non-homogeneous interpolation errors."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.fft as fft
from scipy.interpolate import interp1d

from nova.frame.baseplot import Plot


@dataclass
class IdsInterp(Plot):
    """Interpolation demonstrator."""

    number: int = 5
    time_number: int = 100
    time_range: tuple[int | float, int | float] = (0, 5)
    rng_sead: int = 2025
    data: dict[str, dict[str, np.ndarray]] = field(init=False, repr=False,
                                                   default_factory=dict)

    def __post_init__(self):
        """Construct random number generator."""
        self.rng = np.random.default_rng(2025)
        self.build_source()
        self.build_uniform()
        self.build_random()
        super().__post_init__()

    @cached_property
    def time(self):
        """Return time vector."""
        return np.linspace(*self.time_range, self.time_number)

    @cached_property
    def uniform_time(self):
        """Return uniform time vector."""
        return np.linspace(*self.time_range, self.number)

    @cached_property
    def elongation_time(self):
        """Return random elongation time."""
        return np.sort(self.rng.uniform(*self.time_range, self.number))

    @cached_property
    def minor_radius_time(self):
        """Return random minor radius time."""
        return np.sort(self.rng.uniform(*self.time_range, self.number))

    @cached_property
    def elongation(self):
        """Return elongation interpolator."""
        return interp1d(self.time, self.data['source']['elongation'])

    @cached_property
    def minor_radius(self):
        """Return elongation interpolator."""
        return interp1d(self.time, self.data['source']['minor_radius'])

    def coef(self, modes: int, loc=0.0, scale=1.0):
        """Return Fourier coefficents."""
        normal = self.rng.normal(loc, scale, size=(modes, 2))
        return normal[:, 0] + normal[:, 1]*1j

    def build_source(self):
        """Build source waveforms."""
        elongation = 1.8 + fft.irfft(self.coef(5, scale=2), self.time_number)
        minor_radius = 1.4 + fft.irfft(self.coef(5, scale=2), self.time_number)
        major_radius = elongation * minor_radius
        self.data['source'] = {'elongation': elongation,
                               'minor_radius': minor_radius,
                               'major_radius': major_radius}

    def build_uniform(self):
        """Build uniform interpolants."""
        elongation = self.elongation(self.uniform_time)
        minor_radius = self.minor_radius(self.uniform_time)
        major_radius = elongation * minor_radius
        self.data['uniform'] = {'elongation': elongation,
                                'minor_radius': minor_radius,
                                'major_radius': major_radius}

    def build_random(self):
        """Build non-homogeneous interpolants."""
        elongation = self.elongation(self.elongation_time)
        minor_radius = self.minor_radius(self.minor_radius_time)

        _elongation = interp1d(self.elongation_time, elongation,
                               fill_value='extrapolate')(self.uniform_time)
        _minor_radius = interp1d(self.minor_radius_time, minor_radius,
                                 fill_value='extrapolate')(self.uniform_time)
        major_radius = _elongation * _minor_radius
        self.data['random'] = {'elongation': elongation,
                               'minor_radius': minor_radius,
                               'major_radius': major_radius}

    def plot(self):
        """Plot waveforms."""
        self.set_axes('1d')
        for attr, value in self.data['source'].items():
            self.axes.plot(self.time, value, label=attr)

        for i, (attr, value) in enumerate(self.data['uniform'].items()):
            self.axes.plot(self.uniform_time, value, f'C{i}o')

        self.axes.plot(self.elongation_time,
                       self.data['random']['elongation'], 'C0d')
        self.axes.plot(self.minor_radius_time,
                       self.data['random']['minor_radius'], 'C1d')
        self.axes.plot(self.uniform_time,
                       self.data['random']['major_radius'], 'C2d')
        self.axes.plot([], 'ko', color='gray', label='homogeneous')
        self.axes.plot([], 'kd', color='gray', label='non-homogeneous')

        self.axes.legend(ncol=1)

        self.axes.set_xlabel('time, s')
        self.axes.set_ylabel('value')

if __name__ == '__main__':

    interp = IdsInterp()
    interp.plot()
