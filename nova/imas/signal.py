"""Manage signal methods."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from rdp import rdp
from sklearn.preprocessing import minmax_scale
import scipy.signal
import xarray

from nova.frame.baseplot import Plot
from nova.imas.equilibrium import Equilibrium


@dataclass
class Select:
    """Select subset of data based on coordinate."""

    data: xarray.Dataset = field(repr=False)

    def attrs(self, coord: str):
        """Return attribute list selected according to coord."""
        if coord[0] == '~':
            return [attr for attr, value in self.data.items()
                    if coord[1:] not in value.coords]
        return [attr for attr, value in self.data.items()
                if coord in value.coords]

    def select(self, coord: str, data=None):
        """Return data subset including all data variables with coord."""
        if data is None:
            data = self.data
        return data[self.attrs(coord)]


@dataclass
class Defeature:
    """Defeature dataset using a clustered RDP algoritum."""

    data: xarray.Dataset = field(repr=False)
    epsilon: float = 1e-3
    features: list[str] | None = None

    def __post_init__(self):
        """Extract feature list if None."""
        self.check_features()

    def check_features(self):
        """Check features, update it None."""
        match self.features:
            case None:
                self.features = [attr for attr, value in self.data.items()
                                 if value.coords.dims == ('time',)]
            case list():
                assert np.all(self.data[attr].coords.dims == ('time',)
                              for attr in self.features)
            case _:
                raise TypeError(f'features {self.feaures} not list')

    @cached_property
    def time(self):
        """Return time vector with shape (n, 1)."""
        return np.copy(self.data.time.data[:, np.newaxis])

    def maxmin(self, array: np.ndarray):
        """Return minmax scaled array."""

        return minmax_scale(self['sample'][self.features].to_array(), axis=1).T

    def extract(self):
        """Extract turning points from a single attribute waveform."""

        #time =
        time /= time[-1] - time[0]
        array = np.append(time, self.maxmin_scale(), axis=1)
        mask = rdp(array, self.epsilon, return_mask=True)
        self['rdp'] = self['sample'].sel({'time': mask})


@dataclass
class Sample(Plot, Select):
    """Re-sample signal using a polyphase filter."""

    data: xarray.Dataset = field(repr=False)
    delta: int | float = -100
    savgol: tuple[int, int] | None = (10, 1)
    epsilon: float = 0.01
    features: list[str] = field(default_factory=lambda: [
        'elongation', 'ip'])
    profile_data: dict[str, xarray.Dataset] = field(default_factory=dict)

    def __post_init__(self):
        """Interpolate data onto uniform time-base and resample."""
        self.data = self.clip('li_3', 0)
        self.interpolate()
        self.resample()
        #self.defeature()
        super().__post_init__()

    def __getitem__(self, attr: str) -> xarray.Dataset:
        """Return dataset from profile_data dict."""
        return self.profile_data[attr]

    def __setitem__(self, attr: str, data: xarray.Dataset):
        """Set item in profiles dict."""
        self.profile_data[attr] = data

    def clip(self, attr: str, value: float | str):
        """Select data as abs(attr) > value."""
        time = self.data.time[abs(self.data[attr]) > value]
        return self.data.sel({'time': time})

    @cached_property
    def minimum_timestep(self) -> float:
        """Return minimum timestep present in source data."""
        return np.diff(self.data.time).min()

    @property
    def factor(self):
        """Return re-sample factor."""
        match self.delta:
            case int() if self.delta < 0:
                return -self.delta / float(self.interp_data.dims['time'])
            case int() | float() if self.delta > 0:
                return self.minimum_timestep / self.delta
            case _:
                raise ValueError(f'delta {self.delta} is '
                                 'not a negative int or float')

    @property
    def updown(self) -> tuple[int, int]:
        """Return up and downsampling factors."""
        match self.factor:
            case float(factor) if factor == 0:
                return 1, 1
            case float(factor) if factor > 1:
                return int(10*round(factor, 1)), 10
            case float(factor) if factor < 1:
                return 10, int(10*round(1/factor, 1))
            case _:
                raise ValueError(f'invalid sample factor {self.factor}')

    def interpolate(self):
        """Interpolate data onto uniform time-base."""
        time = np.arange(self.data.time[0], self.data.time[-1],
                         self.minimum_timestep)
        self['uniform'] = self.data.interp({'time': time}).assign_coords(
            {'itime': range(len(time))})

    def resample(self):
        """Return dataset re-sampled using a polyphase filter."""
        updown = self.updown
        timestep = self.minimum_timestep * updown[1] / float(updown[0])
        time = np.arange(self.data.time[0], self.data.time[-1], timestep)
        time_sample = xarray.Dataset(coords={'time': time})
        time_sample.coords['itime'] = 'time', np.arange(len(time))
        for attr, value in self.select('time', self['uniform']).items():
            dims = value.coords.dims
            value = scipy.signal.resample_poly(value, *updown, padtype='line')
            if self.savgol is not None:
                value = scipy.signal.savgol_filter(value, *self.savgol, axis=0)
            time_sample[attr] = dims, value
        self['sample'] = xarray.merge([self.select('~time', self['uniform']),
                                       time_sample])

    def defeature(self):
        """Defeature sample waveform using rdp algorithum."""
        #Defeature(self['sample'], self.features)

    def plot(self, attrs=None):
        """Plot source, interpolated, and sampled datasets."""
        if attrs is None:
            attrs = self.attrs('time')
        if isinstance(attrs, str):
            attrs = [attrs]
        self.set_axes('1d')
        for i, attr in enumerate(attrs):
            dims = self.data[attr].coords.dims
            if 'time' not in dims or len(dims) != 1:
                continue
            self.axes.plot(self.data.time, self.data[attr],
                           '-', color='gray', lw=1.5)
            self.axes.plot(self['uniform'].time, self['uniform'][attr],
                           '-', color=f'C{i}', lw=0.5)
            self.axes.plot(self['sample'].time, self['sample'][attr],
                           '-', color=f'C{i}', lw=2)
            #self.axes.plot(self['rdp'].time, self['rdp'][attr],
            #               'o-', color='k', lw=1, ms=6)


if __name__ == '__main__':

    pulse, run = 135013, 2

    equilibrium = Equilibrium(pulse, run)
    sample = Sample(equilibrium.data, 2.5, (3, 1))

    sample.plot(['elongation', 'triangularity_upper', 'triangularity_lower'])
    #sample.plot('ip')
