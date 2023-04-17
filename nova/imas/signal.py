"""Manage signal methods."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from rdp import rdp
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import minmax_scale
import scipy.signal
import xarray

from nova.frame.baseplot import Plot
from nova.imas.database import Database, IdsEntry
from nova.imas.equilibrium import Equilibrium
from nova.imas.metadata import Metadata


@dataclass
class Select:
    """Select subset of data based on coordinate."""

    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)

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

    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)
    epsilon: float = 1e-3
    cluster: int | float | None = None
    features: list[str] | None = None

    def __post_init__(self):
        """Extract feature list if None."""
        self.check_features()
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

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

    def defeature(self):
        """Return clustered turning point dataset."""
        indices = []
        index = np.arange(self.data.dims['time'])
        for attr in self.features:
            array = np.c_[self.time, minmax_scale(self.data[attr].data)]
            mask = rdp(array, self.epsilon, return_mask=True)
            indices.extend(index[mask])
        indices = np.unique(indices)
        if self.cluster is not None:
            indices = self._cluster(indices)
        return self.data.isel({'time': indices})

    def _cluster(self, indices):
        """Apply DBSCAN clustering algorithum to indices."""
        time = self.time[indices]
        clustering = DBSCAN(eps=self.cluster, min_samples=1).fit(time)
        labels = np.unique(clustering.labels_)
        centroid = np.zeros(len(labels), int)
        label_index = np.arange(len(indices))
        for i, label in enumerate(labels):
            centroid[i] = int(np.mean(
                label_index[label == clustering.labels_]))
        return indices[centroid]


@dataclass
class Signal(Plot, Defeature, Select):
    """Re-sample signal."""

    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)
    dtime: int | float | None = None
    savgol: tuple[int, int] | None = None
    epsilon: float = 0.05
    cluster: int | float | None = 0.5
    features: list[str] = field(default_factory=lambda: [
        'minor_radius', 'elongation',
        'triangularity_upper', 'triangularity_lower',
        'li_3', 'beta_normal', 'ip'])
    samples: dict[str, xarray.Dataset] = field(default_factory=dict)

    def __post_init__(self):
        """Interpolate data onto uniform time-base and resample."""
        self['source'] = self.data
        self.clip('li_3', 0)
        if self.dtime is not None:
            self.interpolate()
            self.resample()
        self.defeature()
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    def __getitem__(self, attr: str) -> xarray.Dataset:
        """Return dataset from samples dict."""
        return self.samples[attr]

    def __setitem__(self, attr: str, data: xarray.Dataset):
        """Set item in profiles dict."""
        self.data = data
        self.samples[attr] = data

    def clip(self, attr: str, value: float | str):
        """Select data as abs(attr) > value."""
        time = self.data.time[abs(self.data[attr]) > value]
        self['clip'] = self.data.sel({'time': time})

    @cached_property
    def minimum_timestep(self) -> float:
        """Return minimum timestep present in source data."""
        return np.diff(self.data.time).min()

    @property
    def factor(self):
        """Return re-sample factor."""
        match self.dtime:
            case int() if self.dtime < 0:
                return -self.dtime / float(self.data.dims['time'])
            case int() | float() if self.dtime > 0:
                return self.minimum_timestep / self.dtime
            case _:
                raise ValueError(f'dtime {self.dtime} is '
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
        factor = updown[0] / updown[1]
        ntime = int(np.ceil(self['uniform'].dims['time'] * factor))
        time = np.linspace(self.data.time[0], self.data.time[-1], ntime)
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
        self['rdp'] = super().defeature()

    def plot(self, attrs=None):
        """Plot source, interpolated, and sampled datasets."""
        if attrs is None:
            attrs = [attr for attr in self.features if attr != 'ip']
        if isinstance(attrs, str):
            attrs = [attrs]
        self.set_axes('1d')
        for i, attr in enumerate(attrs):
            dims = self.data[attr].coords.dims
            if 'time' not in dims or len(dims) != 1:
                continue
            self.axes.plot(self['clip'].time, self['clip'][attr],
                           '-', color=f'C{i}', lw=2, label=attr)
            if self.dtime is not None:
                self.axes.plot(self['uniform'].time, self['uniform'][attr],
                               '-', color='gray', alpha=0.75, lw=2.5)
                self.axes.plot(self['sample'].time, self['sample'][attr],
                               '-', color=f'C{i}', lw=2, label=attr)
            self.axes.plot(self['rdp'].time, self['rdp'][attr],
                           'o-', color='k', lw=1.5, ms=6,
                           zorder=-10)
        self.axes.legend(ncol=3)
        self.axes.set_xlabel('time s')
        self.axes.set_ylabel('value')

    def write_ids(self, **ids_attrs):
        """Write signal data to pulse_schedule ids."""
        ids_attrs |= {'occurrence':  Database(**ids_attrs).next_occurrence(),
                      'name': 'pulse_schedule'}
        ids_entry = IdsEntry(**ids_attrs)

        metadata = Metadata(ids_entry.ids_data)
        comment = 'Feature preserving reduced order waveforms'
        source = ','.join([str(value) for value in ids_attrs.values()])
        metadata.put_properties(comment, source, homogeneous_time=1)
        code_parameters = {attr: getattr(self, attr) for attr in
                           ['dtime', 'savgol', 'epsilon', 'cluster',
                            'features']}
        metadata.put_code('Geometry extraction and RDP order reduciton',
                          code_parameters)

        ids_entry.ids_data.time = self.data.time.data

        with ids_entry.node('flux_control.*.reference.data'):
            ids_entry['i_plasma'] = self.data.ip.data
            for attr in ['li_3', 'beta_normal']:
                ids_entry[attr] = self.data[attr].data

        with ids_entry.node('position_control.geometric_axis.'
                            '*.reference.data'):
            for i, attr in enumerate('rz'):
                ids_entry[attr] = self.data.geometric_axis[:,  i].data

        with ids_entry.node('position_control.*.reference.data'):
            for attr in ['minor_radius', 'elongation', 'triangularity_upper',
                         'triangularity_lower']:
                ids_entry[attr] = self.data[attr].data

        ids_entry.put_ids()


if __name__ == '__main__':

    pulse, run = 135013, 2

    equilibrium = Equilibrium(pulse, run)
    signal = Signal(equilibrium.data)

    signal.write_ids(**equilibrium.ids_attrs)
    signal.plot()

    # signal.plot('ip')
