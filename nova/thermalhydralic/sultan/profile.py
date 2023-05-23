"""Manage sultan point data."""
from dataclasses import dataclass, field
from typing import Union
from types import SimpleNamespace

import numpy as np
import pandas

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
import matplotlib.pyplot as plt


@dataclass
class Profile:
    """Offset and normalize sultan timeseries data."""

    sample: Union[Sample, Trial, Campaign, str]
    _data_offset: Union[bool, tuple[float]] = True
    _offset: tuple[float | int, float | int] = \
        field(init=False, default=(0, 0), repr=False)
    _normalize: bool = False
    _lowpass_filter: bool = True
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Calculate offset."""
        self.reload.__init__(waveform=True)
        if not isinstance(self.sample, Sample):
            self.sample = Sample(self.sample)
        self.sample.sampledata.lowpass_filter = self._lowpass_filter
        self.normalize = self._normalize
        self.offset = self._data_offset
        del self._lowpass_filter

    @property
    def columns(self):
        """Return timeseries column names."""
        return {'time': ('t', 's'), 'data': ('Qdot', 'W')}

    @property
    def offset(self):
        """
        Manage data offset.

        Parameters
        ----------
        offset : bool or tuple[float, float]

            - True: offset data to heatinstance.start
            - False: no offset
            - tuple: custom offset

        Returns
        -------
        offset: tuple[float, float]
            offset.

        """
        self.sample.sampledata._reload()
        if self.sample.sampledata.reload.offset:
            self.offset = self._data_offset
        return self._offset

    @offset.setter
    def offset(self, offset):
        self.sample.sampledata.reload.offset = False
        self._data_offset = offset
        if isinstance(offset, bool):
            self._offset = (0, 0)
            if offset:
                start_index = self.heatindex.start
                self._offset = self.sample.sampledata.lowpass.loc[
                    start_index, [self.time_label, self.data_label]].values
        else:
            self._offset = offset
        self.reload.waveform = True
        self.sample.sampledata.reload.offset = False

    @property
    def normalize(self):
        """Manage data normalization."""
        return self._normalize

    @normalize.setter
    def normalize(self, normalize):
        self._normalize = normalize
        self.sample.sampledata.reload.offset = True
        self.reload.waveform = True

    @property
    def lowpass_filter(self):
        """Return status of lowpass filter, read-only."""
        return self.sample.sampledata.lowpass_filter

    def _assert_lowpass(self):
        """Ensure lowpass filtering is enabled."""
        assert self.lowpass_filter

    def _get_column_label(self, key):
        """Return time label."""
        try:
            label = self.columns[key]
        except KeyError as keyerror:
            raise KeyError(f'{key} not in self.columns {self.columns}') \
                from keyerror
        return label

    @property
    def time_label(self):
        """Return time label."""
        return self._get_column_label('time')

    @property
    def data_label(self):
        """Return data label."""
        data_label = self._get_column_label('data')
        if self.normalize:
            label = (data_label[0]+'_norm', data_label[1])
        else:
            label = data_label
        return label

    def _locate(self, prefix, index=slice(None)):
        if prefix not in self.columns:
            raise NameError(f'prefix {prefix} not in {self.columns}')
        label = getattr(self, f'{prefix}_label')
        offset_index = list(self.columns).index(prefix)
        return self.sample.sampledata.data.loc[index, label].values - \
            self.offset[offset_index]

    @property
    def time(self):
        """Return time array."""
        return self._locate('time')

    @property
    def data(self):
        """Return data array."""
        return self._locate('data')

    def timeseries(self, index=slice(None)):
        """Return data timeseries."""
        return self.time[index], self.data[index]

    @property
    def heatindex(self):
        """Return sample heatindex."""
        return self.sample.heatindex

    @property
    def start(self):
        """Return timesample at index.start."""
        self._assert_lowpass()
        return self.timeseries(self.heatindex.start)

    @property
    def stop(self):
        """Return timesample at index.stop."""
        self._assert_lowpass()
        return self.timeseries(self.heatindex.stop)

    @property
    def minimum(self):
        """Return minimum heat timesample."""
        self._assert_lowpass()
        return self.timeseries(np.argmin(self.timeseries()[1]))

    @property
    def maxindex(self):
        """Return maximum heat index."""
        self._assert_lowpass()
        return np.argmax(self.timeseries()[1])

    @property
    def maximum(self):
        """Return maximum heat timesample."""
        self._assert_lowpass()
        return self.timeseries(self.maxindex)

    @property
    def heatdelta(self):
        """Return heatup delta."""
        self._assert_lowpass()
        if self.maxindex > self.heatindex.start:
            index = slice(self.heatindex.start, self.maxindex)
        else:
            index = slice(self.heatindex.start, None)
        heatup = self.timeseries(index)[1]
        maximum = np.max(heatup)
        minimum = np.min(heatup)
        heatdelta = maximum-minimum
        return heatdelta

    @property
    def cooldown(self):
        """Return cooldown timeseries."""
        self._assert_lowpass()
        return self.timeseries(slice(self.maxindex, None))

    @property
    def cooldelta(self):
        """Return cooldown delta."""
        self._assert_lowpass()
        cooldown = self.cooldown[1]
        maximum = cooldown[0]
        minimum = np.max([np.min(cooldown), self.start[1]])
        return maximum-minimum

    @property
    def coldindex(self):
        """Return 95% cooldown index."""
        self._assert_lowpass()
        return self.maxindex + np.argmax(
            self.cooldown[1] <= self.maximum[1] - 0.95*self.cooldelta)

    @property
    def cold(self):
        """Return 95% cooldown timesample."""
        self._assert_lowpass()
        return self.timeseries(self.coldindex)

    @property
    def pulse_energy(self):
        """Return intergral power."""
        self._assert_lowpass()
        pulse = self.timeseries(slice(self.heatindex.start, self.coldindex))
        return np.trapz(pulse[1], pulse[0])

    @property
    def minimum_ratio(self):
        """Return ratio of stop-minimum to heat delta."""
        self._assert_lowpass()
        return (self.stop[1]-self.minimum[1]) / self.heatdelta

    @property
    def maximum_ratio(self):
        """Return ratio of offset maximum to heat delta."""
        self._assert_lowpass()
        return (self.maximum[1]-self.start[1]) / self.heatdelta

    @property
    def limit_ratio(self):
        """Return ration of index delta to heat delta."""
        self._assert_lowpass()
        return (self.stop[1]-self.start[1]) / self.heatdelta

    @property
    def steady(self):
        """Return steady flag."""
        self._assert_lowpass()
        return self.status.steady.all()

    @property
    def status(self):
        """Return pandas.DataFrame detailing stability metrics."""
        status = pandas.DataFrame(index=['maximum', 'minimum', 'limit'],
                                  columns=['ratio', 'steady'])
        self._assert_lowpass()
        for name in status.index:
            status.loc[name, 'ratio'] = getattr(self, f'{name}_ratio')
            status.loc[name, 'steady'] = status.loc[name, 'ratio'] >= 0.95
        return status

    @property
    def coefficents(self) -> pandas.Series:
        """Return profile coefficents."""
        coefficents = {}
        with self.sample.sampledata(lowpass_filter=True):
            for attribute in ['start', 'stop', 'cold', 'minimum', 'maximum']:
                timesample = getattr(self, attribute)
                for i, postfix in enumerate(['time', 'value']):
                    coefficents[f'{attribute}_{postfix}'] = timesample[i]
            for attribute in ['minimum_ratio', 'maximum_ratio', 'limit_ratio']:
                coefficents[attribute] = getattr(self, attribute)
            coefficents['energy_data'] = self.pulse_energy
        return pandas.Series(coefficents)

    def plot_point(self, index, *args, **kwargs):
        """
        Plot point data.

        Parameters
        ----------
        index : int
            Data index.
        **kwargs : any
            plot keyword args.

        Returns
        -------
        None.

        """
        axes = kwargs.pop('axes', None)
        if axes is None:
            axes = plt.subplots(1, 1)[1]
        axes.plot(*self.timeseries(index), *args, **kwargs)

    def plot_single(self, axes=None, lowpass_filter=True, **kwargs):
        """
        Plot single waveform.

        Parameters
        ----------
        axes : axes, optional
            Plot axes. The default is None, plt.gca().
        lowpass_filter : bool, optional
            Serve lowpass filtered data. The default is False.

        Returns
        -------
        None.

        """
        if axes is None:
            axes = plt.gca()
        if lowpass_filter:
            color = 'C0'
            label = 'lowpass'
            linewidth = 1.5
        else:
            color = 'C9'
            label = 'raw'
            linewidth = 1
        kwargs = {'color': color, 'linestyle': '-', 'label': label,
                  'lw': linewidth} | kwargs
        with self.sample.sampledata(lowpass_filter=lowpass_filter):
            axes.plot(*self.timeseries(), **kwargs)

    def plot(self, axes=None, heat=True, lowpass=True):
        """Plot shot profile."""
        if axes is None:
            axes = plt.gca()
        self.plot_single(lowpass_filter=False, axes=axes)
        if lowpass:
            self.plot_single(lowpass_filter=True, axes=axes)
        if heat:
            self.plot_heat()
        axes.legend(loc='upper right')
        axes.set_xlabel('$t$ s')
        axes.set_ylabel(r'$\dot{Q}$ W')
        plt.despine()
        plt.title(self.sample.label)

    def plot_heat(self, axes=None, **kwargs):
        """Shade heated zone."""
        if axes is None:
            axes = plt.gca()
        timeseries = self.timeseries(self.heatindex.index)
        time = timeseries[0]
        upper = timeseries[1]
        lower = np.min([np.min(upper), 0]) * np.ones(len(time))
        kwargs = {'color': 'lightgray', 'alpha': 0.85,
                  'label': 'heat', 'zorder': -1} | kwargs
        axes.fill_between(time, lower, upper, **kwargs)


if __name__ == '__main__':

    trial = Trial('CSJA12', 'ac0')
    sample = Sample(trial, 0, 'Left')
    profile = Profile(sample, _normalize=False)

    profile.plot(lowpass=True, heat=True)
