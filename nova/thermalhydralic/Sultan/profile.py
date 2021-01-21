"""Manage sultan point data."""
from dataclasses import dataclass, field
from typing import Union
from types import SimpleNamespace

import numpy as np

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.utilities.pyplot import plt


@dataclass
class Profile:
    """Offset and normalize sultan timeseries data."""

    sample: Union[Sample, Trial, Campaign, str]
    _data_offset: Union[bool, tuple[float]] = True
    _offset: tuple[float] = field(init=False, default=(0, 0), repr=False)
    _normalize: bool = True
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
        self.sample.sampledata.propagate_reload()
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
                start_index = self.sample.heatindex.start
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

    def plot_single(self, axes=None, lowpass_filter=False):
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
        color = 'C1' if lowpass_filter else 'C0'
        label = 'lowpass' if lowpass_filter else 'raw'
        with self.sample.sampledata(lowpass_filter=lowpass_filter):
            axes.plot(*self.timeseries(), color=color, label=label)

    def plot(self, axes=None):
        """Plot shot profile."""
        if axes is None:
            axes = plt.gca()
        self.plot_single(lowpass_filter=False, axes=axes)
        self.plot_single(lowpass_filter=True, axes=axes)
        axes.legend(loc='upper right')
        axes.set_xlabel('$t$ s')
        axes.set_ylabel(r'$\hat{\dot{Q}}$ W')
        plt.despine()


if __name__ == '__main__':
    profile = Profile('CSJA12')
    profile.sample.shot = 0
    #profile.sample.shot = -0
    profile.plot()
