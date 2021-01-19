"""Manage sultan point data."""
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.utilities.pyplot import plt


@dataclass
class Profile:
    """Offset and normalize sultan timeseries data."""

    sample: Union[Sample, Trial, Campaign, str]
    offset_data: Union[bool, tuple[float]] = True
    _offset: tuple[float] = field(init=False, default=(0, 0), repr=False)
    normalize: bool = True
    lowpass_filter: bool = True
    columns: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Calculate offset."""
        self.columns |= {'time': ('t', 's'), 'data': ('Qdot', 'W')}
        if not isinstance(self.sample, Sample):
            self.sample = Sample(self.sample)
        self.sample.sampledata.lowpass_filter = self.lowpass_filter

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
            self.offset = self.offset_data
        return self._offset

    @offset.setter
    def offset(self, offset):
        self.sample.sampledata.reload.offset = False
        if isinstance(offset, bool):
            self._offset = (0.0, 0.0)
            if offset:
                start_index = self.sample.heatindex.start
                self._offset = self.sample.sampledata.data.loc[
                    start_index, [self.time_label, self.data_label]].values
        else:
            self._offset = offset

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

    def _locate(self, prefix, index):
        variables = ['time', 'data']
        if prefix not in variables:
            raise NameError(f'prefix {prefix} not in [time, data]')
        label = getattr(self, f'{prefix}_label')
        offset_index = variables.index(prefix)
        return self.sample.sampledata.data.loc[index, label] - \
            self.offset[offset_index]

    def profile(self, index=slice(None)):
        """Return data timeseries."""
        time = self._locate('time', index)
        data = self._locate('data', index)
        return time, data

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
        axes.plot(*self.profile(index), *args, **kwargs)

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
        bg_color = 0.4 * np.ones(3) if lowpass_filter else 'lightgray'
        color = 'C3' if lowpass_filter else 'C0'
        label = 'lowpass' if lowpass_filter else 'raw'
        with self.sample.sampledata(lowpass_filter=lowpass_filter):
            axes.plot(*self.profile(), color=bg_color)
            axes.plot(*self.profile(self.sample.heatindex.index),
                      color=color, label=label)
        axes.legend()
        axes.set_xlabel('$t$ s')
        axes.set_ylabel(r'$\hat{\dot{Q}}$ W')
        plt.despine()

    def plot(self, axes=None):
        """Plot shot profile."""
        self.plot_single(lowpass_filter=False, axes=axes)
        self.plot_single(lowpass_filter=True, axes=axes)


if __name__ == '__main__':
    profile = Profile('CSJA13')
    profile.sample.shot = -11
    profile.sample.shot = -0
    profile.plot()

    print(profile.sample.sampledata.lowpass_filter)
