"""Manage sultan point data."""
from dataclasses import dataclass
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
    _offset: Union[bool, tuple[float]] = True
    normalize: bool = True
    lowpass: bool = True
    time_label: tuple[str] = ('t', 's')
    _data_label: tuple[str] = ('Qdot', 'W')

    def __post_init__(self):
        """Calculate offset."""
        if not isinstance(self.sample, Sample):
            self.sample = Sample(self.sample)
        self.sample.dataframe.lowpass_filter = self.lowpass
        self.offset = self._offset

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
        ### update offset when sample is updated ###
        return self._offset

    @offset.setter
    def offset(self, offset):
        if isinstance(offset, bool):
            self._offset = (0, 0)
            if offset:
                with self.sample.dataframe(True):
                    self._offset = self.data(self.sample.heatindex.start)
        else:
            self._offset = offset

    @property
    def data_label(self):
        """Return data label."""
        if self.normalize:
            label = (self._data_label[0]+'_norm', self._data_label[1])
        else:
            label = self._data_label
        return label

    def _locate(self, prefix, index):
        variables = ['time', 'data']
        if prefix not in variables:
            raise NameError(f'prefix {prefix} not in [time, data]')
        label = getattr(self, f'{prefix}_label')
        offset_index = variables.index(prefix)
        return self.sample.dataframe.data.loc[index, label] - \
            self.offset[offset_index]

    def data(self, index):
        """Return time, data tuple."""
        return self._locate('time', index), self._locate('data', index)

    @property
    def profile(self):
        """Return data timeseries."""
        return np.array(self.data(slice(None)))

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
        axes.plot(*self.data(index), *args, **kwargs)

    def plot_single(self, axes=None, lowpass=False):
        """
        Plot single waveform.

        Parameters
        ----------
        axes : axes, optional
            Plot axes. The default is None, plt.gca().
        lowpass : bool, optional
            Serve lowpass filtered data. The default is False.

        Returns
        -------
        None.

        """
        if axes is None:
            axes = plt.gca()
        bg_color = 0.4 * np.ones(3) if lowpass else 'lightgray'
        color = 'C3' if lowpass else 'C0'
        label = 'lowpass' if lowpass else 'raw'
        with self.sample.dataframe(lowpass):
            axes.plot(*self.profile, color=bg_color)
            axes.plot(*self.profile[:, self.sample.heatindex.index],
                      color=color, label=label)
        axes.legend()
        axes.set_xlabel('$t$ s')
        axes.set_ylabel(r'$\hat{\dot{Q}}$ W')
        plt.despine()

    def plot(self, axes=None):
        """Plot shot profile."""
        self.plot_single(lowpass=False, axes=axes)
        self.plot_single(lowpass=True, axes=axes)


if __name__ == '__main__':
    profile = Profile('CSJA13')
    profile.sample.shot = -11
    profile.plot()
