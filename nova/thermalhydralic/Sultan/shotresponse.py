"""Methods to manage single shot Sultan waveform data."""
from dataclasses import dataclass, field
from typing import Union
from types import SimpleNamespace

import numpy as np
import pandas

from nova.thermalhydralic.sultan.shotprofile import ShotProfile
from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.thermalhydralic.sultan.testplan import TestPlan
from nova.utilities.pyplot import plt


@dataclass
class WaveForm:
    """Manage response waveform."""

    profile: ShotProfile = field(repr=False)
    _threshold: float = 0.25
    _upsample: float = 11
    _index: slice = field(init=False)
    _data: pandas.DataFrame = field(init=False, repr=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init time and heat data fields."""
        self.reload.__init__(threshold=True, upsample=True,
                             index=True, data=True)
        self._extract_index()

        #self.shotname = profile.shotname
        #self.frequency = profile.frequency
        #self._check_threshold(self.cooldown_threshold)
        #self._extract_index(profile)
        #self._extract_data(profile)

    @property
    def threshold(self):
        """
        Manage cooldown threshold parameter.

        Parameters
        ----------
        threshold : float

            - -1: stop = end of heating, heatindex.stop
            - 0-1: cooldown maximum <= minimum * threshold * delta

        Raises
        ------
        ValueError
            threshold must lie between 0 and 1 or equal -1.

        """
        if self.reload.threshold:
            self.threshold = self._threshold
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        if threshold != -1 and (threshold < 0 or threshold > 1):
            raise ValueError(f'cooldown threshold {threshold} out of range.')
        self._threshold = threshold
        self.reload.threshold = False
        self.reload.index = True
        self.reload.data = True

    @property
    def upsample(self):
        """Manage upsample factor."""
        if self.reload.upsample:
            self.upsample = self._upsample
        return self._upsample

    @upsample.setter
    def upsample(self, upsample):
        self._upsample = upsample
        self.reload.data = True

    def _reload(self):
        """Set data chain reload flags."""
        if self.profile.reload.waveform:
            self.reload.index = True
            self.reload.data = True
            self.profile.reload.waveform = False

    @property
    def index(self):
        """
        Return waveform index.

        Parameters
        ----------
        profile : ShotProfile
            Shot profile.

        """
        self._reload()
        if self.reload.index:
            self._extract_index()
        return self._index

    def _extract_index(self):
        start = self.profile.heatindex.start
        if self.threshold == -1:
            stop = self.profile.heatindex.stop
        else:
            pulse = self.profile.lowpassdata.loc[
                start:, self.profile.shotdata.data_label].values
            cooldown = pulse[pulse.argmax():]
            minmax = cooldown.min(), cooldown.max()
            delta = np.diff(minmax)[0]
            stop = np.argmax(cooldown <= minmax[0] + self.threshold*delta)
        self._index = slice(start, stop)
        self.reload.index = False

    @property
    def data(self):
        """
        Return upsampled data.

        Returns
        -------
        None.

        """
        self._reload()
        if self.reload.data:
            self._extract_data()
        return self._data

    def _extract_data(self):
        data = self.profile.lowpassdata.loc[
            self.index, [self.profile.shotdata.time_label,
                         self.profile.shotdata.data_label]]
        data = data.droplevel(1, axis=1)
        data.rename(columns={self.profile.shotdata.time_label[0]: 'time'},
                    inplace=True)
        self._upsample_data(data, self.profile.frequency)
        self._data = data
        self.reload.data = False

    def _upsample_data(self, data, excitation_frequency):
        timestep = np.diff(data.time).mean()
        sample_frequency = 1/timestep
        if sample_frequency < self.upsample*excitation_frequency:
            print('upsample')

        print('upsample', sample_frequency, self.upsample*excitation_frequency)

    # @property
    # def time(self):
    #     return self.data.loc[:, self.label['time'])

    #def _values(self, label):
    #    data = self.profile.lowpassdata.loc[self.index, label].values
    #    return data-data[0]

    ##@property
    #def index(self):
    #    start = self.profile.heatindex.start

   # @property
   # def stop(self):

    def plot(self):
        """Plot input waveform."""
        plt.plot(self.time, self.heat)


@dataclass
class ShotResponse:
    """Calculate single shot heat response."""

    profile: Union[ShotProfile, ShotInstance, TestPlan, str]
    zero: bool = True
    steady_threshold: float = 1.05
    cooldown_threshold: float = 0.25
    _waveform: WaveForm = field(init=False)

    def __post_init__(self):
        """Init profile."""
        if not isinstance(self.profile, ShotProfile):
            self.profile = ShotProfile(self.profile)

    @property
    def waveform(self):
        """Return offset heat step response data."""
        if self.profile.reload.response:
            self._waveform = WaveForm(self.profile, self.cooldown_threshold)
            self.profile.reload.response = False
        return self._waveform

    @property
    def plan(self):
        """Return testplan, read-only."""
        return self.profile.instance.testplan.plan

    @property
    def heatindex(self):
        """Return profile heatindex, read-only."""
        return self.profile.heatindex

    @property
    def lowpassdata(self):
        """Return profile low-pass data, read-only."""
        return self.profile.lowpassdata

    @property
    def shotdata(self):
        """Return shotdata."""
        return self.profile.shotdata

    @property
    def start(self):
        """Return offset heat datainstance at index.start."""
        return self.shotdata.point(self.heatindex.start)

    @property
    def stop(self):
        """Return end heat datainstance at index.stop."""
        return self.shotdata.point(self.heatindex.stop)

    @property
    def maximum(self):
        """Return maximum heat datainstance."""
        return self.shotdata.point(
            np.argmax(self.lowpassdata[self.shotdata.data_label].abs()))

    @property
    def minimum(self):
        """Return minimum heat datainstance."""
        return self.shotdata.point(
            np.argmin(self.lowpassdata[self.shotdata.data_label].abs()))

    @property
    def delta(self):
        """Return delta heating within self.index."""
        indexheat = self.lowpassdata.loc[self.heatindex.index,
                                         self.shotdata.data_label]
        maximum_heat = np.max(indexheat)
        minimum_heat = np.min(indexheat)
        return maximum_heat-minimum_heat

    @property
    def energy(self):
        """Return intergral power."""
        startindex = self.heatindex.start
        time = self.lowpassdata.loc[startindex:, self.shotdata.time_label]
        heat = self.lowpassdata.loc[startindex:, self.shotdata.data_label]
        return np.trapz(heat, time)

    @property
    def minimum_ratio(self):
        """Return ratio of stop-minimum to heat delta."""
        return (self.stop.value-self.minimum.value) / self.delta

    @property
    def maximum_ratio(self):
        """Return ratio of offset maximum to heat delta."""
        return (self.maximum.value-self.start.value) / self.delta

    @property
    def limit_ratio(self):
        """Return ration of index delta to heat delta."""
        return (self.stop.value-self.start.value) / self.delta

    @property
    def steady(self):
        """Return steady flag."""
        return self.status.steady.all()

    @property
    def status(self):
        """Return pandas.DataFrame detailing stability metrics."""
        status = pandas.DataFrame(index=['maximum', 'minimum', 'limit'],
                                  columns=['ratio', 'steady'])
        for name in status.index:
            status.loc[name, 'ratio'] = getattr(self, f'{name}_ratio')
            status.loc[name, 'steady'] = \
                status.loc[name, 'ratio'] < self.steady_threshold
        return status

    def plot(self):
        """Plot shot response."""
        axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]},
                            sharex=True)[1]

        self.profile.plot(axes=axes[1])

        self.shotdata.plot(self.heatindex.start, 'ko', axes=axes[1],
                           label='start')
        self.shotdata.plot(self.heatindex.stop, 'ks', axes=axes[1],
                           label='start')

        #axes[1].plot(self.cool.time, self.cool.value, 'kd', label='cool')
        plt.legend()


    '''
    @property
    def dataseries(self):
        """Return response data series."""
        #(pole, gain, delay), step
        return pandas.Series([self.stop.value-self.start.value,
                              self.maximum.value-self.start.value,
                              self.steady],
                             index=['stop', 'maximum', 'steady'])
    '''


if __name__ == '__main__':

    response = ShotResponse('CSJA13')
    response.profile.instance.index = -5
    response.plot()
