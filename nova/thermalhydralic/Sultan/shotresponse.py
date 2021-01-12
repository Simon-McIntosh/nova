"""Methods to manage single shot Sultan waveform data."""
from dataclasses import dataclass, field, InitVar
from types import SimpleNamespace
from typing import Union

import numpy as np
import pandas

from nova.thermalhydralic.sultan.shotprofile import ShotProfile
from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.thermalhydralic.sultan.testplan import TestPlan
from nova.utilities.pyplot import plt


@dataclass
class DataInstance:
    """Extract profile.data instance."""

    data: InitVar[pandas.DataFrame]
    index: int
    normalize: bool = True
    offset: tuple[float] = field(default=(0, 0))
    time: float = field(init=False)
    value: float = field(init=False)

    def __post_init__(self, data):
        """Set data values."""
        time_label = ('t', 's')
        data_label = ('Qdot_norm', 'W') if self.normalize else ('Qdot', 'W')
        self.time = data.loc[self.index, time_label] - self.offset[0]
        self.value = data.loc[self.index, data_label] - self.offset[1]


@dataclass
class WaveForm:
    """Manage response waveform."""

    profile: InitVar[ShotProfile] = field(repr=False)
    cooldown_threshold: float = 0.25
    upsample: float = 11
    normalize: bool = True
    shotname: str = field(init=False)
    frequency: float = field(init=False)
    data: pandas.DataFrame = field(init=False, repr=False)

    def __post_init__(self, profile):
        """Init time and heat data fields."""
        self.shotname = profile.shotname
        self.frequency = profile.frequency
        self._check_threshold(self.cooldown_threshold)
        self._extract_data(profile)

    @staticmethod
    def _check_threshold(threshold):
        """
        Check cooldown threshold parameter.

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
        if threshold != -1 and (threshold < 0 or threshold > 1):
            raise ValueError(f'cooldown threshold {threshold} out of range.')

    @property
    def datalabel(self):
        """Return datalabel."""
        if self.normalize:
            return ('Qdot_norm', 'W')
        return ('Qdot', 'W')

    @staticmethod
    def extract_index(profile, threshold, datalabel):
        """
        Return waveform slice.

        Parameters
        ----------
        profile : ShotProfile
            Shot profile.
        threshold : float
            Cooldown threshold.

            - -1: stop = end of heating, heatindex.stop
            - 0-1: cooldown maximum <= minimum * threshold * delta
        datalabel : str
            Data label.

        Returns
        -------
        index : slice
            Waveform index.

        """
        start = profile.heatindex.start
        if threshold == -1:
            stop = profile.heatindex.stop
        else:
            pulse = profile.lowpassdata.loc[start:, datalabel].values
            cooldown = pulse[pulse.argmax():]
            minmax = cooldown.min(), cooldown.max()
            delta = np.diff(minmax)[0]
            stop = np.argmax(cooldown <= minmax[0] + threshold*delta)
        return slice(start, stop)

    def _extract_data(self, profile):
        index = self.extract_index(profile, self.cooldown_threshold,
                                   self.datalabel)
        data = profile.lowpassdata.loc[index, self.label.values()]
        data = data.droplevel(1, axis=1)
        data.rename(columns={self.label['time'][0]: 'time'}, inplace=True)

        self._upsample(data, profile.frequency)

    def _upsample(self, data, excitation_frequency):

        print(self.label['time'])
        timestep = np.diff(data.time).mean()
        sample_frequency = 2*np.pi / timestep
        if sample_frequency < self.upsample*excitation_frequency:
            print('upsample', sample_frequency, excitation_frequency)

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
    def offset(self):
        """Return time, value offset."""
        if self.zero:
            offset_data = DataInstance(self.lowpassdata, self.heatindex.start)
            return offset_data.time, offset_data.value
        return 0, 0

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
    def start(self):
        """Return offset heat datainstance at index.start."""
        return DataInstance(self.lowpassdata, self.heatindex.start,
                            self.offset)

    @property
    def stop(self):
        """Return end heat datainstance at index.stop."""
        return DataInstance(self.lowpassdata, self.heatindex.stop,
                            self.offset)

    @property
    def maximum(self):
        """Return maximum heat datainstance."""
        max_index = np.argmax(self.lowpassdata[DataInstance.data_label].abs())
        return DataInstance(self.lowpassdata, max_index,
                            self.offset)

    @property
    def minimum(self):
        """Return minimum heat datainstance."""
        min_index = np.argmin(self.lowpassdata[DataInstance.data_label].abs())
        return DataInstance(self.lowpassdata, min_index,
                            self.offset)

    #@property
    #def heat_waveform(self):
    #    """Return heating waveform, square rate of

    @property
    def delta(self):
        """Return delta heating within self.index."""
        indexheat = self.lowpassdata.loc[self.heatindex.index,
                                         DataInstance.data_label]
        maximum_heat = np.max(indexheat)
        minimum_heat = np.min(indexheat)
        return maximum_heat-minimum_heat

    @property
    def energy(self):
        """Return intergral power."""
        startindex = self.heatindex.start
        time = self.lowpassdata.loc[startindex:, DataInstance.time_label]
        heat = self.lowpassdata.loc[startindex:, DataInstance.data_label]
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

    def plot(self, offset=True):
        """Plot shot response."""
        axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]},
                            sharex=True)[1]

        self.profile.plot(offset=self.zero, axes=axes[1])

        axes[1].plot(self.start.time, self.start.value, 'ko', label='start')
        axes[1].plot(self.stop.time, self.stop.value, 'ks', label='stop')
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
    response.profile.instance.index = -7
    response.plot()
