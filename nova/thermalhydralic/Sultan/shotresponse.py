"""Methods to manage single shot Sultan waveform data."""
from dataclasses import dataclass, field, InitVar
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
    time: float = field(init=False)
    value: float = field(init=False)
    offset: tuple[float] = field(default=(0, 0))
    time_label: tuple[str] = field(default=('t', 's'), repr=False)
    data_label: tuple[str] = field(default=('Qdot_norm', 'W'), repr=False)

    def __post_init__(self, data):
        """Set data values."""
        self.time = data.loc[self.index, self.time_label] - self.offset[0]
        self.value = data.loc[self.index, self.data_label] - self.offset[1]


@dataclass
class Waveform:
    """Manage response waveform."""

    profile: ShotProfile
    _threshold: float = 0.25
    _index: slice = field(init=False)
    time_label: str = DataInstance.time_label
    heat_label: str = DataInstance.data_label
    reload: SimpleNamespace = field(init=True, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init time and heat data fields."""
        self.reload.__init__(threshold=True, index=True)

        start_index = profile.heatindex.start

        index = slice(self.start, self.stop)

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
        if threshold == -1:
            stop = self.profile.heatindex.stop
        elif threshold >= 0 and threshold <= 1:


    def cooldown_index(self):
        """Return cool heat datainstance."""
        heat = self.profile.lowpassdata.loc[self.profile.heatindex.index,
                                            self.heat_label]

        max_index = self.maximum.index
        cooldown = self.lowpassdata.loc[max_index-1:, DataInstance.data_label]
        minmax = cooldown.min(), cooldown.max()
        delta = np.diff(minmax)[0]
        cool_index = np.argmax(cooldown <=
                               minmax[0] + self.cool_threshold*delta)
        return DataInstance(self.lowpassdata, max_index+cool_index,
                            self.offset)



    @property
    def time(self):
        return self._loc(self.time_label)

    @property
    def heat(self):
        return self._loc(self.heat_label)

    def _loc(self, label):
        data = self.profile.lowpassdata.loc[self.index, label].values
        return data-data[0]


    @property
    def index(self):
        start = self.profile.heatindex.start

    @property
    def stop(self):



    def _set_index(self, profile, termination_threshold):
        start = profile.heatindex.start


    def _zero(self):
        self.time -= self.time[0]
        self.heat -= self.heat[0]

    def plot(self):
        """Plot input waveform."""
        plt.plot(self.time, self.heat)


@dataclass
class ShotResponse:
    """Calculate single shot heat response."""

    profile: Union[ShotProfile, ShotInstance, TestPlan, str]
    zero: bool = True
    steady_threshold: float = 1.05

    def __post_init__(self):
        """Init profile."""
        if not isinstance(self.profile, ShotProfile):
            self.profile = ShotProfile(self.profile)

    @property
    def offset(self):
        """Return time, value offset."""
        if self.zero:
            offset_data = DataInstance(self.lowpassdata, self.heatindex.start)
            return offset_data.time, offset_data.value
        else:
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
    def waveform(self):
        """Return offset heat step response data."""
        index = slice(self.start.index, self.cool.index)
        time = self.lowpassdata.loc[index, DataInstance.time_label].values
        heat = self.lowpassdata.loc[index, DataInstance.data_label].values
        return time-time[0], heat-heat[0]

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
        axes[1].plot(self.cool.time, self.cool.value, 'kd', label='cool')
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
