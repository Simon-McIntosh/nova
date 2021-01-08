"""Methods to manage single shot Sultan waveform data."""
from dataclasses import dataclass, field, InitVar
from typing import Union

import numpy as np
import pandas

from nova.thermalhydralic.sultan.shotprofile import ShotProfile
from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.thermalhydralic.sultan.testplan import TestPlan


@dataclass
class DataInstance:
    """Extract profile.data instance."""

    data: InitVar[pandas.DataFrame]
    index: int
    time: float = field(init=False)
    value: float = field(init=False)
    time_label: tuple[str] = field(default=('t', 's'), repr=False)
    data_label: tuple[str] = field(default=('Qdot_norm', 'W'), repr=False)

    def __post_init__(self, data):
        """Set data values."""
        self.time = data.loc[self.index, self.time_label]
        self.value = data.loc[self.index, self.data_label]


@dataclass
class ShotResponse:
    """
    Calculate single shot heat response.

    Parameters
    ----------
    data: pandas.DataFrame
        Input dataframe, lowpassdata.
    heat_index: HeatIndex
        Input heating index and time delta.
    steady_threshold: float
        Threshold below which input is considered steady.

        - maximum.value < threshold * stop.value.

    minimum_threshold: float
        Threshold below which input is considered steady.

        - (stop.value-minimum.value) / delta

    start: HeatInstance
        Time and heat output values at the start of input heating
    stop: HeatInstance
        Time and heat output values at the end of input heating.
    maximum: HeatInstance
        Time and heat output values at maximum input heating.
    minimum: HeatInstance
        Time and heat output values at minimum indexed input heating.
    delta: float
        Delta heating within index.

    """

    profile: Union[ShotProfile, ShotInstance, TestPlan, str]
    steady_threshold: float = 1.05

    def __post_init__(self):
        """Init profile."""
        if not isinstance(self.profile, ShotProfile):
            self.profile = ShotProfile(self.profile)

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
        return DataInstance(self.lowpassdata, self.heatindex.start)

    @property
    def stop(self):
        """Return end heat datainstance at index.stop."""
        return DataInstance(self.lowpassdata, self.heatindex.stop)

    @property
    def maximum(self):
        """Return maximum heat datainstance."""
        max_index = np.argmax(self.lowpassdata[('Qdot_norm', 'W')].abs())
        return DataInstance(self.lowpassdata, max_index)

    @property
    def minimum(self):
        """Return minimum heat datainstance."""
        min_index = np.argmin(self.lowpassdata[('Qdot_norm', 'W')].abs())
        return DataInstance(self.lowpassdata, min_index)

    @property
    def delta(self):
        """Return delta heating within self.index."""
        indexheat = self.lowpassdata.loc[self.heatindex.index,
                                         ('Qdot_norm', 'W')]
        maximum_heat = np.max(indexheat)
        minimum_heat = np.min(indexheat)
        return maximum_heat-minimum_heat

    @property
    def energy(self):
        """Return intergral power."""
        startindex = self.heatindex.start
        time = self.lowpassdata.loc[startindex:, ('t', 's')]
        heat = self.lowpassdata.loc[startindex:, ('Qdot_norm', 'W')]
        return np.trapz(heat, time)

    @property
    def stepdata(self):
        """Return offset heat step response data."""
        time = self.lowpassdata.loc[self.heatindex.index, ('t', 's')].values
        heat = self.lowpassdata.loc[self.heatindex.index,
                                    ('Qdot_norm', 'W')].values
        return time-time[0], heat-heat[0]

    @property
    def maximum_ratio(self):
        """Return ratio of offset maximum to heat delta."""
        return (self.maximum.value-self.start.value) / self.delta

    @property
    def minimum_ratio(self):
        """Return ratio of stop-minimum to heat delta."""
        return (self.stop.value-self.minimum.value) / self.delta

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
        """Extend profile.plot."""
        self.profile.plot(offset=offset)


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
