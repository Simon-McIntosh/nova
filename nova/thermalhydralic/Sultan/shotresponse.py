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
