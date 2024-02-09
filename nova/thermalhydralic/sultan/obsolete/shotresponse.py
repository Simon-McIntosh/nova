"""Methods to manage single shot Sultan waveform data."""

from dataclasses import dataclass, field
from typing import Union


from nova.thermalhydralic.sultan.shotprofile import ShotProfile
from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.thermalhydralic.sultan.testplan import TestPlan
import matplotlib.pyplot as plt


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

    def plot(self):
        """Plot shot response."""
        axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, sharex=True)[1]

        self.profile.plot(axes=axes[1])

        self.shotdata.plot(self.heatindex.start, "ko", axes=axes[1], label="start")
        self.shotdata.plot(self.heatindex.stop, "ks", axes=axes[1], label="start")

        # axes[1].plot(self.cool.time, self.cool.value, 'kd', label='cool')
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


if __name__ == "__main__":
    response = ShotResponse("CSJA13")
    response.profile.instance.index = -5
    response.plot()
