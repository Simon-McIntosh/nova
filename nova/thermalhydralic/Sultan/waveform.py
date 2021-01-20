"""Methods to manage single shot Sultan waveform data."""
from dataclasses import dataclass, field
from typing import Union
from types import SimpleNamespace

import numpy as np
import pandas
import scipy.interpolate

from nova.thermalhydralic.sultan.profile import Profile
from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.utilities.pyplot import plt


@dataclass
class WaveForm:
    """Manage response waveform."""

    profile: Union[Profile, Sample, Trial, Campaign, str] = field(repr=False)
    _threshold: float = 0.25
    _upsample: float = 31
    _index: slice = field(init=False)
    _data: pandas.DataFrame = field(init=False, repr=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init time and heat data fields."""
        self.reload.__init__(threshold=True, index=True, data=True)
        if not isinstance(self.profile, Profile):
            self.profile = Profile(self.profile)
        self._extract_index()

    @property
    def threshold(self):
        """
        Manage cooldown threshold parameter.

        Parameters
        ----------
        threshold : float

            - -1-0: index -> heatindex.stop + threshold * delta_postheat
            - 0-1: index -> cooldown minimum + threshold * delta_cooldown

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
        if threshold is not None:
            if threshold < -1 or threshold > 1:
                raise ValueError(f'cooldown threshold {threshold} '
                                 'out of range.')
        self._threshold = threshold
        self.reload.threshold = False
        self.reload.index = True
        self.reload.data = True

    @property
    def upsample(self):
        """Return upsample factor, read-only."""
        return self._upsample

    def propagate_reload(self):
        """Propagate reload flags."""
        if self.profile.sample.sampledata.reload.waveform:
            self.reload.index = True
            self.reload.data = True
            self.profile.sample.sampledata.reload.waveform = False
        if self.profile.reload.waveform:
            self.reload.data = True
            self.profile.reload.waveform = False

    @property
    def frequency(self):
        """Return sample frequency, read-only."""
        return self.profile.sample.frequency

    @property
    def index(self):
        """
        Return waveform index.

        Parameters
        ----------
        profile : ShotProfile
            Shot profile.

        """
        self.propagate_reload()
        if self.reload.index:
            self._extract_index()
        return self._index

    def _extract_index(self):
        start = self.profile.sample.heatindex.start
        if self.threshold is None:
            start = stop = None
        elif self.threshold < 0:
            threshold = 1+self.threshold
            index = slice(self.profile.sample.heatindex.stop, None)
            pulse = self.profile.timeseries(index=index)[1]
            argmax = pulse.argmax()
            postheat = pulse[:argmax]
            minmax = postheat.min(), postheat.max()
            delta = np.diff(minmax)[0]
            stop = np.argmax(postheat >= minmax[0] + threshold*delta)
            stop += self.profile.sample.heatindex.stop
        else:
            pulse = self.profile.timeseries(index=slice(start, None))[1]
            argmax = pulse.argmax()
            cooldown = pulse[argmax:]
            minmax = cooldown.min(), cooldown.max()
            delta = np.diff(minmax)[0]
            stop = np.argmax(cooldown <= minmax[1] - self.threshold*delta)
            stop += start+argmax
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
        self.propagate_reload()
        if self.reload.data:
            self._extract()
        return self._data

    def _resample(self, timeseries, excitation_frequency):
        """
        Resample timeseries.

        Resample iff sample frequency < upsample*excitation_frequency

        Parameters
        ----------
        timeseries : tuple[array-like[float]]
            time, data arrays.
        excitation_frequency : float
            Excitation frequency.

        Returns
        -------
        timeseries : tuple[array-like[float]]
            Timeseries.
        samplenumber : int
            Sample number.

        """
        timestep = np.diff(timeseries[0]).mean()
        sample_frequency = 1/timestep
        if sample_frequency < self.upsample*excitation_frequency:
            timestep = 1 / (self.upsample*excitation_frequency)
            timebounds = timeseries[0][0], timeseries[0][-1]
            timedelta = np.diff(timebounds)[0]
            samplenumber = int(timedelta / timestep)
            time = np.linspace(*timebounds, samplenumber)
            data = scipy.interpolate.interp1d(*timeseries)(time)
            timeseries = (time, data)
        else:
            samplenumber = len(timeseries[0])
        return timeseries, samplenumber

    def heatindex(self, time):
        """Return waveform heatindex, read-only."""
        start_time = self.profile.time[self.profile.sample.heatindex.start]
        stop_time = self.profile.time[self.profile.sample.heatindex.stop]
        start_index = np.argmax(time >= start_time)
        if stop_time > time[-1]:
            stop_index = len(time)
        else:
            stop_index = np.argmax(time >= stop_time)
        return slice(start_index, stop_index)

    def _extract(self):
        timeseries = self.profile.timeseries(self.index)
        timeseries, samplenumber = self._resample(timeseries, self.frequency)
        heatindex = self.heatindex(timeseries[0])
        phase_offset = np.arcsin(self.profile.sample.heatindex.threshold)
        step_input = np.cos(timeseries[0]*self.frequency*2*np.pi +
                            phase_offset)
        if not self.profile.normalize:
            step_input *= self.profile.sample.sourcedata.excitation_field_rate
        step_input *= step_input  # squared rate of field variation
        step_input[:heatindex.start] = 0
        sultan_input = np.zeros(samplenumber)
        sultan_input[heatindex] = step_input[heatindex]
        self._data = pandas.DataFrame(
            np.array([timeseries[0], sultan_input,
                      step_input, timeseries[1]]).T,
            columns=['time', 'sultan_input', 'step_input', 'sultan_output'])
        self.reload.data = False

    def _vector(self, column):
        """Return vector from data dataframe."""
        return self.data.loc[:, column]

    @property
    def time(self):
        """Return waveform time array."""
        return self._vector('time')

    @property
    def sultan_input(self):
        """Return waveform sultan input array."""
        return self._vector('sultan_input')

    @property
    def step_input(self):
        """Return waveform step input array."""
        return self._vector('step_input')

    def plot(self):
        """Plot input waveform."""
        axes = plt.subplots(2, 1, sharex=True,
                            gridspec_kw={'height_ratios': [1, 3]})[1]
        self.profile.plot(axes[1])

        axes[0].plot(self.time, self.sultan_input, 'k', label='sultan')
        axes[0].plot(self.time, self.step_input, 'C6--', label='step')
        axes[0].legend()
        #data_label = self.profile.columns['data']
        #plt.plot(self.time, self.heat)


if __name__ == '__main__':

    sample = Sample('CSJA12', 0)
    waveform = WaveForm(sample, 0.5)

    waveform.plot()

    waveform.profile.normalize = False

    waveform.plot()

