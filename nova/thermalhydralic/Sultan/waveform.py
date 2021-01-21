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
    pulse: bool = True
    _data: pandas.DataFrame = field(init=False, repr=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init time and heat data fields."""
        self.reload.__init__(threshold=True, data=True)
        if not isinstance(self.profile, Profile):
            self.profile = Profile(self.profile)

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
        self.reload.data = True

    def propagate_reload(self):
        """Propagate reload flags."""
        if self.profile.sample.sourcedata.reload.waveform:
            self.reload.data = True
            self.profile.sample.sourcedata.reload.waveform = False
        if self.profile.sample.sampledata.reload.waveform:
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
        """Return waveform index."""
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
        return slice(start, stop)

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

    def _resample(self, timeseries, excitation_frequency, upsample=31):
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
        if sample_frequency < upsample*excitation_frequency:
            timestep = 1 / (upsample*excitation_frequency)
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
        if self.profile.normalize:
            waveform_amplitude = 1
        else:
            waveform_amplitude = \
                self.profile.sample.sourcedata.excitation_field_rate
        waveform_input = waveform_amplitude * np.cos(
            timeseries[0]*self.frequency*2*np.pi + phase_offset)
        waveform_input *= waveform_input  # square input
        waveform_input[:heatindex.start] = 0  # trim start (threshold==None)
        if self.pulse:
            zeroindex = np.full(samplenumber, True)
            zeroindex[heatindex] = False
            waveform_input[zeroindex] = 0
        self._data = pandas.DataFrame(
            np.array([timeseries[0], waveform_input, timeseries[1]]).T,
            columns=['time', 'waveform_input', 'heat_output'])
        self._data.attrs['waveform_amplitude'] = waveform_amplitude**2
        self._data.attrs['frequency'] = self.frequency
        self._data.attrs['massflow'] = self.profile.sample.metadata[
            (f'dm/dt {self.profile.sample.side[0]}', 'g/s')]
        self._data.attrs['samplenumber'] = samplenumber

        self.reload.data = False

    @property
    def time(self):
        """Return time array."""
        return self.data.loc[:, 'time']

    @property
    def waveform_input(self):
        """Return waveform input array."""
        return self.data.loc[:, 'waveform_input']

    @property
    def heat_output(self):
        """Return heat output array."""
        return self.data.loc[:, 'heat_output']

    def plot(self):
        """Plot target waveform."""
        axes = plt.subplots(2, 1, sharex=True,
                            gridspec_kw={'height_ratios': [1, 3]})[1]
        self.profile.plot(axes[1])
        axes[0].plot(self.time, self.waveform_input, 'C7')
        axes[1].plot(self.time, self.heat_output, 'C1', lw=5)


if __name__ == '__main__':

    sample = Sample('CSJA13', 0)
    waveform = WaveForm(sample, 0.9, pulse=True)
    waveform.plot()



