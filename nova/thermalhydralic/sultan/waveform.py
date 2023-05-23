"""Methods to manage single shot Sultan waveform data."""
from dataclasses import dataclass, field, InitVar
from typing import Union
from types import SimpleNamespace

import numpy as np
import pandas
import scipy.interpolate

from nova.thermalhydralic.sultan.profile import Profile
from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
import matplotlib.pyplot as plt


@dataclass
class WaveForm:
    """Manage response waveform."""

    profile: Union[Profile, Sample, Trial, Campaign, str]
    threshold: InitVar[float] = 0.95
    pulse: InitVar[bool] = True
    _data: pandas.DataFrame = field(init=False, repr=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self, threshold, pulse):
        """Init time and heat data fields."""
        self.reload.__init__(threshold=True, data=True)
        if not isinstance(self.profile, Profile):
            self.profile = Profile(self.profile)
        self.threshold = threshold
        self.pulse = pulse

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

    @property
    def pulse(self):
        """Return pulse flag."""
        return self._pulse

    @pulse.setter
    def pulse(self, pulse):
        self._pulse = pulse
        self.reload.data = True

    def _reload(self):
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
        elif self.threshold == -1:
            stop = self.profile.sample.heatindex.stop
        elif self.threshold == 1:
            stop = len(self.profile.time)
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
            stop = np.argmax(cooldown <= minmax[1]-self.threshold*delta)
            stop += start+argmax
        if stop < self.profile.sample.heatindex.stop:
            stop = self.profile.sample.heatindex.stop
        return slice(start, stop)

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

    def _extract_heatindex(self, time):
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
        heatindex = self._extract_heatindex(timeseries[0])
        phase_offset = np.arcsin(self.profile.sample.heatindex.threshold)
        if self.profile.normalize:
            field_amplitude = 1
            fieldrate_amplitude = 1
        else:
            field_amplitude = \
                self.profile.sample.sourcedata.excitation_field
            fieldrate_amplitude = \
                self.profile.sample.sourcedata.excitation_field_rate
        # build profiles
        field = field_amplitude * np.sin(
            timeseries[0]*self.frequency*2*np.pi + phase_offset)
        fieldrate = fieldrate_amplitude * np.cos(
            timeseries[0]*self.frequency*2*np.pi + phase_offset)
        # trim start (threshold==None)
        field[:heatindex.start] = 0
        fieldrate[:heatindex.start] = 0
        # set pulse
        if self.pulse:
            zeroindex = np.full(samplenumber, True)
            zeroindex[heatindex] = False
            field[zeroindex] = 0
            fieldrate[zeroindex] = 0
        # squared
        fieldsq = field**2
        fieldratesq = fieldrate**2
        # store
        self._data = pandas.DataFrame(
            np.array([timeseries[0], field, fieldsq, fieldrate, fieldratesq,
                      timeseries[1]]).T,
            columns=['time', 'field', 'fieldsq', 'fieldrate', 'fieldratesq',
                     'output'])
        self._data.attrs['filename'] = self.profile.sample.filename
        self._data.attrs['heatindex'] = heatindex
        self._data.attrs['field_amplitude'] = field_amplitude
        self._data.attrs['fieldsq_amplitude'] = field_amplitude**2
        self._data.attrs['fieldrate_amplitude'] = fieldrate_amplitude
        self._data.attrs['fieldratesq_amplitude'] = fieldrate_amplitude**2
        self._data.attrs['frequency'] = self.frequency
        self._data.attrs['massflow'] = self.profile.sample.massflow
        self._data.attrs['samplenumber'] = samplenumber
        self.reload.data = False

    @property
    def heatindex(self):
        """Return waveform heatindex."""
        return self.data.attrs['heatindex']

    def timeseries(self, threshold=None, pulse=None,
                   input_variable='fieldratesq'):
        """
        Return waveform timeseries.

        Parameters
        ----------
        threshold : float
            Cooldown threshold >=-1 and <=1.
        pulse : bool
            Heating flag. The default is True
        input_variable : str, optional
            Input variable. The default is 'fieldratesq'.

        Returns
        -------
        time : array-like
            Time array.
        data : array-like
            Data array.

        """
        if threshold is None:
            threshold = self.threshold
        if pulse is None:
            pulse = self.pulse
        _threshold, _pulse = self.threshold, self.pulse
        self.threshold, self.pulse = threshold, pulse
        time, variable = self.data.time, self.data[input_variable]
        output = self.data.output
        self.threshold, self.pulse = _threshold, _pulse
        return time, variable, output

    def plot(self, input_variable='fieldratesq'):
        """Plot target waveform."""
        axes = plt.subplots(2, 1, sharex=True,
                            gridspec_kw={'height_ratios': [4, 1]})[1]
        if input_variable not in self.data:
            raise IndexError(f'Input variable {input_variable} not in '
                             f'{self.data.columns}')
        axes[1].plot(self.data.time, self.data[input_variable], 'C3')
        axes[0].plot(self.data.time, self.data.output, 'C0')
        self.plot_heat(axes[0])
        variable_label = {'field': '$B$ T', 'fieldsq': '$B^2$ T',
                          'fieldrate': r'$\dot{B}$ Ts$^{-1}$',
                          'fieldratesq': r'$\dot{B}^2$ T$^2$s$^{-2}$'}

        input_label = variable_label[input_variable]
        if self.profile.normalize:
            input_label = input_label.split()[0]
        plt.despine()
        axes[1].set_ylabel(input_label)
        axes[0].set_ylabel(r'$\dot{Q}$ W')
        axes[1].set_xlabel('$t$ s')
        axes[0].set_title(self.profile.sample.label)

    def plot_heat(self, axes=None, **kwargs):
        """Shade heated zone."""
        if axes is None:
            axes = plt.gca()
        _threshold = self.threshold
        self.threshold = 1
        time = self.data.time[self.heatindex]
        upper = self.data.output[self.heatindex]
        self.threshold = _threshold
        lower = np.min([np.min(upper), 0]) * np.ones(len(time))
        kwargs = {'color': 'lightgray'} | kwargs
        axes.fill_between(time, lower, upper, **kwargs)


if __name__ == '__main__':

    sample = Sample('CSJA12', 0)
    waveform = WaveForm(sample, 1, pulse=True)
    waveform.profile.normalize = True
    waveform.plot('fieldratesq')
