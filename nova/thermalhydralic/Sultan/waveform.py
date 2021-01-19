"""Methods to manage single shot Sultan waveform data."""
from dataclasses import dataclass, field
from typing import Union
from types import SimpleNamespace

import numpy as np
import pandas

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
    _upsample: float = 21
    _index: slice = field(init=False)
    _data: pandas.DataFrame = field(init=False, repr=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init time and heat data fields."""
        self.reload.__init__(threshold=True, upsample=True,
                             index=True, data=True)
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
        if threshold < -1 or threshold > 1:
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

    def propagate_reload(self):
        """Propagate reload flags."""
        if self.profile.sample.sampledata.reload.waveform:
            self.reload.index = True
            self.reload.data = True
            self.profile.sample.sampledata.reload.waveform = False

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
        if self.threshold < 0:
            threshold = 1+self.threshold
            index = slice(self.profile.sample.heatindex.stop, None)
            pulse = self.profile.profile(index=index)[1].values
            argmax = pulse.argmax()
            postheat = pulse[:argmax]
            minmax = postheat.min(), postheat.max()
            delta = np.diff(minmax)[0]
            print(minmax[0] - self.threshold*delta)
            stop = np.argmax(postheat >= minmax[0] + threshold*delta)
            stop += self.profile.sample.heatindex.stop
        else:
            pulse = self.profile.profile(index=slice(start, None))[1].values
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

    def excitation(self, axes):
        time = self.profile.profile(slice(None, self.index.stop))[0]
        print(time[:10])
        heatindex = self.profile.sample.heatindex.index
        data = np.zeros(len(time))
        data[heatindex] = np.cos(
            time[heatindex]*self.profile.sample.sourcedata.frequency*2*np.pi)
        axes.plot(time, data**2)

    def plot(self):
        """Plot input waveform."""
        axes = plt.subplots(2, 1, sharex=True,
                            gridspec_kw={'height_ratios': [1, 3]})[1]
        self.profile.plot(axes[1])

        axes[1].plot(*self.profile.profile(self.index), 'C1')
        self.excitation(axes[0])
        #plt.plot(self.time, self.heat)


if __name__ == '__main__':

    sample = Sample('CSJA13', 0)
    waveform = WaveForm(sample, 1)

    waveform.plot()

