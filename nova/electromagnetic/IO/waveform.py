
from dataclasses import dataclass, field

import numpy as np
import scipy
import pandas

from nova.utilities.localdata import LocalData
from nova.electromagnetic.IO.read_scenario import scenario_data
from nova.utilities.pyplot import plt


@dataclass
class WaveForm:
    """Extract waveform from DINA simulation."""

    scenario: str
    _sample_frequency: float
    vertical_control: bool = True
    database: LocalData = field(init=False)
    source: pandas.DataFrame = field(init=False, repr=False)
    sample: pandas.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        """Init data structures."""
        self.database = LocalData('waveforms', 'DINA')
        self._extract_source()
        self.sample_frequency = self._sample_frequency

    @property
    def source_frequency(self):
        """Return source sample frequency."""
        return self.source.attrs['fs']

    @property
    def sample_frequency(self):
        """Manage sample frequency."""
        return self._sample_frequency

    @sample_frequency.setter
    def sample_frequency(self, sample_frequency):
        self._sample_frequency = sample_frequency
        self._downsample()

    def _extract_source(self):
        scenario = scenario_data(self.scenario)
        source_columns = ['t', 'Rp', 'ap', 'kp', 'li(3)', 'BETAp',
                          'Rcur', 'Zcur']
        current_iloc = np.unique(scenario.Ic_iloc)
        source_columns.extend([label for label in scenario.index[current_iloc]
                               if 'tf' not in label])
        self.source = scenario.frame.loc[:, source_columns]
        self.source.attrs['dt'] = scenario.dt
        self.source.attrs['fs'] = 1/scenario.dt
        if self.vertical_control:
            self._add_control_currents('VS1')
            self._add_control_currents('VS2')

    def _add_control_currents(self, loop):
        """
        Add vertical stability control currents.

        Parameters
        ----------
        loop : str
            Vertical stability loop index VS1 or VS2.

        Returns
        -------
        None.

        """
        loop_label = f'I{loop.lower()}'
        if loop_label not in self.source:
            raise IndexError(f'loop label {loop_label} '
                             f'not found in {self.source.columns}')
        loop_current = self.source[loop_label]
        if loop == 'VS1':
            coils = [f'Ipf{index}' for index in range(2, 6)]
            factors = [1, 1, -1, -1]
        elif loop == 'VS2':
            coils = ['Ics2u', 'Ics2l']
            factors = [1, -1]
        else:
            raise IndexError(f'loop {loop} not in [VS1, VS2]')
        for coil, factor in zip(coils, factors):
            self.source.loc[:, coil] = self.source.loc[:, coil] + \
                factor * loop_current
        self.source.drop(columns=loop_label, inplace=True)

    def _downsample(self):
        downsample = int(self.source_frequency / self.sample_frequency)
        self.sample = pandas.DataFrame(columns=self.source.columns)
        self.sample.t = self.source.t[::downsample]
        for label in self.source.columns[1:]:
            self.sample.loc[:, label] = scipy.signal.decimate(
                self.source[label], downsample, ftype='iir')
        self.sample.attrs['dt'] = downsample * self.source.attrs['dt']
        self.sample.attrs['fs'] = 1/self.sample.attrs['dt']

    def plot_spectrum(self, substring, attr='source', axes=None,
                      legend=False, title=False, **kwargs):
        """
        Plot signal power spectrum using Welch's method.

        Parameters
        ----------
        substring : str
            Attribute selection substring.
        dataframe : str
            Dataframe identifier, source or sample.

        Returns
        -------
        None.

        """
        if axes is None:
            axes = plt.gca()
        dataframe = getattr(self, attr)
        labels = [label for label in dataframe.columns if substring in label]
        for label in labels:
            derivitive = np.gradient(dataframe[label], dataframe.attrs['dt'])
            f, Pxx = scipy.signal.welch(derivitive, fs=dataframe.attrs['fs'])
            axes.plot(f[1:], Pxx[1:], label=label.upper(), **kwargs)
        plt.xlabel('$f$ Hz')
        plt.ylabel('$P(f)$')
        plt.despine()
        if legend:
            plt.legend(loc='upper right', ncol=2)
        if title:
            plt.title(f'{self.scenario}\n{attr}: '
                      f'fs={dataframe.attrs["fs"]:1.1f}Hz')


if __name__ == '__main__':

    wf = WaveForm('15MA DT-DINA2017-04_v1.2', 30)
    wf = WaveForm('15MA DT-DINA2017-04_v1.2', 30)

    #wf.plot_spectrum('I', attr='source', color='C0')
    #wf.plot_spectrum('I', attr='sample', color='C1')
