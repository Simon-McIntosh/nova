
from dataclasses import dataclass, field, InitVar
from typing import List, Tuple
from types import SimpleNamespace

import numpy as np
import pandas

from nova.thermalhydralic.Sultan.sultanshot import SultanShot
from nova.thermalhydralic.Sultan.sultanprofile import SultanProfile
from nova.thermalhydralic.Sultan.sultanplot import SultanPlot

from nova.utilities.pyplot import plt


@dataclass
class HeatIndex:
    """Index external heating."""

    data: pandas.DataFrame = field(repr=False)
    _threshold: float = 0.95
    _index: slice = field(init=False, default=None)
    reload: SimpleNamespace = field(
        init=False, repr=False,
        default=SimpleNamespace(threshold=True, index=True))

    @property
    def threshold(self):
        """
        Manage heat threshold parameter.

        Parameters
        ----------
        threshold : float
            Heating idexed as current.abs > threshold * current.abs.max.

        Raises
        ------
        ValueError
            threshold must lie between 0 and 1.

        Returns
        -------
        threshold : float

        """
        if self.reload.threshold:
            self.threshold = self._threshold
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        if threshold < 0 or threshold > 1:
            raise ValueError(f'heat threshold {threshold} '
                             'must lie between 0 and 1')
        self._threshold = threshold
        self.reload.threshold = False
        self.reload.index = True

    @property
    def index(self):
        """
        Return heat index, slice, read-only.

        Evaluated as current.abs() > threshold * current.abs().max()

        """
        if self.reload.index:
            current = self.data.loc[:, ('Ipulse', 'A')]
            abs_current = current.abs()
            max_current = abs_current.max()
            threshold_index = np.where(abs_current >=
                                       self.threshold*max_current)[0]
            self._index = slice(threshold_index[0], threshold_index[-1]+1)
        self.reload.index = False
        return self._index

    @property
    def start(self):
        """Return start index."""
        return self.index.start

    @property
    def stop(self):
        """Return stop index."""
        return self.index.stop

    @property
    def time(self):
        """Return start / end time tuple of input heating period, read-only."""
        return self.data.loc[[self.start, self.stop], ('t', 's')].values

    @property
    def time_delta(self):
        """Return heating period, read-only."""
        return np.diff(self.time)[0]


@dataclass
class DataInstance:
    """Extract profile.data instance."""

    data: InitVar[pandas.DataFrame]
    index: int
    time: float = field(init=False)
    value: float = field(init=False)
    time_column: Tuple[str] = field(default=('t', 's'), repr=False)
    data_column: Tuple[str] = field(default=('Qdot', 'W'), repr=False)

    def __post_init__(self, data):
        """Set data values."""
        self.time = data.loc[self.index, self.time_column]
        self.value = data.loc[self.index, self.data_column]


@dataclass
class Response:
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
        Time and heat output values at max input heating.
    minimum: HeatInstance
        Time and heat output values at minimum input heating.
    delta: float
        Delta heating within index.
    reload: SimpleNamespace
        Boolean reload flags.

    """

    data: pandas.DataFrame = field(repr=False)
    heat_index: HeatIndex
    steady_threshold: float = 1.05

    @property
    def start(self):
        """Return offset heat datainstance at index.start."""
        return DataInstance(self.data, self.heat_index.start)

    @property
    def stop(self):
        """Return end heat datainstance at index.stop."""
        return DataInstance(self.data, self.heat_index.stop)

    @property
    def maximum(self):
        """Return max heat datainstance."""
        max_index = np.argmax(self.data[('Qdot', 'W')].abs())
        return DataInstance(self.data, max_index)

    @property
    def minimum(self):
        """Return minimum heat datainstance."""
        min_index = np.argmin(self.data[('Qdot', 'W')].abs())
        return DataInstance(self.data, min_index)

    @property
    def delta(self):
        """Return delta heating within self.index."""
        index_heat = self.data.loc[self.heat_index.index, ('Qdot', 'W')]
        maximum_heat = np.max(index_heat)
        minimum_heat = np.min(index_heat)
        return maximum_heat-minimum_heat

    @property
    def maximum_ratio(self):
        """Return ratio of max to stop heat."""
        return self.maximum.value/self.stop.value

    @property
    def minimum_ratio(self):
        """Return ratio of stop-minimum to delta heat."""
        return (self.stop.value-self.minimum.value) / self.delta

    @property
    def index_ratio(self):
        """Return ration of start-stop heat."""
        return self.start.value / self.stop.value

    @property
    def steady(self):
        """
        Return steady flag.

        False if any:
        - max/end heat ratio > steady_threshold
        -
        """
        if self.maximum_ratio > self.steady_threshold:
            steady = False
        elif self.minimum_ratio > self.steady_threshold:
            steady = False
        elif self.index_ratio > 1:
            steady = False
        else:
            steady = True
        return steady

    @property
    def steady_status(self):
        """Return pandas.DataFrame detailing stability metrics."""
        status = pandas.DataFrame(index=['maximum', 'minimum', 'index'],
                                  columns=['ratio', 'steady'])
        for name in status.index:
            status.loc[name, 'ratio'] = getattr(self, f'{name}_ratio')
            status.loc[name, 'steady'] = \
                status.loc[name, 'ratio'] < self.steady_threshold
        return status


    def plot(self, ax=None):
        """

        Parameters
        ----------
        ax : axis, optional
            plot axis. The default is None (plt.gca())

        Returns
        -------
        None

        """
        if ax is None:
            ax = plt.gca()
        ax.plot(t_eoh, Qdot_eoh, **self._get_marker(steady, 'eoh'))
        ax.plot(t_max, Qdot_max, **self._get_marker(steady, 'max'))


if __name__ == '__main__':

    shot = SultanShot('CSJA_3')
    profile = SultanProfile(shot)

    response = Response(profile.lowpassdata, HeatIndex(profile.rawdata))



    #post = SultanPostProcess(profile)
    #post.extract_response(plot=True)
