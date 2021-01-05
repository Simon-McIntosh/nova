"""Methods to manage single shot Sultan waveform data."""
from dataclasses import dataclass, field, InitVar
from types import SimpleNamespace

import numpy as np
import pandas


@dataclass
class HeatIndex:
    """Index external heating."""

    data: pandas.DataFrame = field(repr=False)
    _threshold: float = 0.9
    _index: slice = field(init=False, default=None)
    reload: SimpleNamespace = field(init=True, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init reload namespace."""
        self.reload.__init__(threshold=True, index=True)

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
        """Return maximum heat datainstance."""
        max_index = np.argmax(self.data[('Qdot_norm', 'W')].abs())
        return DataInstance(self.data, max_index)

    @property
    def minimum(self):
        """Return minimum heat datainstance."""
        min_index = np.argmin(self.data[('Qdot_norm', 'W')].abs())
        return DataInstance(self.data, min_index)

    @property
    def delta(self):
        """Return delta heating within self.index."""
        index_heat = self.data.loc[self.heat_index.index, ('Qdot_norm', 'W')]
        maximum_heat = np.max(index_heat)
        minimum_heat = np.min(index_heat)
        return maximum_heat-minimum_heat

    @property
    def impulse(self):
        """Return average impulse power response."""
        start_index = self.heat_index.start
        time = self.data.loc[start_index:, ('t', 's')]
        Qdot = self.data.loc[start_index:, ('Qdot_norm', 'W')]
        return np.trapz(Qdot, time) / (self.stop.time - self.start.time)

    @property
    def step(self):
        """Return heat step response."""
        t = self.data.loc[self.heat_index.index, ('t', 's')].values
        Qdot = self.data.loc[self.heat_index.index, ('Qdot_norm', 'W')].values
        return t-t[0], Qdot-Qdot[0]

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

    @property
    def dataseries(self):
        """Return response data series."""
        return pandas.Series([self.stop.value-self.start.value,
                              self.maximum.value-self.start.value,
                              self.impulse-self.start.value,
                              self.steady],
                             index=['stop', 'maximum', 'impulse', 'steady'])
