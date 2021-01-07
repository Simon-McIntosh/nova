"""Methods to manage single shot Sultan waveform data."""
from dataclasses import dataclass, field, InitVar

import numpy as np
import pandas

from nova.thermalhydralic.sultan.stepresponse import ModelResponse
from nova.thermalhydralic.sultan.shotprofile import HeatIndex


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

    data: pandas.DataFrame = field(repr=False)
    heat_index: HeatIndex
    npole: int = 6
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
    def energy(self):
        """Return intergral power."""
        start_index = self.heat_index.start
        time = self.data.loc[start_index:, ('t', 's')]
        Qdot = self.data.loc[start_index:, ('Qdot_norm', 'W')]
        return np.trapz(Qdot, time)

    @property
    def stepdata(self):
        """Return offset heat step response data."""
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

    def stepresponse(self):
        """
        Return thermo-hydralic model parameters.

        Parameters
        ----------
        npole : int, optional
            Number of repeated poles. The default is 6.

        Returns
        -------
        vector : array-like
            Optimization vector [pole, gain, delay].
        steady_state : float
            Step response steady state.

        """
        response = ModelResponse(*self.stepdata, self.npole)
        vector = response.fit()
        steady_state = response.model.steady_state
        return vector, steady_state

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
        (pole, gain, delay), step
        return pandas.Series([self.stop.value-self.start.value,
                              self.maximum.value-self.start.value,
                              self.steady],
                             index=['stop', 'maximum', 'steady'])
