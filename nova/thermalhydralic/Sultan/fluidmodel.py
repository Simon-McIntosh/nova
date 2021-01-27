"""Apply LTI model to fluid timeseries."""
from dataclasses import dataclass, field
from typing import Union

import scipy
import numpy as np

from nova.thermalhydralic.sultan.model import Model
from nova.utilities.pyplot import plt


@dataclass
class FluidModel:
    """Calculate signal response of input LTI model."""

    _model: Union[Model, list[int], int]
    _timeseries: tuple[np.array, np.array] = \
        field(repr=False, default_factory=tuple)
    _output: np.array = field(init=False, repr=False)
    reload: bool = field(default=True)

    def __post_init__(self):
        """Init LTI model."""
        if not isinstance(self._model, Model):
            self._model = Model(self._model)

    @property
    def model(self):
        """Return LTI model."""
        return self._model

    @model.setter
    def model(self, model):
        self.reload = True
        self._model = model

    @property
    def timeseries(self):
        """Return input timeseries."""
        return self._timeseries

    @timeseries.setter
    def timeseries(self, timeseries):
        self.reload = True
        self._timeseries = timeseries

    @property
    def time(self):
        """Return time array."""
        return self._timeseries[0]

    @property
    def signal(self):
        """Return signal array."""
        return self._timeseries[1]

    @property
    def output(self):
        """Return model response."""
        if self.reload or self.model.reload:
            self.reload = False
            self.model.reload = False
            self._output = scipy.signal.lsim2(self.model.lti, self.signal,
                                              T=self.time, atol=5e-4)[1]
            if self.model.time_delay > 0:
                self._output = self._timeshift(self._output)
        return self._output

    def _timeshift(self, output):
        """
        Return response shifted in time by delay.

        Parameters
        ----------
        time : array-like
            time array.
        response : array-like
            model step response.

        Returns
        -------
        response : array-like
            model step response shifted by self.time_delay seconds.

        """
        bounds = (output[0], output[-1])
        return scipy.interpolate.interp1d(
            self.time+self.model.time_delay, output, fill_value=bounds,
            bounds_error=False)(self.time)

    def plot(self, axes=None, **kwargs):
        """Plot model output."""
        if axes is None:
            axes = plt.gca()
        axes.plot(self.time, self.output, **kwargs)
