"""Manage linear time invariant model."""
from dataclasses import dataclass, field, InitVar
from typing import Union
from collections.abc import Iterable

import pandas
import scipy
import numpy as np


@dataclass
class Model:
    """Manage linear time-invariant model."""

    order: Union[tuple[int], list[int], int] = field(default=(6,))
    _pole: InitVar[Union[float, list[float]]] = 0.6
    _dcgain: InitVar[float] = 1
    _time_delay: InitVar[float] = 5
    _vector: list[float] = field(default_factory=list)
    lti: scipy.signal.lti = field(init=False, repr=False)
    reload: bool = False

    def __post_init__(self, _pole, _dcgain, _time_delay):
        """Init linear time-invariant model."""
        if not isinstance(self.order, Iterable):
            self.order = [self.order]
        if not self._vector:  # vector unset - initialize with pole, time_delay
            self.vector = self._sead(_pole, _dcgain, _time_delay)
        else:
            self.vector = self._vector

    def _sead(self, _pole, _dcgain, _time_delay):
        """
        Return sead optimization vector.

        Parameters
        ----------
        pole : float
            Location of repeated pole (positive).
        dcgain : float
            DC gain.
        time_delay : float
            Time delay.

        Returns
        -------
        sead_vector : array-like
            Sead optimization vector [*pole, dcgain, time_delay].

        """
        if not pandas.api.types.is_list_like(_pole):
            _pole = [_pole for __ in range(self.system_number)]
        else:
            _pole = list(_pole)
        vector = _pole + [_dcgain, _time_delay]
        if len(vector) != self.parameter_number:
            raise IndexError(f'sead length {len(vector)} != '
                             f'parameter number {self.parameter_number}\n'
                             f'check _pole kwarg {_pole}')
        return vector

    @property
    def vector(self):
        """
        Manage optimization vector.

        Parameters
        ----------
        vector : list
            Optimization vector [pole, gain, time_delay].

        """
        return self._vector

    @vector.setter
    def vector(self, vector):
        self._vector = vector
        self._generate()  # regenerate lti

    @property
    def system_number(self):
        """Return system number, read-only."""
        return len(self.order)

    @property
    def parameter_number(self):
        """Return number of optimization parameters, read-only."""
        return len(self.order) + 2

    @property
    def repeated_pole(self):
        """Return repeated system poles."""
        return self._vector[:self.system_number]

    @property
    def pole(self) -> list[float]:
        """Return negated poles."""
        pole_list = []
        for pole, order in zip(self.repeated_pole, self.order):
            pole_list += [-pole for __ in range(order)]
        return pole_list

    @property
    def dcgain(self):
        """Return system steady state gain, read-only."""
        return self._vector[-2]

    @property
    def system_gain(self):
        """Return LTI model system gain, k."""
        return self.dcgain * \
            np.prod(np.array(self.repeated_pole)**self.order)

    @property
    def time_delay(self):
        """Return time delay."""
        return self._vector[-1]

    @property
    def label(self):
        """Return transfer function text descriptor."""
        numerator = f'{self.system_gain:1.5f}'
        denominator = ''.join([fr'(s+{pole:1.3f})^{order}'
                               for pole, order in
                               zip(self.repeated_pole, self.order)])
        transferfunction = fr'$\frac{{{numerator}}}{{{denominator}}}$'
        return transferfunction

    def _generate(self):
        """
        Generate linear time-invariant model.

        Parameters
        ----------
        x : array-like
            zeros, poles, gain.
        npole : int
            Number of poles.

        Returns
        -------
        lti : lti
            linear time-invariant model.

        """
        self.lti = scipy.signal.ZerosPolesGain([], self.pole, self.system_gain)
        self.reload = True


if __name__ == '__main__':

    model = Model(6, _dcgain=20.5)
    print(model.label)
    print(scipy.signal.step(model.lti, T=[0, 1e3])[1])
    print(model.dcgain)

