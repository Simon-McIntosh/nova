"""Manage linear time invariant model."""
from dataclasses import dataclass, field, InitVar
from typing import Union

import pandas
import scipy
import numpy as np

from nova.utilities.pyplot import plt


@dataclass
class Model:
    """Manage linear time-invariant model."""

    order: tuple[int] = field(default=(6,))
    _vector: list[float] = field(init=False, default_factory=list)
    _pole: InitVar[Union[float, list[float]]] = 0.5
    _delay: InitVar[float] = 5
    lti: scipy.signal.lti = field(init=False, repr=False)

    def __post_init__(self, _pole, _delay):
        """Init linear time-invariant model."""
        if not self._vector:  # vector unset
            self.vector = self._sead(_pole, _delay)
        else:
            self.vector = self._vector

    def _sead(self, _pole, _delay):
        """
        Return sead optimization vector.

        Parameters
        ----------
        pole : float
            Location of repeated pole (positive).
        delay : float
            Time delay.

        Returns
        -------
        sead_vector : array-like
            Sead optimization vector [*pole, gain, delay].

        """
        if not pandas.api.types.is_list_like(_pole):
            _pole = [_pole for __ in range(self.system_number)]
        else:
            _pole = list(_pole)
        vector = _pole + [1, _delay]
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
            Optimization vector [pole, gain, delay].

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
    def pole_gain(self):
        """Return steady state pole gain, read-only."""
        return np.prod(np.array(self.repeated_pole)**self.order)

    @property
    def gain(self):
        """Return gain."""
        return self._vector[-2]

    @property
    def delay(self):
        """Return time delay."""
        return self._vector[-1]

    @property
    def step(self):
        """Manage steady state step response."""
        return self.gain / self.pole_gain

    @step.setter
    def step(self, step):
        self._vector[-2] = self.pole_gain*step

    @property
    def label(self):
        """Return transfer function text descriptor."""
        numerator = f'{self.gain:1.4f}'
        denominator = ''.join([fr'(s+{pole:1.3f})^{order}'
                               for pole, order in
                               zip(self.repeated_pole, self.order)])
        plt.title(fr'${denominator}$')
        #return (fr'$\frac{{{self.gain:1.4f}}}'
        #        fr'{{(s+{self.pole:1.3f})^{self.npole}}}'
        #        fr'{{\rm e}}^{{-{self.delay:1.2f}s}}$')

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
        self.lti = scipy.signal.ZerosPolesGain([], self.pole, self.gain)


if __name__ == '__main__':

