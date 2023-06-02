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

    order: Union[tuple[int], list[int], int] = field(default=(5,))
    delay: bool = True
    _pole: InitVar[Union[float, list[float]]] = 0.6
    _dcgain: InitVar[float] = 1.0
    _time_delay: InitVar[float] = 5.0
    _vector: list[float] = field(default_factory=list)
    lti: scipy.signal.lti = field(init=False, repr=False)
    reload: bool = False

    def __post_init__(self, _pole, _dcgain, _time_delay):
        """Init linear time-invariant model."""
        if not isinstance(self.order, Iterable):
            self.order = [self.order]
        if not self._vector:  # vector unset - initialize with pole, time_delay
            self._vector = self._sead(_pole, _dcgain, _time_delay)
        self.vector = self._vector[: self.parameter_number]

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
        if len(vector) != self.system_number + 2:
            raise IndexError(
                f"sead length {len(vector)} != "
                f"maximum parameter number "
                f"{self.system_number + 2}\n"
                f"check _pole kwarg {_pole}"
            )
        return vector

    @property
    def vector(self):
        """
        Manage optimization vector.

        Parameters
        ----------
        vector : list
            Optimization vector [*repeated_pole, dcgain, time_delay].

        """
        return self._vector[: self.parameter_number]

    @vector.setter
    def vector(self, vector):
        self._vector[: self.parameter_number] = vector[: self.parameter_number]
        self._generate()  # regenerate lti

    def update_pole(self, pole):
        """Update pole."""
        self._vector[: self.system_number] = pole * np.ones(self.system_number)
        self._generate()  # regenerate lti

    def update_dcgain(self, dcgain):
        """Update dc gain."""
        self._vector[self.system_number] = dcgain
        self._generate()  # regenerate lti

    @property
    def system_number(self):
        """Return system number, read-only."""
        return len(self.order)

    @property
    def parameter_number(self):
        """Return number of optimization parameters, read-only."""
        parameter_number = self.system_number + 1
        if self.delay:
            parameter_number += 1
        return parameter_number

    @property
    def repeated_pole(self):
        """Return repeated system poles, read-only."""
        return self._vector[: self.system_number]

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
        """Return LTI model system gain, read-only."""
        return self.dcgain * np.prod(np.array(self.repeated_pole) ** self.order)

    @property
    def time_delay(self):
        """Return time delay, read-only."""
        delay = self._vector[-1] if self.delay else 0
        return delay

    @property
    def label(self):
        """Return transfer function text descriptor."""
        numerator = f"{self.system_gain:1.5f}"
        denominator = "".join(
            [
                rf"(s+{pole:1.3f})^{order}"
                for pole, order in zip(self.repeated_pole, self.order)
            ]
        )
        transferfunction = rf"$\frac{{{numerator}}}{{{denominator}}}$"
        return transferfunction

    def get_label(self, massflow):
        """Return label factored by massflow."""
        pole_coefficent = np.array(self.repeated_pole) / massflow
        dcgain = self.dcgain
        time_delay = self.time_delay
        numerator = rf"{dcgain:1.2f}"
        denominator = "".join(
            [
                rf"(s+{coefficent:1.3f}\dot{{m}})^{order}"
                for coefficent, order in zip(pole_coefficent, self.order)
            ]
        )
        numerator += "".join(
            [
                rf"({coefficent:1.3f}\dot{{m}})^{order}"
                for coefficent, order in zip(pole_coefficent, self.order)
            ]
        )
        if self.delay:
            numerator += rf"\,\,{{e}}^{{-{time_delay:1.2f}s}}"
        transferfunction = rf"$\frac{{{numerator}}}{{{denominator}}}$"
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

    @property
    def coefficents(self) -> pandas.Series:
        """Return model coefficents."""
        coefficents = {}
        for attr in ["order", "repeated_pole", "dcgain", "time_delay"]:
            coefficents[attr] = getattr(self, attr)
        coefficents["delay"] = 1 if self.delay else 0
        return pandas.Series(coefficents)


if __name__ == "__main__":
    model = Model(6, False, _dcgain=20.5)
    print(model, model.vector)
    model.delay = True
    print(model, model.vector)
    model.delay = False
    print(model, model.vector)
