import string

import numpy as np
from pandas import DataFrame
from pandas.api.types import is_list_like
from astropy import units


class PoloidalLimit:
    "operating limits for poloidal field coils (PF and CS)"

    _limit_key = {"I": "current", "F": "force", "B": "field"}
    _limit_unit = {"current": "kA", "force": "MN", "field": "T"}
    _limit_bound = 1e16
    _limit_columns = ["limit", "label", "lower", "upper", "unit"]

    def __init__(self):
        self.initialise_limitframe()

    def initialise_limitframe(self):
        "initialize limit dataframe - store limit inputs"
        self._limit = DataFrame(columns=self._limit_columns)
        self._limit.set_index(["limit", "label"], inplace=True)

    def _set_limit(self, variable, label, unit):
        "add limit to dataframe with bounding values"
        if (variable, label) not in self._limit.index:
            self._limit.loc[(variable, label), ["lower", "upper"]] = [
                -self._limit_bound,
                self._limit_bound,
            ]
        self._set_unit(variable, label, unit)

    def _set_unit(self, variable, label, unit):
        if unit is None:
            unit = self._limit_unit[variable]
        self._limit.loc[(variable, label), "unit"] = unit

    def _get_index(self, limit):
        variable = self._limit_key[limit[0]]
        label = limit[1:]
        return variable, label

    def add_limit(self, bound="both", eps=1e-2, unit=None, **limits):
        """
        Attributes:
            limits (dict): listing of limits key: value
                           set limit key as ICSsum for [I][CSsum] etc...
            bound (str): set bounds [lower, upper, both, equal]
        """
        if bound == "both" or bound == "equal":
            bounds = ["lower", "upper"]
        else:
            bounds = bound
        for limit in limits:
            variable, label = self._get_index(limit)
            value = limits[limit]
            self._set_limit(variable, label, unit)
            if bound == "equal":
                value = value + eps * np.array([-1, 1])
            elif bound == "both":
                value = abs(value) * np.array([-1, 1])
            self._limit.loc[(variable, label), bounds] = value

    def drop_limit(self, limits=None):
        if limits is None:  # clear all limit
            self.initialise_limitframe()
        else:  # drop specified limits
            if not is_list_like(limits):
                limits = [limits]
            for limit in limits:
                variable, label = self._get_index(limit)
                self._limit.drop(index=(variable, label), inplace=True)

    def initialize_limit(self, variable, index):
        # initalize limit with default bounds
        limit = DataFrame(index=index, columns=self._limit_columns[2:])
        for bound, sign in zip(["lower", "upper"], [-1, 1]):
            limit.loc[:, bound] = sign * self._limit_bound
        limit.loc[:, "unit"] = self._limit_unit[variable]
        return limit

    def update_unit(self, limit, output_unit=None):
        if output_unit is not None:
            for input_unit in limit.unit.unique():
                if input_unit != output_unit:
                    index = limit.unit == input_unit
                    limit.loc[index, ["lower", "upper"]] *= units.Unit(input_unit).to(
                        output_unit
                    )
                    limit.loc[index, "unit"] = output_unit

    def get_limit(self, variable, index=None, unit=None):
        if index is None:
            if not hasattr(self, "coil"):
                raise IndexError(
                    "coil_index must be specified " "when coilset not present"
                )
            else:
                index = self.coil.index
        limit = self.initialize_limit(variable, index)
        if not self._limit.empty:
            _limit = self._limit.xs(variable)
            for label in limit.index:
                if label in _limit.index:  # single label
                    limit.loc[label, :] = _limit.loc[label, :]
                else:
                    part_label = label.rstrip(string.ascii_letters)
                    part_label = part_label.rstrip(string.digits)
                    if part_label in _limit.index:  # group
                        limit.loc[label, :] = _limit.loc[part_label, :]
        self.update_unit(limit, output_unit=unit)
        return limit

    def load_ITER_limits(self):
        "add default limits for ITER coil-set"
        # kA current limits
        self.add_limit(ICS=45)
        self.add_limit(IPF1=48, IPF2=55, IPF3=55, IPF4=55, IPF5=52, IPF6=52)
        self.add_limit(Iimb=22.5)  # imbalance current
        self.add_limit(FCSsep=240, bound="upper")  # force limits
        self.add_limit(FCSsum=60, bound="both")
        self.add_limit(
            FPF1=-150, FPF2=-75, FPF3=-90, FPF4=-40, FPF5=-10, FPF6=-190, bound="lower"
        )
        self.add_limit(
            FPF1=110, FPF2=15, FPF3=40, FPF4=90, FPF5=160, FPF6=170, bound="upper"
        )

    """
    def get_PFz_limit(self):
        PFz_limit = np.zeros((self.nPF, 2))
        for i, coil in enumerate(self.PFcoils):
            if coil in self._limit['F']:  # per-coil
                PFz_limit[i] = self._limit['F'][coil]
            elif coil[:2] in self._limit['F']:  # per-set
                PFz_limit[i] = self._limit['F'][coil[:2]]
            else:  # no limit
                PFz_limit[i] = [-self._bound, self._bound]
        return PFz_limit

    def get_CSsep_limit(self):
        CSsep_limit = np.zeros((self.nCS - 1, 2))
        for i in range(self.nCS - 1):  # gaps, bottom-top
            gap = 'CS{}sep'.format(i)
            if gap in self._limit['F']:  # per-gap
                CSsep_limit[i] = self._limit['F'][gap]
            elif 'CSsep' in self._limit['F']:  # per-set
                CSsep_limit[i] = self._limit['F']['CSsep']
            else:  # no limit
                CSsep_limit[i] = [-self._bound, self._bound]
        return CSsep_limit

    def get_CSsum_limit(self):
        CSsum_limit = np.zeros((1, 2))
        if 'CSsum' in self._limit['F']:  # per-set
            CSsum_limit = self._limit['F']['CSsum']
        else:  # no limit
            CSsum_limit = [-self._bound, self._bound]
        return CSsum_limit
    
    def get_CSaxial_limit(self):
        CSaxial_limit = np.zeros((self.nCS + 1, 2))
        for i in range(self.nCS + 1):  # gaps, top-bottom
            gap = 'CS{}axial'.format(i)
            if gap in self._limit['F']:  # per-gap
                CSaxial_limit[i] = self._limit['F'][gap]
            elif 'CSaxial' in self._limit['F']:  # per-set
                CSaxial_limit[i] = self._limit['F']['CSaxial']
            else:  # no limit
                CSaxial_limit[i] = [-self._bound, self._bound]
        return CSaxial_limit
    """


if __name__ == "__main__":
    pl = PoloidalLimit()
    pl.add_limit(ICS=40, bound="lower")

    pl.load_ITER_limits()

    print(pl.get_limit("current", ["CS1U", "CS2U"], "A"))
    print(pl._limit)
