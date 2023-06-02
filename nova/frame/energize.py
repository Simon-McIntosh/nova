"""Manage frame currents."""
from dataclasses import dataclass, field

import pandas

import nova.frame.metamethod as metamethod
from nova.frame.dataframe import DataFrame
from nova.frame.error import SpaceKeyError


@dataclass
class Energize(metamethod.Energize):
    """Manage dependant frame energization parameters."""

    frame: DataFrame = field(repr=False)

    def initialize(self):
        """Set attribute avalibility flags and columns."""
        for attr in self.available:
            self.available[attr] = attr in self.frame.columns

    def _get_key(self, key, col=None):
        if col is None:
            return key
        if isinstance(key, tuple):
            if isinstance(key[-1], int):
                if not isinstance(col, int):
                    col = self.frame.columns.get_loc(col)
            return (*key[:-1], col)
        return col

    def _set_item(self, indexer, key, value):
        if self.generate and self.frame.get_col(key) == "It":
            if self.frame.lock("energize") is False and self.available["nturn"]:
                value /= indexer.__getitem__(self._get_key(key, "nturn"))
                try:
                    self.frame["Ic"] = value
                except SpaceKeyError:
                    if not isinstance(value, pandas.Series):
                        index = self.frame.loc[key[0], key[1]].index
                        value = pandas.Series(value, index)
                    else:
                        index = value.index
                    index = index.intersection(self.frame.subspace.index)
                    self.frame.subspace.loc[index, "Ic"] = value[index]
                return
        return indexer.__setitem__(key, value)

    def _get_item(self, indexer, key):
        if self.generate and self.frame.get_col(key) == "It":
            if self.available["Ic"] and self.available["nturn"]:
                line_current = indexer.__getitem__(self._get_key(key, "Ic"))
                turn_number = indexer.__getitem__(self._get_key(key, "nturn"))
                with self.frame.setlock(True, ["energize", "subspace"]):
                    self._set_item(indexer, key, line_current * turn_number)
        return indexer.__getitem__(key)
