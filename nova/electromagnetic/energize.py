"""Manage frame currents."""
from dataclasses import dataclass, field

import numpy as np
import pandas

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.error import SpaceKeyError


@dataclass
class Energize(MetaMethod):
    """Manage dependant frame energization parameters."""

    name = 'energize'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['It', 'nturn'])
    additional: list[str] = field(default_factory=lambda: ['Ic'])
    available: dict[str, bool] = field(default_factory=lambda: {
        'Ic': False, 'nturn': False})
    require_all: bool = False

    def __post_init__(self):
        """Update energize key."""
        if self.generate:
            self.frame.metaframe.energize = ['It']  # set metaframe key
            if np.array([attr in self.frame.metaframe.subspace
                         for attr in self.required]).any():
                self.frame.metaframe.metadata = \
                    {'subspace': self.required+self.additional}
        else:
            self.update_available(self.additional)
        super().__post_init__()

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
        if self.generate and self.frame.get_col(key) == 'It':
            if self.frame.lock('energize') is False \
                    and self.available['nturn']:
                value /= indexer.__getitem__(self._get_key(key, 'nturn'))
                try:
                    self.frame['Ic'] = value
                except SpaceKeyError:
                    if not isinstance(value, pandas.Series):
                        index = self.frame.loc[key[0], key[1]].index
                        value = pandas.Series(value, index)
                    else:
                        index = value.index
                    index = index.intersection(self.frame.subspace.index)
                    self.frame.subspace.loc[index, 'Ic'] = value[index]
                return
        return indexer.__setitem__(key, value)

    def _get_item(self, indexer, key):
        if self.generate and self.frame.get_col(key) == 'It':
            if self.available['Ic'] and self.available['nturn']:
                line_current = indexer.__getitem__(self._get_key(key, 'Ic'))
                turn_number = indexer.__getitem__(self._get_key(key, 'nturn'))
                with self.frame.setlock(True, ['energize', 'subspace']):
                    self._set_item(indexer, key, line_current*turn_number)
        return indexer.__getitem__(key)
