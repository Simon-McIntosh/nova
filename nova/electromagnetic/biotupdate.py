"""Update methods for BiotFrame."""
from dataclasses import dataclass, field

import numpy as np
import pandas

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.error import SubSpaceLockError


@dataclass
class BiotUpdate(MetaMethod):
    """Manage Biot update flags."""

    name = 'biotupdate'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['Ic', 'nturn'])
    update: dict[str: bool] = field(init=False, default_factory=lambda: {
        'Ic': True, 'It': True, 'nturn': True})
    plasma_turns: bool = field(init=False, default=True)
    coil_current: bool = field(init=False, default=True)
    plasma_current: bool = field(init=False, default=True)

    def initialize(self):
        """Provide initialize interface."""
        pass


    def _set_item(self, indexer, key, value):
        if self.generate and self.frame.get_col(key) == 'It':
            if self.frame.lock('energize') is False \
                    and self.available['nturn']:
                value /= indexer.__getitem__(self._get_key(key, 'nturn'))
                try:
                    self.frame['Ic'] = value
                except SubSpaceLockError:
                    if not isinstance(value, pandas.Series):
                        index = self.frame.loc[key[0], key[1]].index
                        value = pandas.Series(value, index)
                    else:
                        index = value.index
                    index = index.intersection(self.frame.subspace.index)
                    self.frame.subspace.loc[index, 'Ic'] = value[index]
                return
        return indexer.__setitem__(key, value)

