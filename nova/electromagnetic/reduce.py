"""Calculate reduction indices."""
from dataclasses import dataclass, field

import numpy as np
import pandas

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame


@dataclass
class Reduce(MetaMethod):
    """Calculate reduction indices for reduceat."""

    name = 'reduce'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['frame', 'ref'])
    require_all: bool = True
    index: pandas.Index = field(default=pandas.Index([]))
    indices: list[int] = field(init=False, repr=False)

    def initialize(self):
        ref = np.array(self.frame.ref)

        if np.all(ref[:-1] <= ref[1:]):
           self.indices = np.unique(ref, return_index=True)
           return

        '''
            if (coil == self._default_attributes['coil']).all():
                _reduction_index = np.arange(self._nC)
            else:
                _name = coil[0]
                _reduction_index = [0]
                for i, name in enumerate(coil):
                    if name != _name:
                        _reduction_index.append(i)
                        _name = name
            self._reduction_index = np.array(_reduction_index)
            self._plasma_iloc = np.arange(len(self._reduction_index))[
                self.plasma[self._reduction_index]]
            filament_indices = np.append(self._reduction_index, self.coil_number)
            plasma_filaments = filament_indices[self._plasma_iloc+1] - \
                filament_indices[self._plasma_iloc]
            self._plasma_reduction_index = \
                np.append(0, np.cumsum(plasma_filaments)[:-1])
        '''
