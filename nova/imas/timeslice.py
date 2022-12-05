"""Manage access to timeslice data."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING


import numpy as np
if TYPE_CHECKING:
    import xarray

from nova.imas.database import ImasIds


@dataclass
class TimeSlice:
    """Manage methods for accessing time slice data."""

    ids: ImasIds
    data: xarray.Dataset

    def __call__(self, name: str, itime: int, index=0):
        """Return time slice data array."""
        if index is None:
            return getattr(self.ids.time_slice[itime], name)
        return getattr(self.ids.time_slice[itime], name).array[index]

    def initialize(self, name: str, attrs: list[str], index=0, postfix=''):
        """Create xarray time slice data entries."""
        time_slice = self(name, 0, index)
        coords = self.data.attrs[name]
        shape = tuple(self.data.dims[coordinate] for coordinate in coords)
        for attr in list(attrs):
            if isinstance((value := getattr(time_slice, attr)), float) or \
                    len(value) > 0:
                attr_name = attr + postfix
                self.data[attr_name] = coords, np.zeros(shape, float)
            else:
                attrs.remove(attr)

    def build(self, name: str, attrs: list[str], index=0, postfix=''):
        """Populate xarray dataset with profile data."""
        self.initialize(name, attrs, index, postfix)
        for itime in range(self.data.dims['time']):
            time_slice = self(name, itime, index)
            for attr in attrs:
                attr_name = attr + postfix
                self.data[attr_name][itime] = getattr(time_slice, attr)
