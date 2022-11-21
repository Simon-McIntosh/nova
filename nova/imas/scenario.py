"""Manage access to scenario data."""
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field

import xarray

from nova.imas.database import IdsData
from nova.imas.timeslice import TimeSlice


@dataclass
class Scenario(IdsData):
    """Manage access to scenario data (load, store, build)."""

    machine: str = 'iter'
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)
    time_slice: TimeSlice | None = field(init=False, repr=False, default=None)

    @contextmanager
    def build_scenario(self):
        """Manage dataset creation and storage."""
        self.data = xarray.Dataset()
        self.time_slice = TimeSlice(self.ids_data, self.data)
        self.data.attrs |= self.ids_attrs
        yield
        self.store()

    @abstractmethod
    def build(self):
        """Build netCDF group from ids."""

    def store(self):
        """Extend FilePath.store."""
        return super().store(group=self.name)

    def load(self):
        """Extend FilePath.load."""
        return super().load(group=self.name)
