"""Manage access to scenario data."""
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import os

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
        self.time_slice = TimeSlice(self.ids, self.data)
        self.data.attrs |= self.ids_attrs
        yield
        self.store()

    @abstractmethod
    def build(self):
        """Build netCDF group from ids."""

    def store(self, mode='a'):
        """Store data within hdf file."""
        if self.filename is None:
            return
        file = self.file(self.filename)
        if not os.path.isfile(file):
            mode = 'w'
        self.data.to_netcdf(file, group=self.name, mode=mode)

    def load(self):
        """Load dataset from file (lazy)."""
        file = self.file(self.filename)
        with xarray.open_dataset(file, group=self.name) as data:
            self.data = data
        return self
