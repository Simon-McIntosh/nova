"""Manage access to scenario data."""
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import os

import xarray

from nova.imas.database import Database
from nova.imas.timeslice import TimeSlice


@dataclass
class Scenario(Database):
    """Manage access to scenario data (load, store, build)."""

    tokamak: str = 'iter'
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)
    time_slice: TimeSlice = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Load data."""
        super().__post_init__()
        try:
            self.load()
        except (FileNotFoundError, OSError, KeyError):
            self.build()

    def load_scenario(self):
        """Load ids_data and timeslice."""
        self.ids_data = self.load_ids_data()
        print('homogeneous', self.ids_data.ids_properties.homogeneous_time)
        self.time_slice = TimeSlice(self.ids_data, self.data)

    @contextmanager
    def build_scenario(self):
        """Manage dataset creation and storage."""
        self.close_dataset()
        self.data = xarray.Dataset()
        self.load_scenario()
        self.data.attrs |= self.ids_attrs
        yield
        self.store()

    @abstractmethod
    def build(self):
        """Build netCDF group from ids_data."""

    def store(self, mode='a'):
        """Store data within hdf file."""
        file = self.file(self.filename)
        if not os.path.isfile(file):
            mode = 'w'
        self.data.to_netcdf(file, group=self.ids_name, mode=mode)

    def load(self):
        """Load dataset from file (lazy)."""
        file = self.file(self.filename)
        with xarray.open_dataset(file, group=self.ids_name) as data:
            self.data = data
        return self

    def close_dataset(self):
        """Force dataset closure."""
        try:
            xarray.open_dataset(self.file(self.filename)).close()
        except FileNotFoundError:
            pass
