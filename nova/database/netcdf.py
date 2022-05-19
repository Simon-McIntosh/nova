"""Facilitate structured access to netCDF data."""
from dataclasses import dataclass, field

import xarray

from nova.database.filepath import FilePath


@dataclass
class netCDF(FilePath):
    """Provide regulated access to netCDF database."""

    name: str = None
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    def store(self, filename: str, path=None):
        """Store data as netCDF in hdf5 file."""
        file = self.file(filename, path)
        group = self.netcdf_path(self.name)
        self.data.to_netcdf(file, mode='a', group=group)

    def load(self, filename: str, path=None):
        """Load data from hdf5."""
        file = self.file(filename, path)
        with xarray.open_dataset(
                file, group=self.netcdf_path(self.name)) as data:
            self.data = data
            self.data.load()
