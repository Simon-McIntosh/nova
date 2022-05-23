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

    def store(self, filename=None, path=None):
        """Store data as netCDF in hdf5 file."""
        group = self.netcdf_path(self.name)
        return super().store(filename, path, group)

    def load(self, filename=None, path=None):
        """Load data from hdf5."""
        group = self.netcdf_path(self.name)
        return super().load(filename, path, group)
