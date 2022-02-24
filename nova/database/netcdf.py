"""Facilitate structured access to netCDF data."""
from dataclasses import dataclass, field

import xarray

from nova.database.filepath import FilePath


@dataclass
class netCDF(FilePath):
    """Provide regulated access to netCDF database."""

    name: str = None
    group: str = 'frameset'
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    @property
    def netcdf_group(self) -> str:
        """Return netCDF group."""
        return f'{self.group}/{self.name}'

    def store(self, filename: str, path=None):
        """Store data as netCDF in hdf5 file."""
        file = self.file(filename, path)
        self.data.to_netcdf(file, mode='a', group=self.netcdf_group)

    def load(self, filename: str, path=None):
        """Load data from hdf5."""
        file = self.file(filename, path)
        with xarray.open_dataset(file, group=self.netcdf_group) as data:
            data.load()
            self.data = data
