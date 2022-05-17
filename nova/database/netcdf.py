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

    def store(self, filename: str, path=None, group=None):
        """Store data as netCDF in hdf5 file."""
        file = self.file(filename, path)
        if group is None:
            group = ''
        print('name', self.name)
        self.data.to_netcdf(file, mode='a', group=f'{self.name}/{group}')

    def load(self, filename: str, path=None, group=None):
        """Load data from hdf5."""
        file = self.file(filename, path)
        if group is None:
            group = ''
        print('name', self.name)
        with xarray.open_dataset(file, group=f'{self.name}/{group}') as data:
            self.data = data
