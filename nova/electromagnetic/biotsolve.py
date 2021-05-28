"""Biot methods."""
from dataclasses import dataclass, field

import xarray

from nova.electromagnetic.biotdata import BiotMatrix


@dataclass
class BiotSolve:
    """Biot data IO."""

    name: str = field(default=None)
    data: BiotMatrix = field(init=False, repr=False)

    def store(self, file):
        """Store data as netCDF in hdf5 file."""
        self.data.to_netcdf(file, mode='a', group=self.name)

    def load(self, file):
        """Load data from hdf5."""
        with xarray.open_dataset(file, group=self.name) as data:
            data.load()
            self.data = data
