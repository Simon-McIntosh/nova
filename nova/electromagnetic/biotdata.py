"""Biot methods."""
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import xarray

from nova.electromagnetic.biotsolve import BiotMatrix
from nova.electromagnetic.filepath import FilePath
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class BiotData(FilePath, FrameSetLoc):
    """Biot solution abstract base class."""

    name: str = field(default=None)
    data: BiotMatrix = field(init=False, repr=False)

    @abstractmethod
    def solve(self, index=slice(None)):
        """Solve biot interaction - update self.data."""

    def store(self, file: str, path=None):
        """Store data as netCDF in hdf5 file."""
        file = self.file(file, path)
        self.data.to_netcdf(file, mode='a', group=self.name)

    def load(self, file: str, path=None):
        """Load data from hdf5."""
        file = self.file(file, path)
        with xarray.open_dataset(file, group=self.name) as data:
            data.load()
            self.data = data

    def update_turns(self):
        """Update plasma turns."""
        for attr in ['Psi', 'Br', 'Bz']:
            try:
                self.data[attr][:, -1] = np.sum(
                    getattr(self.data, f'_{attr}') *
                    self.loc['plasma', 'nturn'], axis=1)
            except AttributeError:
                pass
