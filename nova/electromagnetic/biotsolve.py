"""Biot methods."""
from dataclasses import dataclass, field

import numpy as np
import xarray

from nova.electromagnetic.biotdata import BiotMatrix
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class BiotSolve(FrameSetLoc):
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

    def update_turns(self):
        """Update plasma turns."""
        for attr in ['Psi', 'Br', 'Bz']:
            try:
                self.data[attr][:, -1] = np.sum(
                    getattr(self.data, f'_{attr}') *
                    self.loc['plasma', 'nturn'], axis=1)
            except AttributeError:
                pass
