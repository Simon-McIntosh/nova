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
    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    data: BiotMatrix = field(init=False, repr=False)

    def __getattr__(self, attr):
        """Return attribute data."""
        if (Attr := attr.capitalize()) in self.attrs:
            return self.data[Attr].values @ self.sloc['Ic']
        raise AttributeError(f'attribute {attr} not specified in {self.attrs}')

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
        for attr in self.attrs:
            try:
                self.data[attr][:, -1] = np.sum(
                    getattr(self.data, f'_{attr}') *
                    self.loc['plasma', 'nturn'], axis=1)
            except AttributeError:
                pass
