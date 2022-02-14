"""Biot data storage class."""
from abc import abstractmethod
from dataclasses import dataclass, field

import xarray

from nova.electromagnetic.filepath import FilePath
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class BiotData(FilePath, FrameSetLoc):
    """Biot solution abstract base class."""

    name: str = field(default=None)
    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Init path and link line current and plasma index."""
        self.subframe.metaframe.metadata = \
            {'additional': ['plasma', 'nturn'],
             'array': ['plasma', 'nturn'], 'subspace': ['Ic']}
        self.subframe.update_columns()
        super().__post_init__()

    @abstractmethod
    def solve(self, *args):
        """Solve biot interaction - extened by subclass."""
        try:
            self.data.attrs['plasma_index'] = next(
                self.frame.subspace.index.get_loc(name) for name in
                self.subframe.frame[self.aloc['plasma']].unique())
        except StopIteration:
            self.data.attrs['plasma_index'] = -1

    def store(self, filename: str, path=None):
        """Store data as netCDF in hdf5 file."""
        file = self.file(filename, path)
        self.data.to_netcdf(file, mode='a', group=self.name)

    def load(self, filename: str, path=None):
        """Load data from hdf5."""
        file = self.file(filename, path)
        with xarray.open_dataset(file, group=self.name) as data:
            data.load()
            self.data = data
