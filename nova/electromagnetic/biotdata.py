"""Biot data storage class."""
from abc import abstractmethod
from dataclasses import dataclass, field

import xarray

from nova.database.netcdf import netCDF
from nova.database.filepath import FilePath
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class BiotData(netCDF, FilePath, FrameSetLoc):
    """Biot solution abstract base class."""

    name: str = field(default=None)
    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

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
