"""Biot data storage class."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import xarray

from nova.database.netcdf import netCDF
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class BiotData(ABC, netCDF, FrameSetLoc):
    """Biot solution abstract base class."""

    name: str = field(default='')
    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    classname: str = field(init=False)
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    def __post_init__(self):
        """Init path and link line current and plasma index."""
        self.subframe.metaframe.metadata = \
            {'additional': ['plasma', 'nturn'],
             'array': ['plasma', 'nturn'], 'subspace': ['Ic']}
        self.subframe.update_columns()
        self.classname = self.__class__.__name__
        super().__post_init__()

    def __len__(self):
        """Return dataset length."""
        return len(self.data)

    #@abstractmethod
    #def solve(self):
    #    """Solve biot interaction."""

    def post_solve(self):
        """Post process biot solution - extened by subclass."""
        self.data.attrs['classname'] = self.classname
        try:
            self.data.attrs['plasma_index'] = next(
                self.frame.subspace.index.get_loc(name) for name in
                self.subframe.frame[self.aloc['plasma']].unique())
        except StopIteration:
            self.data.attrs['plasma_index'] = -1
