"""Biot methods."""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import numpy.typing as npt
import xarray

from nova.electromagnetic.biotsolve import BiotMatrix
from nova.electromagnetic.filepath import FilePath
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class BiotData(FilePath, FrameSetLoc):
    """Biot solution abstract base class."""

    name: str = field(default=None)
    attrs: Union[list[str], dict[str, int]] = field(
        default_factory=lambda: ['Br', 'Bz', 'Psi'])
    data: xarray.Dataset = field(init=False, repr=False)
    current: npt.ArrayLike = field(init=False, repr=False)
    nturn: npt.ArrayLike = field(init=False, repr=False)
    plasma: npt.ArrayLike = field(init=False, repr=False)

    def __post_init__(self):
        """Init path and link line current and plasma index."""
        super().__post_init__()
        if isinstance(self.attrs, list):
            self.attrs = {attr: id(0) for attr in self.attrs}
        self.current = self.sloc['Ic']
        self.nturn = self.loc['nturn']
        self.plasma = self.loc['plasma']

    def __getattr__(self, attr):
        """Return attribute data."""
        if (Attr := attr.capitalize()) in self.attrs:
            version = self.subframe.metaframe.data.get('seperatrix_id', id(-1))
            if self.attrs[Attr] != version:
                self.update_turns(Attr)
                self.attrs[Attr] = version
            return getattr(self, Attr) @ self.current
        raise AttributeError(f'attribute {Attr} not specified in {self.attrs}')

    @abstractmethod
    def _solve(self, *args):
        """Solve biot interaction - extened by subclass."""

    def solve(self, *args):
        """Solve biot interaction - update attrs."""
        self._solve(*args)
        self.set_attrs()

    def set_attrs(self):
        """Update data attributes."""
        for attr in self.data.data_vars:
            setattr(self, attr, self.data[attr].data)

    def store(self, filename: str, path=None):
        """Store data as netCDF in hdf5 file."""
        file = self.file(filename, path)
        self.data.to_netcdf(file, mode='a', group=self.name)

    def load(self, file: str, path=None):
        """Load data from hdf5."""
        file = self.file(file, path)
        with xarray.open_dataset(file, group=self.name) as data:
            data.load()
            self.data = data

    def update_turns(self, attr: str):
        """Update plasma turns."""
        # TODO fix plasma indexing and test
        getattr(self, attr)[:, -2] = \
            getattr(self, f'_{attr}') @ self.nturn[self.plasma]
        '''
        for attr in self.attrs:
            try:
                #self.data[attr].data[:, -2] = np.sum(
                #    getattr(self.data, f'_{attr}').data *
                #    self.loc[self.subframe.filament, 'nturn'], axis=1)
                self.Psi[:, -2] = self._Psi @ self.nturn[self.plasma]
            except AttributeError:
                pass
        '''
