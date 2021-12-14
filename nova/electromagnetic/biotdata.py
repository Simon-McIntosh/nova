"""Biot methods."""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Union

from numpy import typing as npt
import xarray

from nova.electromagnetic.filepath import FilePath
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class BiotData(FilePath, FrameSetLoc):
    """Biot solution abstract base class."""

    name: str = field(default=None)
    attrs: Union[list[str], dict[str, int]] = field(
        default_factory=lambda: ['Br', 'Bz', 'Psi'])
    data: xarray.Dataset = field(init=False, repr=False)
    array: dict[str, npt.ArrayLike] = field(init=False, default_factory=dict)
    plasma_index: list[int] = field(init=False, default_factory=list)
    plasma_turns: list[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Init path and link line current and plasma index."""
        super().__post_init__()
        if isinstance(self.attrs, list):
            self.attrs = {attr: id(None) for attr in self.attrs}

    def __getattr__(self, attr):
        """Return attribute data."""
        if (Attr := attr.capitalize()) in self.attrs:
            self.update_indexer()
            if self.attrs[Attr] != (plasma_version :=
                                    self.subframe.version['plasma']):
                self.update_turns(Attr)
                self.attrs[Attr] = plasma_version
            return self.array[Attr] @ self.saloc.Ic
        raise AttributeError(f'attribute {Attr} not specified in {self.attrs}')

    @abstractmethod
    def _solve(self, *args):
        """Solve biot interaction - extened by subclass."""

    def solve(self, *args):
        """Solve biot interaction - update attrs."""
        self._solve(*args)
        self.update()

    def update(self):
        """Update data attributes."""
        for attr in self.data.data_vars:
            self.array[attr] = self.data[attr].data
        self.plasma_index = [self.frame.index.get_loc(name) for name in
                             self.subframe.frame[self.aloc.plasma].unique()]
        if len(self.plasma_index) == 1:
            self.plasma_index = self.plasma_index[0]
            return
        self.plasma_turns = [(self.subframe.frame == name).to_numpy()
                             for name in self.frame.index[self.plasma_index]]

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
        self.update()

    def update_turns(self, attr: str):
        """Update plasma turns."""
        # TODO fix plasma indexing and test
        try:
            self.array[attr][:, self.plasma_index] = \
                self.array[f'_{attr}'] @ self.aloc.nturn[self.aloc.plasma]
        except ValueError:  # multi-frame plasma
            for index, turns in zip(self.plasma_index, self.plasma_turns):
                nturn = (self.aloc.nturn * turns)[self.aloc.plasma]
                self.array[attr][:, index] = self.array[f'_{attr}'] @ nturn
