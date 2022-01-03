"""Biot methods."""
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import xarray

from nova.electromagnetic.filepath import FilePath
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class BiotData(FilePath, FrameSetLoc):
    """Biot solution abstract base class."""

    name: str = field(default=None)
    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    version: dict[str, int] = field(init=False, repr=False,
                                    default_factory=dict)
    data: xarray.Dataset = field(init=False, repr=False)
    array: dict = field(init=False, default_factory=dict)
    plasma_index: int = field(init=False, default=None)

    def __post_init__(self):
        """Init path and link line current and plasma index."""
        self.version |= {attr: id(None) for attr in self.attrs}
        self.version['Bn'] = id(None)
        self.subframe.metaframe.metadata = \
            {'additional': ['plasma', 'nturn'],
             'array': ['plasma', 'nturn'], 'subspace': ['Ic']}
        self.subframe.update_columns()
        super().__post_init__()

    def __getattr__(self, attr):
        """Return variales data."""
        if (Attr := attr.capitalize()) in self.version:
            self.update_indexer()
            if Attr == 'Bn':
                return self.get_norm()
            if self.version[Attr] != self.subframe.version['plasma']:
                self.update_turns(Attr)
                self.version[Attr] = self.subframe.version['plasma']
            return self.array[Attr] @ self.current
        raise AttributeError(f'attribute {Attr} not specified in {self.attrs}')

    def get_norm(self):
        """Return cached field L2 norm."""
        version = hash(self.current.data.tobytes())
        if self.version['Bn'] != version or 'Bn' not in self.array:
            self.array['Bn'] = self.calculate_norm()
            self.version['Bn'] = version
        return self.array['Bn']

    def calculate_norm(self):
        """Return calculated L2 norm."""
        return np.linalg.norm([self.Br, self.Bz], axis=0)

    @abstractmethod
    def solve_biot(self, *args):
        """Solve biot interaction - extened by subclass."""

    def solve(self, *args):
        """Solve biot interaction - update attrs."""
        self.solve_biot(*args)
        self.update()

    def update(self):
        """Update data attributes."""
        for attr in self.data.attrs['attributes']:
            self.array[attr] = self.data[attr].data
            self.array[f'_{attr}'] = self.data[f'_{attr}'].data
        self.update_indexer()
        try:
            self.plasma_index = next(
                self.frame.subspace.index.get_loc(name) for name in
                self.subframe.frame[self.aloc.plasma].unique())
        except StopIteration:
            pass

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
        if self.data.attrs['plasma_index'] is None:
            return
        nturn = self.aloc['nturn'][self.aloc['plasma']]
        index = self.plasma_index
        self.array[attr][:, index] = self.array[f'_{attr}'] @ nturn
