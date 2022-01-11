"""Biot methods."""
from abc import abstractmethod
from dataclasses import dataclass, field

import xarray

from nova.electromagnetic.filepath import FilePath
from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.utilities.xpu import array_module, asnumpy


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
    xpu: str = None

    def __post_init__(self):
        """Init path and link line current and plasma index."""
        global xp
        xp = array_module(self.xpu)
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
            return self.array[Attr] @ xp.asarray(self.current)
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
        return xp.linalg.norm([self.Br, self.Bz], axis=0)

    @abstractmethod
    def solve_biot(self, *args):
        """Solve biot interaction - extened by subclass."""

    def solve(self, *args):
        """Solve biot interaction - update attrs."""
        self.solve_biot(*args)
        self.link_array()

    def link_data(self):
        """Update data attributes."""
        for attr in self.data.attrs['attributes']:
            if attr[0] == '_':
                continue
            self.data[attr].data = asnumpy(self.array[attr])

    def link_array(self):
        """Update array attributes."""
        for attr in self.data.attrs['attributes']:
            self.array[attr] = xp.array(self.data[attr].data, xp.float32)
            self.array[f'_{attr}'] = xp.array(self.data[f'_{attr}'].data,
                                              xp.float32)
            rank = int(len(self.data[f'_{attr}']) / 500)
            self.array[f'_U{attr}'] = xp.array(
                self.data[f'_U{attr}'].data[:, :rank], xp.float32)
            self.array[f'_s{attr}'] = xp.array(
                self.data[f'_s{attr}'].data[:rank], xp.float32)
            self.array[f'_V{attr}'] = xp.array(
                self.data[f'_V{attr}'].data[:rank, :], xp.float32)

        self.update_indexer()
        try:
            self.plasma_index = next(
                self.frame.subspace.index.get_loc(name) for name in
                self.subframe.frame[asnumpy(self.aloc.plasma)].unique())
            self.plasma_slice = slice(self.aloc['plasma'].argmax(),
                                      -self.aloc['plasma'][::-1].argmax())
        except StopIteration:
            pass

    def store(self, filename: str, path=None):
        """Store data as netCDF in hdf5 file."""
        file = self.file(filename, path)
        self.link_data()
        self.data.to_netcdf(file, mode='a', group=self.name)

    def load(self, file: str, path=None):
        """Load data from hdf5."""
        file = self.file(file, path)
        with xarray.open_dataset(file, group=self.name) as data:
            data.load()
            self.data = data
        self.link_array()

    def update_turns(self, attr: str):
        """Update plasma turns."""
        if self.plasma_index is None:
            return
        nturn = xp.array(self.aloc['nturn'][self.aloc['plasma']],
                         dtype=xp.float32)
        index = self.plasma_index
        self.array[attr][:, index] = self.array[f'_{attr}'] @ nturn

        #self.array[attr][:, index] = \
        #    self.array[f'_U{attr}'] @ (self.array[f'_s{attr}'] *
        #                               (self.array[f'_V{attr}'] @ nturn))
