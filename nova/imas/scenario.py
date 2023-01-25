"""Manage access to timeslice data."""
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from operator import attrgetter

import numpy as np
import xarray

from nova.imas.database import ImasIds
from nova.imas.database import IdsData
from nova.imas.getslice import GetSlice


@dataclass
class IdsArray:
    """Methods for accessing array data from witnin an ids."""

    ids_data: ImasIds
    ids_node: str = 'time_slice'
    transpose: bool = field(init=False, default=False)
    length: int = field(init=False, default=0)
    shapes: dict[str, tuple[int] | tuple[()]] = \
        field(init=False, default_factory=dict)

    def __post_init__(self):
        """Calculate length of time vector."""
        assert self.ids_data.ids_properties.homogeneous_time == 1
        self.ids = self.ids_node
        self.length = len(self.ids)

    @property
    def ids(self):
        """Return ids_data node."""
        return attrgetter(self.ids_node)(self.ids_data)

    @ids.setter
    def ids(self, ids_node: str | None):
        """Update ids node."""
        if ids_node is not None:
            self.transpose = ids_node != 'time_slice'
            self.ids_node = ids_node

    def __getitem__(self, path: str) -> tuple[int] | tuple[()]:
        """Return cached dimension length."""
        try:
            return self.shapes[path]
        except KeyError:
            self.shapes[path] = self._path_shape(path)
            return self[path]

    def shape(self, path) -> tuple[int, ...]:
        """Return attribute array shape."""
        return (self.length,) + self[path]

    def _path_shape(self, path: str) -> tuple[int] | tuple[()]:
        """Return data shape at itime=0 on path."""
        match data := self.get_slice(0, path):
            case np.ndarray():
                return data.shape
            case float() | int():
                return ()
            case _:
                raise ValueError(f'unable to determine data length {path}')

    def get_slice(self, index: int, path: str):
        """Return attribute vector at itime."""
        try:
            return attrgetter(path)(self.ids[index])
        except AttributeError:  # __structArray__
            node, path = path.split('.', 1)
            return attrgetter(path)(
                attrgetter(node)(self.ids[index])[0])

    def get_array(self, path: str):
        """Return attribute data array."""
        data = np.zeros(self.shape(path))
        for index in range(self.length):
            try:
                data[index] = self.get_slice(index, path)
            except ValueError:  # empty slice
                pass
        if self.transpose:
            return data.T
        return data

    def empty(self, path: str):
        """Return status based on first data point extracted from ids_data."""
        try:
            data = self.get_slice(0, path)
        except IndexError:
            return True
        if hasattr(data, 'flat'):
            try:
                data = data.flat[0]
            except IndexError:
                return True
        return data is None or np.isclose(data, -9e40)

    def get_path(self, branch: str, attr: str) -> str:
        """Return ids attribute path."""
        if '*' in branch:
            return branch.replace('*', attr)
        return '.'.join((branch, attr))

    def append(self, data: xarray.Dataset, coords: tuple[str, ...],
               attrs: list[str], branch='', postfix='', ids_node=None):
        """Append xarray dataset with ids attributes."""
        self.ids = ids_node
        for attr in attrs:
            path = self.get_path(branch, attr)
            if self.empty(path):
                continue
            data[attr+postfix] = coords, self.get_array(path)


@dataclass
class Scenario(GetSlice, IdsData):
    """Manage access to scenario data (load, store, build)."""

    machine: str = 'iter'
    ids_node: str = 'time_slice'
    ids_array: IdsArray = field(init=False, repr=False)

    @contextmanager
    def build_scenario(self):
        """Manage dataset creation and storage."""
        self.data = xarray.Dataset()
        self.ids_array = IdsArray(self.ids_data, self.ids_node)
        self.data.attrs |= self.ids_attrs
        self.data.coords['time'] = self.ids_data.time
        self.data.coords['itime'] = 'time', range(len(self.data['time']))
        yield
        self.store()

    @abstractmethod
    def build(self):
        """Build netCDF group from ids."""
