"""Load ids data as xarray datasets."""
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field

import xarray

from nova.imas.database import IdsData, IdsIndex
from nova.imas.getslice import GetSlice


@dataclass
class Scenario(GetSlice, IdsData):
    """Manage access to scenario data (load, store, build)."""

    machine: str = 'iter'
    ids_node: str = 'time_slice'
    ids_index: IdsIndex = field(init=False, repr=False)

    @contextmanager
    def build_scenario(self):
        """Manage dataset creation and storage."""
        self.data = xarray.Dataset()
        self.ids_index = IdsIndex(self.ids_data, self.ids_node)
        self.data.attrs |= self.ids_attrs
        self.data.coords['time'] = self.ids_data.time
        self.data.coords['itime'] = 'time', range(len(self.data['time']))
        yield
        self.store()

    def append(self, coords: tuple[str, ...], attrs: list[str],
               branch='', postfix='', ids_node=None):
        """Append xarray dataset with ids attributes."""
        self.ids = ids_node
        for attr in attrs:
            path = self.ids_index.get_path(branch, attr)
            if self.ids_index.empty(path):
                continue
            self.data[attr+postfix] = coords, self.ids_index.array(path)

    @abstractmethod
    def build(self):
        """Build netCDF group from ids."""
