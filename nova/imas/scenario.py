"""Load ids data as xarray datasets."""
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property

from nova.imas.database import IdsData, IdsIndex
from nova.imas.getslice import GetSlice


@dataclass
class Scenario(GetSlice, IdsData):
    """Manage access to scenario data (load, store, build)."""

    machine: str = "iter"
    ids_node: str = "time_slice"

    @cached_property
    def ids_index(self):
        """Return cached ids_index instance."""
        return IdsIndex(self.ids_data, self.ids_node)

    @contextmanager
    def build_scenario(self, vtk=False):
        """Manage dataset creation and storage."""
        self.data.attrs[self.name] = ",".join(
            [str(value) for value in self.ids_attrs.values()]
        )
        self.data.attrs[
            "homogeneous_time"
        ] = self.ids_data.ids_properties.homogeneous_time
        if self.data.attrs["homogeneous_time"] == 1:
            self.data.coords["time"] = self.ids_data.time
            self.data.coords["itime"] = "time", range(len(self.data["time"]))
        yield
        if self.pulse != 0 and self.run != 0:  # don't store passed ids_data
            self.store()

    def append(
        self,
        coords: tuple[str, ...],
        attrs: list[str] | str,
        branch="",
        prefix="",
        postfix="",
        ids_node=None,
    ):
        """Append xarray dataset with ids attributes."""
        self.ids = ids_node
        if isinstance(attrs, str):
            attrs = [attrs]
        for attr in attrs:
            path = self.ids_index.get_path(branch, attr)
            if not self.ids_index.valid(path) or self.ids_index.empty(path):
                continue
            self.data[prefix + attr + postfix] = coords, self.ids_index.array(path)

    @abstractmethod
    def build(self):
        """Build netCDF group from ids."""
