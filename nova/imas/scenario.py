"""Load ids data as xarray datasets."""

from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property

from nova.imas.database import IdsData
from nova.imas.getslice import GetSlice
from nova.imas.ids_index import IdsIndex


@dataclass
class Scenario(GetSlice, IdsData):
    """Manage access to scenario data (load, store, build)."""

    machine: str = "iter"
    ids_node: str = "time_slice"
    time_node: str = ""
    _time_node: str = field(init=False, repr=False, default="")

    @cached_property
    def ids_index(self):
        """Return cached ids_index instance."""
        return IdsIndex(self.ids, self.ids_node)

    @property  # type: ignore[no-redef]
    def time_node(self) -> str | None:  # noqa
        """Manage time_node."""
        return self._time_node

    @time_node.setter
    def time_node(self, time_node: str | None):
        if type(time_node) is property:
            time_node = self._time_node
        self._time_node = time_node
        try:
            del self.time_label, self.itime_label
        except AttributeError:
            pass
        if self.ids is not None:
            with self.ids_index.node(self.time_node):
                time = self.ids_index.array("time")
                self.data.coords["time"] = time
                self.data.coords["itime"] = range(len(time))

    @cached_property
    def homogeneous_time(self):
        """Return ids homogeneous time."""
        return self.ids.ids_properties.homogeneous_time

    @cached_property
    def time_label(self):
        """Return dataset time coordinate label for current time_node."""
        match self.homogeneous_time:
            case 0:  # inhomogeneous time
                return "_".join([node for node in [self.time_node, "time"] if node])
            case 1:  # homogeneous time
                return "time"
            case 2:  # constant IDS
                raise ValueError(
                    "no time coordinate for constant IDS " "(homogeneous_time=2)."
                )

    @cached_property
    def itime_label(self):
        """Return dataset itime coordinate label for current time_node."""
        index = self.time_label.rfind("time")
        return self.time_label[:index] + "i" + self.time_label[index:]

    @property
    def time_coord(self):
        """Return time coordinate."""
        return self.data.coords[self.time_label]

    @property
    def itime_coord(self):
        """Return itime coordinate."""
        return self.data.coords[self.itime_label]

    @contextmanager
    def build_scenario(self, time_node: str = ""):
        """Manage dataset creation and storage."""
        self.data.attrs[self.name] = ",".join(
            [str(value) for value in self.ids_attrs.values()]
        )
        self.data.attrs["homogeneous_time"] = self.homogeneous_time
        self.time_node = time_node
        yield
        if self.pulse != 0 and self.run != 0:  # don't store passed ids
            self.store()

    def append(
        self,
        coords: tuple[str, ...],
        attrs: list[str] | str,
        branch="",
        prefix="",
        postfix="",
    ):
        """Append xarray dataset with ids attributes."""
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
