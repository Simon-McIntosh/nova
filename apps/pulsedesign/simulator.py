"""Manage interface from PulseDesign class to pulsedesign app."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

from bokeh.models import ColumnDataSource
import numpy as np

from nova.imas.equilibrium import EquilibriumData
from nova.imas.pulsedesign import PulseDesign
from nova.imas.sample import Sample


@dataclass
class Simulator(PulseDesign):
    """Compose PulseDesign to interface with Bokeh data sources."""

    lock: bool = field(init=False, default=False)
    sample: Sample = field(init=False)
    source: dict[str, ColumnDataSource] = field(
        init=False, repr=False, default_factory=dict
    )

    bokeh_attrs: ClassVar[list[str]] = [
        "coil",
        "plasma",
        "levelset",
        "wall",
        "profiles",
        "x_points",
        "o_points",
        "points",
        "current",
        "vertical_force",
        "field",
    ]
    persist: ClassVar[list[str]] = ["data", "_data"]

    def __post_init__(self):
        """Create Bokeh data sources."""
        self.load_ids(**self.ids_attrs)

    def load_ids(self, **ids_attrs):
        """Update sample dataset.

        Parameters
        ----------
        attrs : Ids | bool | str
            Descriptor for geometry ids.
        """
        ids_attrs = self.merge_ids_attrs(ids_attrs, self.ids_attrs)
        equilibrium = EquilibriumData(**ids_attrs)  # load source equilibrium
        self.sample = Sample(equilibrium.data)  # extract key features
        self.ids = self.sample.equilibrium_ids()
        for attr, value in ids_attrs.items():
            setattr(self, attr, value)
        with self.load_source():
            super().__post_init__()
        self.itime = 0

    @contextmanager
    def load_source(self):
        """Load source data and link to Bokeh column data sources."""
        self.source = {}
        self.lock = False
        try:
            del self._data
        except AttributeError:
            pass
        yield
        for attr in self.bokeh_attrs:
            self.source[attr] = ColumnDataSource()
        self.source["coil"].data = self.coil_data
        self.source["plasma"].data = self.plasma_data
        self.source["wall"].data = self.wall_outline
        self.source["profiles"].data = {
            "psi_norm": self.data.psi_norm.data,
            "dpressure_dpsi": np.zeros_like(self.data.psi_norm.data),
            "f_df_dpsi": np.zeros_like(self.data.psi_norm.data),
        }
        self.source["current"].data = 1e-3 * self._data.current.to_pandas()
        self.source["vertical_force"].data = (
            1e-6 * self._data.vertical_force.to_pandas()
        )
        self.source["field"].data = self._data.field.to_pandas()

    def update(self):
        """Extend PulseDesign update to include bokeh datasources."""
        super().update()
        if not self.source:
            return
        self.source["points"].data = dict(zip("xz", self.control_points.T))
        self.source["levelset"].data = self.levelset.contour()
        self.source["x_points"].data = dict(zip("xz", self.levelset.x_points.T))
        self.source["profiles"].patch(
            {
                "dpressure_dpsi": [(slice(None), self["dpressure_dpsi"])],
                "f_df_dpsi": [(slice(None), self["f_df_dpsi"])],
            }
        )
        self.source["plasma"].patch(
            {
                "ionize": [(slice(None), 0.5 * self.plasma.ionize.astype(float))],
            }
        )
        self.source["current"].data = 1e-3 * self._data.current.to_pandas()
        self.source["vertical_force"].data = (
            1e-6 * self._data.vertical_force.to_pandas()
        )
        self.source["field"].data = self._data.field.to_pandas()

    @cached_property
    def wall_outline(self):
        """Return firstwall outline."""
        return self.geometry["wall"](**self.wall).outline

    @property
    def coil_data(self):
        """Return coil polygons."""
        return self.subframe.polygeo.polygons("free")

    @property
    def plasma_data(self):
        """Return plasma polygons."""
        ionize = self.plasma.ionize.astype(float)
        return self.subframe.polygeo.polygons("plasma") | {"ionize": ionize}


if __name__ == "__main__":
    pulse, run = 135013, 2
    pulse, run = 135011, 7
    pulse, run = 105028, 1

    simulator = Simulator(pulse, run, "iter", 0)
    simulator.plot()
