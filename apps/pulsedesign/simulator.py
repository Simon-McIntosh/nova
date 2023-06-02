"""Manage interface from PulseDesign class to pulsedesign app."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

from bokeh.models import ColumnDataSource
import numpy as np

from nova.imas.pulsedesign import PulseDesign


@dataclass
class Simulator(PulseDesign):
    """Compose PulseDesign to interface with Bokeh data sources."""

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
    ]

    def __post_init__(self):
        """Create Bokeh data sources."""
        super().__post_init__()
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

    def update(self):
        """Extend PulseDesign update to include bokeh datasources."""
        super().update()
        self.source["points"].data = dict(zip("xz", self.points.T))
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
    simulator = Simulator(135013, 2, "iter", 1)
