"""Manage plasma attributes."""

from dataclasses import dataclass, field

import numpy as np

from nova.graphics.plot import Plot
from nova.frame.framesetloc import FrameSetLoc
from nova.frame.poloidalgrid import PoloidalGrid
from nova.geometry.polygon import Polygon


@dataclass
class PlasmaGrid(PoloidalGrid):
    """Mesh rejoin interior to firstwall."""

    turn: str = "hexagon"
    polysection_policy: str = ""
    _configured_polysection_policy: str = field(init=False, repr=False, default="")
    tile: bool = field(init=False, default=True)
    required: list[str] = field(default_factory=lambda: ["x", "z", "dl", "dt"])
    default: dict = field(
        init=False,
        default_factory=lambda: {
            "nturn": 1,
            "part": "plasma",
            "name": "Plasma",
            "plasma": True,
            "active": True,
        },
    )

    def __post_init__(self):
        """Freeze the constructor route used by every plasma-grid insert."""
        from nova.biot.polysection import PolySectionPolicy

        self._configured_polysection_policy = PolySectionPolicy.resolve(
            self.polysection_policy
        ).key
        self.polysection_policy = self._configured_polysection_policy
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def set_conditional_attributes(self):
        """Set conditional attrs for plasma grid."""
        self.ifthen("turn", "rectangle", "segment", "cylinder")
        self.ifthen("turn", "hexagon", "segment", "polysection")

    def insert(self, *required, iloc=None, **additional):
        """
        Extend PoloidalGrid.insert.

        Add plasma to coilset and generate bounding plasma grid.

        Plasma inserted into frame with subframe meshed accoriding
        to delta and trimmed to the plasma's boundary curve.

        """
        from nova.biot.polysection import PolySectionPolicy

        current = PolySectionPolicy.resolve(self.polysection_policy).key
        requested = PolySectionPolicy.resolve(
            additional.get("polysection_policy", current)
        ).key
        if current != self._configured_polysection_policy or requested != current:
            raise ValueError(
                "plasma polygon-section policy is fixed by its CoilSet constructor"
            )
        additional["polysection_policy"] = current
        index = super().insert(*required, iloc=iloc, **additional)
        self._route_rectangular_cells(index)
        return index

    def _route_rectangular_cells(self, index):
        """Keep the cylinder shortcut only on complete axis-aligned cells."""
        import shapely.geometry

        membership = np.asarray(self.subframe["frame"], dtype=object)
        positions = np.flatnonzero(np.isin(membership, np.asarray(index)))
        polygons = np.asarray(self.subframe["poly"], dtype=object)
        for position in positions:
            geometry = polygons[position]
            while not isinstance(geometry, shapely.geometry.base.BaseGeometry):
                geometry = geometry.poly
            eligible = (
                isinstance(geometry, shapely.geometry.Polygon)
                and len(geometry.interiors) == 0
                and geometry.equals(shapely.geometry.box(*geometry.bounds))
            )
            self.subframe.iloc[position, "segment"] = (
                "cylinder" if eligible else "polysection"
            )


@dataclass
class FirstWall(Plot, PlasmaGrid, FrameSetLoc):
    """Mesh plasma rejoin."""

    name: str = "firstwall"

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = {
            "additional": ["plasma", "area", "nturn"],
            "array": ["nturn"],
        }
        self.subframe.update_columns()
        super().__post_init__()

    def insert(self, *args, required=None, iloc=None, **additional):
        """Insert plasma and update plasma nturn version (xxhash)."""
        super().insert(*args, required=None, iloc=None, **additional)
        if self.sloc["plasma"].sum() > 1:
            self.normalize_multiframe()
        self.update_aloc_hash("nturn")

    def normalize_multiframe(self):
        """Normalize turn number for multiframe plasmas."""
        self.linkframe(self.Loc["plasma", :].index.tolist())
        self.Loc["plasma", "nturn"] = self.Loc["plasma", "area"] / np.sum(
            self.Loc["plasma", "area"]
        )
        self.loc["plasma", "nturn"] = self.loc["plasma", "area"] / np.sum(
            self.loc["plasma", "area"]
        )

    @property
    def poly(self) -> Polygon:
        """Return firstwall polygon."""
        return self.Loc["plasma", "poly"][0]

    def plot(self, axes=None, plasma=False):
        """Plot plasma boundary and separatrix."""
        self.axes = axes
        self.poly.plot_boundary(self.axes, color="gray", lw=1.5)
        if plasma:
            self.plot("plasma")
