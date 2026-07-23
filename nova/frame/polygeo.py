"""Geometric methods for FrameSpace class."""

from dataclasses import dataclass, field

import numpy as np

import nova.frame.metamethod as metamethod
from nova.geometry.polygeom import PolyGeom
from nova.geometry.polygon import Polygon


@dataclass
class PolyGeo(metamethod.PolyGeo):
    """
    Polygon geometrical methods for FrameSpace.

    Extract geometric features from shapely polygons.
    """

    name = "polygeo"

    frame: object = field(repr=False)
    required: list[str] = field(default_factory=lambda: ["segment", "section", "poly"])
    additional: list[str] = field(
        default_factory=lambda: ["dl", "dt", "rms", "area", "volume"]
    )
    require_all: bool = field(init=False, repr=False, default=False)
    base: list[str] = field(
        init=False, default_factory=lambda: ["x", "y", "z", "segment", "dx", "dy", "dz"]
    )
    features: list[str] = field(
        init=False,
        default_factory=lambda: [
            "x",
            "y",
            "z",
            "dx",
            "dy",
            "dz",
            "area",
            "volume",
            "rms",
        ],
    )

    def initialize(self):
        """Init sectional polygon data."""
        segment = np.asarray(self.frame["segment"], dtype=object)
        section = np.asarray(self.frame["section"], dtype=object)
        mask = (
            ~self.frame.geotype("Geo", "poly")
            & ~self.frame.geotype("Json", "poly")
            & (segment != "")
            & (segment != "winding")
            & (section != "")
        )
        index = np.flatnonzero(mask)
        index_length = len(index)
        if index_length == 0:
            return
        coordinates = {
            attr: np.asarray(self.frame[attr], dtype=float)
            for attr in ["x", "z", "dl", "dt", "dy"]
        }
        poly = np.asarray(self.frame["poly"], dtype=object).copy()
        section = section.copy()
        geom = np.empty((index_length, len(self.features)), dtype=float)
        for k, i in enumerate(index):
            if poly[i] is None:
                poly[i] = Polygon(
                    {
                        f"{section[i]}": [
                            coordinates[a][i] for a in ["x", "z", "dl", "dt"]
                        ]
                    }
                )
                section[i] = poly[i].metadata["section"]
            geometry = PolyGeom(
                poly[i], segment=segment[i], loop_length=coordinates["dy"][i]
            ).geometry
            geom[k] = [geometry[feature] for feature in self.features]
        with self.frame.setlock(True, ["subspace", "array"]):
            self.frame["poly"][index] = poly[index]
            self.frame["section"][index] = section[index]
            for feature_index, feature in enumerate(self.features):
                self.frame[feature][index] = geom[:, feature_index]

    def limit(self, index):
        """Return coil limits [xmin, xmax, zmin, zmax]."""
        geom = self.frame.loc[index, ["x", "z", "dx", "dz"]]
        limit = [
            min(geom["x"] - geom["dx"] / 2),
            max(geom["x"] + geom["dx"] / 2),
            min(geom["z"] - geom["dz"] / 2),
            max(geom["z"] + geom["dz"] / 2),
        ]
        return limit

    def polygons(self, index) -> dict:
        """Return frame geometry in a Bokeh multi polygons format."""
        polyframe = self.frame.loc[index, "poly"]
        return {
            "x": [poly.polygons[0] for poly in polyframe],
            "z": [poly.polygons[1] for poly in polyframe],
        }
