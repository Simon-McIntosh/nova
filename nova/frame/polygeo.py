"""Geometric methods for FrameSpace class."""
from dataclasses import dataclass, field

import numpy as np

import nova.frame.metamethod as metamethod
from nova.frame.dataframe import DataFrame
from nova.geometry.polygeom import PolyGeom
from nova.geometry.polygon import Polygon


@dataclass
class PolyGeo(metamethod.PolyGeo):
    """
    Polygon geometrical methods for FrameSpace.

    Extract geometric features from shapely polygons.
    """

    name = "polygeo"

    frame: DataFrame = field(repr=False)
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
        index = self.frame.index[
            ~self.frame.geotype("Geo", "poly")
            & ~self.frame.geotype("Json", "poly")
            & (self.frame.segment != "")
            & (self.frame.segment != "winding")
            & (self.frame.section != "")
        ]
        if (index_length := len(index)) > 0:
            section = self.frame.loc[index, "section"].values
            poly_data = self.frame.loc[index, ["x", "z", "dl", "dt"]].values
            segment = self.frame.loc[index, "segment"].values
            length = self.frame.loc[index, ["dy"]].values[0]
            poly = self.frame.loc[index, "poly"].values
            poly_update = self.frame.loc[index, "poly"].isna()
            geom = np.empty((index_length, len(self.features)), dtype=float)
            # itterate over index - generate poly as required
            for i in range(index_length):
                if poly_update.iloc[i]:
                    poly[i] = Polygon({f"{section[i]}": poly_data[i]})
                    section[i] = poly[i].metadata["section"]
                geometry = PolyGeom(poly[i], segment[i], length[i]).geometry
                geom[i] = [geometry[feature] for feature in self.features]
            if poly_update.any():
                self.frame.loc[index, "poly"] = poly
            self.frame.loc[index, self.features] = geom
            self.frame.loc[index, "section"] = section

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
