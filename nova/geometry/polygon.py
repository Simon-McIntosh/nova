"""Manage single instance polygon data."""
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import numpy.typing as npt
import shapely.geometry

from nova.geometry.polyframe import PolyFrame
from nova.geometry.polygen import PolyGen


@dataclass
class Polygon(PolyFrame):
    """Generate bounding polygon.

    Parameters
    ----------
    poly :
        - PolyFrame, shapely.geometry.Polygon
        - dict[str, list[float]], polyname: *args
        - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
        - array-like, shape(n,2) bounding loop [x, z]

    """

    poly: Union[
        PolyFrame,
        shapely.geometry.Polygon,
        dict[str, list[float]],
        list[float],
        npt.ArrayLike,
    ]
    name: str | None = None
    metadata: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Process input geometry."""
        self.correct_aspect()
        self.metadata = self.extract()
        if self.name is not None:
            self.metadata |= dict(name=self.name)
        self.name = self.metadata.get("name", None)
        self.poly = self.translate()
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def correct_aspect(self):
        """Correct bounds to equal aspect geometries."""
        if isinstance(self.poly, dict):
            for section in self.poly:
                if PolyGen(section).shape in ["square", "disc"]:
                    if len(self.poly[section]) == 4:
                        length = PolyGen.boxbound(*self.poly[section][-2:])
                        self.poly[section] = tuple(self.poly[section][:2]) + (length,)

    def extract(self) -> dict:
        """Return metadata extracted from input polygon."""
        if isinstance(self.poly, (Polygon, PolyFrame)):
            return self.poly.metadata
        if isinstance(self.poly, shapely.geometry.Polygon):
            return dict(name="polygon")
        if isinstance(self.poly, shapely.geometry.MultiPolygon):
            return dict(name="multipoly")
        if isinstance(self.poly, dict):
            metadata = dict(names=[PolyGen(name).shape for name in self.poly])
            if len(self.poly) == 1:
                metadata["name"] = metadata["names"][0]
                metadata |= {
                    attr: value
                    for attr, value in zip(
                        ["x_centroid", "z_centroid", "length", "thickness"],
                        self.poly[next(iter(self.poly))],
                    )
                }
                metadata["section"] = metadata["name"]
                return metadata
            metadata["name"] = "-".join(metadata["names"])
            return metadata
        loop = np.array(self.poly)
        if loop.ndim == 1 and len(loop) == 4:  # bounding box
            metadata = self.bounding_box(*loop)
            metadata["section"] = metadata["name"]
            return metadata
        return dict(name="polyloop")

    def translate(self):
        """Translate input geometry to shapely.geometry.Polygon.

        Parameters
        ----------
        poly :
            - PolyFrame, shapely.geometry.Polygon
            - dict[str, list[float]], polyname: *args
            - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        Raises
        ------
        IndexError
            Malformed bounding box, shape is not (4,).
            Malformed bounding loop, shape is not (n, 2).

        Returns
        -------
        polygon : shapely.geometry.Polygon

        """
        if isinstance(
            self.poly, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            return self.poly
        if hasattr(self.poly, "poly"):
            return self.poly.poly
        if isinstance(self.poly, dict):
            names = list(self.poly)
            polys = [PolyGen(section)(*self.poly[section]) for section in names]
            if len(polys) == 1:
                return polys[0]
            poly = shapely.ops.unary_union(polys)
            if not poly.is_valid:
                raise AttributeError(
                    "non-overlapping polygons specified in " f"{self.poly}"
                )
            return poly
        loop = np.array(self.poly)  # to numpy array
        if loop.ndim == 1:  # poly bounding box
            if len(loop) == 4:  # [xmin, xmax, zmin, zmax]
                bbox = self.bounding_box(*loop)
                return PolyGen(bbox["name"])(*list(bbox.values())[1:])
            raise IndexError(
                "malformed bounding box\n"
                f"loop: {loop}\n"
                "require [xmin, xmax, zmin, zmax]"
            )
        if loop.shape[1] != 2:
            loop = loop.T
        if loop.ndim == 2 and loop.shape[1] == 2:  # loop
            return shapely.geometry.Polygon(shapely.geometry.LinearRing(loop))
        raise IndexError(
            "malformed bounding loop\n" f"shape(loop): {loop.shape}\n" "require (n,2)"
        )

    @staticmethod
    def bounding_box(xmin, xmax, zmin, zmax) -> dict:
        """Return characteristic dimensions of bounding box."""
        xlim, zlim = [xmin, xmax], [zmin, zmax]
        x_centroid = np.mean(xlim)
        z_centroid = np.mean(zlim)
        length = np.diff(xlim)[0]
        thickness = np.diff(zlim)[0]
        if np.isclose(length, thickness):
            name = "square"
        else:
            name = "rectangle"
        return dict(
            name=name,
            x_centroid=x_centroid,
            z_centroid=z_centroid,
            length=length,
            thickness=thickness,
        )
