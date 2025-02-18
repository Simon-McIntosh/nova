"""Manage 3D coil windings."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import shapely.ops
import vedo

from nova.frame.coilsetattrs import CoilSetAttrs
from nova.geometry.polygeom import Polygon, PolyGeom
from nova.geometry.polyline import PolyLine

from nova.geometry.volume import Sweep


@dataclass
class Winding(CoilSetAttrs):
    """Insert 3D coil winding."""

    delta: float = 0
    turn: str = "rectangle"
    segment: str = "winding"
    required: list[str] = field(
        default_factory=lambda: ["x", "y", "z", "dx", "dy", "dz", "volume", "vtk"]
    )
    default: dict = field(
        init=False,
        default_factory=lambda: {"label": "Swp", "part": "coil", "active": True},
    )
    attributes: list[str] = field(
        init=False,
        default_factory=lambda: [
            "turn",
            "section",
            "segment",
            "length",
        ],
    )
    array: list[str] = field(
        init=False,
        default_factory=lambda: [
            "x0",
            "y0",
            "z0",
            "dx",
            "dy",
            "dz",
            "x1",
            "y1",
            "z1",
            "x2",
            "y2",
            "z2",
        ],
    )

    polyline_attrs: ClassVar[list[str]] = [
        "arc_eps",
        "line_eps",
        "rdp_eps",
        "minimum_arc_nodes",
        "quadrant_segments",
        "arc_resolution",
        "align",
        "filament",
    ]

    def set_conditional_attributes(self):
        """Set conditional attrs - not required for winding."""

    def insert(
        self,
        path=None,
        cross_section=None,
        polyline=None,
        required=None,
        iloc=None,
        **additional,
    ):
        """
        Add 3D coils to frameset.

        Lines described by x, y, z coordinates meshed into n elements based on delta.

        Parameters
        ----------
        path : np.ndarray, shape(n,3)
            Swept path.

        cross_section :
            - shapely.geometry.Polygon
            - dict[str, list[float]], polyname: *args
            - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]


        required : list[str]
            Required attribute names (args). The default is None.

        iloc : int, optional
            Index before which coils are inserted. The default is None (-1).

        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        index : pandas.Index
            FrameSpace index.

        """
        if cross_section is None or (path is None and polyline is None):
            raise ValueError(
                "winding.insert requires cross_section and "
                "path or polyline attributes"
            )
        if not isinstance(cross_section, PolyGeom):
            cross_section = PolyGeom(cross_section, name="sweep")
        if cross_section.section == "skin":
            cross_section.section = "box"

        polyline_kwargs = {
            attr: additional.pop(attr)
            for attr in self.polyline_attrs
            if attr in additional
        }
        match polyline:
            case PolyLine():
                polyline.cross_section = cross_section.points
            case _:
                polyline = PolyLine(
                    path, cross_section=cross_section.points, **polyline_kwargs
                )
        self.polyline = polyline

        align = polyline_kwargs.pop("align", PolyLine.align)
        vtk = Sweep(cross_section.points, polyline.path, align=align)
        frame_data = self.vtk_data(vtk)
        poly = Polygon(
            shapely.ops.unary_union(
                [polygon.poly for polygon in polyline.volume_geometry["poly"]]
            )
        )
        self.attrs = additional | dict(
            section=cross_section.section,
            area=cross_section.area,
            width=cross_section.width,
            height=cross_section.height,
        )
        with self.insert_required(required):
            index = self.frame.insert(*frame_data, iloc=iloc, poly=poly, **self.attrs)
        with self.insert_required([]):
            subattrs = (
                self.attrs
                | {"label": index[0], "frame": index[0], "link": True}
                | polyline.path_geometry
                | polyline.volume_geometry
                | dict(
                    zip(
                        ("dx", "dy", "dz"),
                        [getattr(polyline, f"delta_{attr}") for attr in "xyz"],
                    )
                )
            )
            subattrs.pop("name", None)
            subindex = self.subframe.insert(**subattrs)
            if cross_section.section in ["box"] and polyline.filament is False:
                subattrs["width"] *= 1 - cross_section.thickness
                subattrs["height"] *= 1 - cross_section.thickness
                self.subframe.insert(**subattrs | {"link": subindex[0]}, factor=-1)
        self.update_loc_indexer()
        return index

    @staticmethod
    def vtk_data(vtk: vedo.Mesh):
        """Extract data from vtk object."""
        centroid = vtk.center_of_mass()
        vtk.triangulate()
        bounds = np.array(vtk.bounds())
        bbox = bounds[1::2] - bounds[::2]
        volume = vtk.volume()
        return *centroid, *bbox, volume, vtk
