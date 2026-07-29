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
                "winding.insert requires cross_section and path or polyline attributes"
            )
        if not isinstance(cross_section, PolyGeom):
            cross_section = PolyGeom(cross_section, name="sweep")
        if "section" not in cross_section.metadata:
            # A boundary loop or a shapely polygon carries no section NAME, only
            # corners, and the frame's biot cross-section registry is keyed by one.
            # ``polygon`` is the key that says exactly that, and naming it here is
            # what lets a section no (width, height) pair can express reach an
            # element at all.
            cross_section.section = "polygon"

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
            area=cross_section.poly.area,
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
            core = self.interior_boundaries(cross_section, polyline)
            core_poly = [polyline.section_footprint(boundary) for boundary in core]
            subattrs["area"] = self.conducting_area(
                polyline.volume_geometry["poly"], core_poly
            )
            subindex = self.subframe.insert(**subattrs)
            for boundary, poly in zip(core, core_poly):
                self.subframe.insert(
                    **subattrs
                    | dict(
                        link=subindex[0],
                        poly=poly,
                        width=float(np.ptp(boundary[:, 0])),
                        height=float(np.ptp(boundary[:, 2])),
                    ),
                    factor=-1,
                )
        self.update_loc_indexer()
        return index

    @staticmethod
    def conducting_area(outer: list, core: list) -> list[float]:
        """Return each segment's conducting area: its footprint less its holes.

        Measured on the SWEPT footprints rather than on the cross-section, because
        that is the polygon every thickened element integrates and the column is
        what sets its current density -- a solid section whose column disagreed with
        its own polygon would spread more or less than the frame's one ampere over
        it, which is a first-order error where the polygon's own departure from the
        curve it approximates is second order.  The two agree to round-off for a
        section symmetric about the poloidal plane and to a few parts in 1e5 for one
        that is not, the difference being the tilt the path's own end tangents give
        the two end stations.
        """
        return [
            poly.poly.area - sum(hole[segment].poly.area for hole in core)
            for segment, poly in enumerate(outer)
        ]

    @staticmethod
    def interior_boundaries(cross_section: PolyGeom, polyline: PolyLine) -> list:
        """Return each interior boundary of a hollow section as ``(n, 3)`` points.

        A ``skin`` or a ``box`` -- or any polygon with a hole -- is an annulus
        between an outer boundary and an inner one, and superposition gives it
        exactly.  The outer boundary is inserted as a solid section carrying current
        density ``+j`` and each interior boundary as a CORE carrying ``-j``, so the
        material between them carries ``j`` and the core cancels.  The density is set
        by the annulus rather than by either boundary::

            j = I / A_annulus,  I_outer = +j A_outer,  I_core = -j A_core

        and ``I_outer + I_core = j (A_outer - A_core) = I``.  Both members carry the
        ANNULUS as their ``area``, which is what sets ``j``, and the ``-1`` factor
        the frame's own link machinery already understands is what makes the core
        subtract: the reference row of a linked group has factor one by definition,
        so the density cannot live in that column and lives in the area instead.
        Both members are solid sections every thickened element already evaluates,
        so nothing here needs a second kernel.

        Reading the boundaries off the section rather than scaling its width and
        height is what makes a ``skin`` a circular annulus instead of a square one,
        and what admits a hollow section no descriptor names at all.

        Empty for a filament, which carries no section, and for a solid one.
        """
        if polyline.filament is not False:
            return []
        return [
            np.c_[coords[:, 0], np.zeros(len(coords)), coords[:, 1]]
            for coords in (
                np.asarray(ring.coords, dtype=float)
                for ring in cross_section.poly.interiors
            )
        ]

    @staticmethod
    def vtk_data(vtk: vedo.Mesh):
        """Extract data from vtk object."""
        centroid = vtk.center_of_mass()
        vtk.triangulate()
        bounds = np.array(vtk.bounds())
        bbox = bounds[1::2] - bounds[::2]
        volume = vtk.volume()
        return *centroid, *bbox, volume, vtk
