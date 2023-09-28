"""Manage 3D coil windings."""
from dataclasses import dataclass, field

import numpy as np
import vedo

from nova.frame.coilsetattrs import CoilSetAttrs
from nova.geometry.polygeom import Polygon
from nova.geometry.polyline import PolyLine

from nova.geometry.volume import Sweep, TriShell


@dataclass
class Winding(CoilSetAttrs):
    """Insert 3D coil winding."""

    delta: float = 0.0
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
        init=False, default_factory=lambda: ["delta", "turn", "section", "segment"]
    )
    array: list[str] = field(
        init=False,
        default_factory=lambda: [
            "x",
            "y",
            "z",
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

    def set_conditional_attributes(self):
        """Set conditional attrs - not required for winding."""

    def insert(self, poly, path, required=None, iloc=None, **additional):
        """
        Add 3D coils to frameset.

        Lines described by x, y, z coordinates meshed into n elements based on delta.

        Parameters
        ----------
        poly :
            - shapely.geometry.Polygon
            - dict[str, list[float]], polyname: *args
            - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        path : npt.ArrayLike, shape(n,3)
            Swept path.

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
        if not isinstance(poly, Polygon):
            poly = Polygon(poly, name="sweep")
        vtk = Sweep(poly.points, path)
        frame_data = self.vtk_data(vtk)
        self.attrs = additional | dict(
            section=poly.section,
            poly=TriShell(vtk).poly,
            area=poly.area,
            dl=poly.width,
            dt=poly.height,
        )
        with self.insert_required(required):
            index = self.frame.insert(*frame_data, iloc=iloc, **self.attrs)

        with self.insert_required([]):
            self.polyline = PolyLine(path, boundary=poly.points, delta=self.delta)
            subattrs = (
                self.attrs
                | {"label": index[0], "frame": index[0], "delim": "_", "link": True}
                | self.polyline.path_geometry
                | self.polyline.volume_geometry
            )
            self.subframe.insert(**subattrs)
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
