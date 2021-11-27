"""Manage 3D coil windings."""
from dataclasses import dataclass, field

import numpy as np
import vedo

from nova.electromagnetic.coilsetattrs import CoilSetAttrs
from nova.geometry.polygeom import Polygon
from nova.geometry.volume import Path, Section, Cell, Sweep, TriShell


@dataclass
class Winding(CoilSetAttrs):
    """Insert 3D coil winding."""

    delta: float = 0.
    turn: str = 'disc'
    required: list[str] = field(
        default_factory=lambda: ['x', 'y', 'z', 'dx', 'dy', 'dz',
                                 'volume', 'vtk'])
    default: dict = field(init=False, default_factory=lambda: {
        'label': 'Swp', 'part': 'coil', 'active': True})

    def set_conditional_attributes(self):
        """Set conditional attrs - not required for winding."""

    def insert(self, poly, path, required=None, iloc=None, **additional):
        """
        Add 3D coils to frameset.

        Lines described by x, y, z coordinates meshed into n elements based on
        dloop (delta).

        Parameters
        ----------
        poly :
            - shapely.geometry.Polygon
            - dict[str, list[float]], polyname: *args
            - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        path : npt.ArrayLike, shape(,3)
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
            poly = Polygon(poly, segment='sweep')
        vtk = Sweep(poly, path)
        frame_data = self.vtk_data(vtk)
        self.attrs = additional | dict(section=poly.section, segment='sweep',
                                       dl=poly.length, dt=poly.thickness)
        with self.insert_required(required):
            index = self.frame.insert(*frame_data, iloc=iloc, **self.attrs)
            subattrs = self.attrs | {'label': index[0], 'frame': index[0],
                                     'delim': '_', 'link': True}

            submesh = Path.from_points(path, delta=self.attrs['delta'])
            section = Section(poly.points).sweep(submesh)

            vtk = [Cell(section.point_array[i:i+2])
                   for i in range(submesh.n_points-1)]
            points = submesh.points
            centroid = (points[1:] + points[:-1]) / 2
            vector = points[1:] - points[:-1]
            volume = [_vtk.clone().triangulate().volume() for _vtk in vtk]
            poly = [TriShell(_vtk).poly for _vtk in vtk]
            area = self.frame.loc[index, 'area']
            self.subframe.insert(*centroid.T, *vector.T, volume, vtk,
                                 poly=poly, area=area, **subattrs)
        return index

    @staticmethod
    def vtk_data(vtk: vedo.Mesh):
        """Extract data from vtk object."""
        centroid = vtk.centerOfMass()
        bounds = np.array(vtk.bounds())
        bbox = bounds[1::2] - bounds[::2]
        volume = vtk.clone().triangulate().volume()
        return *centroid, *bbox, volume, vtk
