"""Methods for ploting 3D FrameSpace data."""
from dataclasses import dataclass, field

import numpy as np
import pygmsh
import pyvista as pv
from rdp import rdp
import vedo

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame


class Geom(pygmsh.occ.Geometry):
    """Extend OCC Geometry."""

    def get_poly(self, boundary):
        """Extract polygon from frame."""
        try:
            return self._get_poly(boundary)
        except NotImplementedError:
            outer = self._get_poly(boundary[0])
            inner = self._get_poly(boundary[1])
            return self.boolean_difference(outer, inner)[0]

    def _get_poly(self, boundary):
        """Return occ polygon constructed from frame.poly boundary."""
        coords = np.array(boundary.xy)
        boundary = np.array([coords[0], np.zeros(len(coords[0])), coords[1]]).T
        index = np.unique(boundary.round(decimals=6),
                          axis=0, return_index=True)[1]
        boundary = boundary[np.sort(index)]
        boundary = rdp(boundary, 1e-2)
        return self.add_polygon(boundary, mesh_size=0.75)

@dataclass
class VtkPlot(MetaMethod):
    """Methods for ploting 3D FrameSpace data."""

    name = 'vtkplot'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['volume'])
    additional: list[str] = field(
        default_factory=lambda: ['a', 'bi', 'cj', 'dk'])

    def initialize(self):
        """Initialize metamethod."""
        self.update_columns()

    def update_columns(self):
        """Update frame columns."""
        unset = [attr not in self.frame.columns
                 for attr in self.required + self.additional]
        if np.array(unset).any():
            self.frame.update_columns()

    def __call__(self, index=slice(None), axes=None, **kwargs):
        """Plot frame if not empty."""
        if not self.frame.empty:
            self.plot(index, axes, **kwargs)

    def pf_coil(self, x, z, dx, dz):
        """Return vtk Poloidal Field coil - rectangular section."""
        outer = pv.Cylinder(center=(0, 0, z), direction=[0, 0, 1],
                            radius=x+dx/2, height=dz).triangulate()
        inner = pv.Cylinder(center=(0, 0, z), direction=[0, 0, 1],
                            radius=x-dx/2, height=1.1*dz).triangulate()
        return outer-inner

    def plot(self, index=slice(None), axes=None, **kwargs):
        """Plot frame."""
        index = self.frame.segment == 'ring'
        vmesh = []
        for item in self.frame.index[index]:
            with Geom() as geom:
                poly = geom.get_poly(self.frame.loc[item, 'poly'].boundary)
                segment = []
                for sign in [-1, 1]:
                    segment.append(geom.revolve(poly,
                                                [0.0, 0.0, 1.0],
                                                [0.0, 0.0, 0.0],
                                                sign*np.pi)[1])
                geom.boolean_union(segment)

                #geom.revolve(poly, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], -np.pi)

                mesh = geom.generate_mesh(dim=3)
            vmesh.append(vedo.Mesh(mesh, alpha=0.5))

        vedo.show(vmesh)

        '''
        mesh = pv.PolyData()
        index = self.frame.segment == 'volume'
        for item in self.frame.index[index]:
            volume = self.frame.loc[item, 'volume']
            radius = (3 / (4*np.pi) * volume)**(1/3)
            center = self.frame.loc[item, ['x', 'y', 'z']].to_list()
            mesh += pv.Sphere(radius, center)

        index = self.frame.segment == 'ring'
        for item in self.frame.index[index]:
            mesh += self.pf_coil(*self.frame.loc[item, ['x', 'z', 'dx', 'dz']])


        if mesh.n_points > 0:
            mesh.plot(color='w')
        '''
