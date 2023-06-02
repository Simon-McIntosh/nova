"""Extend plasma grid plotting with ."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pyvista
import xarray

from nova.biot.biotplasmagrid import BiotPlasmaGrid


@dataclass
class BiotPlasmaVTK(BiotPlasmaGrid):
    """Extend BiotPlasmaGrid dataset with cutplane VTK methods."""

    data: xarray.Dataset = field(repr=False, default_factory=xarray.Dataset)
    mesh: pyvista.PolyData = field(init=False, repr=False)
    classnames: ClassVar[list[str]] = ["PlasmaGrid", "PlasmaVTK"]

    def __post_init__(self):
        """Load biot dataset."""
        super().__post_init__()
        self.load_data()
        assert self.data.attrs["classname"] in self.classnames
        self.build_mesh()

    def build_mesh(self):
        """Build vtk mesh."""
        points = np.c_[self.data.x, np.zeros(self.data.dims["x"]), self.data.z]
        faces = np.c_[np.full(self.data.dims["tri_index"], 3), self.data.triangles]
        self.mesh = pyvista.PolyData(points, faces=faces)

    def plot(self, **kwargs):
        """Plot vtk mesh."""
        self.mesh["psi"] = self.psi
        kwargs = dict(color="purple", line_width=2, render_lines_as_tubes=True) | kwargs
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.mesh)
        plotter.add_mesh(self.mesh.contour(), **kwargs)
        plotter.show()
