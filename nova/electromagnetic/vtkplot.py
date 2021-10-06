"""Methods for ploting 3D FrameSpace data."""
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame


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
        mesh = pv.PolyData()
        index = self.frame.segment == 'volume'
        for item in self.frame.index[index]:
            volume = self.frame.loc[item, 'volume']
            radius = (3 / (4*np.pi) * volume)**(1/3)
            center = self.frame.loc[item, ['x', 'y', 'z']].to_list()
            mesh += pv.Sphere(radius, center)

        index = self.frame.segment == 'circle'
        for item in self.frame.index[index]:
            mesh += self.pf_coil(*self.frame.loc[item, ['x', 'z', 'dx', 'dz']])


        if mesh.n_points > 0:
            mesh.plot(color='w')
        pv.ParametricTorus
