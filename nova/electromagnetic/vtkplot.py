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
            center = self.frame.loc[item, ['x', 'y', 'z']].to_list()
            outer = pv.Cylinder(center=center, direction=[1, 1, 1],
                                radius=1, height=2).triangulate()
            inner = pv.Cylinder(center=center, direction=[1, 1, 1],
                                radius=0.5, height=2).triangulate()
            #mesh += inner
            mesh += outer.boolean_difference(inner)

        if mesh.n_points > 0:
            mesh.plot()
        pv.ParametricTorus
