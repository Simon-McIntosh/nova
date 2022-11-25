"""Methods for ploting 3D FrameSpace data."""
from dataclasses import dataclass, field

import matplotlib
import numpy as np
import vedo

from nova.frame.baseplot import Display
from nova.frame.metamethod import MetaMethod
from nova.frame.dataframe import DataFrame


@dataclass
class VtkPlot(MetaMethod):
    """Methods for ploting 3D FrameSpace data."""

    name = 'vtkplot'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['vtk'])
    additional: list[str] = field(default_factory=lambda: [])

    def initialize(self):
        """Initialize metamethod."""
        self.update_columns()

    def update_columns(self):
        """Update frame columns."""
        unset = [attr not in self.frame.columns
                 for attr in self.required + self.additional]
        if np.array(unset).any():
            self.frame.update_columns()

    def __call__(self):
        """Plot frame if not empty."""
        if not self.frame.empty:
            return self.plot()

    def plot(self):
        """Plot vtk instances."""
        colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        self.frame.vtkgeo.generate_vtk()
        index = self.frame.geotype('Geo', 'vtk')
        color = {f'C{i}': c for i, c in enumerate(colors)}
        vtk = [vtk.c(color[Display.get_facecolor(part)])
               for vtk, part in self.frame.loc[index, ['vtk', 'part']].values]
        if len(vtk) > 0:
            return vedo.show(*vtk)
