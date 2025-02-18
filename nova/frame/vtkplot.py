"""Methods for ploting 3D FrameSpace data."""

from dataclasses import dataclass, field

import matplotlib
import numpy as np
import vedo

from nova.graphics.plot import BasePlot, Properties
import nova.frame.metamethod as metamethod
from nova.frame.dataframe import DataFrame


@dataclass
class VtkPlot(metamethod.VtkPlot, BasePlot):
    """Methods for ploting 3D FrameSpace data."""

    name = "vtkplot"

    frame: DataFrame = field(repr=False)
    additional: list[str] = field(default_factory=lambda: [])

    def initialize(self):
        """Initialize metamethod."""
        self.update_columns()

    def update_columns(self):
        """Update frame columns."""
        unset = [
            attr not in self.frame.columns for attr in self.required + self.additional
        ]
        if np.array(unset).any():
            self.frame.update_columns()

    def __call__(self, index=slice(None), new=True, **kwargs):
        """Plot frame if not empty."""
        if not self.frame.empty:
            return self.plot(index, new=new, **kwargs)

    def plot(self, index=slice(None), decimate=1e5, plotter=None, cut=None, **kwargs):
        """Plot vtk instances."""
        if cut is None:
            cut = []
        if cut is True:
            cut = self.frame.loc[:, "part"].unique()
        colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        self.frame.vtkgeo.generate_vtk()
        index = self.frame.geotype("Geo", "vtk") & self.get_index(index)
        color = {f"C{i}": c for i, c in enumerate(colors)}
        vtk = [
            vtk.c(color[Properties.get_facecolor(part)]).alpha(
                Properties.get_alpha(part)
            )
            for vtk, part in self.frame.loc[index, ["vtk", "part"]].values
        ]
        vtk = [
            (vtk[i].cut_with_plane(normal=[0, 1, 0]) if part in cut else vtk[i])
            for i, part in enumerate(self.frame.loc[index, ["part"]].values)
        ]
        if decimate is not None:
            vtk = [_vtk.decimate(n=decimate, preserve_volume=True) for _vtk in vtk]
        if len(vtk) > 0:
            if plotter is not None:
                return plotter.add(vtk)
            return vedo.show(*vtk, **kwargs)
