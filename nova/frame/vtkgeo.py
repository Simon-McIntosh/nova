"""Geometric VTK methods for FrameSpace class."""
from dataclasses import dataclass, field
from typing import ClassVar

import pandas
import vedo

import nova.frame.metamethod as metamethod
from nova.frame.dataframe import DataFrame
from nova.geometry.volume import TriShell, Ring
from nova.geometry.vtkgen import VtkFrame


@dataclass
class VtkGeo(metamethod.VtkGeo):
    """Volume vtk geometry."""

    frame: DataFrame = field(repr=False)
    additional: list[str] = field(
        default_factory=lambda: [
            *TriShell.features,
            "part",
            "segment",
            "section",
            "poly",
        ]
    )
    features: list[str] = field(init=False, default_factory=lambda: TriShell.features)
    qhull: ClassVar[list[str]] = ["panel"]
    ahull: ClassVar[list[str]] = ["insert"]
    geom: ClassVar[list[str]] = ["insert", "panel", "vtk", "stl"]

    def initialize(self):
        """Init vtk data."""
        index = self.frame.index[
            ~self.frame.geotype("Geo", "vtk")
            & ~self.frame.geotype("Json", "vtk")
            & ~pandas.isna(self.frame.vtk)
        ]
        if (index_length := len(index)) > 0:
            frame = self.frame.loc[index, :].copy()
            for i in range(index_length):
                tri = TriShell(
                    frame.vtk.iloc[i],
                    qhull=frame.segment.iloc[i] in self.qhull,
                    ahull=frame.segment.iloc[i] in self.ahull,
                )
                mesh = vedo.Mesh(
                    [tri.vtk.points(), tri.vtk.cells()],
                    c=tri.vtk.c(),
                    alpha=tri.vtk.opacity(),
                )
                frame.loc[frame.index[i], "vtk"] = VtkFrame(mesh)
                if frame.segment.iloc[i] in self.geom:
                    frame.loc[frame.index[i], self.features] = tri.geom
                    frame.loc[frame.index[i], ["section"]] = ""
                    frame.loc[frame.index[i], "poly"] = tri.poly
                else:
                    frame.loc[frame.index[i], "volume"] = tri.volume
            self.frame.loc[index, :] = frame
            self.generate_vtk()

    def generate_vtk(self):
        """Generate vtk data from poly."""
        index = self.frame.index[
            ~self.frame.geotype("Geo", "vtk") & self.frame.geotype("Geo", "poly")
        ]
        if len(index) > 0:
            self.frame.loc[index, "vtk"] = [
                Ring(polyframe.poly)
                for polyframe in self.frame.loc[index, "poly"].values
            ]
            self.frame.loc[index, "volume"] = [
                vtk.clone().triangulate().volume()
                for vtk in self.frame.loc[index, "vtk"].values
            ]
