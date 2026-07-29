"""Geometric VTK methods for FrameSpace class."""

from dataclasses import dataclass, field
from importlib import import_module
from typing import ClassVar

import numpy as np

import nova.frame.metamethod as metamethod
from nova.frame.dataframe import DataFrame


def _is_mesh(blob) -> bool:
    """Return True for a raw vtk blob (not an empty None / NaN sentinel).

    Replaces the ``~pandas.isna`` guard: an object column loaded from netCDF
    with no vtk payload carries NaN floats, which must be skipped alongside
    None rather than fed to the mesh pipeline.
    """
    if blob is None:
        return False
    if isinstance(blob, float) and np.isnan(blob):
        return False
    return True


def _volume():
    """Return the volume module, imported on first use (needs vedo/vtk)."""
    return import_module("nova.geometry.volume")


def _trishell_features() -> list[str]:
    """Return the TriShell feature column names (lazy, avoids eager vedo)."""
    return list(_volume().TriShell.features)


@dataclass
class VtkGeo(metamethod.VtkGeo):
    """Volume vtk geometry."""

    frame: DataFrame = field(repr=False)
    additional: list[str] = field(
        default_factory=lambda: [
            *_trishell_features(),
            "part",
            "segment",
            "section",
            "poly",
        ]
    )
    features: list[str] = field(init=False, default_factory=_trishell_features)
    qhull: ClassVar[list[str]] = ["panel"]
    ahull: ClassVar[list[str]] = ["insert", "winding", "arc"]
    geom: ClassVar[list[str]] = ["insert", "panel", "vtk", "stl"]

    def initialize(self):
        """Init vtk data."""
        raw = ~self.frame.geotype("Geo", "vtk") & ~self.frame.geotype("Json", "vtk")
        raw &= np.array([_is_mesh(blob) for blob in self.frame["vtk"]], dtype=bool)
        index = self.frame.index[raw]
        if len(index) == 0:
            return
        vedo = import_module("vedo")
        trishell = _volume().TriShell
        vtkframe = import_module("nova.geometry.vtkgen").VtkFrame
        segment = np.asarray(self.frame.loc[index, "segment"])
        blobs = list(self.frame.loc[index, "vtk"])
        for label, seg, blob in zip(index, segment, blobs):
            tri = trishell(blob, qhull=seg in self.qhull, ahull=seg in self.ahull)
            mesh = vedo.Mesh(
                [tri.vtk.vertices, tri.vtk.cells],
                c=tri.vtk.c(),
                alpha=tri.vtk.opacity(),
            )
            self.frame.at[label, "vtk"] = vtkframe(mesh)
            if seg in self.geom:
                for feature, value in zip(self.features, tri.geom):
                    self.frame.at[label, feature] = value
                self.frame.at[label, "section"] = ""
                self.frame.at[label, "poly"] = tri.poly
            else:
                self.frame.at[label, "volume"] = tri.volume
        self.generate_vtk()

    def generate_vtk(self):
        """Generate vtk data from poly."""
        mask = ~self.frame.geotype("Geo", "vtk") & self.frame.geotype("Geo", "poly")
        index = self.frame.index[mask]
        if len(index) == 0:
            return
        ring = _volume().Ring
        rings = [ring(polyframe.poly) for polyframe in self.frame.loc[index, "poly"]]
        self.frame.loc[index, "vtk"] = rings
        self.frame.loc[index, "volume"] = [
            vtk.clone().triangulate().volume() for vtk in rings
        ]
