"""Manage vtk instances within DataFrame.poly."""

import json
import numpy as np
import vedo

from nova.geometry.geoframe import GeoFrame


class VtkFrame(vedo.Mesh, GeoFrame):
    """Manage vtk serialization via json strings."""

    def __str__(self):
        """Return polygon name."""
        return "vtk"

    def dumps(self) -> str:
        """Return string representation of vtk object."""
        return json.dumps(
            {
                "type": "VTK",
                "points": self.vertices.tolist(),
                "cells": np.array(self.cells, dtype=int).tolist(),
                "color": self.color().tolist(),
                "opacity": self.opacity(),
            }
        )

    @classmethod
    def loads(cls, vtk: str):
        """Load json prepresentation."""
        vtk = json.loads(vtk)
        return cls([vtk["points"], vtk["cells"]], c=vtk["color"], alpha=vtk["opacity"])
