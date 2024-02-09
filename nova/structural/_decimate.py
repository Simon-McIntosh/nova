"""Defeature vtk geometries extracted from ansys .rst files."""

from dataclasses import dataclass
from typing import Union

import pyvista as pv

from nova.structural.ansyspost import AnsysPost


@dataclass
class Decimate:
    """Defeature vtk geometry."""

    mesh: Union[str, pv.PolyData]
    decimate: float = 0.95

    def __post_init__(self):
        """Extract surface and decimate."""
        if isinstance(self.mesh, str):
            self.mesh = AnsysPost("TFCgapsG10", "k0", self.mesh).mesh
        self.mesh = self.mesh.decimate_boundary(self.decimate)


if __name__ == "__main__":
    mesh = Decimate("case_ol", 0.9).mesh
    mesh.plot()
