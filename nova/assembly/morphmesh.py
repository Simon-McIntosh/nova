"""Apply RBF morphing to ITER coilset."""
from dataclasses import dataclass
import os
from typing import Union

import pyvista as pv

from nova.definitions import root_dir
from nova.assembly.ansyspost import AnsysPost
from nova.assembly.morph import Morph
from nova.assembly.plotter import Plotter


@dataclass
class MorphMesh(Plotter):
    """Morph base mesh to fiducial deltas."""

    fiducial: pv.PolyData
    base: pv.PolyData
    version: Union[str, int] = None
    smoothing: float = 1.0

    def __post_init__(self):
        """Init data directory and load morphed mesh."""
        self.load()
        # self.morph()
        # self.mesh.save(self.filename)

    @property
    def name(self):
        """Return mesh name."""
        name = f"{self.fiducial.name}_base"  # '_{self.base.name}'
        if self.version is not None:
            name += f"_{self.version}"
        return name

    @property
    def filename(self):
        """Return full mesh filepath."""
        return os.path.join(
            root_dir, "data/Assembly/toroidal_fiducial", f"{self.name}.vtk"
        )

    def load(self):
        """Load morphed vtk mesh."""
        try:
            self.mesh = pv.read(self.filename)
        except FileNotFoundError:
            self.morph()
            self.mesh.save(self.filename)

    def morph(self):
        """Morph base mesh."""
        # self.mesh = Morph(self.fiducial, self.base,
        #                  smoothing=self.smoothing).mesh
        self.mesh = self.base.copy()
        # Morph(self.fiducial).predict(self.base)
        Morph(self.fiducial).interpolate(self.mesh, neighbors=None)

    # morph.animate('TFC18_morph', 'delta', max_factor=500,
    #              frames=31, opacity=0)


if __name__ == "__main__":
    # fiducial = FiducialCoil('fiducial', 10)
    base = AnsysPost("TFCgapsG10", "k0_wp", "all")
    base.warp(500)
    # morph = MorphMesh(fiducial.mesh, base.mesh)

    # morph.mesh['displacement mm'] = 1e3*morph.mesh['delta']
    # plotter = morph.warp(0.1, 0, plotter=pv.Plotter())
    # plotter.update_scalar_bar_range(clim=[0, 6])
    # plotter.show()

    # morph.animate('TFC18_morph', 'delta', max_factor=500,
    #              frames=51, opacity=0)

    # morph.warp(500, opacity=0)
