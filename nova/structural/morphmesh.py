"""Apply RBF morphing to ITER coilset."""
from dataclasses import dataclass
import os
from typing import Union

import pyvista as pv

from nova.definitions import root_dir
from nova.structural.ansyspost import AnsysPost
from nova.structural.fiducialcoil import FiducialCoil
from nova.structural.morph import Morph
from nova.structural.plotter import Plotter


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

    @property
    def name(self):
        """Return mesh name."""
        name = f'{self.fiducial.name}_{self.base.name}'
        if self.version is not None:
            name += f'_{self.version}'
        return name

    @property
    def filename(self):
        """Return full mesh filepath."""
        return os.path.join(root_dir, 'data/Assembly/toroidal_fiducial',
                            f'{self.name}.vtk')

    def load(self):
        """Load morphed vtk mesh."""
        try:
            self.mesh = pv.read(self.filename)
        except FileNotFoundError:
            self.morph()
            self.mesh.save(self.filename)

    def morph(self):
        """Morph base mesh."""
        self.mesh = Morph(self.fiducial, self.base,
                          smoothing=self.smoothing).mesh

    #morph.animate('TFC18_morph', 'delta', max_factor=500,
    #              frames=31, opacity=0)


if __name__ == '__main__':

    fiducial = FiducialCoil('fiducial', 10)
    base = AnsysPost('TFCgapsG10', 'k0', 'all')

    morph = MorphMesh(fiducial.mesh, base.mesh)
    #morph.mesh = morph.mesh.slice(normal=[0, 0, 1])
    morph.warp(500, opacity=0)
