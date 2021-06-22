"""Apply structural deformation to TF coilset."""
from dataclasses import dataclass, field
import os

import pyvista as pv

from nova.structural.ansyspost import AnsysPost
from nova.structural.datadir import AnsysDataDir
from nova.structural.plotter import Plotter
from nova.structural.windingpack import WindingPack


@dataclass
class TFC18(AnsysDataDir, Plotter):
    """Post-process Ansys output from F4E's 18TF coil model."""

    scenario: dict[str, int] = field(default_factory=dict)
    ansys: pv.PolyData = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load database."""
        super().__post_init__()
        self.load()

    def __str__(self):
        """Return Ansys model descriptor."""
        return AnsysPost(*self.metadata).__str__()

    @property
    def vtk_file(self):
        """Return vtk file path."""
        return os.path.join(self.directory, f'{self.file}_ccl.vtk')

    def load(self):
        """Load vtm data file."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.load_ansys()
            self.load_mesh()
            self.interpolate()
            self.mesh.save(self.vtk_file)

    def load_ansys(self):
        """Load ansys vtk mesh."""
        ansys = AnsysPost(self.file, 'WP').mesh
        self.ansys = ansys.copy()
        self.ansys.clear_point_arrays()
        for scn in self.scenario:
            self.ansys[scn] = ansys[f'displacement-{self.scenario[scn]}']

    def load_mesh(self):
        """Load referance windingpack ccl."""
        self.mesh = WindingPack('TFC1_CL').mesh
        self.mesh.clear_point_arrays()

    def interpolate(self):
        """Interpolate ansys displacements to winding pack centrelines."""
        self.mesh = self.mesh.interpolate(
            self.ansys, sharpness=2, radius=1.0, strategy='closest_point')

    def plot(self):
        """Plot warped shape."""
        self.warp('TFonly')


if __name__ == '__main__':

    tf = TFC18('v4', 'WP', scenario={'deadweight': 0, 'cooldown': 1,
                                     'TFonly': 2})

    tf.plot()
