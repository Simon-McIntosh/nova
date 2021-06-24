"""Apply structural deformation to TF coilset."""
from dataclasses import dataclass, field
import os

import pyvista as pv

from nova.structural.ansyspost import AnsysPost
from nova.structural.datadir import AnsysDataDir
from nova.structural.plotter import Plotter
from nova.structural.windingpack import WindingPack
from nova.structural.uniformwindingpack import UniformWindingPack


@dataclass
class TFC18(AnsysDataDir, Plotter):
    """Post-process Ansys output from F4E's 18TF coil model."""

    scenario: dict[str, int] = field(default_factory=dict)
    ansys: pv.PolyData = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)
    cluster: int = 1

    def __post_init__(self):
        """Load database."""
        super().__post_init__()
        self.subset = 'WP'
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

    def load_ansys(self):
        """Load ansys vtk mesh."""
        ansys = AnsysPost(self.folder, self.file, self.subset).mesh
        self.ansys = ansys.copy()
        self.ansys.clear_point_arrays()
        for scn in self.scenario:
            self.ansys[scn] = ansys[f'displacement-{self.scenario[scn]}']

    def load_windingpack(self):
        """Load conductor windingpack."""
        if self.cluster is not None:
            return UniformWindingPack().mesh
        return WindingPack('TFC1_CL').mesh

    def load_mesh(self):
        """Load referance windingpack ccl."""
        self.mesh = self.load_windingpack()
        self.mesh.clear_point_arrays()
        self.mesh = self.interpolate_coils(self.mesh, self.ansys)
        mesh = self.mesh.copy()
        self.mesh.clear_arrays()
        for scn in self.scenario:
            self.mesh[scn] = mesh[scn]
        self.mesh.save(self.vtk_file)

    def interpolate_coils(self, source, target, sharpness=3, radius=1.5,
                          n_cells=7):
        """Retun mesh interpolant."""
        return source.interpolate(target, sharpness=sharpness, radius=radius,
                                  strategy='closest_point')

    def plot(self):
        """Plot warped shape."""
        self.warp('TFonly-cooldown')

    def animate(self):
        """Animate displacement."""
        filename = os.path.join(self.directory, self.file)
        super().animate(filename, 'TFonly', view='xy')


if __name__ == '__main__':

    tf = TFC18('TFC18', 'v4', scenario={'cooldown': 1, 'TFonly': 2})

    tf.mesh['TFonly-cooldown'] = tf.mesh['TFonly'] - tf.mesh['cooldown']
    tf.warp('TFonly-cooldown', factor=120)
    #tf.animate()
