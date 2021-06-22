"""Apply structural deformation to TF coilset."""
from dataclasses import dataclass, field
import os

import numpy as np
import numpy.typing as npt
import pyvista as pv
from rdp import rdp

from nova.structural.ansyspost import AnsysPost
from nova.structural.datadir import AnsysDataDir
from nova.structural.plotter import Plotter
from nova.structural.windingpack import WindingPack
from nova.utilities.time import clock


@dataclass
class RamerDouglasPeucker:
    """Subsample polyline using RDP algorithum."""

    points: npt.ArrayLike
    epsilon: 0.01

    def __post_init__(self):
        """Apply RDP algorithum."""
        self.samples = np.array(rdp(self.points, self.epsilon))

    def __len__(self):
        """Return sample lenght."""
        return len(self.samples)


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
        self.mesh = self.interpolate_coils(self.mesh, self.ansys)
        mesh = self.mesh.copy()
        self.mesh.clear_arrays()
        for scn in self.scenario:
            self.mesh[scn] = mesh[scn]
        self.mesh.save(self.vtk_file)

    def interpolate_coils(self, source, target, sharpness=3, radius=1.5,
                          n_cells=7):
        """Retun mesh interpolant."""
        mesh = pv.PolyData()
        n_coils = source.n_cells // n_cells
        tick = clock(n_coils, header='Interpolating Ansys displacements.')
        for n_coil in range(n_coils):
            mesh += source.extract_cells(
                range(n_cells*n_coil, n_cells*(n_coil+1))).interpolate(
                target, sharpness=sharpness, radius=radius,
                strategy='closest_point')
            tick.tock()
        return mesh

    def compress(self, epsilon=5e-4):
        """Compress vtk mesh using rdp algorithum."""
        mesh = pv.PolyData()
        n_cells = self.mesh.n_cells
        tick = clock(n_cells, header='Compressing vtk mesh.')
        for cell in range(n_cells):
            cell = self.mesh.extract_cells(cell)
            fit = RamerDouglasPeucker(cell.points, epsilon)
            submesh = pv.Spline(fit.samples)
            submesh.clear_point_arrays()
            mesh += submesh
            tick.tock()
        self.mesh = self.interpolate_coils(mesh, self.mesh)

    def plot(self):
        """Plot warped shape."""
        self.warp('TFonly-cooldown')

    def animate(self):
        filename = os.path.join(self.directory, self.file)
        super().animate(filename, 'TFonly', view='xy')




if __name__ == '__main__':

    tf = TFC18('v0', 'WP', scenario={'cooldown': 1, 'TFonly': 2})
    tf.mesh['TFonly-cooldown'] = tf.mesh['TFonly'] - tf.mesh['cooldown']
    tf.plot()
    #tf.animate()
