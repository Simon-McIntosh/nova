"""Methods to morph vtk meshes to fit fiducial mesurments."""
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
import scipy.interpolate

from nova.structural.ansyspost import AnsysPost
from nova.structural.fiducialmesh import FiducialMesh
from nova.structural.plotter import Plotter


@dataclass
class Morph(Plotter):
    """Apply RBF morphing to VTK objects."""

    fiducial: pv.PolyData
    mesh: pv.PolyData = None
    neighbors: int = None
    smoothing: float = 0.
    kernel: str = 'thin_plate_spline'
    epsilon: float = 1
    rbf: scipy.interpolate.RBFInterpolator = field(init=None)

    def __post_init__(self):
        """Load data."""
        self.build()
        if self.mesh is not None:
            self.interpolate()

    def build(self):
        """Build radial basis function interpolator."""
        index = np.unique(self.fiducial.points, axis=0, return_index=True)[1]
        self.rbf = scipy.interpolate.RBFInterpolator(
            self.fiducial.points[index], self.fiducial['delta'][index],
            neighbors=self.neighbors,
            smoothing=self.smoothing, kernel=self.kernel, epsilon=self.epsilon)

    def interpolate(self):
        """Morph mesh."""
        self.mesh['delta'] = self.rbf(self.mesh.points)


if __name__ == '__main__':

    # mesh = AnsysPost('TFCgapsG10', 'k0', 'E_TF1').mesh
    # mesh = AnsysPost('TFCgapsG10', 'k0', 'E_TF2').mesh
    # mesh = AnsysPost('TFCgapsG10', 'k0', 'all').mesh
    mesh = AnsysPost('TFCgapsG10', 'k0', 'E_WP_1').mesh

    fiducial = FiducialMesh()
    fiducial.load_centerline([0])

    #fiducial = FiducialData()
    #fiducial.warp('delta', factor=500)

    morph = Morph(fiducial.mesh, mesh, kernel='linear', neighbors=1)
    morph.mesh = morph.mesh.slice((0, 0, 1))
    morph.warp(500)

    #plotter = pv.Plotter()
    #plotter.add_mesh(mesh, color='w', opacity=0.1)
    #plotter.add_mesh(fiducial.mesh)
    #plotter.show()


    # morph.warp('delta', factor=500, opacity=0.05)
    # morph.fiducial.mesh.plot()
    # morph.animate('TFC18_morph', 'delta', max_factor=500, frames=2, opacity=0)
    # morph.write_ansys_table()


'''


plotter = pv.Plotter()
plotter.add_mesh(ansys, color='w', opacity=0.05)
plotter.add_mesh(ansys.warp_by_vector('delta', factor=500), scalars='delta',
                 show_edges=True)
plotter.show()


plotter = pv.Plotter()
color = sns.color_palette("hls", 18)

for i in range(1, 4):
    if i in fiducial.data.coil:
        if i % 2 == 1:
            morph_mesh = TF1.copy()
            rotate = i-1
        else:
            morph_mesh = TF2.copy()
            rotate = i-2
        morph_mesh.rotate_z(rotate*20)

        morph_mesh['delta'] = rbf(morph_mesh.points)

        plotter.add_mesh(morph_mesh, color='w', opacity=0.05)
        plotter.add_mesh(morph_mesh.warp_by_vector('delta', factor=500),
                         color=color[i])

#plotter.add_mesh(fiducial.mesh, line_width=8)
plotter.show()
'''
