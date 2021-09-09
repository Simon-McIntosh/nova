"""Methods to morph vtk meshes to fit fiducial mesurments."""
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
import scipy.interpolate
import seaborn as sns

from nova.structural.ansyspost import AnsysPost
from nova.structural.fiducial import FiducialData
from nova.structural.plotter import Plotter
from nova.utilities import ANSYS


@dataclass
class Morph(Plotter):
    """Apply RBF morphing to VTK objects."""

    mesh: pv.PolyData
    fiducial: FiducialData = field(default_factory=FiducialData)
    rbf: scipy.interpolate.RBFInterpolator = field(init=None)

    def __post_init__(self):
        """Load data."""
        self.build_rbf()
        #self.interpolate()

    def build_rbf(self):
        """Build radial basis function interpolator."""
        index = np.unique(self.fiducial.mesh.points, axis=0,
                          return_index=True)[1]
        self.rbf = scipy.interpolate.RBFInterpolator(
            self.fiducial.mesh.points[index],
            self.fiducial.mesh['delta'][index])

    def interpolate(self):
        """Morph mesh."""
        self.mesh['delta'] = self.rbf(self.mesh.points)

    def to_ansys(self, resolution=0.2):
        bounds = self.mesh.bounds
        axes = [np.linspace(*bounds[2*i:2*i+2],
                            int(np.diff(bounds[2*i:2*i+2])/resolution))
                for i in range(3)]

        grid = np.array(np.meshgrid(*axes)).T
        shape = grid.shape
        grid = grid.reshape(-1, 3)

        delta = 500*self.rbf(grid).reshape(*shape).T

        with ANSYS.table('tmp', ext='.mac') as table:
            for i, coord in enumerate('xyz'):
                table.load(f'delta_{coord}', delta[i], [*axes])
                table.write(['x', 'y', 'z'])
                table.write_text('')


if __name__ == '__main__':

    TFC = AnsysPost('TFCgapsG10', 'k0', 'E_TF1').mesh
    #TFC = AnsysPost('TFCgapsG10', 'k0', 'E_TF2').mesh
    #TFC = AnsysPost('TFCgapsG10', 'k0', 'all').mesh

    morph = Morph(TFC)
    #morph.warp('delta', factor=500, opacity=0.05)
    #morph.fiducial.mesh.plot()
    #morph.animate('TFC18_morph', 'delta', max_factor=500, frames=2, opacity=0)
    morph.to_ansys()


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
