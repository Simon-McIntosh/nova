"""Methods to morph vtk meshes to fit fiducial mesurments."""
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pyvista as pv
import scipy.interpolate
from sklearn.gaussian_process import kernels, GaussianProcessRegressor

from nova.structural.ansyspost import AnsysPost
from nova.structural.fiducialmesh import FiducialMesh
from nova.structural.plotter import Plotter


@dataclass
class Morph(Plotter):
    """Morph VTK objects using Gaussian Process Regressor."""

    mesh: pv.PolyData
    length_scale: int = 5
    length_scale_bounds: Union[list[float, float], str] = 'fixed'
    alpha: float = 1e-9
    gpr: GaussianProcessRegressor = field(init=None)

    def __post_init__(self):
        """Init GPR."""
        kernel = kernels.RBF(length_scale=self.length_scale,
                             length_scale_bounds=self.length_scale_bounds)
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, random_state=2025, alpha=self.alpha)
        self.gpr.fit(self.mesh.points, self.mesh['delta'])

    #def fit(self):
    #    pickle.dump(large_object, fileobj, protocol=5)

    def predict(self, mesh: pv.PolyData, index=None):
        """Update Gausian process regression - for subset if index."""
        if 'delta' not in mesh.array_names:
            mesh['delta'] = np.zeros((mesh.n_points, 3))
        if index is None:
            index = np.full(mesh.n_points, True)
        index &= ~np.isnan(mesh.points).any(axis=1)
        mesh['delta'][index, :] = self.gpr.predict(mesh.points[index , :])


if __name__ == '__main__':

    TF1 = AnsysPost('TFCgapsG10', 'k0', 'E_TF1')
    # mesh = AnsysPost('TFCgapsG10', 'k0', 'E_TF2').mesh
    # mesh = AnsysPost('TFCgapsG10', 'k0', 'all').mesh
    #mesh = AnsysPost('TFCgapsG10', 'k0', 'E_WP_1').mesh


    fiducial = FiducialMesh()
    fiducial.load_centerline([0])

    morph = Morph(fiducial.mesh)

    index = TF1.mesh.points[:, 0] < 5.5
    morph.predict(TF1.mesh, index=index)

    TF1.warp(500)

    '''
    ilis = pv.PolyData(fiducial.mesh.points[:1])
    ilis['delta'] = [(0, 0, 0)]

    fiducial.mesh['delta'][0] += (0.001, 0, 0)

    morph = Morph(fiducial.mesh, mesh)

    morph = Morph(morph.mesh, TF1)

    #fiducial.mesh = fiducial.mesh.slice((0, 0, 1))
    #morph.mesh = morph.mesh.slice((0, 0, 1))


    plotter = pv.Plotter()
    morph.warp(500, plotter=plotter)
    fiducial.warp(500, plotter=plotter, color='w')

    plotter.show()

    # morph.warp('delta', factor=500, opacity=0.05)
    # morph.fiducial.mesh.plot()
    # morph.animate('TFC18_morph', 'delta', max_factor=500, frames=2, opacity=0)
    # morph.write_ansys_table()
    '''
