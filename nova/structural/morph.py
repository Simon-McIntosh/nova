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
    neighbors: int = 1
    smoothing: float = 2.
    kernel: str = 'linear'
    epsilon: float = 1
    rbf: scipy.interpolate.RBFInterpolator = field(init=None)

    def __post_init__(self):
        """Load data."""
        self.build()
        if self.mesh is not None:
            self.interpolate()

    def build(self):
        """Build radial basis function interpolator."""
        '''
        index = np.unique(self.fiducial.points, axis=0, return_index=True)[1]
        self.rbf = scipy.interpolate.RBFInterpolator(
            self.fiducial.points[index], self.fiducial['delta'][index],
            neighbors=self.neighbors,
            smoothing=self.smoothing, kernel=self.kernel, epsilon=self.epsilon)
        '''




    def interpolate(self):
        """Morph mesh."""

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel


        kernel = RBF(length_scale=5, length_scale_bounds='fixed')
        gpr = GaussianProcessRegressor(
            kernel=kernel, random_state=0, alpha=0.25**2).fit(
                self.fiducial.points,self.fiducial['delta'])

        index = ~np.isnan(self.mesh.points).any(axis=1)
        print(np.sum(~index))

        self.mesh['delta'] = np.zeros((self.mesh.n_points, 3))
        self.mesh['delta'][index , :] = gpr.predict(self.mesh.points[index , :])



    '''
    def save(self):
        """Save morph instance."""


    def load(self):
        f = open(self.filename, 'rb')
        tmp_dict = cPickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)


    def save(self):
        f = open(self.filename, 'wb')
        cPickle.dump(self.__dict__, f, 2)
        f.close()

            @classmethod
    def loader(cls,f):
        return cPickle.load(f)
    '''


if __name__ == '__main__':

    TF1 = AnsysPost('TFCgapsG10', 'k0', 'E_TF1').mesh
    # mesh = AnsysPost('TFCgapsG10', 'k0', 'E_TF2').mesh
    # mesh = AnsysPost('TFCgapsG10', 'k0', 'all').mesh
    mesh = AnsysPost('TFCgapsG10', 'k0', 'E_WP_1').mesh

    fiducial = FiducialMesh()
    fiducial.load_centerline([0])

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
