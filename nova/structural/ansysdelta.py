from dataclasses import dataclass

import numpy as np
import pyvista as pv

from nova.structural.ansyspost import AnsysPost
from nova.structural.plotter import Plotter


@dataclass
class AnsysDelta(Plotter):

    target: str = 'ccl0_EMerr'
    baseline: str = 'k0'
    part: str = 'wp'
    folder: str = 'TFCgapsG10'

    def __post_init__(self):
        """Calculate solution delta."""
        target_mesh = AnsysPost(self.folder, self.target, self.part).mesh
        baseline_mesh = AnsysPost(self.folder, self.baseline, self.part).mesh

        sort = np.argsort(baseline_mesh['ids'])
        unsort = np.zeros(baseline_mesh.n_points, dtype=int)
        unsort[sort] = np.arange(baseline_mesh.n_points)
        index = np.argsort(target_mesh['ids'])[unsort]

        self.mesh = pv.UnstructuredGrid()
        self.mesh.copy_structure(baseline_mesh)

        self.mesh['delta'] = target_mesh.points[index] - baseline_mesh.points

        for array in [array for array in target_mesh.point_data
                      if array != 'ids' and array in baseline_mesh.point_data]:
            self.mesh[array] = target_mesh[array][index] - baseline_mesh[array]
        '''
        self.mesh['delta'] = self.target.points - self.mesh.points

        print(np.linalg.norm(self.target.points - self.mesh.points, axis=1))
        index = np.argmax(np.linalg.norm(self.target.points -
                                         self.mesh.points, axis=1))
        print(self.target.points[index])
        print(self.mesh.points[index])



        #self.mesh['delta'] =
        #ansys = AnsysPost('TFCgapsG10', 'ccl0_EMerr', 'wp')
        #ansys = AnsysPost('TFCgapsG10', 'ccl0_EMerr', 'wp')
        '''

    def plot(self):
        plotter = pv.Plotter()
        self.target.points = self.mesh.points
        plotter.add_mesh(self.mesh, color='r')
        plotter.add_mesh(self.target, color='b')
        plotter.show()

if __name__ == '__main__':

    delta = AnsysDelta()
    #delta.plot()
    delta.warp(500, opacity=0, displace='disp-3')
