from dataclasses import dataclass

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
        self.target = AnsysPost(self.folder, self.target, self.part).mesh
        self.mesh = AnsysPost(self.folder, self.baseline, self.part).mesh
        self.mesh.clear_data()

        self.mesh['delta'] = self.target.points - self.mesh.points

        print(np.linalg.norm(self.target.points - self.mesh.points, axis=1))
        index = np.argmax(np.linalg.norm(self.target.points -
                                         self.mesh.points, axis=1))
        print(self.target.points[index])
        print(self.mesh.points[index])



        #self.mesh['delta'] =
        #ansys = AnsysPost('TFCgapsG10', 'ccl0_EMerr', 'wp')
        #ansys = AnsysPost('TFCgapsG10', 'ccl0_EMerr', 'wp')

if __name__ == '__main__':

    delta = AnsysDelta()

    #delta.warp(1)
