"""Manage as-designed coil winding pack descriptors."""
from dataclasses import dataclass, field
import os
from typing import Union

import numpy as np
import pandas
import pyvista as pv
from scipy.spatial.transform import Rotation

from nova.definitions import root_dir
from nova.structural.windingpack import WindingPack
from nova.utilities.pyplot import plt


@dataclass
class UniformWindingPack:
    """Simplify TF conductor centerline for EM calculations."""

    wp_mesh: pv.PolyData = field(init=False, repr=False)
    txt_mesh: pv.PolyData = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load winding-pack mesh."""
        self.wp_mesh = WindingPack('TFC1_CL').mesh
        self.read_txt()
        self.load_ccl()

    @property
    def vtk_file(self):
        """Return vtk file path (Uniform Current Centerline)."""
        return os.path.join(root_dir, 'input/geometry/ITER/TF_UCCL.vtk')

    @property
    def txt_file(self):
        """Return txt file path."""
        return os.path.join(root_dir, 'input/geometry/ITER/TFC1_CCL.txt')

    def load_ccl(self):
        """Load single coil current centerline."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.compute_ccl()
            self.extract_turns()
            self.pattern_mesh()
            self.mesh.save(self.vtk_file)

    def read_txt(self):
        """Read TFC1 conductor centerline from txt file."""
        points = pandas.read_csv(self.txt_file, delimiter='\t',
                                 skiprows=1, header=None).to_numpy()
        points = np.insert(points, 1, np.zeros(len(points)), axis=1)
        points = np.append(points, points[:1, :], axis=0)
        self.txt_mesh = pv.Spline(points)
        self.compute_tangent(self.txt_mesh)

    @staticmethod
    def normalize(vector):
        """Return normalized vector."""
        norm = np.linalg.norm(vector, axis=1).reshape(-1, 1) @ np.ones((1, 3))
        return vector / norm

    def compute_tangent(self, mesh):
        """Compute centerline tangent as central diffrence."""
        points = mesh.points
        # forward diffrence
        forward = np.zeros(points.shape)
        forward[:-1] = points[1:] - points[:-1]
        forward[-1] = forward[0]
        forward = self.normalize(forward)
        # backward diffrence
        backward = np.zeros(points.shape)
        backward[1:] = points[1:] - points[:-1]
        backward[0] = backward[-1]
        backward = self.normalize(backward)
        # central diffrence
        mesh['tangent'] = (forward + backward) / 2

    def select_coil(self, n_coil, n_cells=7):
        """Return mesh for single TF coil."""
        return self.wp_mesh.extract_cells(
            range(n_cells*n_coil, n_cells*(n_coil+1)))

    def box_bounds(self, point, edge_length):
        """Return box centered around point."""
        bounds = np.zeros(6)
        bounds[::2] = point - edge_length/2
        bounds[1::2] = point + edge_length/2
        return bounds

    def slice_coil(self, coil, index):
        """Return sliced mesh."""
        if isinstance(coil, int):
            coil = self.select_coil(coil)
        tangent = self.txt_mesh['tangent'][index]
        point = self.txt_mesh.points[index]
        plane = coil.slice(tangent, point)
        cube = pv.Cube(center=[0, 0, -0.03],
                       x_length=0.1, y_length=0.9, z_length=0.7)
        cube = self.rotate(cube, tangent)
        cube.translate(point)
        return plane.clip_box(cube, invert=False)

    @staticmethod
    def rotate(mesh, tangent, referance=[1, 0, 0]):
        """Return Euler rotation angles."""
        rotvec = np.cross(referance, tangent)
        rotvec *= np.arccos(referance @ tangent) / np.linalg.norm(rotvec)
        euler_angles = \
            Rotation.from_rotvec(rotvec).as_euler('xyz', degrees=True)
        mesh.rotate_x(euler_angles[0])
        mesh.rotate_y(euler_angles[1])
        mesh.rotate_z(euler_angles[2])
        return mesh

    def compute_ccl(self):
        """Extract winding centerlines from slices."""
        coil = self.select_coil(0)
        loops = np.zeros((134, self.txt_mesh.n_points, 3))
        index = 0
        for i in range(self.txt_mesh.n_points):
            plane = self.slice_coil(coil, i)
            try:
                loops[:, index] = plane.points
                index += 1
            except ValueError:
                pass
        self.mesh = pv.PolyData()
        for loop in loops[:, :index]:
            self.mesh += pv.Spline(loop)

    def extract_turns(self):
        """Extract low-field midplane turn layout."""
        self.mesh['turns'] = self.slice_coil(0, 0).points

    def plot_turns(self):
        """Plot low-field turn grid in x-y plane."""
        points = self.mesh['turns']
        ax = plt.subplots(1, 1)[1]
        ax.plot(points[:, 0], points[:, 1], 'o')
        plt.axis('off')
        plt.axis('equal')

    def pattern_mesh(self):
        """Pattern TF coils."""
        TFC1 = self.mesh.copy()
        self.mesh = pv.PolyData()
        for i in range(18):
            TFC = TFC1.copy()
            TFC.rotate_z(360*i / 18)
            self.mesh += TFC


if __name__ == '__main__':

    ccl = UniformWindingPack()
    #ccl.plot_turns()

    #points = ccl.mesh.points.reshape(18, 134, -1, 3)
    #loop = points[9, 0]
    #plt.plot(loop[:, 0], loop[:, 2])
