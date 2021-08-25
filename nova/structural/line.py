"""Implemet geometric methods on vtk lines."""
import numpy as np
import pyvista as pv


class Line:
    """Manage geometric line methods."""

    @staticmethod
    def normalize(vector):
        """Return normalized vector."""
        norm = np.linalg.norm(vector, axis=1).reshape(-1, 1) @ np.ones((1, 3))
        return vector / norm

    def tangent(self, mesh: pv.PolyData):
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
        mesh['tangent'] = self.normalize(mesh['tangent'])

    def normal(self, mesh, plane=(0, 1, 0)):
        """Compute normal vector as cross product of tangent and plane."""
        mesh['normal'] = np.cross(plane, mesh['tangent'])
        mesh['normal'] = self.normalize(mesh['normal'])

    def plane(self, mesh):
        """Compute plane vector."""
        self.mesh['plane'] = np.cross(self.mesh['tangent'],
                                      self.mesh['normal'])
        mesh['plane'] = self.normalize(mesh['plane'])

    def vector(self, mesh):
        """Compute line vectors."""
        self.tangent(mesh)
        self.normal(mesh)
        self.plane(mesh)
