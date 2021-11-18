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

    def normal(self, mesh: pv.PolyData):
        """Compute normal vector as cross product of tangent and plane."""
        plane = self.plane(mesh.points)
        mesh['normal'] = np.cross(plane, mesh['tangent'])
        mesh['normal'] = self.normalize(mesh['normal'])

    def cross(self, mesh: pv.PolyData):
        """Compute cross product betweeh tangent and normal vectors."""
        mesh['cross'] = np.cross(mesh['tangent'], mesh['normal'])
        mesh['cross'] = self.normalize(mesh['cross'])

    def plane(self, points):
        """Return plane vector (eigen vector of smallest eigenvalue - PCA)."""
        points = points.copy()
        points -= np.mean(points, axis=0)  # center
        cov = np.cov(points, rowvar=False)  # covariance
        evals, evecs = np.linalg.eigh(cov)  # eigen
        index = np.argmin(evals)
        return evecs[:, index]

    def vector(self, mesh: pv.PolyData):
        """Compute line vectors."""
        self.tangent(mesh)
        self.normal(mesh)
        self.cross(mesh)
