"""Implemet geometric methods on vtk lines."""
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv


@dataclass
class Line:
    """Manage geometric line methods."""

    mesh: pv.PolyData = field(repr=False, default_factory=pv.PolyData)

    def __post_init__(self):
        """Compute vector array data."""
        self.compute_vectors()

    @classmethod
    def from_points(cls, points: np.ndarray):
        """Return line instance generated from points."""
        if len(points) > 2:
            return cls(pv.Spline(points))
        mesh = pv.PolyData(points, lines=np.arange(0, len(points)))
        mesh["arc_length"] = np.array(
            [0, np.linalg.norm(np.diff(points, axis=0))], float
        )
        return cls(mesh)

    @staticmethod
    def normalize(vector):
        """Return normalized vector."""
        norm = np.linalg.norm(vector, axis=1).reshape(-1, 1) @ np.ones((1, 3))
        return vector / norm

    def compute_tangent(self):
        """Compute centerline tangent as central difference."""
        points = self.mesh.points
        # forward difference
        forward = np.zeros(points.shape)
        forward[:-1] = points[1:] - points[:-1]
        forward[-1] = forward[0]
        forward = self.normalize(forward)
        # backward difference
        backward = np.zeros(points.shape)
        backward[1:] = points[1:] - points[:-1]
        backward[0] = backward[-1]
        backward = self.normalize(backward)
        # central difference
        self.mesh["tangent"] = (forward + backward) / 2
        self.mesh["tangent"][0] = forward[0]
        self.mesh["tangent"][-1] = backward[-1]

    def compute_normal(self):
        """Compute normal vector as cross product of tangent and plane."""
        plane = self.fit_plane()
        self.mesh["normal"] = np.cross(self.mesh["tangent"], plane)
        self.mesh["normal"] = self.normalize(self.mesh["normal"])

    def compute_cross(self):
        """Compute cross product betweeh tangent and normal vectors."""
        self.mesh["cross"] = np.cross(self.mesh["tangent"], self.mesh["normal"])
        self.mesh["cross"] = self.normalize(self.mesh["cross"])

    def fit_plane(self):
        """Return plane vector (eigen vector of smallest eigenvalue - PCA)."""
        points = self.mesh.points.copy()
        points -= np.mean(points, axis=0)  # center
        cov = np.cov(points, rowvar=False)  # covariance
        evals, evecs = np.linalg.eigh(cov)  # eigen
        index = np.argmin(evals)
        return evecs[:, index]

    def compute_vectors(self):
        """Compute line vectors."""
        self.compute_tangent()
        self.compute_normal()
        self.compute_cross()
