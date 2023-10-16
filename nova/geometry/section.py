"""Manage sectional transforms."""
from dataclasses import dataclass, field

import numpy as np
import vedo

from nova.geometry.frenet import Frenet
from nova.geometry.rotate import to_vector, to_axes
from nova.geometry.vtkgen import VtkFrame


@dataclass
class Section:
    """Transform 2D sectional data."""

    points: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, float))
    triad: np.ndarray = field(default_factory=lambda: np.identity(3, float))
    mesh_array: list[VtkFrame] = field(init=False, default_factory=list)
    point_array: list[np.ndarray] = field(init=False, default_factory=list)

    def __len__(self):
        """Return length of mesh."""
        return len(self.mesh_array)

    def _append(self):
        """Generate mesh and append mesh to list."""
        self.point_array.append(self.points.tolist())
        self.mesh_array.append(
            VtkFrame([self.points, [[*range(len(self.points))]]]).c(len(self))
        )

    def _rotate_points(self, rotation):
        self.points -= self.origin
        self.points = rotation.apply(self.points)
        self.points += self.origin

    def to_vector(self, vector: np.ndarray, coord: int):
        """Rotate points to vector."""
        rotation = to_vector(self.triad[coord], vector)
        self._rotate_points(rotation)
        self.triad = rotation.apply(self.triad)

    def to_axes(self, axes: np.ndarray):
        """Rotate points to triad."""
        rotation = to_axes(axes, self.triad)
        self._rotate_points(rotation)
        self.triad = rotation.apply(self.triad.T).T

    def to_point(self, point):
        """Translate points to point and store mesh."""
        delta = np.array(point - self.origin, float)
        self.origin = self.origin + delta
        self.points = self.points + delta

    def sweep(self, path: np.ndarray, binormal: np.ndarray, align: str):
        """Sweep section along path."""
        frenet = Frenet(path, binormal)
        for i in range(len(path)):
            self.to_point(frenet.points[i])
            match align:
                case "axes":
                    axes = np.c_[
                        frenet.binormal[i], frenet.tangent[i], frenet.normal[i]
                    ]
                    self.to_axes(axes)
                case "vector":
                    sign = np.sign(np.array([0, 0, 1]) @ self.triad[0])
                    if np.isclose(sign, 0):
                        sign = 1
                    self.to_vector(np.array([0, 0, sign * 1]), 0)
                    self.to_vector(frenet.tangent[i], 1)
                case _:
                    raise ValueError(f"align {align} not in [vector or axes]")
            self._append()
        return self

    def plot(self):
        """Plot mesh instances."""
        vedo.show(*self.mesh_array, new=True, axes=True)
