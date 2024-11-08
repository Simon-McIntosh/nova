"""Manage sectional transforms."""

from dataclasses import dataclass, field

import numpy as np
import scipy.interpolate
import vedo

from nova.geometry.frenet import Frenet
from nova.geometry.rotate import to_vector, to_axes, by_angle
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

    def by_angle(self, axis: np.ndarray, angle: float):
        """Rotate points by angle about axis."""
        rotation = by_angle(axis, angle)
        self._rotate_points(rotation)
        self.triad = rotation.apply(self.triad)

    def to_vector(self, vector: np.ndarray, coord: int):
        """Rotate points to vector."""
        rotation = to_vector(self.triad[coord], vector)
        self._rotate_points(rotation)
        self.triad = rotation.apply(self.triad)

    def to_axes(self, axes: np.ndarray):
        """Rotate points to align triad with axes."""
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

        triad = np.identity(3)
        normal = np.zeros((len(path), 3))
        for i in range(len(path)):
            rotation = to_vector(triad[0], frenet.tangent[i])
            triad = rotation.apply(triad)
            normal[i] = -triad[0]

        normal = frenet.project(normal, triad[1][np.newaxis])
        twist = np.arccos(
            np.clip(
                np.dot(normal[0], normal[-1]),
                -1.0,
                1.0,
            )
        )

        # untwist = scipy.interpolate.interp1d(
        #    [0, 1], [0, -2 * twist], fill_value="extrapolate"
        # )
        # angle = 0

        delta = -13 * twist / len(path)
        for i in range(len(path)):
            self.to_point(frenet.points[i])
            match align:
                case "axes":
                    axes = np.c_[
                        -frenet.normal[i], frenet.tangent[i], frenet.binormal[i]
                    ]
                    self.to_axes(axes)
                case "vector":
                    sign = np.sign(np.array([0, 0, 1]) @ self.triad[0])
                    if np.isclose(sign, 0):
                        sign = 1
                    self.to_vector(np.array([0, 0, sign * 1]), 0)
                    self.to_vector(frenet.tangent[i], 1)
                case "twist":
                    # delta = untwist(frenet.parametric_length[i]) - angle
                    # angle = untwist(frenet.parametric_length[i])
                    self.by_angle(frenet.tangent[i], -delta)
                    self.to_vector(frenet.tangent[i], 1)

                case _:
                    raise ValueError(f"align {align} not in [vector or axes]")
            self._append()
        return self

    def plot(self):
        """Plot mesh instances."""
        vedo.show(*self.mesh_array, new=True, axes=True)
