"""Define curvilinear coordinates for 3D curves."""
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import ClassVar

import numpy as np
import pyvista

from nova.graphics.plot import Plot


@dataclass
class Frenet(Plot):
    """Compute Frenet-Serret coordinates for a 3D curve."""

    points: np.ndarray = field(repr=False)
    binormal: np.ndarray = field(
        repr=False, default_factory=lambda: np.array([0, 1, 0])
    )
    normal: np.ndarray = field(init=False, repr=False)
    tangent: np.ndarray = field(init=False, repr=False)
    curvature: np.ndarray = field(init=False, repr=False)
    torsion: np.ndarray = field(init=False, repr=False)

    frenet_attrs: ClassVar[list[str]] = ["tangent", "normal", "binormal"]

    def __post_init__(self):
        """Extract Frenet-Serret coordinates."""
        if self.binormal is None:
            self.binormal = next(
                field for field in fields(self) if field.name == "binormal"
            ).default
        self.build()

    def __len__(self) -> int:
        """Return length of point vector."""
        return len(self.points)

    def build(self):
        """Build coordinate system."""
        match len(self):
            case 2:
                self._from_binormal()
            case int(length) if length > 2:
                self._from_gradient()
            case _:
                raise IndexError("unable to extract coordinate system from points.")
        # self._normalize()

    @cached_property
    def segment_length(self) -> np.ndarray:
        """Return segment length vector."""
        return np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)

    @cached_property
    def length(self) -> float:
        """Return polyline length."""
        return np.sum(self.segment_length)

    @cached_property
    def parametric_length(self) -> np.ndarray:
        """Return parametric length vector."""
        return np.r_[0, np.cumsum(self.segment_length) / self.length]

    def _difference(self, vector):
        """Return vector finite difference."""
        return (vector[1:] - vector[:-1]) / self.segment_length[:, np.newaxis]

    def _forward(self, difference):
        """Return forward difference."""
        return np.r_[difference, difference[-1:]]

    def _backward(self, difference):
        """Return backard difference."""
        return np.r_[difference[:1], difference]

    def gradient(self, vector):
        """Return central difference gradient of vector."""
        difference = self._difference(vector)
        forward = self._forward(difference)
        backward = self._backward(difference)
        central = (forward + backward) / 2
        central[0] = forward[0]
        central[-1] = backward[-1]
        return central

    def _from_gradient(self):
        """Build parametric Frenet coordinate system from segment gradients."""
        self.tangent = self.gradient(self.points)
        self.tangent /= np.linalg.norm(self.tangent, axis=1)[:, np.newaxis]
        self.normal = self.gradient(self.tangent)
        self.curvature = np.linalg.norm(self.normal, axis=1)
        index = np.isclose(self.curvature, 0, atol=1e-5)
        self.normal[~index] /= self.curvature[~index, np.newaxis]
        if index.all():  # straight line
            self._from_binormal()
        elif index.any():
            for i in range(3):
                self.normal[index, i] = np.interp(
                    self.parametric_length[index],
                    self.parametric_length[~index],
                    self.normal[~index, i],
                    period=1,
                )
        self.normal -= (
            np.einsum("ij,ij->i", self.normal, self.tangent)
            / np.einsum("ij,ij->i", self.tangent, self.tangent)
        )[:, np.newaxis] * self.tangent
        self.normal /= np.linalg.norm(self.normal, axis=1)[:, np.newaxis]
        self.binormal = np.cross(self.tangent, self.normal)

        binorm_dot = self.gradient(self.binormal)
        self.torsion = -np.linalg.norm(binorm_dot, axis=1) * np.sign(
            np.einsum("ij,ij->i", binorm_dot, self.normal)
        )

    def _from_binormal(self):
        """Build parametric coordinate system from two point path."""
        tangent = self.points[-1] - self.points[0]
        tangent /= np.linalg.norm(tangent)
        assert np.shape(self.binormal) == (3,)
        binormal = self.binormal / np.linalg.norm(self.binormal)
        if np.isclose(abs(tangent @ binormal), 1):
            raise ValueError(
                f"straight line tangent {tangent} alligned with binormal {binormal}"
            )
        self.tangent = np.ones((len(self), 1)) * tangent[np.newaxis, :]
        self.binormal = np.ones((len(self), 1)) * self.binormal[np.newaxis, :]
        self.normal = np.cross(self.binormal, self.tangent)

    def _normalize(self):
        """Normalize coordinate system."""
        for axis in self.frenet_attrs:
            self.data[axis] /= np.linalg.norm(self[axis], axis=1)[:, np.newaxis]

    def plot(self, scale=1):
        """Plot polyline."""
        mesh = pyvista.MultipleLines(self.points)
        mag = scale * self.length / len(self)
        plotter = pyvista.Plotter()
        for axis, color in zip(self.frenet_attrs, ["blue", "orange", "green"]):
            plotter.add_arrows(self.points, getattr(self, axis), mag=mag, color=color)
        plotter.add_mesh(mesh)
        plotter.show()


if __name__ == "__main__":
    from nova.assembly.fiducialdata import FiducialData

    fiducial = FiducialData("RE", fill=True)

    curve = fiducial.data.centerline.data

    frenet = Frenet(curve)
    frenet.plot()
