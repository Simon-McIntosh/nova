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
    binormal: np.ndarray | None = field(
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

    def gradient(self, vector):
        """Return gradient of vector."""
        diff = (vector[1:] - vector[:-1]) / self.segment_length[:, np.newaxis]
        return np.r_[diff[:1], diff]

    def _from_gradient(self):
        """Build parametric Frenet coordinate system from segment gradients."""
        self.tangent = self.gradient(self.points)
        self.normal = self.gradient(self.tangent)
        self.curvature = np.linalg.norm(self.normal, axis=1)
        index = np.isclose(self.curvature, 0, atol=1e-5)
        self.normal[~index] /= self.curvature[~index, np.newaxis]
        if index.any():
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
        self.torsion = np.linalg.norm(self.gradient(self.binormal))

        # self.normal /= self["curvature"][:, np.newaxis]
        # self.binormal = np.cross(self.tangent, self.normal)
        """
        tck, parametric_length = splprep(self.points.T, k=2, s=0)
        self.tangent = np.array(splev(parametric_length, tck, 1)).T / self.length
        # self.tangent /= np.linalg.norm(self.tangent, axis=1)[:, np.newaxis]
        self.normal = np.array(splev(parametric_length, tck, 2)).T / self.length**2
        self["curvature"] = np.linalg.norm(self.normal, axis=1)
        # self.normal /= self["curvature"][:, np.newaxis]
        self.binormal = np.cross(self.tangent, self.normal)
        b_tck = splprep(self.binormal.T, u=parametric_length, k=self.degree, s=0)[0]
        self["torsion"] = np.linalg.norm(
            np.array(splev(parametric_length, b_tck, 1)).T / self.length, axis=1
        )
        index = np.isclose(
            np.einsum("ij,ij->i", self.tangent[:-1], self.tangent[1:]),
            0,
            atol=1e-3,
        )
        print(np.einsum("ij,ij->i", self.tangent[:-1], self.tangent[1:]))
        if index.any():
            for i in range(3):
                self.normal[index, i] = np.interp(
                    parametric_length[index],
                    parametric_length[~index],
                    self.normal[~index, i],
                    period=1,
                )
            self.normal[index] -= (
                np.einsum("ij,ij->i", self.normal[index], self.tangent[index])
                / np.einsum("ij,ij->i", self.tangent[index], self.tangent[index])
            )[:, np.newaxis] * self.tangent[index]
            self.binormal[index] = np.cross(
                self.tangent[index], self.normal[index]
            )
            import matplotlib.pylab as plt

            plt.plot(parametric_length, self["curvature"])
            plt.plot(parametric_length[index], self["curvature"][index])
        """

    def _from_binormal(self):
        """Build parametric coordinate system from two point path."""
        self.tangent = (
            np.ones((2, 1)) * (self.points[1] - self.points[0])[np.newaxis, :]
        )
        self.binormal = np.ones((2, 1)) * self.binormal[np.newaxis, :]
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
    """
    points = np.zeros((30, 3))
    points[:, 0] = np.linspace(0, 50, len(points))
    points[-15:, 1] = np.linspace(0, 50, 15)
    frenet = Frenet(points)
    """
    frenet.plot()
