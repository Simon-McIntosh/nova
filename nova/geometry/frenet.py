"""Define curvilinear coordinates for 3D curves."""
from dataclasses import dataclass, field

import numpy as np
import pyvista
from scipy.interpolate import splprep, splev

from nova.graphics.plot import Plot


@dataclass
class Frenet(Plot):
    """Compute Frenet-Serret coordinates for a 3D curve."""

    points: np.ndarray = field(repr=False)
    tck: tuple = field(init=False, repr=False)
    parametric_length: np.ndarray = field(init=False, repr=False)
    data: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        """Fit spline and extract Frenet-Serret coordinates."""
        self.tck, self.parametric_length = splprep(self.points.T)
        self.build()

    def __getitem__(self, key: str):
        """Return item from data dict."""
        return self.data[key]

    def __setitem__(self, key: str, value: np.ndarray):
        """Update item in data dict."""
        self.data[key] = value

    def build(self):
        """Normalize coordinate system."""
        self["tangent"] = np.array(splev(self.parametric_length, self.tck, 1)).T
        self["normal"] = np.array(splev(self.parametric_length, self.tck, 2)).T
        if (
            index := np.isclose(np.linalg.norm(self["normal"], axis=1), 0, rtol=1e-2)
        ).any():
            for i in range(3):
                self["normal"][index, i] = np.interp(
                    self.parametric_length[index],
                    self.parametric_length[~index],
                    self["normal"][~index, i],
                )
        self["binormal"] = np.cross(self["tangent"], self["normal"])
        for axis in self.data:
            self.data[axis] /= np.linalg.norm(self[axis], axis=1)[:, np.newaxis]

    def plot(self):
        """Plot polyline."""
        mesh = pyvista.Spline(self.points)
        mesh["tangent"] = self["tangent"]
        print(mesh.point_data)

        pyvista.plot_arrows(self.points, 100 * self["tangent"])
        pyvista.plot_arrows(self.points, 100 * self["normal"])
        pyvista.plot_arrows(self.points, 100 * self["binormal"])
        """
        self.set_axes("3d")
        self.axes.plot(*self.points.T)
        self.axes.set_box_aspect(np.ptp(self.points, 0))
        self.arrow(self.points, self["tangent"], scale=1e5)
        """


if __name__ == "__main__":
    from nova.assembly.fiducialdata import FiducialData

    fiducial = FiducialData("RE", fill=True)

    curve = fiducial.data.centerline.data

    frenet = Frenet(curve)

    frenet.plot()
