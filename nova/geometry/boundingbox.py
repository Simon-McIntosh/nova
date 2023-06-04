"""Optimize placment of plasma bounding box within first wall."""
from dataclasses import dataclass

import numpy as np

from nova.geometry.plasmaprofile import PlasmaProfile
from nova.geometry.quadrant import Quadrant
from nova.graphics.plot import Plot
from nova.imas.machine import Wall


@dataclass
class Datum:
    """Plasma datum."""

    datum: str
    radius: float
    height: float


@dataclass
class BoundingBox(Plot, Datum):
    """Construct plasma bounding box."""

    minor_radius: float
    elongation: float
    triangularity_upper: float
    triangularity_lower: float
    triangularity_outer: float
    triangularity_inner: float

    @property
    def axis(self):
        """Return geometric axis."""
        match self.datum:
            case "geometric":
                return np.array([self.radius, self.height])
            case "x-point":
                return np.array(
                    [self.radius, self.height]
                ) + self.minor_radius * np.array(
                    [self.triangularity_lower, self.elongation]
                )
            case _:
                raise NotImplementedError()

    @property
    def upper(self):
        """Return upper control point."""
        return self.axis + self.minor_radius * np.array(
            [-self.triangularity_upper, self.elongation]
        )

    @property
    def lower(self):
        """Return upper control point."""
        return self.axis - self.minor_radius * np.array(
            [self.triangularity_lower, self.elongation]
        )

    @property
    def inner(self):
        """Return inner control point."""
        return self.axis + self.minor_radius * np.array([-1, self.triangularity_inner])

    @property
    def outer(self):
        """Return outer control point."""
        return self.axis + self.minor_radius * np.array([1, self.triangularity_outer])

    @property
    def upper_outer(self):
        """Return upper outer control point."""
        return Quadrant(self.outer, self.upper).separatrix_point(
            self["squareness_upper_outer"]
        )

    @property
    def upper_inner(self):
        """Return upper inner control point."""
        return Quadrant(self.inner, self.upper).separatrix_point(
            self["squareness_upper_inner"]
        )

    @property
    def lower_inner(self):
        """Return lower inner control point."""
        return Quadrant(self.inner, self.lower).separatrix_point(
            self["squareness_lower_inner"]
        )

    @property
    def lower_outer(self):
        """Return lower outer control point."""
        return Quadrant(self.outer, self.lower).separatrix_point(
            self["squareness_lower_outer"]
        )

    @property
    def coef(self) -> dict:
        """Return plasma profile coefficents."""
        return {"geometric_radius": self.axis[0], "geometric_height": self.axis[1]} | {
            attr: getattr(self, attr)
            for attr in [
                "minor_radius",
                "elongation",
                "triangularity_upper",
                "triangularity_lower",
            ]
        }

    def plot_plasma_profile(self):
        """Plot analytic profile."""
        plasmaprofile = PlasmaProfile(point_number=121)
        plasmaprofile.limiter(**self.coef).plot()

    def plot(self, index=None, axes=None, **kwargs):
        """Plot control points and first wall."""
        self.get_axes("2d", axes)
        wall = Wall()
        for segment in wall.segments:
            self.axes.plot(
                segment[:, 0], segment[:, 1], "-", ms=4, color="gray", linewidth=1.5
            )
        self.plot_plasma_profile()


if __name__ == "__main__":
    BoundingBox("x-point", 5.5, -3, 1.5, 2, 0.5, 0.5, 0, 0).plot()
