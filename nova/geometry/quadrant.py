"""GGenerate quatrant plasma profile parameters."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from nova.graphics.plot import Plot


@dataclass
class Quadrant(Plot):
    """Manage single quadrant squarness calculations."""

    minor_point: tuple[float, float]
    major_point: tuple[float, float]

    @cached_property
    def axis(self):
        """Return quadrant axis."""
        return np.array([self.major_point[0], self.minor_point[1]])

    @cached_property
    def minor_radius(self):
        """Return minor radius."""
        return self.minor_point[0] - self.axis[0]

    @cached_property
    def major_radius(self):
        """Return minor radius."""
        return self.major_point[1] - self.axis[1]

    @cached_property
    def quadrant(self):
        """Return quadrant index."""
        theta = np.arctan2(self.major_radius, self.minor_radius)
        if theta < 0:
            theta += 2 * np.pi
        return int(2 * theta / np.pi)

    def arc_radius(self, u):
        """Return parametric arc radius."""
        return self.minor_radius * (1 - u**2) / (1 + u**2)

    def arc_height(self, u):
        """Return parametric arc height."""
        return self.major_radius * 2 * u / (1 + u**2)

    @cached_property
    def theta(self):
        """Return quadrant bisection angle."""
        theta = np.arctan(abs(self.major_radius / self.minor_radius))
        if self.quadrant in [1, 3]:
            theta = np.pi / 2 - theta
        return theta

    @cached_property
    def ellipse_point(self):
        """Return arc bisection point."""
        u_bisect = np.tan(np.pi / 8)
        return (
            np.array([self.arc_radius(u_bisect), self.arc_height(u_bisect)]) + self.axis
        )

    @cached_property
    def ellipse_radius(self):
        """Return radius of elliptic arc at quadrant midpoint."""
        return np.linalg.norm(self.ellipse_point - self.axis)

    @cached_property
    def square_radius(self):
        """Return ellipse radius L2 norm."""
        return np.linalg.norm([self.minor_radius, self.major_radius])

    def squareness(self, separatrix_point):
        """Return squarness of separatrix point."""
        radius = np.linalg.norm(np.array(separatrix_point) - self.axis)
        return (radius - self.ellipse_radius) / (
            self.square_radius - self.ellipse_radius
        )

    def separatrix_point(self, squareness):
        """Return separatrix point for a given squareness."""
        radius = (
            squareness * (self.square_radius - self.ellipse_radius)
            + self.ellipse_radius
        )
        theta = self.theta + self.quadrant * np.pi / 2
        return self.axis + radius * np.array([np.cos(theta), np.sin(theta)])

    def plot(self, axes=None):
        """Plot parametric arc."""
        u = np.linspace(0, 1)
        self.set_axes("2d", axes)
        self.axes.plot(*self.minor_point, "ko", ms=4)
        self.axes.plot(*self.major_point, "ko", ms=4)
        self.axes.plot(*self.ellipse_point, "kd", ms=6)
        self.axes.plot(*np.c_[self.axis, self.ellipse_point], "--", color="gray")
        self.axes.plot(
            self.axis[0] + np.array([0, self.minor_radius]),
            self.axis[1] * np.ones(2),
            "-",
            color="gray",
            lw=1,
        )
        self.axes.plot(
            self.axis[0] * np.ones(2),
            self.axis[1] + np.array([0, self.major_radius]),
            "-",
            color="gray",
            lw=1,
        )
        self.axes.plot(
            self.arc_radius(u) + self.axis[0],
            self.arc_height(u) + self.axis[1],
            "--",
            color="gray",
        )
