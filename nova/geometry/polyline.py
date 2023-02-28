"""Impement rdp-like decimator for mixed linear / arc polylines."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from rdp import rdp
import scipy

from nova.frame.baseplot import Plot


@dataclass
class Triple:
    """Manage 3-point arc nodes."""

    point_a: np.ndarray
    point_b: np.ndarray
    point_c: np.ndarray


@dataclass
class Arc(Plot):
    """Fit arc to 3d point cloud."""

    points: np.ndarray | None = None
    arc_axes: np.ndarray = field(init=False)
    center: np.ndarray = field(init=False)
    radius: float = field(init=False)
    theta: float = field(init=False)
    error: float = field(init=False)
    eps: float = 1e-8

    def __post_init__(self):
        """Generate curve."""
        super().__post_init__()
        self.align()
        self.fit()

    def align(self):
        """Align point cloud to 2d plane."""
        self.mean = np.mean(self.points, axis=0)
        delta = self.points - self.mean[np.newaxis, :]
        self.arc_axes = scipy.linalg.svd(delta)[2]

    @staticmethod
    def fit_2d(points):
        """Return center and radius of best fit circle to polyline."""
        coef = np.linalg.lstsq(np.c_[2*points, np.ones(len(points))],
                               np.sum(points**2, axis=1), rcond=None)[0]
        center = coef[:2]
        radius = np.sqrt(coef[2] + np.sum(center**2))
        return center, radius

    @cached_property
    def points_2d(self):
        """Return point locations projected onto 2d plane."""
        return np.c_[np.einsum('j,ij->i', self.arc_axes[0], self.points),
                     np.einsum('j,ij->i', self.arc_axes[1], self.points)]

    @cached_property
    def points_fit(self):
        """Return best-fit points in 3d space."""
        return self.center[np.newaxis, :] + self.radius*(
            np.cos(self.theta)[:, np.newaxis]*self.arc_axes[np.newaxis, 0] +
            np.sin(self.theta)[:, np.newaxis]*self.arc_axes[np.newaxis, 1])

    def fit(self):
        """Align local coordinate system and fit arc to plane point cloud."""
        center, self.radius = self.fit_2d(self.points_2d)
        self.center = center @ self.arc_axes[:2]
        self.center += np.mean(np.einsum('j,ij->i', self.arc_axes[2],
                                         self.points))*self.arc_axes[2]
        center_points = self.points_2d - center
        self.theta = np.unwrap(np.arctan2(center_points[:, 1],
                                          center_points[:, 0]))
        self.error = np.linalg.norm(self.points -
                                    self.points_fit, axis=1).std()

    @property
    def length(self):
        """Return length of discrete polyline."""
        return np.linalg.norm(self.points[1:] - self.points[:-1], axis=1).sum()

    @property
    def match(self):
        """Return status of normalized fit residual."""
        return self.error / self.length < self.eps

    def sample(self, point_number=50):
        """Return sampled polyline."""
        theta = np.linspace(self.theta[0], self.theta[-1],
                            point_number)[:, np.newaxis]
        return self.center[np.newaxis, :] + self.radius*(
            np.cos(theta)*self.arc_axes[0][np.newaxis, :] +
            np.sin(theta)*self.arc_axes[1][np.newaxis, :])

    @property
    def nodes(self):
        """Return best-fit node triple respecting start and end locations."""
        return np.array([self.points[0], self.sample(3)[1], self.points[-1]])

    def plot_circle(self):
        """Plot best fit circle."""
        self.get_axes('2d')
        theta = np.linspace(0, 2*np.pi)
        points_2d = self.radius * np.c_[np.cos(theta), np.sin(theta)]
        points = points_2d @ self.arc_axes[:2]
        points += self.center
        self.axes.plot(points[:, 0], points[:, 1], ':', color='gray')

    def plot(self):
        """Plot point and best-fit data."""
        self.get_axes('2d')
        self.axes.plot(self.points[:, 0], self.points[:, 1], 'o')
        self.axes.plot(self.points_fit[:, 0], self.points_fit[:, 1], 'D')

    def plot_fit(self):
        """Plot best-fit arc and point cloud."""
        self.plot()
        self.plot_circle()


@dataclass
class ThreePointArc(Arc, Triple):
    """Generate arc segments."""

    def __post_init__(self):
        """Generate curve."""
        self.points = np.c_[self.point_a, self.point_b, self.point_c].T
        super().__post_init__()

    def plot(self):
        """Plot arc."""
        self.set_axes('2d')
        points = self.sample()
        self.axes.plot(points[:, 0], points[:, 1])
        self.axes.plot(self.point_a[0], self.point_a[1], 'o')
        self.axes.plot(self.point_b[0], self.point_b[1], 'X')
        self.axes.plot(self.point_c[0], self.point_c[1], 's')


@dataclass
class PolyArc(Plot):
    """Construct polyline from multiple arc segments."""

    points: np.ndarray
    resolution: int = 20
    curve: np.ndarray = field(init=False)

    def __post_init__(self):
        """Build multi-arc polyline."""
        super().__post_init__()
        self.build()

    def build(self):
        """Build multi arc curve."""
        segment_number = (len(self.points) - 1) // 2
        self.curve = np.zeros((segment_number*self.resolution, 3))
        if segment_number > 1:
            self.curve = self.curve[:-1]
        for i in range(segment_number):
            points = self.points[slice(2*i, 2*i+3)]
            start_index = i*self.resolution
            if i > 0:
                start_index -= 1
            self.curve[slice(start_index, start_index+self.resolution)] = \
                ThreePointArc(*points).sample(self.resolution)

    def plot(self):
        """Plot polyline."""
        self.get_axes('2d')
        self.axes.plot(self.points[:, 0], self.points[:, 1], 'o')
        self.axes.plot(self.curve[:, 0], self.curve[:, 1], '-')


@dataclass
class PolyLine(Plot):
    """Decimate polylines using a hybrid arc/line-segment rdp algorithum."""

    points: np.ndarray
    segments: list = field(init=False, default_factory=list)

    def fit_arc(self, points):
        """Return point index prior to first arc mis-match."""
        for i in range(4, len(points)+1):
            if not Arc(points[:i]).match:
                return i-1
        return i

    def append(self, points):
        """Append points to segment list."""
        if len(points) >= 4:
            self.segments.append(Arc(points).nodes)
            return
        if i in range(len(points)):
            self.segments.append(points[i:i+2])

    def plot(self):
        """Plot decimated polyline."""
        self.set_axes('2d')
        self.axes.plot(self.points[:, 0], self.points[:, 1])


        #points = rdp(self.points, 1e-3)
        points = self.points

        start = 0
        while start <= len(points) - 4:
            number = self.fit_arc(points[start:])
            print(number)
            self.append(points[start:start+number])
            start += number-1


if __name__ == '__main__':

    rng = np.random.default_rng(2025)
    points = rng.random((3, 3))
    line = PolyArc(points, 100)

    curve = line.curve
    for i in range(2):
        points = rng.random((8, 3))
        line = PolyArc(points, 100)
        curve = np.append(curve, rng.random((2, 3)), axis=0)
        curve = np.append(curve, line.curve, axis=0)


    #line.plot()

    polyline = PolyLine(curve)
    polyline.plot()

    #arc = Arc(line.curve[:20])
    #arc.plot_fit()

    #line.plot()

    #print(arc.match)
