
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
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

    def __post_init__(self):
        """Generate curve."""
        self.align()
        self.fit()
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

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

    def fit(self):
        """Align local coordinate system and fit arc to plane point cloud."""
        points_2d = np.zeros((len(self.points), 2))
        points_2d[:, 0] = np.einsum('j,ij->i', self.arc_axes[0], self.points)
        points_2d[:, 1] = np.einsum('j,ij->i', self.arc_axes[1], self.points)
        center, self.radius = self.fit_2d(points_2d)
        self.center = center @ self.arc_axes[:2]
        self.center += np.dot(self.arc_axes[2], self.mean)*self.arc_axes[2]

    def plot(self):
        """Plot best-fit arc and point cloud."""
        self.get_axes('2d')
        self.axes.plot(self.points[:, 0], self.points[:, 1], 'o')
        theta = np.linspace(0, 2*np.pi)
        points_2d = self.radius * np.c_[np.cos(theta), np.sin(theta)]
        points = points_2d @ self.arc_axes[:2]
        points += self.center
        self.axes.plot(points[:, 0], points[:, 1], ':', color='gray')


@dataclass
class ThreePointArc(Arc, Triple):
    """Generate arc segments."""

    def __post_init__(self):
        """Generate curve."""
        self.points = np.c_[self.point_a, self.point_b, self.point_c].T
        super().__post_init__()

    @cached_property
    def axis(self) -> np.ndarray | None:
        """Return arc axis, None if points are co-linear."""
        axis = np.cross(np.array(self.point_c) - np.array(self.point_b),
                        np.array(self.point_b) - np.array(self.point_a))
        length = np.linalg.norm(axis)
        if np.isclose(length, 0):
            return None
        return axis / length

    def _norm(self, point):
        """Return normalized vector between center and point."""
        vector = getattr(self, f'point_{point}') - self.center
        return vector / np.linalg.norm(vector)

    def sample(self, point_number=50):
        """Return sampled polyline."""
        axis_a = self._norm('a')
        axis_c = self._norm('c')
        axis_n = np.cross(axis_a, self.axis)
        cos_theta = np.dot(axis_a, axis_c)
        sin_theta = np.sqrt(1 - cos_theta**2)
        point_c = self.center + self.radius*(cos_theta*axis_a +
                                             sin_theta*axis_n)
        theta = np.linspace(0, np.arccos(cos_theta), point_number)
        if not np.allclose(point_c, self.point_c):
            theta = np.linspace(0, 2*np.pi-np.arccos(cos_theta), point_number)
        return self.center[np.newaxis, :] + self.radius*(
            np.cos(theta)[:, np.newaxis]*axis_a[np.newaxis, :] +
            np.sin(theta)[:, np.newaxis]*axis_n[np.newaxis, :])

    def plot(self):
        """Plot arc."""
        self.set_axes('2d')
        points = self.sample()
        self.axes.plot(points[:, 0], points[:, 1])
        self.axes.plot(self.point_a[0], self.point_a[1], 'o')
        self.axes.plot(self.point_b[0], self.point_b[1], 'X')
        self.axes.plot(self.point_c[0], self.point_c[1], 's')


@dataclass
class PolyLine(Plot):
    """Construct polyline from multiple segments."""

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


#@dataclass



if __name__ == '__main__':


    #arc = ThreePointArc((0, 1, -3.4), (1, 0, -3.4), (0, -1, -3.4))
    #arc.plot()

    rng = np.random.default_rng(2025)
    points = rng.random((11, 3))
    #points[:, 2] = 0

    line = PolyLine(points)
    line.plot()

    arc = Arc(line.curve)
    arc.plot()

    '''
    arc = Arc(8.88, (0.001, 7.3, -3), (-np.pi, np.pi), np.pi/2)
    arc.plot()
    #xyz = arc((0, 0, 0), 5, (0, np.pi), (0, np.pi))
    #.poplt.plot(xyz[0], xyz[2])
    print(arc.fit())
    '''
