
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nova.frame.baseplot import Plot


@dataclass
class ThreePointArc(Plot):
    """Generate arc segments."""

    point_a: tuple[float, float, float] | np.ndarray
    point_b: tuple[float, float, float] | np.ndarray
    point_c: tuple[float, float, float] | np.ndarray

    center: np.ndarray = field(init=False)
    radius: float = field(init=False)

    def __post_init__(self):
        """Generate curve."""
        for point in 'abc':
            attr = f'point_{point}'
            setattr(self, attr, np.array(getattr(self, attr)))
        self.center, self.radius = self.fit(
            np.c_[self.point_a, self.point_b, self.point_c].T)
        super().__post_init__()

    @cached_property
    def axis(self) -> tuple[float, float, float] | None:
        """Return arc axis, None if points are co-linear."""
        axis = np.cross(self.point_c - self.point_b,
                        self.point_b - self.point_a)
        length = np.linalg.norm(axis)
        if np.isclose(length, 0):
            return None
        return axis / length

    def _norm(self, point):
        """Return normalized vector between center and point."""
        vector = getattr(self, f'point_{point}') - self.center
        return vector / np.linalg.norm(vector)

    def linspace(self, num=50):
        """Return sampled polyline."""
        axis_a = self._norm('a')
        axis_c = self._norm('c')
        axis_n = np.cross(axis_a, self.axis)
        cos_theta = np.dot(axis_a, axis_c)
        theta = np.linspace(0, np.arccos(cos_theta), num)
        return self.center[np.newaxis, :] + \
            self.radius*np.cos(theta)[:, np.newaxis]*axis_a[np.newaxis, :] + \
            self.radius*np.sin(theta)[:, np.newaxis]*axis_n[np.newaxis, :]

    '''
    def space(self, angle: float | tuple[float, float]):
        """Return angle vectors for tuple inputs."""
        match angle:
            case tuple():
                return np.linspace(angle[0], angle[1], self.number)
            case _:
                return angle

    def generate(self):
        """Generate xyz point array."""
        radius = self.radius * np.ones(self.number)
        phi = self.space(self.phi)
        theta = self.space(self.theta)
        self.points = np.c_[radius*np.sin(theta)*np.cos(phi),
                            radius*np.sin(theta)*np.sin(phi),
                            radius*np.cos(theta)]
        self.points += np.array(self.center)[np.newaxis, :]
    '''

    def plot(self):
        """Plot arc."""
        self.set_axes('2d')
        points = self.linspace()
        self.axes.plot(points[:, 0], points[:, 1])
        self.axes.plot(self.point_a[0], self.point_a[1], 'o')
        self.axes.plot(self.point_b[0], self.point_b[1], 'X')
        self.axes.plot(self.point_c[0], self.point_c[1], 's')


    @staticmethod
    def fit(points):
        """Return center and radius of best fit circle to polyline."""
        coef = np.linalg.lstsq(np.c_[2*points, np.ones(len(points))],
                               np.sum(points**2, axis=1))[0]
        center = coef[:3]
        radius = np.sqrt(coef[3] + np.sum(center**2))
        return center, radius


'''
    radius: float = 1
    center: tuple[float, float, float] = (0, 0, 0)
    normal: tuple[float, float, float] = (0, 0, 1)
    theta: float | tuple[float, float] = (0, np.pi/4)
'''

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    arc = ThreePointArc((0, 1, 0),
                        (1, 0, 0),
                        (0, -1, 0))
    arc.plot()

    '''
    arc = Arc(8.88, (0.001, 7.3, -3), (-np.pi, np.pi), np.pi/2)
    arc.plot()
    #xyz = arc((0, 0, 0), 5, (0, np.pi), (0, np.pi))
    #.poplt.plot(xyz[0], xyz[2])
    print(arc.fit())
    '''
