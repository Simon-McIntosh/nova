"""Coordinate system transform methods for BiotFrame."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

import nova.frame.metamethod as metamethod
from nova.frame.framelink import FrameLink
from nova.graphics.plot import Plot3D


@dataclass
class Space(metamethod.Space, Plot3D):
    """Coordinate system transform methods."""

    name = "space"

    frame: FrameLink = field(repr=False)
    coordinate_axes: np.ndarray = field(init=False, repr=False)
    coordinate_origin: np.ndarray = field(init=False, repr=False)
    quadrant_segments: int = 11

    def initialize(self):
        """Build local coordinate axes."""
        self.coordinate_axes = np.zeros((len(self.frame), 3, 3))
        self.coordinate_axes[..., 0] = self.normal
        self.coordinate_axes[..., 1] = np.cross(self.axis, self.normal)
        self.coordinate_axes[..., 2] = self.axis
        self.coordinate_axes /= np.linalg.norm(self.coordinate_axes, axis=1)[
            :, np.newaxis
        ]
        self.origin = self._column_stack(*list("xyz"))

    def _column_stack(self, *args: tuple[str]):
        """Return stacked array column vectors."""
        return np.column_stack([self.frame.aloc[attr] for attr in args])

    @cached_property
    def axis(self):
        """Return source element axis."""
        return self._column_stack("ax", "ay", "az")

    @cached_property
    def normal(self):
        """Return source element normal."""
        return self._column_stack("nx", "ny", "nz")

    @cached_property
    def start_point(self):
        """Return element start point."""
        return self._column_stack("x1", "y1", "z1")

    @cached_property
    def end_point(self):
        """Return element end point."""
        return self._column_stack("x2", "y2", "z2")

    @cached_property
    def mid_point(self):
        """Return element mid point."""
        end_point = self.to_local(self.end_point)
        print(self.to_local(self.start_point))
        radius = np.linalg.norm(end_point, axis=1)
        theta = np.arctan2(end_point[:, 1], end_point[:, 0])
        points = (
            radius[:, np.newaxis]
            * np.c_[np.cos(theta / 2), np.sin(theta / 2), np.zeros_like(theta)]
        )
        # TODO fix arc axis sign
        print(theta / 2, np.arctan2(points[:, 1], points[:, 0]))
        line_index = self.frame.segment == "line"
        points[line_index] = np.zeros((sum(line_index), 3))
        return self.to_global(points)

    def _rotate_to_local(self, points: np.ndarray):
        """Return point array (n, 3) aligned to local coordinate system."""
        return np.einsum("ij,ijk->ik", points, self.coordinate_axes)

    def to_local(self, points: np.ndarray):
        """Return point array (n, 3) mapped to local coordinate system."""
        return self._rotate_to_local(points - self.origin)

    def _rotate_to_global(self, points: np.ndarray):
        """Return point array (n, 3) aligned to global coordinate system."""
        return np.einsum("ij,ikj->ik", points, self.coordinate_axes)

    def to_global(self, points: np.ndarray):
        """Return point array (n, 3) mapped to global coordinate system."""
        return self._rotate_to_global(points) + self.origin

    def _plot_segment(self, i, start_point, end_point, local=False):
        """Plot segment from end points given in local coordinates."""
        segment = "arc"
        match segment:
            case "arc":
                end_theta = np.arctan2(end_point[1], end_point[0])
                if end_theta <= 0:
                    end_theta += 2 * np.pi
                point_number = int(self.quadrant_segments * end_theta * 2 / np.pi)
                theta = np.linspace(0, end_theta, point_number)
                points = (
                    start_point[0]
                    * np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)]
                )
            case "line":
                points = np.c_[start_point, end_point].T
        if not local:
            points = np.einsum("ij,kj->ik", points, self.coordinate_axes[i])
            points += self.origin[i]
        self.axes.plot(*points.T)

    def plot(self, local=False):
        """Plot 3d segments."""
        self.set_axes("3d")
        start_point = self.to_local(self.start_point)
        end_point = self.to_local(self.end_point)
        for i in range(len(self.frame.segment)):
            self._plot_segment(i, start_point[i], end_point[i], local=local)
        self.set_box_aspect()
