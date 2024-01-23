"""Coordinate system transform methods for BiotFrame."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

import nova.frame.metamethod as metamethod
from nova.frame.framelink import FrameLink
from nova.graphics.plot import Plot3D


@dataclass
class Segment:
    """Define segment start axes base class."""

    normal: np.ndarray
    axis: np.ndarray
    segment_type: str | None = None

    @property
    def axes(self):
        """Return line axes."""
        raise NotImplementedError()

    @property
    def start_axes(self):
        """Return single start axes."""
        match self.segment_type:
            case "line":
                start_axes = Line(self.axis, self.normal).axes
            case "arc":
                start_axes = Arc(self.axis, self.normal).axes
            case None:
                raise ValueError("segment type not set")
            case segment_type:
                raise NotImplementedError(
                    f"segment type {segment_type} " "not implemented"
                )
        return start_axes


@dataclass
class Line(Segment):
    """Define line segment start axes."""

    @property
    def axes(self):
        """Return line axes."""
        return np.stack(
            [self.axis, self.normal, np.cross(self.axis, self.normal)], axis=1
        )


@dataclass
class Arc(Segment):
    """Define arc segment start axes."""

    @property
    def axes(self):
        """Return arc axes."""
        return np.stack(
            [np.cross(self.normal, self.axis), self.normal, self.axis], axis=1
        )


@dataclass
class Space(metamethod.Space, Plot3D):
    """Coordinate system transform methods."""

    name = "space"

    frame: FrameLink = field(repr=False)
    coordinate_axes: np.ndarray = field(init=False, repr=False)
    coordinate_origin: np.ndarray = field(init=False, repr=False)
    quadrant_segments: int = 11

    def initialize(self):
        """Build local coordinate axes (-normal, tangent, binormal)."""
        self.coordinate_axes = np.zeros((len(self.frame), 3, 3))
        self.coordinate_axes[..., 0] = -self.normal
        self.coordinate_axes[..., 1] = np.cross(self.axis, -self.normal)
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
    def _start_point(self):
        """Return start point in local coordinate system."""
        return self.to_local(self.start_point)

    @cached_property
    def end_point(self):
        """Return element end point."""
        return self._column_stack("x2", "y2", "z2")

    @cached_property
    def _end_point(self):
        """Return end point in local coordinate system."""
        return self.to_local(self.end_point)

    @cached_property
    def _arc_index(self):
        """Return arc index."""
        return self.frame.segment == "arc"

    @cached_property
    def _arc_number(self):
        """Return number of arc segments."""
        return sum(self._arc_index)

    @cached_property
    def _line_index(self):
        """Return line index."""
        return self.frame.segment == "line"

    @cached_property
    def _line_number(self):
        """Return number of line segments."""
        return sum(self._line_index)

    @cached_property
    def intermediate_point(self):
        """Return element intermediate point."""
        end_point = self.to_local(self.end_point)
        radius = np.linalg.norm(end_point, axis=1)
        theta = np.arctan2(end_point[:, 1], end_point[:, 0])
        theta[theta < 0] += 2 * np.pi
        points = (
            radius[:, np.newaxis]
            * np.c_[np.cos(theta / 2), np.sin(theta / 2), np.zeros_like(theta)]
        )
        if self._line_number > 0:  # line intermediate point defines Frenet normal
            start_point = self.to_local(self.start_point)
            points[self._line_index] = start_point[self._line_index]
            points[self._line_index] += np.array([-1, 0, 0])[np.newaxis, :]
        return self.to_global(points)

    @cached_property
    def _intermediate_point(self):
        """Return intermediate point in local coordinate system."""
        return self.to_local(self.intermediate_point)

    @property
    def _line_tangent(self):
        """Return line segment start tangents."""
        return self.axis[self._line_index]

    @property
    def _arc_tangent(self):
        """Return arc segment start tangents."""
        return np.cross(self.normal[self._arc_index], self.axis[self._arc_index])

    def _check_segment_types(self):
        """Assert all segment types in [line, arc]."""
        try:
            assert len(self.frame) == self._arc_number + self._line_number
        except AssertionError as error:
            error_index = ~self._arc_index & ~self._line_index
            error_types = self.frame.segment[error_index].unique()
            raise NotImplementedError(
                f"segment types {error_types} not implemented"
            ) from error

    @property
    def start_axes(self):
        """Return local coordinate axes aligned to element start."""
        self._check_segment_types()
        axes = np.zeros((len(self.frame), 3, 3))
        axes[self._line_index] = Line(
            self.normal[self._line_index], self.axis[self._line_index]
        ).axes
        axes[self._arc_index] = Arc(
            self.normal[self._arc_index], self.axis[self._arc_index]
        ).axes
        return axes

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

    def _segment_path(self, index, local=False):
        """Return segment path sampled from local segment endpoints."""
        _start_point = self._start_point[index]
        _end_point = self._end_point[index]
        match self.frame.segment.iloc[index]:
            case "arc":
                end_theta = np.arctan2(_end_point[1], _end_point[0])
                if end_theta <= 0:
                    end_theta += 2 * np.pi
                point_number = np.max(
                    [
                        self.quadrant_segments,
                        int(self.quadrant_segments * end_theta * 2 / np.pi),
                    ]
                )
                theta = np.linspace(0, end_theta, point_number)
                points = (
                    _start_point[0]
                    * np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)]
                )
            case "line":
                points = np.c_[_start_point, _end_point].T
        if local:
            return points
        points = np.einsum("ij,kj->ik", points, self.coordinate_axes[index])
        points += self.origin[index]
        return points

    @cached_property
    def path(self):
        """Return quadrant segment resolved path."""
        if (segment_number := self.frame.shape[0]) == 1:
            return self._segment_path(0)
        return np.r_[
            np.vstack(
                [self._segment_path(index)[:-1] for index in range(segment_number)]
            ),
            self.end_point[-1:],
        ]

    def _plot_segment(self, index, local=False):
        """Plot segment from end points given in local coordinates."""
        self.axes.plot(*self._segment_path(index, local).T)

    def _plot_points(self, *points, local=False):
        """Plot segment point."""
        for _points in points:
            if not local:
                _points = self.to_global(_points)
            self.axes.plot(*_points.T, "o")

    def plot(self, local=False, axes=None):
        """Plot 3d segments."""
        self.set_axes("3d", axes)
        for index in range(len(self.frame.segment)):
            self._plot_segment(index, local=local)
        self._plot_points(
            self._start_point, self._intermediate_point, self._end_point, local=local
        )
        self.set_box_aspect()
