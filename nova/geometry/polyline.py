"""Impement rdp-like decimator for mixed linear / arc polylines."""

import abc
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from overrides import override
import pandas
import scipy
from vedo import Mesh

from nova.geometry.frenet import Frenet
from nova.geometry.polygeom import Polygon
from nova.geometry.rdp import rdp
from nova.geometry.volume import Cell, Sweep, TriShell
from nova.graphics.plot import Plot


@dataclass
class Element(abc.ABC):
    """Element base class."""

    points: np.ndarray = field(default_factory=lambda: np.ndarray([]), repr=False)
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    center: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    start_point: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    end_point: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))

    name: ClassVar[str] = "base"
    keys: ClassVar[dict[str, list[str]]] = {
        "center": ["x", "y", "z"],
        "axis": ["ax", "ay", "az"],
        "normal": ["nx", "ny", "nz"],
        "start_point": ["x1", "y1", "z1"],
        "end_point": ["x2", "y2", "z2"],
    }

    def __post_init__(self):
        """Set default values."""
        self.start_point = self.points[0]
        self.end_point = self.points[-1]

    def __getitem__(self, attr: str):
        """Return item from geometry dict."""
        return self.geometry[attr]

    def _to_dict(self, keys: list[str], attr: str) -> dict:
        """Return dict combining values in attr with keys."""
        return dict(zip(keys, getattr(self, attr)))

    @cached_property
    def geometry(self) -> dict:
        """Return geometry dict."""
        return {"segment": self.name, "length": self.length} | {
            key: value
            for attr, keys in self.keys.items()
            for key, value in self._to_dict(keys, attr).items()
        }

    @property
    @abc.abstractmethod
    def length(self):
        """Return element length."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def path(self):
        """Return discrete element path."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nodes(self):
        """Return minimal-set element defining nodes."""
        raise NotImplementedError


@dataclass
class Line(Plot, Element):
    """
    Manage 3D line element.

    Attributes
    ----------
    points: np.ndarray(2, 3)
        Line endpoints.

    normal: np.ndarray(3)
        A vector normal to the line such that axis @ normal = 0.
    """

    name: ClassVar[str] = "line"

    def __post_init__(self):
        """Assert two point line."""
        super().__post_init__()
        assert len(self.points) == 2
        self.center = np.mean([self.start_point, self.end_point], axis=0)
        self.axis = self.end_point - self.start_point
        self.axis /= np.linalg.norm(self.axis)
        if np.isclose(np.linalg.norm(self.normal), 0):
            self.normal = np.cross(self.axis, [0, 0, 1])
        if not np.isclose(np.linalg.norm(self.normal), 1):
            self.normal /= np.linalg.norm(self.normal)
        if not np.isclose(np.dot(self.axis, self.normal), 0):
            self.normal = np.cross(self.axis, np.cross(self.normal, self.axis))
            self.normal /= np.linalg.norm(self.normal)

    @property
    @override
    def nodes(self):
        """Return line endpoints."""
        return self.points

    @property
    @override
    def length(self):
        """Return line length."""
        return np.linalg.norm(self.points[1] - self.points[0])

    def plot3d(self, quadrant_segments=None):
        """Plot point and best-fit data."""
        self.get_axes("3d")
        self.axes.plot(*self.points.T, "k-o", ms=3)

    @property
    @override
    def path(self):
        """Return discrete element path."""
        return self.nodes


@dataclass
class Arc(Plot, Element):
    """Fit arc to 3d point cloud."""

    arc_axes: np.ndarray = field(init=False, repr=False)
    radius: float = field(init=False, repr=False)
    center: np.ndarray = field(init=False, repr=False)
    theta: float = field(init=False, repr=False)
    error: float = field(init=False, repr=False)
    eps: float = 1e-8
    quadrant_segments: int = 21
    arc_resolution: float = 1.5  # points per arc length

    name: ClassVar[str] = "arc"

    def __post_init__(self):
        """Generate curve."""
        super().__post_init__()
        self.build()

    def build(self, points=None):
        """Align and fit 3d arc to point cloud."""
        if points is not None:
            self.points = points
        if self.points is None:
            return None
        self.fit()
        return self

    def align(self, normal):
        """Align point cloud to 2d plane."""
        points_delta = self.points - np.mean(self.points, axis=0)[np.newaxis, :]
        svd_binormal = scipy.linalg.svd(points_delta)[2][2]
        binormal = Frenet(self.points, svd_binormal).binormal.mean(axis=0)
        self.arc_axes = np.zeros((3, 3), float)
        self.arc_axes[1] = normal
        self.arc_axes[0] = np.cross(normal, binormal)
        if np.allclose(self.arc_axes[0], 0):
            binormal = np.cross(
                np.mean(self.points[1:-1] - self.points[0], axis=0), self.arc_axes[1]
            )
            self.arc_axes[0] = np.cross(self.arc_axes[1], binormal)
        self.arc_axes[2] = np.cross(self.arc_axes[0], self.arc_axes[1])
        self.arc_axes[1] = np.cross(self.arc_axes[2], self.arc_axes[0])
        self.arc_axes /= np.linalg.norm(self.arc_axes, axis=1)[:, np.newaxis]
        self.normal = self.arc_axes[1]
        self.axis = self.arc_axes[2]

    def fit_2d(self):
        """Update center and radius of best fit circle to points on plane."""
        points = self.points_2d
        chord = np.linalg.norm(points[-1] - points[0])
        origin = np.mean(np.c_[points[0], points[-1]], axis=1)
        points -= origin[np.newaxis, :]
        coef = np.linalg.lstsq(
            2 * points[:, 1:2],
            chord**2 / 4 - np.sum(points**2, axis=1),
            rcond=None,
        )[0]
        center_2d = origin
        center_2d[1] -= coef[0]
        self.radius = np.sqrt(chord**2 / 4 + coef[0] ** 2)
        points[:, 1] -= coef[0]
        assert np.allclose(self.radius, np.linalg.norm(points[0]))
        assert np.allclose(self.radius, np.linalg.norm(points[-1]))
        self.center = center_2d @ np.c_[-self.arc_axes[1], self.arc_axes[0]].T
        self.center += (
            np.mean(np.c_[self.points[0], self.points[-1]].T, axis=0)
            @ self.arc_axes[2]
            * self.arc_axes[2]
        )
        self.normal = self.center - self.points[0]  # align normal to local start radius
        self.normal /= np.linalg.norm(self.normal)
        self.arc_axes = np.c_[
            np.cross(self.normal, self.axis), self.normal, self.axis
        ].T
        center_points = self.points_2d - self.center_2d
        self.theta = np.arctan2(center_points[:, 1], center_points[:, 0])
        self.theta[self.theta < 0] += 2 * np.pi
        self.theta = np.unwrap(self.theta)
        self.error = np.linalg.norm(self.points - self.points_fit, axis=1).std()

    def _to_local(self, points):
        """Return points projected onto 2d plane (-normal, tangent)."""
        return np.c_[
            np.einsum("j,ij->i", -self.arc_axes[1], points),
            np.einsum("j,ij->i", self.arc_axes[0], points),
        ]

    @property
    def points_2d(self):
        """Return point locations projected onto 2d plane."""
        return self._to_local(self.points)

    @property
    def center_2d(self):
        """Return point locations projected onto 2d plane."""
        return self._to_local(self.center[np.newaxis, :])

    @property
    def points_fit(self):
        """Return best-fit points in 3d space."""
        return self.center[np.newaxis, :] + self.radius * (
            np.cos(self.theta)[:, np.newaxis] * -self.arc_axes[np.newaxis, 1]
            + np.sin(self.theta)[:, np.newaxis] * self.arc_axes[np.newaxis, 0]
        )

    def fit(self):
        """Align local coordinate system and fit arc to plane point cloud."""
        self.align(self.points[-1] - self.points[0])
        self.fit_2d()

    @cached_property
    def central_angle(self):
        """Return the angle subtended by arc from the arc's center."""
        return self.theta[-1] - self.theta[0]

    @cached_property
    def length(self):
        """Return absolute arc length."""
        return self.radius * abs(self.central_angle)

    @property
    def test(self):
        """Return status of normalized fit residual."""
        return self.error / abs(self.length) < self.eps

    def sample(self, point_number=50):
        """Return sampled polyline."""
        theta = np.linspace(self.theta[0], self.theta[-1], point_number)[:, np.newaxis]
        return self.center[np.newaxis, :] + self.radius * (
            np.cos(theta) * -self.arc_axes[1][np.newaxis, :]
            + np.sin(theta) * self.arc_axes[0][np.newaxis, :]
        )

    @cached_property
    def intermediate_point(self):
        """Return arc mid-point."""
        return self.sample(3)[1]

    @property
    @override
    def nodes(self):
        """Return best-fit node triple respecting start and end locations."""
        return np.array([self.start_point, self.intermediate_point, self.end_point])

    @cached_property
    def chord(self):
        """Return arc chord as Line instance."""
        return Line(np.c_[self.points[0], self.points[-1]].T, self.axis)

    def plot_circle(self):
        """Plot best fit circle."""
        self.get_axes("2d")
        theta = np.linspace(0, 2 * np.pi)
        theta = self.theta
        points_2d = self.radius * np.c_[np.cos(theta), np.sin(theta)]
        points = points_2d @ self.arc_axes[:2]
        points += self.center
        self.axes.plot(points[:, 0], points[:, 2], ":", color="gray")

    def plot(self):
        """Plot point and best-fit data."""
        self.get_axes("2d")
        points = self.sample(21)
        self.axes.plot(points[:, 0], points[:, 2])
        self.axes.plot(self.points[:, 0], self.points[:, 2], "o")
        self.axes.plot(self.points_fit[:, 0], self.points_fit[:, 2], "D")

    def plot3d(self, quadrant_segments=None):
        """Plot point and best-fit data."""
        self.get_axes("3d")
        if quadrant_segments is None:
            quadrant_segments = self.quadrant_segments
        points = self.sample(quadrant_segments)
        self.axes.plot(*points.T)
        self.axes.plot(*self.nodes.T, "o", ms=3)

    def plot_fit(self):
        """Plot best-fit arc and point cloud."""
        self.plot()
        self.plot_circle()

    @property
    @override
    def path(self):
        """Return arc path at sample resolution."""
        resolution = np.max(
            [
                int(self.arc_resolution * self.radius * self.central_angle),
                self.quadrant_segments,
                int(self.quadrant_segments * self.central_angle / (np.pi / 2)),
            ]
        )
        return self.sample(resolution)


@dataclass
class PolyArc(Plot):
    """Construct polyline from multiple arc segments."""

    points: np.ndarray
    resolution: int = 50
    path: np.ndarray = field(init=False)

    def __post_init__(self):
        """Build multi-arc polyline."""
        self.build()

    def build(self):
        """Build multi arc path."""
        segment_number = (len(self.points) - 1) // 2
        self.path = np.zeros((segment_number * self.resolution, 3))
        if segment_number > 1:
            self.path = self.path[:-1]
        for i in range(segment_number):
            points = self.points[slice(2 * i, 2 * i + 3)]
            start_index = i * self.resolution
            if i > 0:
                start_index -= 1
            self.path[slice(start_index, start_index + self.resolution)] = Arc(
                points
            ).sample(self.resolution)

    def plot(self):
        """Plot polyline."""
        self.set_axes("3d")
        self.axes.plot(*self.points.T, "o")
        self.axes.plot(*self.path.T, "-")


@dataclass
class PolyLine(Plot):
    """Decimate polyline using a hybrid arc/line-segment rdp algorithum."""

    points: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    cross_section: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    arc_eps: float = 1e-3
    line_eps: float = 5e-2
    rdp_eps: float = 1e-3
    minimum_arc_nodes: int = 4
    quadrant_segments: int = 16
    arc_resolution: float = 1.5
    filament: bool = True
    segments: list[Line | Arc] = field(init=False, repr=False, default_factory=list)

    path_attrs: ClassVar[list[str]] = [
        "x",
        "y",
        "z",
        "ax",
        "ay",
        "az",
        "nx",
        "ny",
        "nz",
        "x1",
        "y1",
        "z1",
        "x2",
        "y2",
        "z2",
        "segment",
        "length",
    ]
    volume_attrs: ClassVar[list[str]] = [
        "vtk",
        "poly",
        "area",
        "volume",
    ]

    def __post_init__(self):
        """Decimate polyline."""
        if self.points.size > 0:
            self.decimate()

    def __getitem__(self, attr: str):
        """Return path geometry attribute."""
        return self.path_geometry[attr]

    def __len__(self):
        """Return segment number."""
        return len(self.segments)

    def __iadd__(self, other):
        """Return polyline instance augmented by other."""
        self.segments.extend(other.segments)
        self.points = np.append(self.points, other.points)
        return self

    def fit_arc(self, points):
        """Return point index prior to first arc mis-match."""
        point_number = len(points)
        for i in range(self.minimum_arc_nodes, point_number + 1):
            if Arc(points[:i], eps=self.arc_eps).test:
                continue
            if i > self.minimum_arc_nodes:
                return i - 1
            return 2
        return point_number

    def append(self, points, normal=None):
        """Append points to segment list."""
        if len(points) >= self.minimum_arc_nodes and self.minimum_arc_nodes != 0:
            self.segments.append(
                Arc(
                    points,
                    eps=self.arc_eps,
                    quadrant_segments=self.quadrant_segments,
                    arc_resolution=self.arc_resolution,
                )
            )
            return
        for i in range(len(points) - 1):
            self.segments.append(Line(points[i : i + 2], normal[i]))

    def decimate(self):
        """Decimate polyline via multi-segment by an arc fit."""
        point_number = len(self.points)
        start = 0
        self.segments = []
        line_normal = Frenet(self.points).normal
        while (
            start <= point_number - self.minimum_arc_nodes
            and self.minimum_arc_nodes != 0
        ):
            number = self.fit_arc(self.points[start:])
            self.append(
                self.points[start : start + number],
                line_normal[start : start + number],
            )
            start += number - 1
        if point_number - start > 1:
            self.append(self.points[start:], line_normal[start:])
        for i, segment in enumerate(self.segments):
            if isinstance(segment, Line):
                continue
            if abs(segment.central_angle) < self.line_eps:
                self.segments[i] = segment.chord
        self.rdp_merge()

    def _rdp_line_segments(self, nodes, line_normal):
        """Return rdp reduced Line segments."""
        nodes = np.r_[nodes[::2], nodes[-1:]]
        line_normal = np.append(line_normal, line_normal[-1:], axis=0)
        rdp_mask = rdp(nodes, self.rdp_eps, algo="iter", return_mask=True)
        nodes = nodes[rdp_mask]
        line_normal = line_normal[rdp_mask]
        return [Line(nodes[i : i + 2], line_normal[i]) for i in range(len(nodes) - 1)]

    def rdp_merge(self):
        """Merge multiple line segments using the rdp algorithum."""
        segments, nodes, line_normal = [], [], []
        for segment in self.segments:
            match segment:
                case Line():
                    nodes.extend(segment.nodes)
                    line_normal.append(segment.normal)
                case Arc():
                    if nodes:
                        segments.extend(self._rdp_line_segments(nodes, line_normal))
                        nodes, line_normal = [], []
                    segments.append(segment)
        segments.extend(self._rdp_line_segments(nodes, line_normal))
        self.segments = segments

    def _stackattr(self, attr: str):
        """Return stacked segment attribute."""
        if len(self.segments) == 1:
            return getattr(self.segments[0], attr)
        return np.r_[
            np.vstack([getattr(seg, attr)[:-1] for seg in self.segments]),
            self.segments[-1].points[-1:],
        ]

    @property
    def nodes(self):
        """Return segment nodes."""
        return self._stackattr("nodes")

    @property
    def path(self):
        """Return quadseg resolved polyline path."""
        return self._stackattr("path")

    def _to_list(self, attr: str):
        """Return segment attribute list."""
        if attr == "segment" and not self.filament:
            thicken = {"arc": "bow", "line": "beam"}
            return [thicken[segment[attr]] for segment in self.segments]
        return [segment[attr] for segment in self.segments]

    @cached_property
    def vtk(self) -> list[Cell]:
        """Retun list of vtk mesh segments swept along segment paths."""
        return [
            Sweep(self.cross_section, segment.path, segment.normal)
            for segment in self.segments
        ]

    @cached_property
    def poly(self) -> list[Polygon]:
        """Return list of polygon objects for 3D coil projected to 2d poloidal plane."""
        return [
            TriShell(vtk, ahull=segment.name == "arc", alpha=None).poly
            for vtk, segment in zip(self.vtk, self.segments)
        ]

    @cached_property
    def length(self) -> list[float]:
        """Return list of segment lengths."""
        return self._to_list("length")

    @cached_property
    def area(self) -> list[float]:
        """Return list of polygon areas projected to 2d poloiodal plane."""
        return [poly.area for poly in self.poly]

    @property
    def volume(self) -> list[float]:
        """Return subframe volume list."""
        return [_vtk.clone().triangulate().volume() for _vtk in self.vtk]

    @cached_property
    def bounds(self) -> np.ndarray:
        """Return 3d bounding box coordinates for vtk volume objects."""
        return np.c_[[_vtk.clone().triangulate().bounds() for _vtk in self.vtk]]

    @cached_property
    def delta(self) -> np.ndarray:
        """Return 3d bounding box deltas for vtk volume objects."""
        return self.bounds[:, 1::2] - self.bounds[:, ::2]

    @property
    def delta_x(self):
        """Return bounding box x-coordinate delta."""
        return self.delta[:, 0]

    @property
    def delta_y(self):
        """Return bounding box y-coordinate delta."""
        return self.delta[:, 1]

    @property
    def delta_z(self):
        """Return bounding box z-coordinate delta."""
        return self.delta[:, 2]

    @cached_property
    def path_geometry(self) -> dict:
        """Return path geometry attribute dict."""
        return {attr: self._to_list(attr) for attr in self.path_attrs}

    @cached_property
    def volume_geometry(self) -> dict:
        """Return volume geometry attribute dict."""
        if len(self.cross_section) == 0:
            return {}
        return {attr: getattr(self, attr) for attr in self.volume_attrs}

    def to_frame(self):
        """Return segment geometry as a pandas DataFrame."""
        return pandas.DataFrame(self.path_geometry | self.volume_geometry)

    def plot(self, quadrant_segments=101, axes=None):
        """Plot decimated polyline."""
        self.set_axes("3d", axes)
        self.axes.plot(*self.points.T)
        for segment in self.segments:
            segment.plot3d(quadrant_segments)
        self.axes.set_aspect("equal")

    @property
    def frenet(self):
        """Return frenet instance."""
        return Frenet(self.points)

    def vtkplot(self):
        """Plot vtk centerline."""
        Mesh(*[segment.mesh for segment in self.segments]).show()


if __name__ == "__main__":
    from nova.assembly.fiducialdata import FiducialData

    fiducial = FiducialData(fiducial="RE")

    points = fiducial.data.centerline_target.data
    points += 500 * fiducial.data.centerline_delta[3].data
    polyline = PolyLine(points)
    polyline.plot()

    # poly = PolyLine()
    # for segment in polyline.segments:
