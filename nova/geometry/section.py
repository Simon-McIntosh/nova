"""Manage sectional transforms."""

from dataclasses import dataclass, field

import numpy as np
import shapely.geometry
import shapely.ops
import vedo

from nova.geometry.frenet import Frenet
from nova.geometry.rotate import to_vector, to_axes, by_angle
from nova.geometry.vtkgen import VtkFrame


def distinct_corners(points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """Return a closed ring's corners with every zero-length edge dropped.

    A generated section comes back as a shapely ring, so its first corner is
    repeated at the end -- and a generator that sweeps an angle closes on a corner
    that is its own first to within round-off rather than exactly, which leaves the
    ring one corner longer still.  Both are edges of zero length, and the tolerance
    is taken against the SECTION's own extent so a thin plate keeps corners a
    coordinate-absolute one would merge.
    """
    points = np.asarray(points, dtype=np.float64)
    scale = max(float(np.max(np.ptp(points, axis=0))), np.finfo(float).tiny)
    gap = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    return points[gap > tolerance * scale]


def collapse_collinear(points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """Return a planar ring's minimal corner set as an ``(n, 2)`` array.

    Beyond the zero-length edges :func:`distinct_corners` drops, a corner sitting
    part way along a straight edge carries no geometry either.  A projection puts
    them there: the poloidal footprint of a swept hexagon comes back with a corner
    mid-edge, agreeing with the straight line to round-off, and a closed-form
    section reduction pays per corner -- so the run is collapsed before the section
    is handed over rather than paid for.

    Redundancy is measured as twice the area of the triangle a corner spans with
    its two neighbours, against the square of the section's own extent, so it is
    scale-free in the same way the zero-length test is.  A run of any length
    collapses in one pass because every interior point of it fails the test at
    once, and the loop repeats only for the runs a removal creates.
    """
    points = distinct_corners(points, tolerance)
    scale = max(float(np.max(np.ptp(points, axis=0))), np.finfo(float).tiny)
    while len(points) > 3:
        edge = np.roll(points, -1, axis=0) - points
        previous = np.roll(edge, 1, axis=0)
        turn = np.abs(previous[:, 0] * edge[:, 1] - previous[:, 1] * edge[:, 0])
        corner = turn > tolerance * scale**2
        if corner.all() or corner.sum() < 3:
            break
        points = points[corner]
    return points


def poloidal_footprint(loops: np.ndarray, tolerance: float = 1e-10):
    """Return the poloidal footprint of a swept section as a shapely polygon.

    ``loops`` is the ``(loop, corner, 3)`` stack a sweep places the cross-section
    at along its path, in the double precision the cross-section was authored in.
    Each loop projects to ``(r, z)`` by ``r = hypot(x, y)``, and the footprint is
    the union of those projections -- the section as it is actually carried, at
    every station along the path, rather than an alpha shape fitted to a mesh's
    vertices.

    The loops themselves are unioned rather than the solid bands between them,
    which is what keeps a CONCAVE section concave: a band's projection would have
    to be convexified to be cheap, and a hull across the two loops fills in the
    notch of an L or a wedge.  What that gives up is the sliver a band sweeps out
    between two stations, which is bounded by how far the loops move -- nothing at
    all for a sweep at constant radius about the vertical, the case every arc
    element in an axisymmetric frame reads, where every loop projects onto very
    nearly the same ring and the union collapses back onto it.

    The result is collapsed before it is returned: a union of near-identical rings
    leaves corners part way along the section's own edges, one per station that
    moved, so a swept hexagon's union has sixteen corners over a sixteen-station arc
    and leaves with six -- its area agreeing with the section's to 2e-14.
    """
    loops = np.asarray(loops, dtype=np.float64)
    poloidal = np.stack(
        [np.linalg.norm(loops[..., :2], axis=-1), loops[..., 2]], axis=-1
    )
    scale = max(float(np.max(np.ptp(poloidal.reshape(-1, 2), axis=0))), 1.0)
    if np.max(np.abs(poloidal - poloidal[0])) <= tolerance * scale:
        return shapely.geometry.Polygon(collapse_collinear(poloidal[0]))
    station = [shapely.geometry.Polygon(ring).buffer(0) for ring in poloidal]
    footprint = shapely.ops.unary_union(station)
    if not footprint.is_valid:
        footprint = footprint.buffer(0)
    if not isinstance(footprint, shapely.geometry.Polygon):
        return footprint
    return shapely.geometry.Polygon(
        collapse_collinear(np.asarray(footprint.exterior.coords, dtype=np.float64)),
        [
            collapse_collinear(np.asarray(ring.coords, dtype=np.float64))
            for ring in footprint.interiors
        ],
    )


@dataclass
class Section:
    """Transform 2D sectional data."""

    points: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, float))
    triad: np.ndarray = field(default_factory=lambda: np.identity(3, float))
    mesh_array: list[VtkFrame] = field(init=False, default_factory=list)
    point_array: list[np.ndarray] = field(init=False, default_factory=list)

    def __len__(self):
        """Return length of mesh."""
        return len(self.mesh_array)

    def _append(self):
        """Generate mesh and append mesh to list."""
        self.point_array.append(self.points.tolist())
        self.mesh_array.append(
            VtkFrame([self.points, [[*range(len(self.points))]]]).c(len(self))
        )

    def _rotate_points(self, rotation):
        self.points -= self.origin
        self.points = rotation.apply(self.points)
        self.points += self.origin

    def by_angle(self, axis: np.ndarray, angle: float):
        """Rotate points by angle about axis."""
        rotation = by_angle(axis, angle)
        self._rotate_points(rotation)
        self.triad = rotation.apply(self.triad)

    def to_vector(self, vector: np.ndarray, coord: int):
        """Rotate points to vector."""
        rotation = to_vector(self.triad[coord], vector)
        self._rotate_points(rotation)
        self.triad = rotation.apply(self.triad)

    def to_axes(self, axes: np.ndarray):
        """Rotate points to align triad with axes."""
        rotation = to_axes(axes, self.triad)
        self._rotate_points(rotation)
        self.triad = rotation.apply(self.triad.T).T

    def to_point(self, point):
        """Translate points to point and store mesh."""
        delta = np.array(point - self.origin, float)
        self.origin = self.origin + delta
        self.points = self.points + delta

    def sweep(self, path: np.ndarray, binormal: np.ndarray, align: str):
        """Sweep section along path."""
        frenet = Frenet(path, binormal)

        triad = np.identity(3)
        normal = np.zeros((len(path), 3))
        for i in range(len(path)):
            rotation = to_vector(triad[0], frenet.tangent[i])
            triad = rotation.apply(triad)
            normal[i] = -triad[0]

        normal = frenet.project(normal, triad[1][np.newaxis])
        twist = np.arccos(
            np.clip(
                np.dot(normal[0], normal[-1]),
                -1.0,
                1.0,
            )
        )

        # untwist = scipy.interpolate.interp1d(
        #    [0, 1], [0, -2 * twist], fill_value="extrapolate"
        # )
        # angle = 0

        delta = -13 * twist / len(path)
        for i in range(len(path)):
            self.to_point(frenet.points[i])
            match align:
                case "axes":
                    axes = np.c_[
                        -frenet.normal[i], frenet.tangent[i], frenet.binormal[i]
                    ]
                    self.to_axes(axes)
                case "vector":
                    # A cross-section is authored in its own x-z plane -- width
                    # along x, height along z, the y axis its normal -- so the
                    # axis pinned to world vertical is the section's THIRD, and
                    # its normal follows the tangent.  Pinning the first instead
                    # lands the width along z and the height along the path's
                    # own normal, which swaps the two dimensions of every swept
                    # conductor and is invisible only for a square one.
                    sign = np.sign(np.array([0, 0, 1]) @ self.triad[2])
                    if np.isclose(sign, 0):
                        sign = 1
                    self.to_vector(np.array([0, 0, sign * 1]), 2)
                    self.to_vector(frenet.tangent[i], 1)
                case "twist":
                    # delta = untwist(frenet.parametric_length[i]) - angle
                    # angle = untwist(frenet.parametric_length[i])
                    self.by_angle(frenet.tangent[i], -delta)
                    self.to_vector(frenet.tangent[i], 1)

                case _:
                    raise ValueError(f"align {align} not in [vector or axes]")
            self._append()
        return self

    def plot(self):
        """Plot mesh instances."""
        vedo.show(*self.mesh_array, new=True, axes=True)
