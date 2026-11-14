"""Surface-charge reference for linear, homogeneous iron in free space.

The charge is sigma = M.n. Its interior normal field has the jump -sigma/2,
so ((mu+1)/(2*(mu-1)) I - K) sigma = H_external.n, with K the principal-value
normal field of the single layer. Panels integrate a constant charge exactly;
only the charge distribution is discretised. No outer truncation boundary is
introduced. Rotation images are optional; reflection uses the odd charge of
an up-down symmetric circuit.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import gmres
import shapely

from scripts.prototypes.iron_limb_fill.bricks import (
    ARM_HALF_WIDTH,
    ARM_Z,
    COLUMN_RADIUS,
    HUB_APOTHEM,
    LIMB_X,
    MU0,
)


@dataclass
class Panels:
    """Oriented triangles and the geometric operands of their exact field."""

    vertices: np.ndarray

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=float)
        cross = np.cross(
            self.vertices[:, 1] - self.vertices[:, 0],
            self.vertices[:, 2] - self.vertices[:, 0],
        )
        twice_area = np.linalg.norm(cross, axis=1)
        if not np.isfinite(self.vertices).all() or np.any(twice_area < 1e-16):
            raise ValueError("surface triangles must be finite and nondegenerate")
        self.areas = twice_area / 2
        self.normals = cross / twice_area[:, None]
        self.centres = self.vertices.mean(axis=1)
        self.edges = np.roll(self.vertices, -1, axis=1) - self.vertices
        self.lengths = np.linalg.norm(self.edges, axis=2)
        self.outward = np.cross(self.edges, self.normals[:, None])
        self.outward /= self.lengths[:, :, None]


@njit(inline="always")
def _triangle_components(x, y, z, vertices, normal, area, lengths, outward):
    ax, ay, az = vertices[0, 0] - x, vertices[0, 1] - y, vertices[0, 2] - z
    bx, by, bz = vertices[1, 0] - x, vertices[1, 1] - y, vertices[1, 2] - z
    cx, cy, cz = vertices[2, 0] - x, vertices[2, 1] - y, vertices[2, 2] - z
    ra = np.sqrt(ax * ax + ay * ay + az * az)
    rb = np.sqrt(bx * bx + by * by + bz * bz)
    rc = np.sqrt(cx * cx + cy * cy + cz * cz)
    sums = (ra + rb, rb + rc, rc + ra)
    hx, hy, hz = 0.0, 0.0, 0.0
    for k in range(3):
        integral = np.log1p(2 * lengths[k] / max(sums[k] - lengths[k], 1e-300))
        hx += outward[k, 0] * integral
        hy += outward[k, 1] * integral
        hz += outward[k, 2] * integral
    distance = -(ax * normal[0] + ay * normal[1] + az * normal[2])
    if abs(distance) > 1e-12:
        denominator = (
            ra * rb * rc
            + (ax * bx + ay * by + az * bz) * rc
            + (bx * cx + by * cy + bz * cz) * ra
            + (cx * ax + cy * ay + cz * az) * rb
        )
        omega = 2 * np.arctan2(-2 * area * distance, denominator)
        hx -= normal[0] * omega
        hy -= normal[1] * omega
        hz -= normal[2] * omega
    scale = 1 / (4 * np.pi)
    return hx * scale, hy * scale, hz * scale


@njit
def triangle_field(point, vertices, normal, area, lengths, outward):
    """Integrate a constant triangular charge using edge logs and solid angle.

    On the panel plane the principal value is returned. The normal jump is
    handled separately by the boundary equation.
    """
    return np.array(
        _triangle_components(
            point[0], point[1], point[2], vertices, normal, area, lengths, outward
        )
    )


@njit(parallel=True)
def _normal_matrix(
    targets, normals, vertices, source_normals, areas, lengths, outward, signs
):
    transposed = np.zeros((len(vertices), targets.shape[1]))
    correction = np.zeros(len(vertices))
    for j in prange(len(vertices)):
        integral = 0.0
        for i in range(targets.shape[1]):
            even, odd = 0.0, 0.0
            for k in range(len(signs)):
                point, normal = targets[k, i], normals[k, i]
                hx, hy, hz = _triangle_components(
                    point[0],
                    point[1],
                    point[2],
                    vertices[j],
                    source_normals[j],
                    areas[j],
                    lengths[j],
                    outward[j],
                )
                value = hx * normal[0] + hy * normal[1] + hz * normal[2]
                odd += signs[k] * value
                even += value
            transposed[j, i] = odd
            integral += areas[i] * even
        correction[j] = 0.5 - integral / areas[j]
    return np.ascontiguousarray(transposed.T), correction


@njit(parallel=True)
def _field(targets, charge, vertices, normals, areas, lengths, outward, frames, signs):
    result = np.zeros((len(targets), 3))
    for i in prange(len(targets)):
        for k in range(len(signs)):
            frame, sign = frames[k], signs[k]
            point = targets[i] @ frame
            lx, ly, lz = 0.0, 0.0, 0.0
            for j in range(len(vertices)):
                hx, hy, hz = _triangle_components(
                    point[0],
                    point[1],
                    point[2],
                    vertices[j],
                    normals[j],
                    areas[j],
                    lengths[j],
                    outward[j],
                )
                lx += charge[j] * hx
                ly += charge[j] * hy
                lz += charge[j] * hz
            for a in range(3):
                result[i, a] += sign * (
                    frame[a, 0] * lx + frame[a, 1] * ly + frame[a, 2] * lz
                )
    return result


def image_frames(rotations=6, reflect=True):
    """Return orthogonal source frames and the scalar charge parity."""
    frames, signs = [], []
    for angle in np.arange(rotations) * 2 * np.pi / rotations:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        for mirror in (1.0, -1.0) if reflect else (1.0,):
            frames.append(rotation @ np.diag([1.0, 1.0, mirror]))
            signs.append(mirror)
    return np.asarray(frames), np.asarray(signs)


class SurfaceModel:
    """A fixed boundary interaction reused for several relative permeabilities."""

    def __init__(self, panels, external, rotations=6, reflect=True):
        self.panels = panels
        self.frames, self.signs = image_frames(rotations, reflect)
        self.external = external
        targets = np.stack([panels.centres @ frame for frame in self.frames])
        normals = np.stack([panels.normals @ frame for frame in self.frames])
        self.interaction, self.conservation_correction = _normal_matrix(
            targets,
            normals,
            panels.vertices,
            panels.normals,
            panels.areas,
            panels.lengths,
            panels.outward,
            self.signs,
        )
        # Gauss' law fixes each area-weighted column integral to half its source
        # area. Enforce this exact identity to remove centroid-test quadrature
        # bias. Compute the identity with even charge parity before applying
        # the physical odd reflection parity to the solve.
        self.interaction[np.diag_indices_from(self.interaction)] += (
            self.conservation_correction
        )
        self.rhs = np.sum(external(panels.centres) * panels.normals, axis=1) / MU0
        if not np.isfinite(self.interaction).all():
            raise ValueError("nonfinite surface interaction")

    def solve(self, permeability):
        if not np.isfinite(permeability) or permeability <= 1:
            raise ValueError("relative permeability must be finite and above one")
        diagonal = (permeability + 1) / (2 * (permeability - 1))
        matrix = -self.interaction.copy()
        matrix[np.diag_indices_from(matrix)] += diagonal
        charge, info = gmres(
            matrix, self.rhs, rtol=2e-11, atol=0.0, restart=150, maxiter=30
        )
        residual = np.linalg.norm(matrix @ charge - self.rhs) / np.linalg.norm(self.rhs)
        if info or residual > 1e-9:
            raise RuntimeError(
                f"surface solve did not converge: info={info}, {residual=}"
            )
        return charge, float(residual)

    def iron_field(self, points, charge):
        p = self.panels
        return MU0 * _field(
            np.asarray(points, dtype=float),
            charge,
            p.vertices,
            p.normals,
            p.areas,
            p.lengths,
            p.outward,
            self.frames,
            self.signs,
        )


def _split_triangles(triangles, size):
    pending, result = list(triangles), []
    while pending:
        triangle = pending.pop()
        lengths = np.linalg.norm(np.roll(triangle, -1, axis=0) - triangle, axis=1)
        k = int(np.argmax(lengths))
        if lengths[k] <= size * (1 + 1e-12):
            result.append(triangle)
            continue
        a, b, c = triangle[k], triangle[(k + 1) % 3], triangle[(k + 2) % 3]
        midpoint = (a + b) / 2
        pending.extend((np.array([a, midpoint, c]), np.array([midpoint, b, c])))
    return result


def _cap(polygon, z, sign, size):
    bounds = polygon.bounds
    spacing = size / np.sqrt(2)
    xx, yy = np.meshgrid(
        np.arange(bounds[0], bounds[2], spacing),
        np.arange(bounds[1], bounds[3], spacing),
    )
    points = np.column_stack((xx.ravel(), yy.ravel()))
    points = points[shapely.contains_xy(polygon, *points.T)]
    cloud = shapely.GeometryCollection([polygon, shapely.MultiPoint(points)])
    triangles = []
    for cell in shapely.get_parts(shapely.delaunay_triangles(cloud)):
        clipped = cell.intersection(polygon)
        if clipped.area < 1e-15:
            continue
        for triangle in shapely.get_parts(
            shapely.constrained_delaunay_triangles(clipped)
        ):
            xy = np.asarray(triangle.exterior.coords)[:3]
            u, v = xy[1] - xy[0], xy[2] - xy[0]
            if (u[0] * v[1] - u[1] * v[0]) * sign < 0:
                xy = xy[::-1]
            triangles.append(np.column_stack((xy, np.full(3, z))))
    return _split_triangles(triangles, size)


def _sides(polygon, lower, upper, size):
    triangles = []
    polygon = shapely.orient_polygons(polygon)
    for ring in [polygon.exterior, *polygon.interiors]:
        xy = np.asarray(ring.coords)
        for a, b in zip(xy[:-1], xy[1:]):
            if np.linalg.norm(b - a) < 1e-12:
                continue
            # The two radial cuts are image boundaries, never iron-air interfaces.
            on_cut = any(
                abs(a[1] - slope * a[0]) < 1e-10 and abs(b[1] - slope * b[0]) < 1e-10
                for slope in (-np.tan(np.pi / 6), np.tan(np.pi / 6))
            )
            if on_cut:
                continue
            count = max(1, int(np.ceil(np.linalg.norm(b - a) / size)))
            height = max(1, int(np.ceil((upper - lower) / size)))
            for i in range(count):
                left, right = a + (b - a) * i / count, a + (b - a) * (i + 1) / count
                for j in range(height):
                    low = lower + (upper - lower) * j / height
                    high = lower + (upper - lower) * (j + 1) / height
                    q = np.array(
                        [(*left, low), (*right, low), (*right, high), (*left, high)]
                    )
                    triangles.extend((q[[0, 1, 2]], q[[0, 2, 3]]))
    return triangles


def _disk_sector(radius, arc_segments, inner=0.0):
    angle = np.linspace(-np.pi / 6, np.pi / 6, arc_segments + 1)
    outer = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    inside = (
        np.column_stack((inner * np.cos(angle[::-1]), inner * np.sin(angle[::-1])))
        if inner
        else np.zeros((1, 2))
    )
    return shapely.Polygon(np.vstack((inside, outer)))


def circuit_panels(size, annulus=False, missing_limb=None, arc_segments=24, bore=0.0):
    """Mesh the fixed circuit boundary, not a centroid-selected volume.

    The circle is one fixed inscribed polygon for all panel resolutions.
    The control annulus uses the same polygonal circles. An optional bore lets
    both reference methods share the production axisymmetric kernel's bore.
    A missing return limb expands the surface to all six angular sectors.
    """
    if not np.isfinite(size) or size <= 0:
        raise ValueError("panel size must be finite and positive")
    if missing_limb is not None and missing_limb not in range(6):
        raise ValueError("missing limb must be an angular sector from zero to five")
    column = _disk_sector(COLUMN_RADIUS, arc_segments, bore)
    if annulus:
        top = _disk_sector(LIMB_X[1], arc_segments, bore)
        limb = _disk_sector(LIMB_X[1], arc_segments, LIMB_X[0])
    else:
        top = shapely.Polygon(
            [
                (0, 0),
                (HUB_APOTHEM, -ARM_HALF_WIDTH),
                (LIMB_X[1], -ARM_HALF_WIDTH),
                (LIMB_X[1], ARM_HALF_WIDTH),
                (HUB_APOTHEM, ARM_HALF_WIDTH),
            ]
        )
        if bore:
            top = top.difference(_disk_sector(bore, arc_segments))
        limb = shapely.box(LIMB_X[0], -ARM_HALF_WIDTH, LIMB_X[1], ARM_HALF_WIDTH)

    def sector(with_limb):
        lower = shapely.union_all([column, limb]) if with_limb else column
        pieces = [
            *_cap(top, ARM_Z[1], 1, size),
            *_cap(top.difference(lower), ARM_Z[0], -1, size),
            *_sides(top, *ARM_Z, size),
            *_sides(column, 0.0, ARM_Z[0], size),
        ]
        if with_limb:
            pieces.extend(_sides(limb, 0.0, ARM_Z[0], size))
        return np.asarray(pieces)

    if missing_limb is None:
        return Panels(sector(True))
    pieces = []
    for k, frame in enumerate(image_frames(6, False)[0]):
        pieces.append(sector(k != missing_limb) @ frame.T)
    return Panels(np.concatenate(pieces))
