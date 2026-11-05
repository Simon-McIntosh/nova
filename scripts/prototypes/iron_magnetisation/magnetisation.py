"""Axisymmetric magnetised iron as toroidal sheet currents on cell edges.

Each iron cell carries a uniform poloidal magnetisation ``(M_R, M_Z)``. In
axisymmetry the bound volume current ``curl M`` of a uniform cell is zero, so
the cell's field is that of toroidal sheet currents on its edges,

    K_phi = M_Z n_R - M_R n_Z        (outward normal n = (n_R, n_Z)),

which is a line integral of ring filaments along each edge. This module does
that integral by Gauss-Legendre quadrature over nova's exact ring Green's
function; a closed-form edge kernel (Urankar's sheet reduction) is the
production replacement. An edge on the axis carries a ring of zero radius and
contributes nothing, as it should.

The material law is solved by fixed-permeability iteration: for frozen
``mu_r`` per cell the magnetisation ``M = c B`` with ``c = (mu_r - 1)/(mu0 mu_r)``
makes ``(I - c G) M = c B_ext`` linear, and ``mu_r`` is then updated from the
resulting ``|B|`` with under-relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from nova.biot.greens import greens_bz_br

MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class Cells:
    """Polygon cells of an iron region."""

    polygons: tuple[np.ndarray, ...]

    @property
    def centroids(self) -> np.ndarray:
        return np.array([shapely.Polygon(p).centroid.coords[0] for p in self.polygons])

    @property
    def areas(self) -> np.ndarray:
        return np.array([shapely.Polygon(p).area for p in self.polygons])


def mesh(region: shapely.Polygon, size: float, sliver: float = 0.15) -> Cells:
    """Square cells of side ``size`` clipped to ``region``, slivers merged.

    A clipped cell smaller than ``sliver * size**2`` has its centroid almost on
    its own edges, where the edge quadrature is near-singular; it is merged into
    the neighbour with which it shares the longest edge.
    """
    r0, z0, r1, z1 = region.bounds
    pieces = []
    for r in np.arange(r0, r1, size):
        for z in np.arange(z0, z1, size):
            cell = shapely.box(r, z, r + size, z + size).intersection(region)
            for piece in getattr(cell, "geoms", [cell]):
                if piece.geom_type == "Polygon" and piece.area > 0:
                    pieces.append(piece)
    small = [k for k, p in enumerate(pieces) if p.area < sliver * size**2]
    for k in sorted(small, key=lambda k: pieces[k].area):
        piece = pieces[k]
        if piece is None:
            continue
        shared = [
            (piece.intersection(other).length, j)
            for j, other in enumerate(pieces)
            if other is not None and j != k and piece.distance(other) == 0
        ]
        shared = [item for item in shared if item[0] > 0]
        if not shared:
            continue
        _, j = max(shared)
        merged = shapely.union(pieces[j], piece)
        if merged.geom_type == "Polygon":
            pieces[j] = shapely.simplify(merged, 1e-9)
            pieces[k] = None
    polygons = [np.asarray(p.exterior.coords)[:-1] for p in pieces if p is not None]
    return Cells(tuple(polygons))


def edge_rings(cells: Cells, order: int):
    """Quadrature rings on every edge: radius, height, weight, normal, cell."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    radius, height, weight, normal, owner = [], [], [], [], []
    for index, polygon in enumerate(cells.polygons):
        ring = shapely.Polygon(polygon)
        vertices = np.asarray(
            shapely.geometry.polygon.orient(ring, 1.0).exterior.coords
        )
        for start, end in zip(vertices[:-1], vertices[1:]):
            step = end - start
            length = float(np.hypot(*step))
            if length == 0:
                continue
            outward = np.array([step[1], -step[0]]) / length  # CCW boundary
            points = start + np.outer((nodes + 1) / 2, step)
            radius.append(points[:, 0])
            height.append(points[:, 1])
            weight.append(weights * length / 2)
            normal.append(np.repeat(outward[None], order, axis=0))
            owner.append(np.full(order, index))
    return (
        np.concatenate(radius),
        np.concatenate(height),
        np.concatenate(weight),
        np.concatenate(normal),
        np.concatenate(owner),
    )


def ring_field(target_r, target_z, source_r, source_z, chunk=2048):
    """``(B_R, B_Z)`` [T/A] at targets from unit rings, shape (targets, sources)."""
    br = np.empty((target_r.size, source_r.size))
    bz = np.empty_like(br)
    for start in range(0, source_r.size, chunk):
        stop = start + chunk
        with np.errstate(all="ignore"):
            z_part, r_part = greens_bz_br(
                target_r[:, None],
                target_z[:, None],
                source_r[None, start:stop],
                source_z[None, start:stop],
            )
        br[:, start:stop] = np.nan_to_num(r_part)
        bz[:, start:stop] = np.nan_to_num(z_part)
    return br, bz


def interaction(cells: Cells, targets: np.ndarray, order: int = 12) -> np.ndarray:
    """Matrix G with ``(B_R, B_Z)`` at targets = G @ (M_R, M_Z) per cell.

    Rows are ``[B_R(targets), B_Z(targets)]`` and columns
    ``[M_R(cells), M_Z(cells)]``; units T per (A/m).
    """
    r, z, w, n, owner = edge_rings(cells, order)
    keep = r > 0
    r, z, w, n, owner = r[keep], z[keep], w[keep], n[keep], owner[keep]
    br, bz = ring_field(targets[:, 0], targets[:, 1], r, z)
    count = len(cells.polygons)
    # K_phi per unit M_R is -n_Z, per unit M_Z is +n_R
    columns = []
    for factor in (-n[:, 1], n[:, 0]):
        weightings = np.zeros((r.size, count))
        weightings[np.arange(r.size), owner] = w * factor
        columns.append(np.vstack((br @ weightings, bz @ weightings)))
    return np.hstack(columns)


@dataclass(frozen=True)
class Material:
    """Relative permeability as a function of |B| from a B-H table."""

    b_field: np.ndarray
    mu_r: np.ndarray

    def __call__(self, bmod: np.ndarray) -> np.ndarray:
        return np.interp(bmod, self.b_field, self.mu_r)


def solve(
    gram: np.ndarray,
    b_external: np.ndarray,
    material,
    relax: float = 0.5,
    tolerance: float = 1e-4,
    iterations: int = 200,
):
    """Return ``M`` (2N,), ``mu_r`` (N,) and the iteration record."""
    count = b_external.size // 2
    mu_r = material(np.hypot(b_external[:count], b_external[count:]))
    history = []
    for iteration in range(iterations):
        c = np.tile((mu_r - 1) / (MU0 * mu_r), 2)
        system = np.eye(2 * count) - c[:, None] * gram
        magnetisation = np.linalg.solve(system, c * b_external)
        b_total = b_external + gram @ magnetisation
        bmod = np.hypot(b_total[:count], b_total[count:])
        target = material(bmod)
        change = float(np.max(np.abs(target - mu_r) / mu_r))
        history.append(change)
        if change < tolerance:
            break
        mu_r = mu_r + relax * (target - mu_r)
    return magnetisation, mu_r, history


def solve_newton(
    gram: np.ndarray,
    b_external: np.ndarray,
    material,
    tolerance: float = 1e-8,
    iterations: int = 60,
):
    """Damped Newton on ``F(M) = M - c(|B|) B`` with ``B = B_ext + G M``.

    ``c(b) = (mu_r(b) - 1)/(mu0 mu_r(b))`` makes ``M = c B`` the material law.
    The Jacobian uses the differential law ``d(cB)/dB = c I + (c'/b) B B^T`` per
    cell, so the rising branch of a soft-iron curve, which drives the
    fixed-permeability update unstable, is handled. Steps are halved until the
    residual norm falls.
    """
    from scipy.interpolate import PchipInterpolator

    table = PchipInterpolator(
        material.b_field, (material.mu_r - 1) / (MU0 * material.mu_r)
    )
    table_slope = table.derivative()
    top = material.b_field[-1]

    def law(b):
        return table(np.clip(b, 0, top))

    def slope(b):
        return np.where(b < top, table_slope(np.clip(b, 0, top)), 0.0)

    count = b_external.size // 2

    def residual(m):
        b = b_external + gram @ m
        br, bz = b[:count], b[count:]
        bmod = np.hypot(br, bz)
        c = law(bmod)
        return m - np.concatenate((c * br, c * bz)), br, bz, bmod, c

    linear = solve(gram, b_external, material, iterations=1)[0]
    m = linear
    history = []
    for _ in range(iterations):
        f, br, bz, bmod, c = residual(m)
        norm = float(np.linalg.norm(f) / max(np.linalg.norm(m), 1e-30))
        history.append(norm)
        if norm < tolerance:
            break
        held = np.where(bmod > 0, bmod, 1.0)
        dc = slope(bmod) / held
        block = np.zeros((2 * count, 2 * count))
        idx = np.arange(count)
        block[idx, idx] = c + dc * br * br
        block[idx, idx + count] = dc * br * bz
        block[idx + count, idx] = dc * bz * br
        block[idx + count, idx + count] = c + dc * bz * bz
        jacobian = np.eye(2 * count) - block @ gram
        step = np.linalg.solve(jacobian, -f)
        scale = 1.0
        base = np.linalg.norm(f)
        while scale > 1e-4:
            trial = m + scale * step
            if np.linalg.norm(residual(trial)[0]) < base:
                break
            scale /= 2
        m = m + scale * step
    b = b_external + gram @ m
    mu_r_final = 1 / (1 - MU0 * law(np.hypot(b[:count], b[count:])))
    return m, mu_r_final, history
