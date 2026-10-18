"""Exhaustive stationary-point census of a record's own bicubic spline field.

Newton on grad psi from a dense seed lattice, deduplicated, with the Hessian
eigenvalues and a type per point.  The field is a FastField over the record's
own 65x65 grid, so this shares no code with either nova field-swap adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
from matplotlib.path import Path


@dataclass
class CriticalPoint:
    position: np.ndarray      # (r, z)
    psi: float
    gradient: np.ndarray      # (dpsi/dr, dpsi/dz)
    eigenvalues: np.ndarray   # Hessian eigenvalues, ascending
    kind: str                 # 'maximum' | 'minimum' | 'saddle'
    converged: bool
    inside_wall: bool = dc_field(default=False)


def classify(eigenvalues) -> str:
    if eigenvalues[0] > 0 and eigenvalues[1] > 0:
        return "minimum"
    if eigenvalues[0] < 0 and eigenvalues[1] < 0:
        return "maximum"
    return "saddle"


def hessian_type(field, r, z):
    """Hessian eigenvalues and type at a point, from the record's own spline."""
    _, gr, gz, hrr, hrz, hzz = field.jet(np.array([r]), np.array([z]))
    hess = np.array([[hrr[0], hrz[0]], [hrz[0], hzz[0]]])
    eig = np.linalg.eigvalsh(0.5 * (hess + hess.T))
    return eig, classify(eigenvalues=eig), np.array([gr[0], gz[0]])


def newton_points(field, seeds, tol=1e-12, max_iter=80, step_cap=0.5):
    """Vectorised Newton on grad psi; one row per seed.

    A point whose gradient has vanished is frozen and dropped from the working
    set, so a few pathological seeds do not keep every other row in the loop.
    """
    R = seeds[:, 0].astype(float).copy()
    Z = seeds[:, 1].astype(float).copy()
    active = np.arange(len(R))
    while True:
        if active.size == 0:
            break
        ar, az = R[active], Z[active]
        _, gr, gz, hrr, hrz, hzz = field.jet(ar, az)
        det = hrr * hzz - hrz * hrz
        safe = np.abs(det) > 1e-12
        s_r = np.zeros(active.size)
        s_z = np.zeros(active.size)
        s_r[safe] = (hzz[safe] * gr[safe] - hrz[safe] * gz[safe]) / det[safe]
        s_z[safe] = (hrr[safe] * gz[safe] - hrz[safe] * gr[safe]) / det[safe]
        norm = np.hypot(s_r, s_z)
        scale = np.ones_like(norm)
        big = norm > step_cap
        scale[big] = step_cap / norm[big]
        R[active] = ar - s_r * scale
        Z[active] = az - s_z * scale
        done = (norm < tol) | ~safe
        active = active[~done]
        max_iter -= 1
        if max_iter <= 0:
            break
    _, gr, gz, hrr, hrz, hzz = field.jet(R, Z)
    return R, Z, np.hypot(gr, gz)


def census(field, record, n_seed=120, dedup=1e-4, wall_polygon=None):
    """Deduplicated stationary points of the record's spline over the grid box."""
    rmin, rmax = float(record.rgrid.min()), float(record.rgrid.max())
    zmin, zmax = np.min(record.zgrid), np.max(record.zgrid)
    rs = np.linspace(rmin, rmax, n_seed)
    zs = np.linspace(zmin, zmax, n_seed)
    rr, zz = np.meshgrid(rs, zs, indexing="ij")
    seeds = np.column_stack((rr.ravel(), zz.ravel()))

    R, Z, grad = newton_points(field, seeds)
    tol = 1e-9
    keep = (
        (R >= rmin - tol)
        & (R <= rmax + tol)
        & (Z >= zmin - tol)
        & (Z <= zmax + tol)
        & (grad < 1e-6)
        & np.isfinite(R)
        & np.isfinite(Z)
    )
    R, Z = R[keep], Z[keep]

    order = np.lexsort((Z, R))
    R, Z = R[order], Z[order]
    # collapse points that land within `dedup` of an earlier one, by rounding
    # onto a lattice of side `dedup` and averaging each occupied cell
    keys = np.stack((np.round(R / dedup), np.round(Z / dedup)), axis=1).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    cell_r = np.bincount(inverse, weights=R) / np.bincount(inverse)
    cell_z = np.bincount(inverse, weights=Z) / np.bincount(inverse)
    R, Z = cell_r[inverse[first]], cell_z[inverse[first]]

    value, gr, gz, hrr, hrz, hzz = field.jet(R, Z)
    pts = []
    for i in range(len(R)):
        hess = np.array([[hrr[i], hrz[i]], [hrz[i], hzz[i]]])
        eig = np.linalg.eigvalsh(0.5 * (hess + hess.T))
        pts.append(
            CriticalPoint(
                position=np.array([R[i], Z[i]]),
                psi=float(value[i]),
                gradient=np.array([gr[i], gz[i]]),
                eigenvalues=eig,
                kind=classify(eig),
                converged=True,
            )
        )

    if wall_polygon is not None and len(wall_polygon) >= 3:
        inside = Path(wall_polygon).contains_points(
            np.array([p.position for p in pts])
        )
        for point, flag in zip(pts, inside):
            point.inside_wall = bool(flag)

    pts.sort(key=lambda p: (-p.psi if record.polarity > 0 else p.psi))
    return pts