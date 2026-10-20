"""Locate stationary points from a sampled field and exact local derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class StationaryPoint:
    position: tuple[float, float]
    level: float
    kind: str
    eigenvalues: tuple[float, float]
    residual: float


def stationary_points(values, jets, bounds, inside, *, resolution=256):
    """Find all grid sign-change cells, polish their roots, and remove duplicates.

    ``values`` accepts an ``(n, 2)`` array. ``jets`` returns value, gradient,
    Hessian at one point. Grid derivatives only propose seeds; classification
    and levels come from the field's exact point evaluator.
    """
    left, right, bottom, top = bounds
    r = np.linspace(left, right, resolution)
    z = np.linspace(bottom, top, resolution)
    rr, zz = np.meshgrid(r, z, indexing="xy")
    samples = np.stack((rr.ravel(), zz.ravel()), axis=1)
    flux = np.asarray(values(samples)).reshape(rr.shape)
    dz, dr = np.gradient(flux, z, r, edge_order=2)
    crossed = (
        (np.minimum.reduce((dr[:-1, :-1], dr[1:, :-1], dr[:-1, 1:], dr[1:, 1:])) <= 0)
        & (np.maximum.reduce((dr[:-1, :-1], dr[1:, :-1], dr[:-1, 1:], dr[1:, 1:])) >= 0)
        & (np.minimum.reduce((dz[:-1, :-1], dz[1:, :-1], dz[:-1, 1:], dz[1:, 1:])) <= 0)
        & (np.maximum.reduce((dz[:-1, :-1], dz[1:, :-1], dz[:-1, 1:], dz[1:, 1:])) >= 0)
    )
    rows, columns = np.where(crossed)
    found = []
    spacing = min(r[1] - r[0], z[1] - z[0])
    for row, column in zip(rows, columns, strict=True):
        seed = np.array(((r[column] + r[column + 1]) / 2, (z[row] + z[row + 1]) / 2))
        if not bool(inside(seed[None, :])[0]):
            continue

        def gradient(point):
            return np.asarray(jets(point)[1])

        def hessian(point):
            return np.asarray(jets(point)[2])

        polished = root(gradient, seed, jac=hessian, tol=1e-11)
        point = polished.x
        if not bool(inside(point[None, :])[0]):
            continue
        level, first, second = jets(point)
        residual = float(np.linalg.norm(first))
        if residual > 1e-9 or np.linalg.norm(point - seed) > 3 * spacing:
            continue
        if any(np.linalg.norm(point - item.position) < spacing / 10 for item in found):
            continue
        eigen = np.linalg.eigvalsh(second)
        kind = (
            "saddle"
            if eigen[0] < 0 < eigen[1]
            else (
                "maximum"
                if eigen[1] < 0
                else "minimum"
                if eigen[0] > 0
                else "degenerate"
            )
        )
        found.append(
            StationaryPoint(
                tuple(map(float, point)),
                float(level),
                kind,
                tuple(map(float, eigen)),
                residual,
            )
        )
    return found, {"r": r, "z": z, "flux": flux, "candidate_count": len(rows)}
