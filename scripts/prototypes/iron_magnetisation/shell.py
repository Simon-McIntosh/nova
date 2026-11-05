"""Analytic control: a permeable spherical shell around a central dipole.

A small current loop at the centre of a linear shell (inner radius a, outer b)
has its exterior dipole field reduced by the exact factor

    S = ((2 mu + 1)(mu + 2) - 2 (mu - 1)^2 (a/b)^3) / (9 mu),

the same factor that shields a uniform field. This is the configuration the
axisymmetric WEST iron forms around its coils, so it tests the edge-sheet
method where the uniform-field sphere cannot: a closed high-mu shell carrying
a strongly non-uniform magnetisation.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import shapely

from magnetisation import interaction, mesh, ring_field, solve

A, B, LOOP = 0.9, 1.0, 0.05
result = {}
probe = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
MUS = (
    [float(v) for v in sys.argv[2].split(",")]
    if len(sys.argv) > 2
    else [10.0, 100.0, 1000.0]
)
SIZES = (
    [float(v) for v in sys.argv[3].split(",")]
    if len(sys.argv) > 3
    else [0.04, 0.02, 0.01]
)
for mu_r in MUS:
    exact = ((2 * mu_r + 1) * (mu_r + 2) - 2 * (mu_r - 1) ** 2 * (A / B) ** 3) / (
        9 * mu_r
    )
    for size in SIZES:
        half = shapely.box(0, -B, B, B)
        shell = (
            shapely.Point(0, 0)
            .buffer(B, 512)
            .difference(shapely.Point(0, 0).buffer(A, 512))
        ).intersection(half)
        cells = mesh(shell, size)
        centroids = cells.centroids
        count = len(cells.polygons)

        def loop_field(points):
            br, bz = ring_field(
                points[:, 0], points[:, 1], np.array([LOOP]), np.array([0.0])
            )
            return np.concatenate((br[:, 0], bz[:, 0]))

        gram = interaction(cells, centroids)
        m, _, _ = solve(
            gram, loop_field(centroids), lambda b, m=mu_r: np.full_like(b, m)
        )
        air = loop_field(probe)
        total = air + interaction(cells, probe) @ m
        n = probe.shape[0]
        ratio = np.hypot(air[:n], air[n:]) / np.hypot(total[:n], total[n:])
        error = np.abs(ratio / exact - 1)
        key = f"mu{mu_r:g}_h{size:g}"
        result[key] = dict(
            cells=count, exact=exact, measured=ratio.tolist(), error=error.tolist()
        )
        print(
            f"mu_r={mu_r:6g} h={size:5.3f} cells={count:5d} S exact {exact:8.3f} "
            f"measured {np.round(ratio, 3)} rel err {np.round(error, 4)}",
            flush=True,
        )
        json.dump(result, open(sys.argv[1], "w"), indent=1)
