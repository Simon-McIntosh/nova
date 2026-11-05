"""Analytic control: a linear permeable sphere in a uniform field.

The sphere is axisymmetric (a half disc in the poloidal plane, on the axis), so
the edge-sheet method must reproduce the exact interior field
``B_in = 3 mu_r / (mu_r + 2) B_0`` and the exact exterior dipole. A negative
control replaces mu_r by 1, which must return B_0 to rounding.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import shapely

from magnetisation import interaction, mesh, solve

RADIUS, B0 = 0.5, 0.01
result = {}
for mu_r in (1.0, 10.0, 100.0, 1000.0):
    for size in (0.1, 0.05, 0.025):
        disc = (
            shapely.Point(0, 0)
            .buffer(RADIUS, 256)
            .intersection(shapely.box(0, -RADIUS, RADIUS, RADIUS))
        )
        cells = mesh(disc, size)
        centroids = cells.centroids
        gram = interaction(cells, centroids, order=12)
        count = len(cells.polygons)
        b_ext = np.concatenate((np.zeros(count), np.full(count, B0)))
        magnetisation, _, _ = solve(gram, b_ext, lambda b, m=mu_r: np.full_like(b, m))
        b_in = b_ext + gram @ magnetisation
        exact = 3 * mu_r / (mu_r + 2) * B0
        inner = np.hypot(*centroids.T) < 0.6 * RADIUS
        error = np.abs(b_in[count:][inner] / exact - 1)
        # exterior on the equator at 2a: exact B_z = B0 (1 - k a^3 / r^3)
        k = (mu_r - 1) / (mu_r + 2)
        probe = np.array([[2 * RADIUS, 0.0], [0.0, 2 * RADIUS]])
        b_probe = interaction(cells, probe, order=12) @ magnetisation
        bz_equator = B0 + b_probe[2]
        bz_pole = B0 + b_probe[3]
        result[f"mu{mu_r:g}_h{size:g}"] = dict(
            cells=count,
            interior_median_error=float(np.median(error)),
            interior_max_error=float(np.max(error)),
            equator_error=float(abs(bz_equator / (B0 * (1 - k / 8)) - 1)),
            pole_error=float(abs(bz_pole / (B0 * (1 + 2 * k / 8)) - 1)),
        )
        print(
            f"mu_r={mu_r:7g} h={size:5.3f} cells={count:5d} "
            f"interior median {np.median(error):.2e} max {np.max(error):.2e} "
            f"equator {result[f'mu{mu_r:g}_h{size:g}']['equator_error']:.2e} "
            f"pole {result[f'mu{mu_r:g}_h{size:g}']['pole_error']:.2e}",
            flush=True,
        )
json.dump(result, open(sys.argv[1], "w"), indent=1)
