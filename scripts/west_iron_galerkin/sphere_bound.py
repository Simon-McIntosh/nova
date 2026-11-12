"""Quantify the exact-sphere consistency defect of volume-tested polygon cells."""

from nova.jax.config import configure_dtypes

configure_dtypes()

import json  # noqa: E402
import numpy as np  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot.tiledassembly import galerkin_cell_interaction  # noqa: E402
from tests.test_magnetisation import sphere, uniform  # noqa: E402

cells = iron._mesh(sphere(), 0.1)
n = len(cells.areas)
volume = 2 * np.pi * cells.areas * cells.centroids[:, 0]
scale = np.sqrt(np.tile(volume / volume.sum(), 2))
unit = np.r_[np.zeros(n), np.ones(n)]
point = iron._interaction(cells, cells.centroids, iron._PLAN) / iron._MU0
rows = n + np.flatnonzero(np.linalg.norm(cells.centroids, axis=1) < 0.3)
probe_points = np.array([[1.0, 0.0], [0.0, 1.0]])
probe = iron._interaction(cells, probe_points, iron._PLAN) / iron._MU0
recorded = {
    10.0: [9.736922259584446e-7, 1.4861685984079998e-6],
    100.0: [1.299614500682722e-6, 1.8379254441702741e-6],
    1000.0: [1.3400238558780586e-6, 1.8779696518977929e-6],
}
for order in (24, 32, 48, 64):
    gram = (
        galerkin_cell_interaction(
            tuple(p * iron._SHEET_LENGTH_SCALE for p in cells.polygons),
            plan=iron._PLAN,
            order=order,
        )
        / iron._MU0
    )
    weighted = gram * scale[:, None] / scale[None, :]
    eigenvalues, modes = np.linalg.eigh((weighted + weighted.T) / 2)
    defect = gram @ unit - 2 * unit / 3
    modal_defect = modes.T @ (scale * defect)
    transfer = (point[rows] / scale[None, :]) @ modes
    point_defect = point[rows] @ unit - 2 / 3
    for mu in (10.0, 100.0, 1000.0):
        beta = 1 - 1 / mu
        modal_bound = np.max(
            np.sum(
                np.abs(transfer * modal_defect[None, :])
                / (1 - beta * eigenvalues)[None, :],
                axis=1,
            )
        )
        bound = beta * np.max(np.abs(point_defect)) + beta**2 * modal_bound
        state = np.linalg.solve(
            np.eye(2 * n) - beta * gram, beta * 0.01 / iron._MU0 * unit
        )
        exact = 3 * mu / (mu + 2) * 0.01
        interior = np.max(np.abs((0.01 + iron._MU0 * point[rows] @ state) / exact - 1))
        fraction = (mu - 1) / (mu + 2)
        expected = 0.01 * np.array([1 - fraction / 8, 1 + fraction / 4])
        total = (
            (uniform(probe_points).T.reshape(-1) + iron._MU0 * probe @ state)
            .reshape(2, -1)
            .T
        )
        errors = np.abs(total[:, 1] / expected - 1)
        print(
            json.dumps(
                dict(
                    order=order,
                    mu=mu,
                    interior_error=interior,
                    derived_bound=bound,
                    exterior_errors=errors.tolist(),
                    exterior_drift=(errors - np.asarray(recorded[mu])).tolist(),
                )
            ),
            flush=True,
        )
