"""Reproduce the weighted collocation defect on an asymmetric cell pair."""

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402

assert jax.config.jax_enable_x64
cells = iron._Mesh(
    (
        np.array([[1.0, 0.0], [1.3, 0.0], [1.3, 0.2], [1.0, 0.2]]),
        np.array([[1.3, 0.0], [1.7, 0.0], [1.7, 0.2], [1.3, 0.2]]),
    ),
    np.array([[1.15, 0.1], [1.5, 0.1]]),
    np.array([0.06, 0.08]),
)
gram = iron._interaction(cells, cells.centroids, iron._PLAN)
volumes = 2 * np.pi * cells.areas * cells.centroids[:, 0]
w = np.tile(volumes, 2)
s = np.sqrt(w)
weighted = gram * s[:, None] / s[None, :] / iron._MU0
print("MODULE", iron.__file__, flush=True)
print(
    "COLLOCATION asymmetry",
    np.linalg.norm(w[:, None] * gram - (w[:, None] * gram).T)
    / np.linalg.norm(w[:, None] * gram),
    "spectrum",
    np.linalg.eigvalsh((weighted + weighted.T) / 2).tolist(),
    flush=True,
)
from nova.biot.tiledassembly import galerkin_cell_interaction  # noqa: E402

for order in (8, 16, 24, 32):
    g = galerkin_cell_interaction(cells.polygons, plan=iron._PLAN, order=order)
    h = g * s[:, None] / s[None, :] / iron._MU0
    print(
        "GALERKIN order",
        order,
        "matrix",
        (g / iron._MU0).tolist(),
        "spectrum",
        np.linalg.eigvalsh(h).tolist(),
        "symmetry",
        np.linalg.norm(w[:, None] * g - (w[:, None] * g).T)
        / np.linalg.norm(w[:, None] * g),
        flush=True,
    )
