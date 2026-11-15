"""Inspect the affine magnetic energy on a small geometry before shell solves."""

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402

assert jax.config.jax_enable_x64
import shapely  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot.tiledassembly import linear_galerkin_cell_interaction  # noqa: E402
from time import monotonic  # noqa: E402

for size in (1.0,):
    cells = iron._mesh(shapely.box(1, 0, 2, 1), size)
    started = monotonic()
    operator, blocks = linear_galerkin_cell_interaction(cells.polygons, plan=iron._PLAN)
    print("operator", operator / iron._MU0, flush=True)
    print("gram", blocks, flush=True)
    print(
        "spectrum",
        iron.galerkin_spectrum(cells, operator, basis="linear"),
        "wall",
        monotonic() - started,
        flush=True,
    )
