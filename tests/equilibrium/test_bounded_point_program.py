"""The shared point program stays one bounded graph.

The row window, the jets build and the operator bank all evaluate
``fixed_points._source_jets``. Its lowered graph is what a cold compile pays
for, and an unrolled per-term form is the failure this budget records: the
program's size is set by the kernel expression, not by the number of cells, so
a reduced cell count measures the same graph the production rows compile.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, topology

assert jax.config.jax_enable_x64 is True

HEXAGON = np.asarray(
    (
        (5.00, -0.10),
        (5.06, -0.06),
        (5.08, 0.00),
        (5.06, 0.06),
        (5.00, 0.10),
        (4.94, 0.00),
    )
)
POINTS = np.asarray(((5.02, 0.01), (5.03, -0.02), (4.99, 0.03)))
CELLS = 8

# Measured 2026-11-15 at the shared form with eight cells: 5,442,157 bytes; one
# cell gives 5,440,521, so the program is cell-invariant and cells are not the
# unrolled axis. The ceiling allows the constant folding a JAX patch level
# changes and refuses a form that unrolls the term order again, which
# multiplies the graph once per term.
STABLEHLO_CEILING = 6_000_000


@pytest.mark.slow
def test_point_program_lowers_within_its_graph_budget():
    coupling = topology.BiotMomentCoupling.from_polygons(
        tuple(HEXAGON + np.asarray((1.5 + 0.2 * index, 0.0)) for index in range(CELLS))
    )
    program = jax.jit(jax.vmap(fixed_points._source_jets, in_axes=(0, None)))
    lowered = program.lower(jnp.asarray(POINTS), coupling)
    text = lowered.as_text()
    assert len(text.encode()) <= STABLEHLO_CEILING, (
        f"point program lowered to {len(text.encode())} bytes; "
        f"budget is {STABLEHLO_CEILING}"
    )