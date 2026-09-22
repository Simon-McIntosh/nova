"""Production saddle admission on cell carriers near a separatrix edge."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import dual_stencil_census as census_benchmark
from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes


@pytest.mark.parametrize("requested,realised", [(110, 132), (300, 340), (500, 550)])
def test_production_read_admits_analytic_saddle(requested, realised):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    machine, operator, state, _exact = census_benchmark._machine_and_field(
        certificate.DIVERTED_CASE_NAME, requested
    )
    assert len(machine.node) == realised
    pitch = np.sqrt(np.median(np.asarray(machine.area, dtype=np.float64)))
    reference = np.asarray(certificate.DIVERTED_REFERENCE.x_point)
    assert np.ptp(state[:realised]) > 1.0e-10
    masks, topology = operator.read(jnp.asarray(state, dtype=jnp.float64))
    error = float(np.linalg.norm(np.asarray(topology.x_point) - reference) / pitch)
    grid = operator._fixed_design_topology.grid
    physical = jnp.asarray(state[:realised], dtype=jnp.float64)
    table = grid.candidate_table_status(physical)
    crossing = np.asarray(table["ring_crossing_count"])
    print(
        f"saddle_admission realised={realised} error_pitch={error:.12g} "
        f"retained={np.asarray(table['retained_count']).tolist()} "
        f"crossing_count_shape={crossing.shape} labelled_cells={masks.label.size}",
        flush=True,
    )
    assert crossing.size > 0
    assert np.any(crossing == 2)
    assert np.isfinite(error) and error <= 0.10
