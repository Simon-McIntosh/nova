"""Chord current membership follows the shared separatrix geometry."""

from __future__ import annotations

import jax
import numpy as np
from matplotlib.path import Path as PolygonPath

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

import jax.numpy as jnp  # noqa: E402

from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium.forward_operator import set_support_clip_mode  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402


def test_chord_membership_uses_global_separatrix() -> None:
    """Analytic-exterior carriers receive no chord-mode plasma current."""
    case = "diverted-single-null"
    carrier, source, exact = certificate._case(case)
    machine = certificate._case_machine(case, carrier, exact, -500)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case, exact, coordinates)
    operator = fixture.forward_operator(source, machine)

    exterior_cells = np.asarray([7, 64], dtype=np.int32)
    analytic_boundary = PolygonPath(fixture._analytic_separatrix(exact))
    np.testing.assert_array_equal(
        analytic_boundary.contains_points(machine.node[exterior_cells]),
        np.zeros(len(exterior_cells), dtype=bool),
    )

    set_support_clip_mode("chord")
    partition = operator._support_partition(jnp.asarray(analytic))
    moments = operator._partitioned_current_moments(partition)
    current = np.asarray(jax.device_get(moments.cell_current))

    assert len(current) == 550
    assert np.count_nonzero(current) > 0
    np.testing.assert_array_equal(current[exterior_cells], np.zeros(2))
