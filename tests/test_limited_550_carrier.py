"""The limited certificate carrier at the 550-cell rung reaches the solver geometry.

The 550-cell rung of the analytic certificate is the requested -500 hex
generator on the weak-rotation (limited) case, which realises 553 cells.  Three
geometry owners must accept that carrier before a solve can start: the atomic
cell atomiser, the topology geometry built on it, and the moment geometry the
forward operator carries.  Each is asserted here, so a refusal from any of them
fails by name rather than as an obscure downstream error.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.equilibrium import topology  # noqa: E402
from nova.equilibrium.separatrix_clip import AtomicCellMesh  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as oracle_fixture  # noqa: E402
from tests.rotating_equilibrium_references import reference_cases  # noqa: E402

#: The hex generator requested for the coarse certificate rung.
REQUESTED_CELLS = -500

#: The cell count that generator realises on the limited case.
REALISED_CELLS = 553


@pytest.fixture(scope="module")
def limited_machine():
    """Return the limited carrier at the coarse certificate rung."""
    case = reference_cases()["weak-rotation-reactor"].static_limit()
    return oracle_fixture.cached_machine(
        case,
        REQUESTED_CELLS,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )


def test_carrier_realises_the_coarse_rung(limited_machine):
    """The requested -500 generator realises 553 cells on the limited case."""
    assert len(limited_machine.node) == REALISED_CELLS
    assert len(limited_machine.cell_polygons) == REALISED_CELLS


def test_atomiser_accepts_the_carrier(limited_machine):
    """Every carrier cell atomises, keeping one centroid per carrier cell."""
    mesh = AtomicCellMesh.from_cells(limited_machine.cell_polygons)
    assert len(mesh.centroids) == REALISED_CELLS
    assert np.all(mesh.cell_vertex_count >= 3)
    assert np.all(np.isfinite(mesh.node_coordinates))
    assert np.all(np.isfinite(mesh.centroids))


def test_topology_geometry_accepts_the_carrier(limited_machine):
    """The topology geometry builds one centre and one full polygon per cell."""
    geometry = topology.TopologyGeometry.from_cells(
        limited_machine.cell_polygons,
        limited_machine.sampling_vertices,
        (limited_machine.wall_node,),
    )
    assert int(np.asarray(geometry.centre).shape[0]) == REALISED_CELLS
    assert int(np.asarray(geometry.vertex_count).shape[0]) == REALISED_CELLS
    assert np.all(np.asarray(geometry.full_area) > 0.0)


def test_forward_operator_accepts_the_carrier(limited_machine):
    """The forward operator carries one moment cell per carrier cell."""
    case = reference_cases()["weak-rotation-reactor"].static_limit()
    operator = oracle_fixture.forward_operator(case, limited_machine)
    assert int(operator.area.shape[0]) == REALISED_CELLS
