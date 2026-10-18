"""The sweep covers every input, and a wide wall reads with the fitted capacity.

Two properties the case-by-case table rests on, both cheap to check without
running the sweep: that the enumeration behind the table names every input the
manifest lists, crossing records included, and that a record whose wall
truncates nova's own contour-mesh capacity is read rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.equilibrium import topology

from scripts.prototypes.efit_field_swap.crosscheck import (
    _decision,
    cases,
    fitted_geometry,
    lattice,
)
from scripts.prototypes.efit_field_swap.trace_record import read_trace_record

ITER_RECORD = (
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-saddle-admission/traces/iter-dina3420/"
    "boundary_slice_4_iter_18.txt"
)


def test_every_manifest_input_becomes_a_case():
    """No record is dropped from the table, whatever its read will do."""
    named = list(cases())
    assert len(named) == 28
    assert len({case for case, _, _ in named}) == 28, "case names must be unique"
    assert sum(case.startswith(("crossing", "synthetic_")) for case, _, _ in named) == 4
    for _case, _source, path in named:
        assert path.endswith(".txt")


def _wide_wall_cells(resolution=16):
    ring = np.asarray(
        ((2.0, -6.0), (8.0, -6.0), (8.0, 24.0), (2.0, 24.0), (2.0, -6.0)),
        dtype=np.float64,
    )
    cells, samples = lattice(ring, resolution)
    return ring, cells, samples


def test_a_wide_wall_needs_the_fitted_capacity():
    """nova's own capacity admits no vertex here; the fit is what reads it."""
    ring, cells, samples = _wide_wall_cells()
    with pytest.raises(IndexError):
        topology.TopologyGeometry.from_cells(cells, samples, (ring,))
    geometry, fitted = fitted_geometry(cells, samples, (ring,))
    assert geometry is not None
    assert fitted is True
    assert len(np.asarray(geometry.centre)) == len(cells)


def test_a_wide_wall_record_reads_a_decision():
    """An ITER record, unread before the fit, now carries nova's decision."""
    row = _decision(
        "iter-dina3420-s4-18-19:A",
        "boundary_slice_4_iter_18.txt",
        read_trace_record(ITER_RECORD),
        resolutions=(16,),
    )
    assert row.capacity_fit is True
    assert row.nova_label in ("limited", "diverted")
