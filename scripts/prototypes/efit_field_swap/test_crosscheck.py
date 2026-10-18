"""The sweep covers every input, and a wide wall reads with the fitted capacity.

Two properties the case-by-case table rests on, both cheap to check without
running the sweep: that the enumeration behind the table names every input the
manifest lists, crossing records included, and that a record whose wall
truncates nova's own contour-mesh capacity is read rather than dropped.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.prototypes.efit_field_swap.crosscheck import (
    _decision,
    _row,
    cases,
    domain_rings,
    fitted_geometry,
    lattice,
    wall_units,
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


def test_a_wide_wall_record_reads_a_decision():
    """An ITER record, unread before the fit, now carries nova's decision.

    Nova's own contour-mesh capacity admits no vertex for this record's wall, so
    the geometry cannot be built without the fit. The assertion is on the read
    reaching a decision, not on the shape of the failure that preceded it: what
    the repair owes is that this record is comparable, which is the property the
    review found missing.
    """
    record = read_trace_record(ITER_RECORD)
    cells, samples = lattice(record.wall_ring, 16)
    geometry, fitted = fitted_geometry(cells, samples, wall_units(record))
    assert fitted is True, "this record's wall is the case the fit exists for"
    assert geometry is not None
    row = _decision(
        "iter-dina3420-s4-18-19:A",
        "boundary_slice_4_iter_18.txt",
        record,
        resolutions=(16,),
    )
    assert row.capacity_fit is True
    assert row.nova_label in ("limited", "diverted")


def test_missing_record_remains_an_explicit_unread_row(tmp_path):
    row = _row("missing", "missing.txt", str(tmp_path / "missing.txt"))
    assert row.case == "missing"
    assert row.efit_label == "unread"
    assert row.nova_label == "unread"
    assert "FileNotFoundError" in row.note


def test_indexed_wall_loops_keep_the_axis_boundary_and_exclude_obstacles():
    source = read_trace_record(ITER_RECORD)
    obstacle = np.asarray(((0.7, 0.7), (1.3, 0.7), (1.3, 1.3), (0.7, 1.3)))
    vessel = np.asarray(((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)))

    def segments(vertices):
        return np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)

    joined = np.concatenate((segments(obstacle), segments(vessel)))
    record = replace(
        source,
        wall_r=joined[:, :, 0],
        wall_z=joined[:, :, 1],
        wall_index=np.asarray(((1, 4), (5, 8)), dtype=np.int32),
        axis_r=0.0,
        axis_z=0.0,
    )
    rings = record.wall_rings
    assert len(rings) == 2
    np.testing.assert_array_equal(rings[0][:-1], obstacle)
    np.testing.assert_array_equal(rings[1][:-1], vessel)
    with pytest.raises(ValueError, match="2 wall loops"):
        len(record.wall_ring)

    boundary, obstacles = domain_rings(record)
    np.testing.assert_array_equal(boundary, rings[1])
    assert len(obstacles) == 1
    np.testing.assert_array_equal(obstacles[0], rings[0])
    units = wall_units(record)
    assert len(units) == 2
    np.testing.assert_array_equal(units[0], rings[1])
    np.testing.assert_array_equal(units[1], rings[0])
    cells, _ = lattice(boundary, 8, obstacles)
    centres = np.asarray([cell.mean(axis=0) for cell in cells])
    assert len(cells) == 60
    assert not np.any(np.all((centres > 0.7) & (centres < 1.3), axis=1))
