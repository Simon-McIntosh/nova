"""The sweep covers every input, and a wide wall admits a complete contour mesh.

Two properties the case-by-case table rests on, both cheap to check without
running the sweep: that the enumeration behind the table names every input the
manifest lists, crossing records included, and that a record whose wall exceeds a
capacity sized from its tree centres still meshes, because the read sizes the
contour mesh from the geometry it must hold.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.solve_request import TopologyPolicy
from nova.equilibrium.topology import TopologyGeometry
from nova.linalg.tensor_spline import fit_tensor_spline
from scripts.prototypes.efit_field_swap.crosscheck import (
    _decision,
    _row,
    cases,
    domain_rings,
    lattice,
    read_state,
    wall_units,
)
from scripts.prototypes.efit_field_swap.orientation import read_convention
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField
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


def test_a_wide_wall_record_meshes_through_the_plain_read():
    """The read sizes its own contour mesh, so a wide wall meshes directly.

    This record's wall generates more contour vertices than a capacity sized
    from its tree centres and sampling stencil would hold. The read sizes the
    carrier from the geometry it must hold, so the plain read yields a mesh with
    valid vertices. Restoring the fixed capacity formula in the read leaves the
    mesh empty or refuses it, and this assertion fails.
    """
    record = read_trace_record(ITER_RECORD)
    cells, samples = lattice(record.wall_ring, 16)
    geometry = TopologyGeometry.from_cells(cells, samples, wall_units(record))
    assert np.any(np.asarray(geometry.contour_mesh.vertex_valid))


def test_iter_read_uses_its_negative_flux_polarity():
    record = read_trace_record(ITER_RECORD)
    assert record.polarity == -1
    row = _decision("iter", "trace", record)
    assert row.nova_label in ("limited", "diverted"), row.note
    assert row.nova_axis_psi != pytest.approx(row.nova_boundary_psi)
    assert record.polarity * row.nova_axis_psi > record.polarity * row.nova_boundary_psi


def test_field_swap_record_reports_coverage_outcome():
    case, source, path = next(
        item for item in cases() if item[0] == "jt60sa-5slice-s5-133-134:A"
    )
    row = _row(case, source, path, resolutions=(16,))
    assert row.nova_label == "limited", row.note
    assert row.coverage_winding == pytest.approx(1), row.note
    assert row.coverage_index_sum == 1, row.note
    assert row.coverage_cells_cleared > 0, row.note
    assert row.coverage_cells_uncertified == 0, row.note


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


def _paraboloid_field(radial, vertical, centre):
    """A field whose only stationary point is a minimum at ``centre``."""
    r = np.asarray(radial)[None, :]
    z = np.asarray(vertical)[:, None]
    values = (r - centre[0]) ** 2 + (z - centre[1]) ** 2
    spline = fit_tensor_spline(
        jnp.asarray(radial), jnp.asarray(vertical), jnp.asarray(values)
    )
    return TraceSplineField(spline=spline)


def test_invalid_axis_read_publishes_no_label():
    """A read that admits no polarity-valid axis publishes no limited/diverted label.

    The declared positive-polarity convention expects the axis at a flux
    maximum; this field's single interior extremum is a minimum, so the read
    licenses no axis and leaves it NaN. The adapter must refuse - no boundary
    and no limited/diverted label - rather than deciding from that read. The
    negative control restores the label path and this test then fails.
    """
    radial = np.linspace(0.0, 4.0, 9)
    vertical = np.linspace(-2.0, 2.0, 9)
    field = _paraboloid_field(radial, vertical, (2.0, 0.0))
    boundary_ring = np.asarray(
        ((0.25, -1.75), (3.75, -1.75), (3.75, 1.75), (0.25, 1.75))
    )
    outcome = read_state(
        field,
        boundary_ring,
        (),
        read_convention(polarity=1),
        TopologyPolicy(),
        resolutions=(16,),
    )
    assert outcome.label == "refused invalid axis", outcome.attempts
    assert not np.any(np.isfinite(outcome.axis))
    assert not np.any(np.isfinite(outcome.boundary))
