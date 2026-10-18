"""The sweep covers every input, and a wide wall reads with the fitted capacity.

Two properties the case-by-case table rests on, both cheap to check without
running the sweep: that the enumeration behind the table names every input the
manifest lists, crossing records included, and that a record whose wall
truncates nova's own contour-mesh capacity is read rather than dropped.
"""

from __future__ import annotations


from scripts.prototypes.efit_field_swap.crosscheck import (
    _decision,
    _row,
    cases,
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
