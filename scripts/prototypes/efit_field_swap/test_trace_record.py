"""Round-trip a captured imas-efit trace record through the Python codec."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from scripts.prototypes.efit_field_swap.trace_record import (
    HEADER,
    read_trace_record,
    write_trace_record,
)

CAPTURE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-saddle-admission/traces/jt60sa-5slice/"
    "boundary_slice_5_iter_301.txt"
)


def test_source_record_is_available():
    assert CAPTURE.is_file(), f"captured record missing: {CAPTURE}"


def test_header_grid_axis_polarity_and_wall_round_trip(tmp_path):
    record = read_trace_record(CAPTURE)
    assert record.rgrid.size == 65 and record.zgrid.size == 65
    assert record.psi.shape == (65, 65)
    assert record.polarity in (1, -1)
    assert record.verdict in (1, 2)
    # The ordered wall ring retains segment endpoints and closes at its start.
    ring = record.wall_ring
    assert ring.shape[1] == 2
    np.testing.assert_array_equal(ring[0], ring[-1])
    assert np.all(np.any(np.diff(ring, axis=0) != 0, axis=1))

    out = tmp_path / "roundtrip.txt"
    write_trace_record(out, record)
    again = read_trace_record(out)

    assert again.slice == record.slice
    assert again.iteration == record.iteration
    assert again.verdict == record.verdict
    assert again.polarity == record.polarity
    assert again.sweep_armed == record.sweep_armed
    assert again.wall_axis_connected == record.wall_axis_connected
    assert again.handover_reverted == record.handover_reverted
    assert again.saddle_source_node == record.saddle_source_node
    assert again.wall_source_node == record.wall_source_node
    assert again.wall_source_segment == record.wall_source_segment

    np.testing.assert_array_equal(again.rgrid, record.rgrid)
    np.testing.assert_array_equal(again.zgrid, record.zgrid)
    np.testing.assert_array_equal(again.psi, record.psi)
    np.testing.assert_array_equal(again.wall_r, record.wall_r)
    np.testing.assert_array_equal(again.wall_z, record.wall_z)
    np.testing.assert_array_equal(again.wall_index, record.wall_index)
    np.testing.assert_array_equal(again.cell_r, record.cell_r)
    np.testing.assert_array_equal(again.cell_z, record.cell_z)
    np.testing.assert_array_equal(again.polygon_r, record.polygon_r)
    np.testing.assert_array_equal(again.polygon_z, record.polygon_z)
    for name in ("axis_psi", "axis_r", "axis_z", "saddle_refined", "wall_level"):
        assert getattr(again, name) == getattr(record, name)
    assert again.fed_back_psi == record.fed_back_psi


def test_layout_matches_the_fortran_writer(tmp_path):
    """The emitted lines reproduce the Fortran writer's field-by-field layout."""
    record = read_trace_record(CAPTURE)
    out = tmp_path / "layout.txt"
    write_trace_record(out, record)
    lines = out.read_text().splitlines()

    assert lines[0] == HEADER
    dimensions = lines[1].split()
    assert dimensions[:3] == ["5", "301", "2"]
    assert [int(token) for token in dimensions[3:]] == [65, 65, 115, 3, 386, 115]

    flags = lines[2].split()
    assert flags[3] in ("T", "F") and flags[4] in ("T", "F")
    assert flags[6] in ("T", "F")
    assert int(flags[5]) in (1, -1)

    status = lines[3].split()
    assert len(status) == 7
    assert all("E" in token for token in status)

    # Real lines reuse the Fortran ES26.17E3 spelling: 17 fractional digits and
    # a three-digit exponent.
    token = lines[4].split()[0]
    assert "E" in token
    assert len(token.split("E")[0].split(".")[1]) == 17
    assert len(token.split("E")[1]) == 4  # sign and three exponent digits

    # The grid pair is 65 r values then 65 z values, then the psi block.
    nw = nh = 65
    nwall, nlim, ncell, npoly = 115, 3, 386, 115
    expected = 4 + nw + nh + nwall + nlim + ncell + npoly + nw * nh
    assert len(lines) == expected


def test_rejects_an_unknown_header(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("BOUNDARY_LEVEL_TRACE 1\n")
    with pytest.raises(ValueError):
        read_trace_record(bad)


def test_wallless_crossing_uses_its_polygon_and_closes_it():
    crossing = read_trace_record(
        Path(CAPTURE.parents[2]) / "crossing" / "synthetic_crossing_a.txt"
    )
    assert crossing.wall_r.shape[0] == 0
    assert crossing.polygon_r.size == 4
    ring = crossing.wall_ring
    np.testing.assert_array_equal(ring[:-1, 0], crossing.polygon_r)
    np.testing.assert_array_equal(ring[0], ring[-1])


def test_open_trace_wall_keeps_its_last_endpoint_before_closing():
    source = read_trace_record(CAPTURE)
    wall_r = source.wall_r.copy()
    wall_r[-1, 1] += 0.1
    ring = replace(source, wall_r=wall_r).wall_ring
    endpoints = np.stack((wall_r, source.wall_z), axis=-1)
    for endpoint in endpoints.reshape(-1, 2):
        assert np.any(np.all(ring == endpoint, axis=1))
    np.testing.assert_array_equal(ring[-2], endpoints[-1, 1])
    np.testing.assert_array_equal(ring[0], ring[-1])
    assert np.all(np.any(np.diff(ring, axis=0) != 0, axis=1))

    closed_r = source.wall_r.copy()
    closed_z = source.wall_z.copy()
    closed_r[-1, 1] = closed_r[0, 0]
    closed_z[-1, 1] = closed_z[0, 0]
    closed = replace(source, wall_r=closed_r, wall_z=closed_z).wall_ring
    np.testing.assert_array_equal(closed[-1], closed[0])
    assert not np.array_equal(closed[-2], closed[-1])
    assert np.all(np.any(np.diff(closed, axis=0) != 0, axis=1))
