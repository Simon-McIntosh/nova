"""A nova-decided export is read back through the record codec.

The export writer is only useful if the record it emits is one imas-efit's own
list-directed reader could open, so the check is a full re-read rather than a
comparison of the file against the code the writer shares with the reader: the
emitted record is parsed again and its decision fields, grid, wall and psi are
compared against what was written.
"""

from __future__ import annotations


import numpy as np
import pytest

from scripts.prototypes.efit_field_swap.crosscheck import load_pairs, resolve_trace
from scripts.prototypes.efit_field_swap.export_states import export_row
from scripts.prototypes.efit_field_swap.export_tie_islands import (
    bisected_brackets,
    bisected_offset,
    validate_carrier,
)
from scripts.prototypes.efit_field_swap.trace_record import (
    HEADER,
    read_trace_record,
    write_trace_record,
)


def _source_and_pairs():
    return load_pairs()


def _raw_fields(path):
    """Read the Fortran text positions without using the trace codec."""
    lines = path.read_text().splitlines()
    dims = [int(value) for value in lines[1].split()]
    nr, nz, nwall, nlim, ncell, npoly = dims[3:]
    cursor = 4
    radial = np.asarray([float(line) for line in lines[cursor : cursor + nr]])
    cursor += nr
    vertical = np.asarray([float(line) for line in lines[cursor : cursor + nz]])
    cursor += nz
    wall = np.asarray(
        [
            [float(value) for value in line.split()]
            for line in lines[cursor : cursor + nwall]
        ]
    )
    cursor += nwall + nlim + ncell + npoly
    psi = np.asarray([float(line) for line in lines[cursor:]]).reshape(nr, nz)
    return (
        dims,
        [float(value) for value in lines[3].split()],
        radial,
        vertical,
        wall,
        psi,
    )


def test_export_round_trips_through_the_codec(tmp_path):
    pairs = _source_and_pairs()
    row = {
        "case": f"{pairs[0][0]}:A",
        "nova_label": "diverted",
        "nova_axis_psi": 1.25,
        "nova_axis_r": 3.5,
        "nova_axis_z": 0.25,
        "nova_saddle_psi": 0.5,
        "nova_boundary_psi": 0.5,
    }
    path = export_row(row, pairs, tmp_path)
    assert path.is_file()
    assert path.read_text().splitlines()[0] == HEADER

    read_back = read_trace_record(path)
    source = read_trace_record(resolve_trace(pairs[0][1]))
    assert read_back.verdict == 2
    assert read_back.diverted is True
    assert read_back.axis_psi == 1.25
    assert (read_back.axis_r, read_back.axis_z) == (3.5, 0.25)
    assert read_back.saddle_refined == 0.5
    assert read_back.fed_back_psi == 0.5
    assert read_back.slice == source.slice
    assert read_back.iteration == source.iteration
    assert np.array_equal(read_back.rgrid, source.rgrid)
    assert np.array_equal(read_back.zgrid, source.zgrid)
    assert np.array_equal(read_back.psi, source.psi)
    assert np.array_equal(read_back.wall_r, source.wall_r)
    assert np.array_equal(read_back.polygon_r, source.polygon_r)

    raw_export = _raw_fields(path)
    raw_source = _raw_fields(resolve_trace(pairs[0][1]))
    assert raw_export[0][2] == 2
    assert raw_export[1][0:3] == [1.25, 3.5, 0.25]
    assert raw_export[1][4] == raw_export[1][6] == 0.5
    for exported, original in zip(raw_export[2:], raw_source[2:], strict=True):
        np.testing.assert_array_equal(exported, original)


def test_written_export_is_bit_stable(tmp_path):
    pairs = _source_and_pairs()
    source = read_trace_record(resolve_trace(pairs[0][1]))
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    write_trace_record(first, source)
    write_trace_record(second, read_trace_record(first))
    assert first.read_text() == second.read_text()


def _receipt(adjacent, unchanged, right=None):
    if right is None:
        right = float(np.nextafter(0.1613, np.inf))
    return {
        "key": "both",
        "requested_cells": 250,
        "achieved_cells": 282,
        "bisected": [
            {
                "bracket": [0.1613, right],
                "steps": 42,
                "adjacent": adjacent,
                "unchanged": unchanged,
                "left": {"boundary_class": 1, "saddle_identity": 0},
                "right": {"boundary_class": 0, "saddle_identity": 2},
            }
        ],
    }


def test_an_unbisected_offset_is_refused(tmp_path):
    """A tie-island state is a bisected one, so an interval is not exportable."""
    for adjacent, unchanged in ((False, False), (True, True)):
        with pytest.raises(SystemExit):
            bisected_offset(_receipt(adjacent, unchanged), tmp_path / "receipt.json")
    with pytest.raises(SystemExit, match="adjacent binary64"):
        bisected_offset(_receipt(True, False, 0.1613), tmp_path / "receipt.json")
    same_key = _receipt(True, False)
    same_key["bisected"][0]["right"] = same_key["bisected"][0]["left"]
    with pytest.raises(SystemExit, match="decision transition"):
        bisected_offset(same_key, tmp_path / "receipt.json")


def test_a_bisected_offset_is_taken_from_the_receipt(tmp_path):
    offset, entry = bisected_offset(_receipt(True, False), tmp_path / "receipt.json")
    assert offset == entry["bracket"][0] == 0.1613
    assert len(list(bisected_brackets(_receipt(True, False), "receipt.json"))) == 1


def test_receipt_cell_counts_must_match_the_carrier():
    receipt = _receipt(True, False)
    validate_carrier(receipt, 250, 282)
    with pytest.raises(SystemExit, match="250/550"):
        validate_carrier(receipt, 250, 550)
    with pytest.raises(SystemExit, match="500/282"):
        validate_carrier(receipt, 500, 282)
