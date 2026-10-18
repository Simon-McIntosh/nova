"""A nova-decided export is read back through the record codec.

The export writer is only useful if the record it emits is one imas-efit's own
list-directed reader could open, so the check is a full re-read rather than a
comparison of the file against the code the writer shares with the reader: the
emitted record is parsed again and its decision fields, grid, wall and psi are
compared against what was written.
"""

from __future__ import annotations


import numpy as np

from scripts.prototypes.efit_field_swap.crosscheck import load_pairs, resolve_trace
from scripts.prototypes.efit_field_swap.export_states import export_row
from scripts.prototypes.efit_field_swap.trace_record import (
    HEADER,
    read_trace_record,
    write_trace_record,
)


def _source_and_pairs():
    return load_pairs()


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


def test_written_export_is_bit_stable(tmp_path):
    pairs = _source_and_pairs()
    source = read_trace_record(resolve_trace(pairs[0][1]))
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    write_trace_record(first, source)
    write_trace_record(second, read_trace_record(first))
    assert first.read_text() == second.read_text()
