"""Versioned boundary trace records preserve each format's decision fields."""

from pathlib import Path

import numpy as np
import pytest

from scripts.prototypes.efit_axis_adjudication.records import parse_record
from scripts.prototypes.efit_field_swap.trace_record import read_trace_record


@pytest.mark.parametrize(
    ("version", "metadata", "decision"),
    [
        (2, "7 8 9 T F -1 T", "1 2 3 4 5 6 7"),
        (3, "-1", "1 2 3 4 5 6 7"),
    ],
)
def test_boundary_trace_headers(
    tmp_path: Path, version: int, metadata: str, decision: str
):
    path = tmp_path / "trace.txt"
    path.write_text(
        f"BOUNDARY_LEVEL_TRACE {version}\n"
        "1 2 1 2 2 0 0 0 0\n"
        f"{metadata}\n{decision}\n"
        "0\n1\n0\n1\n10\n11\n12\n13\n"
    )
    record = read_trace_record(path)
    adapted = parse_record(path)
    assert record.version == version
    assert record.polarity == adapted.polarity == -1
    np.testing.assert_array_equal(record.psi, [[10, 11], [12, 13]])
    np.testing.assert_array_equal(adapted.psi, record.psi)
    if version == 2:
        assert record.saddle_source_node == adapted.saddle_source_node == 7
        assert record.wall_level == adapted.wall_level == 6
        assert record.level_psi == 6
        assert record.anchor_r is None
    else:
        assert record.saddle_source_node is adapted.saddle_source_node is None
        assert record.wall_level is adapted.wall_level is None
        assert (record.level_psi, record.anchor_r, record.anchor_z) == (4, 5, 6)
