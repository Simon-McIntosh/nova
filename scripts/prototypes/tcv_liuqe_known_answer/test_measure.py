"""A limited slice must be scored, not dropped for want of a saddle.

LIUQE's limited slices store no X-point, so nova's census admits no saddle on
their field. The driver must still return a reading, because the limited slices
are half of the known-answer comparison. This test runs the driver on TCV 77549
slice 34, which LIUQE marks limited, and requires the read to come back
``limited`` rather than ``unqualified``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.prototypes.tcv_liuqe_known_answer import measure

LIUQE_STORE = Path(
    os.environ.get("TCV_LIUQE_STORE", "/home/ITER/mcintos/public/imasdb/tcv/3/77549/1")
)
WALL_STORE = Path(
    os.environ.get(
        "TCV_WALL_STORE", "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/TCV/77549"
    )
)
LIMITED_SLICE = 34


@pytest.mark.skipif(
    not (LIUQE_STORE / "equilibrium.h5").exists(),
    reason="the TCV LIUQE reconstruction is not present on this machine",
)
def test_limited_slice_reads_without_a_saddle():
    import imas

    db = imas.DBEntry(f"imas:hdf5?path={LIUQE_STORE}", "r", dd_version="3.41.0")
    try:
        view = measure.slice_view(db.get("equilibrium"), LIMITED_SLICE)
    finally:
        db.close()
    assert not view["xpoints"], "the chosen slice must be a limited one"
    reading = measure.measure_slice(view, measure.read_wall(str(WALL_STORE)), (16,))
    assert reading["nova_label"] == "limited", reading["attempts"]
    assert reading["nova_saddles"] == 0
