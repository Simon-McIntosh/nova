"""Typed wall-unit collection contracts."""

import numpy as np
import pytest

from nova.equilibrium.wall_mask import (
    WallUnit,
    pack_wall_units,
    wall_units_from_ids,
)
from nova.imas.diiid_machine_ids import write_wall_units
from nova.imas.mast_geometry import _author_wall as author_mast_wall
from nova.imas.test_utilities import mark


def test_packed_units_keep_offsets_without_bridge_edges():
    vessel = WallUnit(
        r=[1.0, 2.0, 2.0, 1.0, 1.0],
        z=[-1.0, -1.0, 1.0, 1.0, -1.0],
        kind="vessel",
        closed=True,
        name="vessel",
    )
    blade = WallUnit(
        r=[1.4, 1.6],
        z=[0.0, 0.0],
        kind="material",
        closed=False,
        name="blade",
    )

    vertices, offsets = pack_wall_units((vessel, blade))

    np.testing.assert_array_equal(offsets, [0, 5, 7])
    np.testing.assert_array_equal(vertices[:5], vessel.vertices)
    np.testing.assert_array_equal(vertices[5:], blade.vertices)
    assert not np.array_equal(vertices[4], vertices[5])


def test_open_vessel_unit_is_rejected():
    with pytest.raises(ValueError, match="vessel WallUnit must be closed"):
        WallUnit(r=[1.0, 2.0], z=[0.0, 0.0], kind="vessel", closed=False)


@mark["imas"]
def test_mast_writer_preserves_a_typed_unit_collection():
    import imas

    units = (
        WallUnit(
            r=[1.0, 2.0, 2.0, 1.0, 1.0],
            z=[-1.0, -1.0, 1.0, 1.0, -1.0],
            kind="vessel",
            closed=True,
            name="vessel",
        ),
        WallUnit(
            r=[1.4, 1.6],
            z=[0.0, 0.0],
            kind="material",
            closed=False,
            name="blade",
        ),
    )

    wall = author_mast_wall(imas.IDSFactory("4.1.1"), {"limiter_units": units})
    restored = wall_units_from_ids(wall)

    assert tuple(unit.name for unit in restored) == ("vessel", "blade")
    assert tuple(unit.closed for unit in restored) == (True, False)
    for source, target in zip(units, restored, strict=True):
        np.testing.assert_array_equal(target.vertices, source.vertices)


@mark["imas"]
def test_real_multi_unit_descriptions_and_jt60sa_round_trip():
    import imas

    sources = (
        (
            "/home/ITER/mcintos/public/imasdb/jt-60sa_md",
            "4.1.0",
            (89, 6, 6, 10, 8),
            (True, False, False, False, False),
        ),
        (
            "/home/ITER/mcintos/public/imasdb/iter_md/3/116000/2",
            "3.37.0",
            (19, 37),
            (False, False),
        ),
    )
    loaded = []
    for path, dd_version, vertex_counts, closure in sources:
        with imas.DBEntry(
            f"imas:hdf5?path={path}", "r", dd_version=dd_version
        ) as entry:
            units = wall_units_from_ids(
                entry.get("wall", 0, lazy=False, autoconvert=False)
            )
        assert tuple(unit.r.size for unit in units) == vertex_counts
        assert tuple(unit.closed for unit in units) == closure
        loaded.append(units)

    factory = imas.IDSFactory("4.1.0")
    target = factory.new("wall")
    target.description_2d.resize(1)
    write_wall_units(target.description_2d[0], loaded[0])
    round_tripped = wall_units_from_ids(target)

    assert len(round_tripped) == 5
    for source, restored in zip(loaded[0], round_tripped, strict=True):
        assert restored.name == source.name
        assert restored.closed == source.closed
        assert restored.kind == source.kind
        np.testing.assert_array_equal(restored.r, source.r)
        np.testing.assert_array_equal(restored.z, source.z)
