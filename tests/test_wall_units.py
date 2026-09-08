"""Typed wall-unit collection contracts."""

import numpy as np
import pytest

from nova.equilibrium.wall_mask import WallUnit, pack_wall_units


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
