"""Position-selected reads in the IMAS package refuse ambiguous containers.

A wall read that takes ``description_2d[0]``, ``limiter.unit[0]`` or
``description_2d[0].vessel`` with no count check selects by position on an
axis whose length nothing has measured. These tests drive the three readers
on synthetic two-description and two-unit containers and require a loud
refusal rather than a silent first-element selection, mirroring the count
guard the DIII-D description author has always carried.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nova.imas import diiid_machine_ids, mast_chain_factory
from nova.imas.machine import Wall
from nova.imas.test_utilities import mark


def _static_ids_stub() -> SimpleNamespace:
    """Return the ids surface every static description object must expose."""

    return SimpleNamespace(
        validate=lambda: None,
        ids_properties=SimpleNamespace(homogeneous_time=0),
    )


def _wall_stub(*, descriptions: int = 1, limiter_units: int = 1) -> SimpleNamespace:
    """Return a synthetic wall ids with the requested container lengths."""

    wall = _static_ids_stub()
    if descriptions != 1:
        wall.description_2d = [object() for _ in range(descriptions)]
    else:
        units = [
            SimpleNamespace(
                name="vessel" if index == 0 else "blade",
                closed=1 if index == 0 else 0,
                outline=SimpleNamespace(
                    r=np.array([1.0, 2.0, 2.0, 1.0, 1.0])
                    if index == 0
                    else np.array([1.4, 1.6]),
                    z=np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
                    if index == 0
                    else np.array([0.0, 0.0]),
                ),
            )
            for index in range(limiter_units)
        ]
        wall.description_2d = [SimpleNamespace(limiter=SimpleNamespace(unit=units))]
    return wall


def _wall_ids_mapping() -> dict[str, SimpleNamespace]:
    """Return the three static description ids the DIII-D validator expects."""

    magnetics = _static_ids_stub()
    magnetics.b_field_pol_probe = []
    magnetics.flux_loop = []
    magnetics.ip = []
    return {
        "wall": _static_ids_stub(),
        "pf_active": _static_ids_stub(),
        "magnetics": magnetics,
    }


class _CachedWall(Wall):
    """A wall whose meshed geometry is already reported cached."""

    def load(self):
        """Report a cache hit without touching the store."""
        return self

    def build(self):
        """Fail the test: a cache hit must never rebuild."""
        raise AssertionError("cache hit rebuilt the wall")


def _diiid_bundle(wall: SimpleNamespace) -> diiid_machine_ids.DiiidMachineIds:
    """Return a validator bundle carrying the given wall ids."""

    ids = _wall_ids_mapping()
    ids["wall"] = wall
    return diiid_machine_ids.DiiidMachineIds(
        ids=ids,
        source_path=Path("unused.nc"),
        source_dd_version="4.1.1",
        dd_version="4.1.1",
        absent=(),
        limiter_repair=SimpleNamespace(
            published_ring_sha256="",
            published_vertex_count=0,
        ),
    )


@mark["imas"]
def test_mast_wall_grid_refuses_a_two_description_wall():
    """A second wall description must raise rather than be ignored."""

    with pytest.raises(ValueError, match="exactly one wall description"):
        mast_chain_factory._wall_grid({"wall": _wall_stub(descriptions=2)}, 3, 3)


@mark["imas"]
def test_mast_wall_grid_preserves_a_two_unit_limiter():
    """A blade remains a separate unit and is excluded from the occupiable mask."""

    grid = mast_chain_factory._wall_grid({"wall": _wall_stub(limiter_units=2)}, 31, 31)

    assert len(grid.wall_units) == 2
    np.testing.assert_array_equal(grid.wall_unit_offsets, [0, 5, 7])
    centre_r = int(np.argmin(np.abs(grid.rg - 1.5)))
    centre_z = int(np.argmin(np.abs(grid.zg)))
    assert not grid.inside_limiter[centre_z, centre_r]


@mark["imas"]
def test_diiid_validate_refuses_a_two_description_wall():
    """The validator's own wall read repeats the description count guard."""

    with pytest.raises(ValueError, match="exactly one wall description"):
        _diiid_bundle(_wall_stub(descriptions=2)).validate()


@mark["imas"]
def test_diiid_validate_accepts_a_two_unit_limiter():
    """The validator reads every unit while retaining first-ring provenance."""

    wall = _wall_stub(limiter_units=2)
    ring = np.column_stack(
        (
            wall.description_2d[0].limiter.unit[0].outline.r,
            wall.description_2d[0].limiter.unit[0].outline.z,
        )
    )
    bundle = _diiid_bundle(wall)
    object.__setattr__(
        bundle.limiter_repair,
        "published_ring_sha256",
        diiid_machine_ids._chain_sha256(ring),
    )
    object.__setattr__(bundle.limiter_repair, "published_vertex_count", len(ring))

    bundle.validate()


@mark["imas"]
def test_machine_wall_vessel_refuses_a_two_description_wall(monkeypatch, tmp_path):
    """The machine-agnostic vessel read refuses a multi-description wall."""

    monkeypatch.setenv("IMAS_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "cache"))
    wall = _CachedWall(pulse=999999, run=1, user="public")
    wall.ids = _wall_stub(descriptions=2)
    with pytest.raises(ValueError, match="exactly one"):
        wall.vessel


@mark["imas"]
def test_machine_wall_vessel_selects_a_single_description(monkeypatch, tmp_path):
    """A single-description wall still serves its vessel to the boundary read."""

    monkeypatch.setenv("IMAS_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "cache"))
    wall = _CachedWall(pulse=999999, run=1, user="public")
    wall.ids = SimpleNamespace(description_2d=[SimpleNamespace(vessel="vessel-node")])
    assert wall.vessel == "vessel-node"
