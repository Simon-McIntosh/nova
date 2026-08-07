"""Where the catalog says each flux loop is, and how a copied block is caught.

A flux loop is a fixture on a coil, so the coil it sits nearest is the coil its
name identifies.  That relation holds for every loop in both of the archive's
tables and is what makes a transcription visible without a tolerance: a block of
loops carrying another block's coordinates is not slightly displaced, it is beside
the wrong coil.  The tests here pin the relation on the real catalog, pin the four
positions the repair restores, and pin the join accounting the repair moves --
because the value of the repair is exactly the channels it turns from unusable
into usable, and a count is the only honest statement of that.

The failure mode worth guarding is the opposite one: a repair that fires when it
should not.  Restoring a position writes a sensor pose the machine description
then carries, so a block whose replacement cannot be identified uniquely has to
stay as published and let the join refuse it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely

from nova.catalog.mast_geometry import (
    DEFAULT_LEVEL2_ROOT,
    LoopPlacement,
    component_mount,
    loop_mount,
    physical_snapshot,
    placed_loop_positions,
    shot_loop_placements,
)
from nova.imas.mast_flux_loop_adjudication import join_accounting
from nova.imas.mast_solve_inputs import reconstruction_loop_positions

REPRESENTATIVE_SHOT = 11766
"""Shot the packaged registry's single physical configuration is read from."""

SAMPLED_SHOTS = (11766, 15656, 20019, 24037, 27773, 28770)
"""Shots spanning the campaign the placement relation is checked on.

Both catalogs have to carry the shot, and the level-2 catalog is the sparser of
the two, so these are drawn from its own listing rather than from the level-1
store the response fits read.
"""

COPIED_CHANNELS = ("FL_P4L_1", "FL_P4L_2", "FL_P4L_3", "FL_P4L_4")
"""The level-2 block that carries another block's coordinates."""

RESTORED_POSITIONS = (
    (1.59840, -1.04443),
    (1.59840, -1.06943),
    (1.40050, -1.15943),
    (1.40050, -1.14403),
)
"""What the reconstruction puts the four P4-lower loops at, in name order."""

_needs_store = pytest.mark.skipif(
    not Path(DEFAULT_LEVEL2_ROOT).is_dir(),
    reason=f"MAST level-2 catalog not present at {DEFAULT_LEVEL2_ROOT}",
)


# --- what a loop's name says about where it is ------------------------------


@pytest.mark.parametrize(
    ("name", "mount"),
    [
        ("FL_P4L_1", "p4_lower"),
        ("FL_P2U_3", "p2_upper"),
        ("FL_P6L_2", "p6_lower"),
        ("FL_CC01", "sol"),
        ("FL_CC010", "sol"),
    ],
)
def test_a_loop_name_names_the_coil_it_is_mounted_on(name, mount):
    """The name carries the mounting, which is what makes a misplacement visible."""

    assert loop_mount(name) == mount


def test_an_unreadable_loop_name_is_refused_rather_than_left_unmounted():
    """A name nobody can mount is a catalog change, not a loop to place by proximity."""

    with pytest.raises(ValueError, match="flux-loop name"):
        loop_mount("FL_WHAT_1")


@pytest.mark.parametrize(
    ("component", "mount"),
    [
        ("p2_inner_upper", "p2_upper"),
        ("p2_outer_lower", "p2_lower"),
        ("p4_lower", "p4_lower"),
        ("sol", "sol"),
    ],
)
def test_a_coil_set_gathers_the_packs_a_loop_encircles(component, mount):
    """A loop around P2 upper encircles both its packs, so the two share a mounting."""

    assert component_mount(component) == mount


# --- the placement relation on the real catalog -----------------------------


@_needs_store
@pytest.mark.parametrize("shot", SAMPLED_SHOTS)
def test_only_the_copied_block_sits_beside_the_wrong_coil(shot):
    """The relation is the evidence, so it must hold everywhere else it is applied."""

    placements = shot_loop_placements(shot)
    misplaced = [row.name for row in placements if row.published_mount != row.mount]

    assert sorted(misplaced) == sorted(COPIED_CHANNELS)
    assert all(row.published_mount == "p3_lower" for row in placements if row.restored)


@_needs_store
@pytest.mark.parametrize("shot", SAMPLED_SHOTS)
def test_the_copied_block_is_restored_to_the_coil_it_is_named_for(shot):
    """The four positions are the deliverable: they turn two channels usable."""

    restored = {
        row.name: (round(row.r, 5), round(row.z, 5))
        for row in shot_loop_placements(shot)
        if row.restored
    }

    assert restored == dict(zip(COPIED_CHANNELS, RESTORED_POSITIONS, strict=True))


@_needs_store
@pytest.mark.parametrize("shot", SAMPLED_SHOTS)
def test_every_other_loop_is_served_exactly_as_the_catalog_published_it(shot):
    """A repair that moved a loop it was not asked about would be silent corruption."""

    for row in shot_loop_placements(shot):
        if row.restored:
            continue
        assert (row.r, row.z) == (row.published_r, row.published_z)


@_needs_store
def test_the_repair_removes_the_duplicated_described_positions():
    """Eight loops on four points is what a copied block looks like in the payload."""

    snapshot = physical_snapshot(REPRESENTATIVE_SHOT)
    positions = [(row[0], row[1]) for row in snapshot["magnetics"]["flux_loops"]]

    assert len(positions) == 44
    assert len(set(positions)) == 44


# --- what the repair is worth, in channels ----------------------------------


@pytest.fixture(scope="module")
def accounting():
    """Return every channel's join against the repaired description."""

    return join_accounting(
        physical_snapshot(REPRESENTATIVE_SHOT),
        reconstruction_loop_positions(REPRESENTATIVE_SHOT),
    )


@_needs_store
def test_the_join_serves_every_channel_the_description_has_a_loop_for(accounting):
    """The channel count is the measurable worth of the repair, so it is pinned."""

    served = [row for row in accounting if row.served]

    assert len(accounting) == 46
    assert len(served) == 43
    assert len({row.described_index for row in served}) == 43


@_needs_store
def test_the_repaired_block_is_what_the_join_gained(accounting):
    """Two of the four publish on every shot, which is the rank this unlocks."""

    served = {row.channel for row in accounting if row.served}

    assert {"fl_p4l_1", "fl_p4l_2", "fl_p4l_3", "fl_p4l_4"} <= served


@_needs_store
def test_every_refused_channel_names_the_condition_that_refused_it(accounting):
    """A refusal is only actionable if it says what would lift it."""

    refused = {row.channel: row for row in accounting if not row.served}

    assert sorted(refused) == ["fl_p5l_3", "fl_p6u_1", "fl_p6u_2"]
    assert refused["fl_p5l_3"].described_on_mount == 4
    assert "75 mm" in refused["fl_p5l_3"].cause
    for channel in ("fl_p6u_1", "fl_p6u_2"):
        assert refused[channel].described_on_mount == 0
        assert "no flux loop on p6_upper" in refused[channel].cause


@_needs_store
def test_no_channel_claims_a_described_loop_another_channel_holds(accounting):
    """Loops of a pair sit millimetres apart, so many-to-one would hide a sensor."""

    claimed = [row.described_index for row in accounting if row.served]

    assert len(claimed) == len(set(claimed))


# --- the repair refuses rather than guesses ---------------------------------


def _outlines(mounts: dict[str, tuple[float, float]]) -> dict[str, str]:
    """Return one unit-square outline per named component, centred as asked."""

    return {
        name: shapely.box(r - 0.05, z - 0.05, r + 0.05, z + 0.05).wkb_hex
        for name, (r, z) in mounts.items()
    }


def test_a_block_with_no_free_reconstruction_row_stays_as_published():
    """Nothing identifies the replacement, so the join must refuse the channel."""

    outlines = _outlines({"p3_lower": (1.0, -1.0), "p4_lower": (2.0, -1.0)})
    placements = placed_loop_positions(
        ("FL_P3L_1", "FL_P4L_1"),
        np.array([1.0, 1.0]),
        np.array([-1.0, -1.0]),
        outlines,
        np.array([[1.0, -1.0]]),
    )

    assert [row.restored for row in placements] == [False, False]
    assert [(row.r, row.z) for row in placements] == [(1.0, -1.0), (1.0, -1.0)]


def test_a_block_with_more_free_rows_than_members_stays_as_published():
    """Two candidates for one loop is a layout this rule cannot read, not a choice."""

    outlines = _outlines({"p3_lower": (1.0, -1.0), "p4_lower": (2.0, -1.0)})
    placements = placed_loop_positions(
        ("FL_P3L_1", "FL_P4L_1"),
        np.array([1.0, 1.0]),
        np.array([-1.0, -1.0]),
        outlines,
        np.array([[1.0, -1.0], [2.0, -1.0], [2.0, -1.02]]),
    )

    assert [row.restored for row in placements] == [False, False]


def test_a_block_whose_replacement_is_unique_is_restored():
    """One free row beside the named coil for one member is an identification."""

    outlines = _outlines({"p3_lower": (1.0, -1.0), "p4_lower": (2.0, -1.0)})
    placements = placed_loop_positions(
        ("FL_P3L_1", "FL_P4L_1"),
        np.array([1.0, 1.0]),
        np.array([-1.0, -1.0]),
        outlines,
        np.array([[1.0, -1.0], [2.0, -1.01]]),
    )

    assert [row.restored for row in placements] == [False, True]
    assert placements[1].r == pytest.approx(2.0)
    assert placements[1].z == pytest.approx(-1.01)


def test_a_row_a_correctly_mounted_loop_already_sits_on_is_not_reused():
    """Two loops at one point is the fault being repaired, not an acceptable repair."""

    outlines = _outlines({"p4_lower": (2.0, -1.0), "p4_upper": (2.0, 1.0)})
    placements = placed_loop_positions(
        ("FL_P4L_1", "FL_P4U_1"),
        np.array([2.0, 2.0]),
        np.array([1.0, 1.0]),
        outlines,
        np.array([[2.0, 1.0]]),
    )

    assert [row.restored for row in placements] == [False, False]


def test_a_placement_reports_the_mounting_the_published_position_had():
    """The refusal has to name what was wrong, so both mountings travel with the row."""

    outlines = _outlines({"p3_lower": (1.0, -1.0), "p4_lower": (2.0, -1.0)})
    placements = placed_loop_positions(
        ("FL_P4L_1",),
        np.array([1.0]),
        np.array([-1.0]),
        outlines,
        np.array([[2.0, -1.01]]),
    )
    row = placements[0]

    assert isinstance(row, LoopPlacement)
    assert row.mount == "p4_lower"
    assert row.published_mount == "p3_lower"
    assert row.restored


def test_a_catalog_with_no_active_component_places_nothing_by_proximity():
    """With no outline the mounting is unknown, so a loop is served as published."""

    placements = placed_loop_positions(
        ("FL_P4L_1",),
        np.array([1.0]),
        np.array([-1.0]),
        {},
        np.array([[2.0, -1.01]]),
    )

    assert placements[0].published_mount == ""
    assert not placements[0].restored
