"""Unit-aware media wall contracts."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nova.equilibrium.wall_mask import material_unit, vessel_unit
from nova.media.poloidal import (
    draw_nulls,
    draw_separatrix_branches,
    draw_wall,
)
from nova.media.sources.frame import MachineGeometry, inside_wall_units
from nova.media.sources.plasma_mesh import clip_to_boundary


def _units():
    vessel = vessel_unit([0.0, 4.0, 4.0, 0.0], [-2.0, -2.0, 2.0, 2.0], name="vessel")
    blade = material_unit([1.0, 1.5, 1.5, 1.0], [-0.5, -0.5, 0.5, 0.5], name="blade")
    open_blade = material_unit([2.0, 3.0], [0.0, 0.0], closed=False, name="blade-tip")
    return vessel, blade, open_blade


def test_draw_wall_keeps_units_separate_and_closes_only_closed_units():
    figure, axes = plt.subplots()
    draw_wall(axes, units=_units())

    assert len(axes.lines) == 3
    np.testing.assert_array_equal(
        axes.lines[0].get_xydata()[0], axes.lines[0].get_xydata()[-1]
    )
    np.testing.assert_array_equal(
        axes.lines[1].get_xydata()[0], axes.lines[1].get_xydata()[-1]
    )
    assert not np.array_equal(
        axes.lines[2].get_xydata()[0], axes.lines[2].get_xydata()[-1]
    )
    assert axes.lines[2].get_linestyle() == "--"
    plt.close(figure)


def test_containment_matches_vessel_minus_material_units():
    units = _units()
    points = np.array([[0.5, 0.0], [1.25, 0.0], [3.5, 0.0], [2.5, 0.0]])

    np.testing.assert_array_equal(
        inside_wall_units(points, units), [True, False, True, False]
    )


def test_machine_geometry_and_mesh_accept_the_unit_collection():
    units = _units()
    geometry = MachineGeometry(limiter=units[0].vertices, wall_units=units)
    assert geometry.wall_units == units
    cells = (
        np.array([[0.2, -0.2], [0.8, -0.2], [0.8, 0.2], [0.2, 0.2]]),
        np.array([[1.1, -0.2], [1.4, -0.2], [1.4, 0.2], [1.1, 0.2]]),
    )
    clipped = clip_to_boundary(cells, units)
    assert len(clipped) == 1


def _padded_branch_set():
    """One closed lobe, one leg, and a padded slot that must never be drawn."""
    zero = np.zeros((4, 2))
    closed = np.stack(
        (
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, -1.0], [0.0, -1.0], [0.0, 0.0]]),
            zero,
            zero,
        )
    )
    open_controls = np.stack(
        (
            np.stack(
                (
                    np.array([[0.0, 0.0], [0.2, -0.5], [0.4, -1.0], [0.5, -1.5]]),
                    zero,
                    zero,
                )
            ),
            np.stack((zero, zero, zero)),
        )
    )
    return {
        "closed_controls_rz": closed,
        "closed_valid": np.array([True, True, False, True]),
        "open_controls_rz": open_controls,
        "open_valid": np.array([[True, False, False], [False, False, False]]),
        "open_branch_valid": np.array([True, False]),
    }


def test_separatrix_branches_draw_the_closed_lobe_and_dashed_legs():
    """A padded branch set draws its lobe and its legs, and never the pad
    slot that carries a cleared mask."""
    figure, axes = plt.subplots()
    tally = draw_separatrix_branches(axes, _padded_branch_set())

    assert tally["closed_drawn"] == 1
    assert tally["open_drawn"] == 1
    assert len(axes.lines) == 2
    lobe = axes.lines[0].get_xydata()
    assert lobe.shape[0] == 2 * 8 + 1
    np.testing.assert_allclose(lobe[0], lobe[-1])
    assert axes.lines[0].get_linestyle() == "-"
    assert axes.lines[1].get_linestyle() == "--"
    leg = axes.lines[1].get_xydata()
    assert leg.shape[0] == 1 * 8 + 1
    assert not np.allclose(leg[0], leg[-1])
    plt.close(figure)


def test_separatrix_branches_skip_a_pad_that_lost_its_mask():
    """An all-zero slot is padding even when the mask says yes."""
    zero = np.zeros((2, 4, 2))
    figure, axes = plt.subplots()
    tally = draw_separatrix_branches(
        axes,
        {
            "closed_controls_rz": zero,
            "closed_valid": np.array([False, True]),
            "open_controls_rz": zero,
            "open_valid": np.zeros((2, 4), dtype=bool),
            "open_branch_valid": np.array([False, False]),
        },
    )
    assert tally == {"closed_drawn": 0, "open_drawn": 0}
    assert len(axes.lines) == 0
    plt.close(figure)


def test_other_qualified_nulls_draw_hollow_beyond_the_wall():
    """A saddle outside the limiter's interior draws hollow and unfilled."""
    figure, axes = plt.subplots()
    tally = draw_nulls(
        axes,
        magnetic_axis=np.array([0.4, 0.0]),
        x_points=np.array([[0.5, 0.0]]),
        other_x_points=np.array([[1.25, 1.23]]),
        contain=_units(),
    )

    assert tally["x_points_drawn"] == 1
    assert tally["other_x_points_drawn"] == 1
    assert len(axes.lines) == 3
    hollow = axes.lines[-1]
    assert hollow.get_markerfacecolor() == "none"
    plt.close(figure)


def test_nulls_use_the_occupiable_region_for_markers():
    figure, axes = plt.subplots()
    tally = draw_nulls(
        axes,
        magnetic_axis=np.array([1.25, 0.0]),
        x_points=np.array([[0.5, 0.0], [1.25, 0.0]]),
        contain=_units(),
    )

    assert tally["x_points_drawn"] == 1
    assert tally["x_points_dropped_outside_wall"] == 1
    assert len(axes.lines) == 1
    plt.close(figure)
