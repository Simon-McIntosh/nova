import numpy as np
import pytest

from nova.frame.coilset import CoilSet


@pytest.fixture(scope="module")
def solved_square_wall():
    """Return a wall whose hexagonal tiling cannot resemble its full outline."""
    coilset = CoilSet(dplasma=-20, tplasma="hex", nwall=2)
    coilset.firstwall.insert(
        [[1.0, 5.0, 5.0, 1.0, 1.0], [1.0, 1.0, 5.0, 5.0, 1.0]],
        Ic=1.0,
    )
    coilset.plasmawall.solve()
    coilset.plasmagrid.solve()
    return coilset


def test_plasmawall_reads_the_full_first_wall(solved_square_wall):
    wall = solved_square_wall
    np.testing.assert_allclose(
        wall.plasmawall.boundary,
        [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0], [1.0, 1.0]],
    )
    assert not np.allclose(
        wall.plasmawall.boundary, wall.aloc["plasma", "poly"][0].boundary
    )


def test_nwall_counts_nodes_per_first_wall_segment(solved_square_wall):
    wall = solved_square_wall
    segment_count = len(wall.plasmawall.boundary) - 1
    assert wall.plasmawall.data.sizes["target"] == wall.nwall * segment_count == 8


def test_plasmagrid_tessellation_uses_the_full_first_wall(solved_square_wall):
    grid = solved_square_wall.plasmagrid.data

    assert grid.triangles.shape == (58, 3)
    assert np.unique(grid.triangles).size == grid.sizes["target"] == 38
    np.testing.assert_array_equal(
        grid.stencil,
        [
            [3, 5, 21, 15, 11, 9, 2],
            [11, 3, 15, 13, 10, 7, 9],
            [13, 15, 16, 14, 12, 10, 11],
            [15, 21, 27, 16, 13, 11, 3],
            [16, 27, 29, 20, 14, 13, 15],
            [21, 22, 28, 27, 15, 3, 5],
            [27, 28, 31, 29, 16, 15, 21],
            [28, 24, 33, 31, 27, 21, 22],
            [29, 31, 32, 30, 20, 16, 27],
            [31, 33, 35, 32, 29, 27, 28],
        ],
    )
