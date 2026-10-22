"""The section quadrature integrates a shaped section however its outline bends."""

import numpy as np
import pytest
from shapely import contains_xy
from shapely.geometry import Polygon

from nova.equilibrium.wall_mask import polygon_area
from nova.imas.mast_passive_inductance import PassiveError, section_quadrature

#: a notched L whose first corner cannot see across its own notch
L_SHAPE = np.array(
    [[2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0], [0.0, 0.0], [2.0, 0.0]]
)

#: an L-shaped ribbon 0.05 m thick, bent through a right angle
BENT_STRIP = np.array(
    [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0 + 0.05],
        [0.05, 0.05],
        [0.05, 1.5],
        [0.0, 1.5],
    ]
)

SECTIONS = (L_SHAPE, BENT_STRIP)


def _first_moments(vertices):
    """Exact ``(area, integral r dA, integral z dA)`` of a simple polygon."""

    r, z = vertices[:, 0], vertices[:, 1]
    next_r, next_z = np.roll(r, -1), np.roll(z, -1)
    cross = r * next_z - next_r * z
    sign = -1.0 if cross.sum() < 0.0 else 1.0
    return (
        sign * 0.5 * float(cross.sum()),
        sign * float(((r + next_r) * cross).sum()) / 6.0,
        sign * float(((z + next_z) * cross).sum()) / 6.0,
    )


def test_every_point_lies_inside_and_the_weights_sum_to_the_area():
    for vertices in SECTIONS:
        point_r, point_z, weight = section_quadrature(vertices)

        outline = Polygon(vertices)
        assert np.all(contains_xy(outline, point_r, point_z))
        area = polygon_area(vertices[:, 0], vertices[:, 1])
        assert float(weight.sum()) == pytest.approx(area, rel=1e-12)


def test_a_linear_field_integrates_exactly():
    for vertices in SECTIONS:
        point_r, point_z, weight = section_quadrature(vertices)
        area, moment_r, moment_z = _first_moments(vertices)

        assert float(np.dot(weight, point_r)) == pytest.approx(moment_r, rel=1e-12)
        assert float(np.dot(weight, point_z)) == pytest.approx(moment_z, rel=1e-12)
        assert float(weight.sum()) == pytest.approx(area, rel=1e-12)


def test_a_section_enclosing_no_area_is_refused():
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    with pytest.raises(PassiveError, match="encloses no area"):
        section_quadrature(collinear)