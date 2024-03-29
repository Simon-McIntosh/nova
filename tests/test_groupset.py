from itertools import product
import pytest

import numpy as np

from nova.biot.biotframe import Source
from nova.geometry.polyline import PolyLine


@pytest.fixture
def multiline():
    points = np.array([[-2, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]], float)
    polyline = PolyLine(points, minimum_arc_nodes=4)
    source = Source(polyline.path_geometry)
    source.set_target(1)
    return source


@pytest.fixture
def multiarc():
    points = np.array(
        [[-1, 2, 0], [-1, 1, -1], [-1, 0, 0], [0, 1, 0], [1, 0, 0]], float
    )
    polyline = PolyLine(points, minimum_arc_nodes=3)
    source = Source(polyline.path_geometry)
    source.set_target(1)
    return source


@pytest.mark.parametrize("source, axis", product(["multiline", "multiarc"], ["n", "a"]))
def test_source_axes(source, axis, request):
    source = request.getfixturevalue(source)
    vector = {"n": [-1, 0, 0], "a": [0, 0, 1]}[axis]
    points = source.stack(*[f"{axis}{coord}" for coord in "xyz"])
    local_points = source.space._rotate_to_local(points[0])
    global_points = source.space._rotate_to_global(local_points)
    assert np.allclose(local_points, np.tile(vector, (1, len(points))))
    assert np.allclose(points, global_points)


def test_space_plot(multiline):
    space = multiline.space
    with space.test_plot():
        space.plot()


def test_local_arc_start_point_theta(multiarc):
    start_point = multiarc.space.to_local(multiarc.space.start_point)
    theta = np.arctan2(start_point[:, 1], start_point[:, 0])
    assert np.allclose(theta, 0)


@pytest.mark.parametrize(
    "source, point",
    [
        ("multiline", [[-2, 0, 0], [-1, 0, 0], [0, 1, 0]]),
        ("multiarc", [[-1, 1, -1], [0, 1, 0]]),
    ],
)
def test_space_intermediate_point(source, point, request):
    space = request.getfixturevalue(source).space
    if source == "multiline":  # start point + normal
        point += space.normal
    assert np.allclose(space.intermediate_point, point)


if __name__ == "__main__":
    pytest.main([__file__])
