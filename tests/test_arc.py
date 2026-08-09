import pytest

import numpy as np

from nova.biot.arc import Arc
from nova.biot.biotframe import Source, Target
from nova.frame.coilset import CoilSet


@pytest.fixture
def source():
    coilset = CoilSet()
    coilset.winding.insert(
        np.array([[5, 0, 3.2], [0, 5, 3.2], [-5, 0, 3.2]]),
        {"c": (0, 0, 0.5)},
        minimum_arc_nodes=3,
    )
    coilset.winding.insert(
        np.array([[5, 0, 3.2], [0, 0, -1.8], [-5, 0, 3.2]]),
        {"c": (0, 0, 0.5)},
        minimum_arc_nodes=3,
    )
    return Source(coilset.subframe)


@pytest.fixture
def arc(source):
    return Arc(source, Target({"x": np.linspace(2, 5, 7), "z": -0.3}))


def test_space_axes_shape(source):
    assert np.shape(source.space.coordinate_axes) == (2, 3, 3)


def test_terminal_points(source):
    assert np.allclose(source.start_point, [5, 0, 3.2])
    assert np.allclose(source.end_point, [-5, 0, 3.2])


def test_local_start_points(source):
    start_point = source.space.start_point
    assert np.allclose(source.space.to_local(start_point)[:, 1], 0)


def test_coordinate_transform_roundtrip(source):
    assert np.allclose(
        source.start_point,
        source.space.to_global(source.space.to_local(source.start_point)),
    )


def test_coordinate_transform_local_plane(source):
    assert np.isclose(
        source.space.to_local(source.start_point)[0, 2],
        source.space.to_local(source.end_point)[0, 2],
    )


def test_coordinate_transform_local_axis(source):
    assert np.allclose(source.space._rotate_to_local(source.axis)[1], [0, 0, 1])


@pytest.mark.parametrize("local_radius", [0.0, np.nan])
def test_arc_rejects_nonpositive_or_nonfinite_source_radius(source, local_radius):
    index = source.index[0]
    source.loc[index, "x1"] = source.loc[index, "x0"] + local_radius
    source.loc[index, "y1"] = source.loc[index, "y0"]
    source.loc[index, "z1"] = source.loc[index, "z0"]
    target = Target({"x": [2.0], "z": [-0.3]})
    with pytest.raises(ValueError, match="radius must be finite and positive"):
        Arc(source, target)


@pytest.mark.parametrize("dl", [0.0, -1.0, np.nan, np.inf])
def test_arc_rejects_nonpositive_or_nonfinite_dl(source, dl):
    source.loc[source.index[0], "dl"] = dl
    target = Target({"x": [2.0], "z": [-0.3]})
    with pytest.raises(ValueError, match="dl must be finite and positive"):
        Arc(source, target)


def test_arc_rejects_coincident_stored_endpoints(source):
    index = source.index[0]
    source.loc[index, ["x2", "y2", "z2"]] = source.loc[
        index, ["x1", "y1", "z1"]
    ].to_numpy()
    target = Target({"x": [2.0], "z": [-0.3]})
    with pytest.raises(ValueError, match="coincident endpoints.*ambiguous topology"):
        Arc(source, target)


if __name__ == "__main__":
    pytest.main([__file__])
