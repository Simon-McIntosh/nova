import pytest
import shapely.geometry
import shapely

from nova.frame.shellgrid import ShellGrid


def test_circle_rdp_features():
    x, z = shapely.geometry.Point(0, 0).buffer(3.5).boundary.xy
    shellgrid = ShellGrid(x, z, 0, 0.1, eps=1e-3)
    assert len(shellgrid.rdp) == 33


def test_circle_rdp_features_upscale():
    x, z = shapely.geometry.Point(0, 0).buffer(30.5).boundary.xy
    shellgrid = ShellGrid(x, z, 0, 0.1)
    assert len(shellgrid.rdp) == 33


def test_min_ndiv():
    shellgrid = ShellGrid([1, 3], [3, 5], -1, 0.1, ndiv=5)
    assert len(shellgrid.ldiv) == 5


def test_negative_ndiv():
    shellgrid = ShellGrid([1, 3], [3, 5], -5, 0.1, ndiv=1)
    assert len(shellgrid.dataframe) == 5


def test_bump_sharp_corners():
    shellgrid = ShellGrid([1, 1.5, 2, 2, 4, 4],
                          [0, 0.1, 0, 1, -1, 0], -5, 0.1, eps=1e-3)
    assert len(shellgrid.dataframe) == 5


def test_frame_columns():
    shellgrid = ShellGrid([1, 1.5], [0, 0.1], -3, 0.1)
    assert all([attr in shellgrid.dataframe for attr in [
        'x', 'z', 'dl', 'dt', 'dx', 'dz', 'rms', 'area', 'section', 'poly']])


def test_unequal_input_error():
    with pytest.raises(IndexError):
        ShellGrid([1, 1.5], [0, 0.1, 3], -3, 0.1)


def test_zero_thickness_input_error():
    with pytest.raises(ValueError):
        ShellGrid([1, 1.5], [0, 3], -3, 0)


def test_negative_thickness_input_error():
    with pytest.raises(ValueError):
        ShellGrid([1, 1.5], [0, 0.1], -3, -7.2)


def test_thickness_max_space():
    shellgrid = ShellGrid([0, 10], [0, 0], -5, 2.5)
    assert len(shellgrid.dataframe) == 4


def test_thickness_max_space_subsegment():
    shellgrid = ShellGrid([0, 10], [0, 0], -5, 2.5, delta=-100)
    subsegment = next(shellgrid.divide())
    assert len(subsegment.dataframe) == 1


def test_subsegment_length():
    shellgrid = ShellGrid([0, 10], [0, 0], -2, 0.01, delta=-7)
    subsegment = next(shellgrid.divide())
    assert len(subsegment.dataframe) == 7


if __name__ == '__main__':

    pytest.main([__file__])
