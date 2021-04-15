import pytest
import shapely.geometry
import shapely

from nova.electromagnetic.shellgrid import ShellGrid


def test_circle_rdp_features():
    x, z = shapely.geometry.Point(0, 0).buffer(3.5).boundary.xy
    shellgrid = ShellGrid([x, z], 0, 0.1)
    assert len(shellgrid.unit_length) == 33


def test_circle_rdp_features_upscale():
    x, z = shapely.geometry.Point(0, 0).buffer(30.5).boundary.xy
    shellgrid = ShellGrid([x, z], 0, 0.1)
    assert len(shellgrid.unit_length) == 33


def test_min_ndiv():
    shellgrid = ShellGrid([[1, 3], [3, 5]], 0, 0.1, ndiv=5)
    assert len(shellgrid.unit_length) == 5


def test_negative_ndiv():
    shellgrid = ShellGrid([[1, 3], [3, 5]], -5, 0.1, ndiv=1)
    assert len(shellgrid.unit_length) == 5


def test_bump_sharp_corners():
    shellgrid = ShellGrid([[1, 1.5, 2, 2, 4, 4],
                           [0, 0.1, 0, 1, -1, 0]], -5, eps=1e-3)
    assert len(shellgrid.unit_length) == 9


def test_frame_x_z():
    shellgrid = ShellGrid([[1, 1.5], [0, 0.1]], -3)
    assert shellgrid.segment.frame.columns.to_list() == ['x', 'z']

def test_frame_x_z_dt():
    shellgrid = ShellGrid([[1, 1.5], [0, 0.1]], -3, 0.1)
    assert shellgrid.segment.frame.columns.to_list() == ['x', 'z', 'dt']


def test_frame_x_z_dt_rho():
    shellgrid = ShellGrid([[1, 1.5], [0, 0.1]], -3, 0.1, 1e-6)
    assert shellgrid.segment.frame.columns.to_list() == ['x', 'z', 'dt', 'rho']

if __name__ == '__main__':

    pytest.main([__file__])