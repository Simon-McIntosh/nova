import matplotlib.pylab
import numpy as np
import pytest

from nova.imas.pulsedesign import Constraint, Control


def test_constraint_point_index():
    constraint = Constraint(np.ones((8, 2)))
    constraint.poloidal_flux = 4, range(4)
    constraint.radial_field = 0, [0, 2]
    constraint.vertical_field = 0, [1, 3]
    constraint.radial_field = 0, [3]
    assert np.allclose(constraint.point_index, range(8))
    assert np.allclose(constraint.index("null"), 3)
    assert np.allclose(constraint.index("radial"), [0, 2])
    assert np.allclose(constraint.index("vertical"), 1)
    assert np.allclose(constraint.index("br"), [0, 2, 3])
    assert np.allclose(constraint.index("bz"), [1, 3])


def test_null_plot():
    constraint = Constraint()
    with matplotlib.pylab.ioff():
        constraint.plot()
    assert constraint.mpl_axes._axes is None


def test_plot_constraint():
    theta = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    points = np.c_[np.cos(theta), np.sin(theta)]
    constraint = Constraint(points)
    constraint.poloidal_flux = 4, range(4)
    constraint.radial_field = 0, [0, 2]
    constraint.vertical_field = 0, [1, 3]
    constraint.radial_field = 0, [3]
    with matplotlib.pylab.ioff():
        constraint.plot()


def test_flux_update():
    constraint = Constraint(np.ones((3, 2)))
    constraint.poloidal_flux = 3.2
    constraint.poloidal_flux = 1.3, range(1, 3)
    constraint.radial_field = 2.2, [0, 2]
    constraint.vertical_field = [-4.6, 5.5], [1, 2]
    assert np.allclose(constraint.poloidal_flux, [3.2, 1.3, 1.3])
    assert np.allclose(constraint.radial_field, [2.2, 2.2])
    assert np.allclose(constraint.vertical_field, [-4.6, 5.5])


def test_point_index_error():
    constraint = Constraint(np.ones((3, 2)))
    with pytest.raises(IndexError):
        constraint.radial_field = 3.3, [3]


def test_pds_control():
    control = Control()

    control.data

    # control.plot()


# test_pds_control()
# assert False

if __name__ == "__main__":
    pytest.main([__file__])
