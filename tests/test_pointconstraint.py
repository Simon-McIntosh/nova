from itertools import product

import matplotlib.pylab
import numpy as np
import pytest
import xarray

from nova.imas.pulsedesign import Constraint, Control
from nova.imas.test_utilities import mark


@pytest.fixture
def data():
    data = xarray.Dataset()
    data["time"] = [0]
    data["point"] = ["r", "z"]
    data["index"] = [0, 1]

    data["boundary_type"] = "time", [1]
    data["geometric_axis"] = ("time", "point"), [[5.8, 0.3]]
    data["x_point"] = ("time", "point"), [[5, -2.5]]
    data["minor_radius"] = "time", [1.0]
    data["elongation_upper"] = "time", [0.1]
    data["elongation_lower"] = "time", [0.1]
    data["triangularity"] = "time", [0.4]
    data["elongation"] = "time", [2.1]
    data["squareness_upper_outer"] = "time", [0.1]
    data["squareness_upper_inner"] = "time", [0.1]
    data["squareness_lower_inner"] = "time", [0.1]
    data["squareness_lower_outer"] = "time", [0.1]

    data["strike_point"] = ("time", "point", "index"), [[[4.8, -3.5], [5.5, -3.5]]]
    return data.copy(deep=True)


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


@mark["wall"]
@pytest.mark.parametrize("strike,square", product([True, False], [True, False]))
def test_control_plot(data, strike, square):
    control = Control(data=data, strike=strike, square=square)
    control.itime = 0
    with matplotlib.pylab.ioff():
        control.plot()


@mark["wall"]
def test_control_normal(data):
    control = Control(data=data, strike=True, square=True)
    control.itime = 0
    assert np.allclose(np.linalg.norm(control.normal, axis=1), 1)


@mark["wall"]
def test_control_midpoints(data):
    control = Control(data=data, strike=True, square=True)
    data["boundary_type"][0] = 1
    control.itime = 0
    control.fit()
    index = control.point_index
    control_midpoints = control.control_midpoints(index)
    gaps = control.point_gap(index)
    approximate_gaps = np.linalg.norm(
        control.control_points[index] - control_midpoints, axis=1
    )
    assert np.allclose(gaps, approximate_gaps, 5e-3, 5e-3)


@mark["wall"]
@pytest.mark.parametrize(
    "strike,square,boundary_type", product([True, False], [True, False], [0, 1])
)
def test_control_fit(data, strike, square, boundary_type):
    data["boundary_type"][0] = boundary_type
    control = Control(data=data)
    control.itime = 0
    control.fit()
    index = control.point_index
    gaps = control.point_gap(index)
    if boundary_type == 0:  # limiter
        mingap = 0
    else:
        mingap = control["minimum_gap"]
    assert np.all([gap >= mingap - 1e-8 for gap in gaps])


if __name__ == "__main__":
    pytest.main([__file__])
