import pytest

import matplotlib.pylab
import numpy as np

from nova.frame.coilset import CoilSet


def test_grid_shape():
    coilset = CoilSet(dcoil=-5, tcoil="hex")
    coilset.coil.insert(3, 0, 0.1, 0.1)
    coilset.grid.solve(10)
    assert coilset.grid.shape == (
        coilset.grid.data.dims["x"],
        coilset.grid.data.dims["z"],
    )


def test_grid_shaped_array():
    coilset = CoilSet(dcoil=-5, tcoil="hex")
    coilset.coil.insert(3, 0, 0.1, 0.5)
    coilset.grid.solve(9)
    assert coilset.grid.shape == coilset.grid.psi_.shape


def test_grid_shaped_array_address():
    coilset = CoilSet(dcoil=-5, tcoil="hex")
    coilset.coil.insert(5, -2, 0.7, 0.5)
    coilset.grid.solve(5)
    psi_ = coilset.grid.psi_
    coilset.sloc["Ic"] = 10
    assert psi_.ctypes.data == coilset.grid.psi_.ctypes.data


def test_point_shaped_array():
    coilset = CoilSet(dcoil=-5, tcoil="hex")
    coilset.coil.insert(5, -2, 0.7, 0.5, Ic=10)
    coilset.point.solve(((1, 2), (4, 5), (7, 3)))
    assert len(coilset.point.shape) == 1


def test_point_shaped_array_address():
    coilset = CoilSet(dcoil=-5, tcoil="hex")
    coilset.coil.insert(5, -2, 0.7, 0.5, Ic=-10)
    coilset.point.solve(((1, 12), (4, 5), (7, -3)))
    assert coilset.point.psi.ctypes.data == coilset.point.psi_.ctypes.data


def test_nturn_hash_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5)
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    nturn_hash = coilset.subframe.version["nturn"]
    coilset.plasma.separatrix = dict(o=[5, 1, 2.5])
    assert coilset.subframe.version["nturn"] != nturn_hash


def test_nturn_Psi_update():
    coilset = CoilSet(dcoil=-5, dplasma=-15, tcoil="hex", tplasma="hex")
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()
    Psi = coilset.plasmagrid.data["Psi"].values.copy()
    coilset.plasma.separatrix = dict(o=[5, 1, 2.5])
    coilset.plasmagrid.update_turns("Psi")
    assert np.not_equal(coilset.plasmagrid.data["Psi"].values, Psi).all()


def test_nturn_skip_Psi_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5, tcoil="hex", tplasma="hex")
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()
    coilset.sloc["Ic"] = 1
    psi_hash = coilset.aloc_hash["nturn"]
    psi = coilset.plasmagrid.psi
    coilset.plasma.separatrix = dict(o=[5, 1, 2.5])
    coilset.plasmagrid.version["Psi"] = psi_hash  # skip update
    assert np.allclose(coilset.plasmagrid.psi, psi)


def test_nturn_current_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5, tcoil="hex", tplasma="hex")
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()
    coilset.sloc["Ic"] = 1e6
    psi = coilset.plasmagrid.psi.copy()
    coilset.sloc["Ic"] = 2e6
    assert np.not_equal(coilset.plasmagrid.psi, psi).all()


def test_nturn_skip_current_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5, tcoil="hex", tplasma="hex")
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()
    coilset.sloc["Ic"] = 1
    current_hash = coilset.aloc_hash["Ic"]
    psi = coilset.plasmagrid.psi
    coilset.sloc["Ic"] = 2
    coilset.plasmagrid.version["psi"] = current_hash  # skip updated
    assert np.allclose(coilset.plasmagrid.psi, psi)


def test_ngap_zero():
    coilset = CoilSet(ngap=10, mingap=0, maxgap=5)
    np.allclose(coilset.plasmagap.nodes, np.linspace(0, 5, 10))


def test_ngap_positive():
    coilset = CoilSet(ngap=10, mingap=0.5, maxgap=5)
    assert np.allclose(coilset.plasmagap.nodes, np.geomspace(0.5, 5, 10))


def test_ngap_negative():
    coilset = CoilSet(mingap=-0.5, ngap=10)
    with pytest.raises(ValueError):
        coilset.plasmagap.nodes


def test_ngap_zero_negative_gap():
    coilset = CoilSet(ngap=-3.4, mingap=4.3, maxgap=5)
    assert coilset.plasmagap.node_number == int(4.3 / 3.4) + 1
    assert coilset.plasmagap.mingap == 0


def test_ngap_zero_positive_float_gap():
    coilset = CoilSet(dplasma=-4, tplasma="hex", ngap=3.4)
    with pytest.raises(IndexError):
        coilset.plasmagap


def test_plot_plasma_gaps():
    coilset = CoilSet(dplasma=-4, tplasma="hex", ngap=3, mingap=0, maxgap=5)
    coilset.firstwall.insert(dict(o=[5, 1, 3]))
    theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    coilset.plasmagap.solve(np.c_[5 + 3 * np.cos(theta), 1 + 3 * np.sin(theta)], theta)
    with matplotlib.pylab.ioff():
        coilset.plasmagap.plot()


def test_plasmagap_matrix():
    coilset = CoilSet(dplasma=-4, tplasma="hex", ngap=3, mingap=0, maxgap=5)
    coilset.firstwall.insert(dict(o=[5, 1, 3]))
    theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    coilset.plasmagap.solve(np.c_[5 + 3 * np.cos(theta), 1 + 3 * np.sin(theta)], theta)
    assert coilset.plasmagap.matrix(0.25 * np.ones(5)).shape == (5, 1)


def test_plasmagap_kd_points():
    coilset = CoilSet(dplasma=-4, tplasma="hex", ngap=13, mingap=0, maxgap=5)
    coilset.firstwall.insert(dict(o=[5, 1, 3]))
    theta = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    coilset.plasmagap.solve(np.c_[5 + 3 * np.cos(theta), 1 + 3 * np.sin(theta)], theta)
    assert coilset.plasmagap.kd_points.shape == (13 * 7, 2)


def test_plasmagap_kd_query():
    coilset = CoilSet(dplasma=-4, tplasma="hex", ngap=13, mingap=0, maxgap=5)
    coilset.firstwall.insert(dict(o=[5, 1, 3]))
    theta = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    coilset.plasmagap.solve(np.c_[5 + 3 * np.cos(theta), 1 + 3 * np.sin(theta)], theta)

    theta_fine = np.linspace(0, 2 * np.pi, 70, endpoint=False)
    points = np.c_[5 + 3 * np.cos(theta_fine), 1 + 3 * np.sin(theta_fine)]
    assert len(coilset.plasmagap.kd_query(points)) == 7


if __name__ == "__main__":
    pytest.main([__file__])
