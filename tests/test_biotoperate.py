import pytest

import matplotlib.pylab
import numpy as np
import scipy.special

from nova.biot.constants import Constants
from nova.frame.coilset import CoilSet
from nova.frame.polygrid import PolyTarget

QUARTER_TURN = np.pi / 2


def _arc_winding(coilset, end: float, span: float, radius=3.945, height=2.0):
    """Insert a finite-cross-section arc whose far end sits at azimuth ``end``.

    ``filament=False`` is what routes the winding through the Bow kernel; the
    default filament arc never forms an incomplete third kind at all.
    """
    azimuth = np.linspace(end - span, end, 4)
    points = np.stack(
        [
            radius * np.cos(azimuth),
            radius * np.sin(azimuth),
            np.full_like(azimuth, height),
        ],
        axis=-1,
    )
    coilset.winding.insert(
        points,
        {"rect": (0, 0, 0.06, 0.03)},
        nturn=1,
        minimum_arc_nodes=4,
        Ic=1,
        filament=False,
        ifttt=False,
    )
    return radius, height


def _stepped(value: float, steps: int) -> float:
    """Return ``value`` moved ``steps`` representable numbers along the line."""
    toward = np.inf if steps > 0 else -np.inf
    for _ in range(abs(steps)):
        value = np.nextafter(value, toward)
    return value


def test_grid_shape():
    coilset = CoilSet(dcoil=-5, tcoil="hex")
    coilset.coil.insert(3, 0, 0.1, 0.1)
    coilset.grid.solve(10)
    assert coilset.grid.shape == (
        coilset.grid.data.sizes["x"],
        coilset.grid.data.sizes["z"],
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
    coilset.point.solve(np.array([(1, 2), (4, 5), (7, 3)]))
    assert len(coilset.point.shape) == 1


def test_point_shaped_array_address():
    coilset = CoilSet(dcoil=-5, tcoil="hex")
    coilset.coil.insert(5, -2, 0.7, 0.5, Ic=-10)
    coilset.point.solve(np.array([(1, 12), (4, 5), (7, -3)]))
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
    psi = coilset.plasmagrid.psi
    coilset.plasma.separatrix = dict(o=[5.1, 1, 2.5])
    psi_hash = coilset.aloc_hash["nturn"]
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


def test_inductance_quadrature_keeps_original_plasma_block_dimensions():
    """Kernel nodes contract before the live plasma-turn matrices are extracted."""
    coilset = CoilSet(dplasma=-8, tplasma="hex")
    coilset.firstwall.insert({"circle": [3.0, 0.0, 0.5]}, Ic=1.0)
    logical_cells = int(np.asarray(coilset.subframe.plasma).sum())
    coilset.inductance.solve(1)
    data = coilset.inductance.data
    assert data.sizes["source_plasma"] == logical_cells
    assert data.sizes["target_plasma"] == logical_cells
    assert data["Psi_"].shape == (1, logical_cells)
    assert data["_Psi"].shape == (logical_cells, 1)
    assert data["_Psi_"].shape == (logical_cells, logical_cells)


def test_force_keeps_the_physical_poly_target_and_dcoil_cells():
    """Linked-flux quadrature changes neither force targets nor conducting sources."""
    coilset = CoilSet(dcoil=-3)
    coilset.coil.insert(3.0, 0.0, 0.4, 0.2, nturn=12, name="PF")
    expected = PolyTarget(*coilset.frames, index="coil", delta=-2).target
    source_index = coilset.subframe.index.tolist()
    source_x = np.asarray(coilset.subframe.x).copy()
    source_z = np.asarray(coilset.subframe.z).copy()
    coilset.force.solve(2)
    assert coilset.force.target.index.tolist() == expected.index.tolist()
    np.testing.assert_array_equal(coilset.force.target.x, expected.x)
    np.testing.assert_array_equal(coilset.force.target.z, expected.z)
    assert coilset.subframe.index.tolist() == source_index
    np.testing.assert_array_equal(coilset.subframe.x, source_x)
    np.testing.assert_array_equal(coilset.subframe.z, source_z)


def test_nturn_skip_current_update():
    coilset = CoilSet(dcoil=-5, dplasma=-5, tcoil="hex", tplasma="hex")
    coilset.firstwall.insert(dict(o=[5, 1, 5]), Ic=15e6)
    coilset.plasmagrid.solve()
    coilset.sloc["Ic"] = 1
    psi = coilset.plasmagrid.psi
    coilset.sloc["Ic"] = 2
    current_hash = coilset.aloc_hash["Ic"]
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


def test_ellippinc_amplitude_below_quarter_turn():
    """The third kind evaluates one representable step short of a quarter turn.

    A target a rounding error past an arc's END PLANE folds to exactly this
    amplitude, so the step below a quarter turn is a configuration a segmented
    winding reaches at its own joints rather than a contrived argument.
    """
    n, m = np.array([0.3]), np.array([0.4])
    value = Constants.ellippinc(n, np.array([_stepped(QUARTER_TURN, -1)]), m)
    quarter = Constants.ellippinc(n, np.array([QUARTER_TURN]), m)
    assert np.all(np.isfinite(value))
    assert np.allclose(value, quarter, rtol=1e-14)


def test_ellippinc_reduces_across_the_quarter_turn():
    """Folding is continuous through the quarter turn, from either side."""
    steps = np.array([_stepped(QUARTER_TURN, step) for step in range(-4, 5)])
    value = Constants.ellippinc(
        np.full_like(steps, 0.3), steps, np.full_like(steps, 0.4)
    )
    assert np.all(np.isfinite(value))
    assert np.ptp(value) < 1e-14 * abs(value[0])


@pytest.mark.parametrize("m", [0.0, 0.4, 0.9])
def test_ellippinc_first_kind_limit(m):
    """A vanishing characteristic is the FIRST kind, which scipy evaluates.

    The gap to a quarter turn is swept down to 1e-13 because that is where the
    amplitude's own cosine has to be carried rather than recovered from its
    sine: ``1 - sin^2`` has no digits left there.
    """
    gap = 10.0 ** -np.arange(2.0, 14.0)
    amplitude = QUARTER_TURN - gap
    value = Constants.ellippinc(
        np.zeros_like(amplitude), amplitude, np.full_like(amplitude, m)
    )
    reference = scipy.special.ellipkinc(amplitude, m)
    assert np.allclose(value, reference, rtol=1e-13, atol=0)


@pytest.mark.parametrize("n", [-0.7, 0.3, 0.9])
def test_ellippinc_zero_modulus_closed_form(n):
    """A vanishing modulus is elementary: ``arctan(sqrt(1-n) tan phi)/sqrt(1-n)``.

    The reference is taken through the gap's own tangent, so it keeps its
    relative accuracy where the amplitude approaches a quarter turn and the
    comparison pins the evaluation rather than the reference.
    """
    gap = 10.0 ** -np.arange(2.0, 14.0)
    amplitude = QUARTER_TURN - gap
    root = np.sqrt(1 - n)
    value = Constants.ellippinc(
        np.full_like(amplitude, n), amplitude, np.zeros_like(amplitude)
    )
    reference = np.arctan(root / np.tan(gap)) / root
    assert np.allclose(value, reference, rtol=1e-13, atol=0)


def test_ellippinc_amplitude_pair_matches_the_angle():
    """An amplitude supplied as its own (sine, cosine) agrees with the angle.

    A caller that forms the pair from the geometry — the arc's amplitude is
    ``(pi + psi)/2`` for an azimuthal separation ``psi``, so its cosine is
    ``-sin(psi/2)`` exactly — keeps relative accuracy the angle cannot.
    """
    amplitude = QUARTER_TURN - 10.0 ** -np.arange(2.0, 14.0)
    n, m = np.full_like(amplitude, 0.3), np.full_like(amplitude, 0.4)
    value = Constants.ellippinc(
        n, amplitude, m, sine=np.sin(amplitude), cosine=np.cos(amplitude)
    )
    assert np.allclose(value, Constants.ellippinc(n, amplitude, m), rtol=1e-15)


def test_ellippinc_half_turn_quasi_periodicity():
    """Every half turn of amplitude adds two complete integrals, exactly."""
    turns = np.array([1.0, 10.0, 1e3, 1e6])
    n, m = np.full_like(turns, 0.3), np.full_like(turns, 0.4)
    base = Constants.ellippinc(n, np.full_like(turns, 0.7), m)
    value = Constants.ellippinc(n, 0.7 + turns * np.pi, m)
    assert np.allclose(value, base + 2 * turns * Constants.ellipp(n, m), rtol=1e-13)


def test_point_solve_on_the_arc_end_plane():
    """A finite-section arc solves at targets sitting on its own end plane.

    The end plane is where the folded amplitude reaches a quarter turn, and the
    representable steps either side of it are what the fold has to carry.
    """
    coilset = CoilSet(field_attrs=["Ay", "Br", "Bz"])
    end = 0.7
    radius, height = _arc_winding(coilset, end, 1.1)
    azimuth = np.array([_stepped(end, step) for step in range(-64, 65)])
    points = np.stack(
        [
            (radius + 0.2) * np.cos(azimuth),
            (radius + 0.2) * np.sin(azimuth),
            np.full_like(azimuth, height + 0.1),
        ],
        axis=-1,
    )
    coilset.point.solve(points)
    assert np.all(np.isfinite(coilset.point.br))


def test_grid_solve_finite_section_winding():
    """A winding grid solve completes through the Bow kernel."""
    coilset = CoilSet(field_attrs=["Ay", "Br", "Bz"])
    _arc_winding(coilset, 0.0, 2.0)
    assert coilset.subframe.segment[0] == "bow"
    coilset.grid.solve(30, 0.5)
    assert np.all(np.isfinite(coilset.grid.br))
    assert np.any(coilset.grid.br != 0)


if __name__ == "__main__":
    pytest.main([__file__])
