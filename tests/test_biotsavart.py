from dataclasses import dataclass
from functools import cached_property
from itertools import product
from numpy import allclose
import numpy as np
import pytest
import scipy.special


from nova.biot.biotframe import BiotFrame
from nova.biot.circle import Circle
from nova.biot.grid import Grid
from nova.biot.matrix import Matrix
from nova.biot.point import Point
from nova.biot.solve import Solve
from nova.frame.coilset import CoilSet
from nova.geometry.polyshape import PolyShape

segments = ["circle", "cylinder"]


def axial_vertical_field(radius, height, current):
    """Return analytic axial vertical field."""
    return (
        Matrix.mu_0
        * current
        * radius**2
        / (2 * (radius**2 + height**2) ** (3 / 2))
    )


@dataclass
class AnalyticField:
    """
    Provide access to analytic magnetic field solutions.

    Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop

    Analytic expressions for the magnetic induction and its spatial derivatives
    for a circular loop carrying a static current are presented in Cartesian,
    spherical and cylindrical coordinates.
    The solutions are exact throughout all space outside the conductor.
    """

    radius: float
    height: float
    current: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __post_init__(self):
        """Initialize C coefficent."""
        self.C = Matrix.mu_0 * self.current / np.pi
        self.phi = np.arctan2(self.y, self.x)

    @property
    def p2(self):
        """Return p2 coefficent."""
        return self.x**2 + self.y**2

    @property
    def p(self):
        """Return p coefficent."""
        return np.sqrt(self.p2)

    @property
    def r2(self):
        """Return r2 coefficent."""
        return self.x**2 + self.y**2 + (self.z - self.height) ** 2

    @property
    def a2(self):
        """Return a2 coefficent."""
        return self.radius**2 + self.r2 - 2 * self.radius * self.p

    @property
    def b2(self):
        """Return b2 coefficent."""
        return self.radius**2 + self.r2 + 2 * self.radius * self.p

    @property
    def b(self):
        """Return b coefficent."""
        return np.sqrt(self.b2)

    @property
    def k2(self):
        """Return k2 coefficient."""
        return 1 - self.a2 / self.b2

    @property
    def gamma(self):
        """Return gamma coefficient."""
        return self.x**2 - self.y**2

    @cached_property
    def bx(self):
        """Return x-component of magnetic field vector."""
        return (
            self.C
            * self.x
            * (self.z - self.height)
            / (2 * self.a2 * self.b * self.p2)
            * (
                (self.radius**2 + self.r2) * scipy.special.ellipe(self.k2)
                - self.a2 * scipy.special.ellipk(self.k2)
            )
        )

    @cached_property
    def by(self):
        """Return y-component of magnetic field vector."""
        return self.y / self.x * self.bx

    @property
    def br(self):
        """Return radial magnetic field."""
        return self.bx * np.cos(self.phi) - self.by * np.sin(self.phi)

    @property
    def bz(self):
        """Return z-component of magnetic field vector."""
        return (
            self.C
            / (2 * self.a2 * self.b)
            * (
                (self.radius**2 - self.r2) * scipy.special.ellipe(self.k2)
                + self.a2 * scipy.special.ellipk(self.k2)
            )
        )


def test_matrix_getitem():
    biotframe = BiotFrame()
    biotframe.insert(1, 7.3)
    biot = Circle(biotframe, biotframe)
    assert np.isclose(biot["zs"].item(), 7.3)


def test_biotreduce():
    biotframe = BiotFrame()
    biotframe.insert(range(3), 0)
    biotframe.insert(range(3), 1, link=True)
    biotframe.insert(range(3), 2, link=False)
    biotframe.insert(range(3), 3, link=True)
    biotframe.multipoint.link(["Coil0", "Coil11", "Coil2", "Coil8"])
    assert biotframe.biotreduce.indices == [0, 1, 2, 3, 6, 7, 8, 9, 11]
    assert list(biotframe.biotreduce.link) == [2, 6, 8]
    assert biotframe.biotreduce.index.to_list() == [
        f"Coil{i}" for i in [0, 1, 3, 6, 7, 9]
    ]


def test_subframe_lock():
    biotframe = BiotFrame(subspace=["Ic"])
    biotframe.insert([1, 3], 0, dl=0.95, dt=0.95, section="hex")
    assert biotframe.lock("subspace") is False


def test_link_negative_factor():
    biotframe = BiotFrame(label="C")
    biotframe.insert(1, 0)
    biotframe.insert(1, 0)
    biotframe.multipoint.link(["C0", "C1"], -1)
    biot = Circle(biotframe, biotframe, reduce=[True, True])
    assert np.isclose(biot.compute("Psi")[0][0, 0], 0)


def test_random_segment_error():
    biotframe = BiotFrame(label="C")
    biotframe.insert(1, 0, segment="circle")
    biotframe.insert(1, 0, segment="random")
    with pytest.raises(NotImplementedError):
        Solve(biotframe, biotframe)


@pytest.mark.parametrize("segment", segments)
def test_ITER_subinductance_matrix(segment):
    """
    Test inductance calculation against DDD values for 2 CS and 1 PF coil.

    Baseline (old) CS geometory used.
    """
    coilset = CoilSet(dcoil=0.25)
    coilset.coil.insert(
        3.9431,
        7.5641,
        0.9590,
        0.9841,
        nturn=248.64,
        name="PF1",
        part="PF",
        segment=segment,
    )
    coilset.coil.insert(
        1.722, 5.313, 0.719, 2.075, nturn=554, name="CS3U", part="CS", segment=segment
    )
    coilset.coil.insert(
        1.722, 3.188, 0.719, 2.075, nturn=554, name="CS2U", part="CS", segment=segment
    )
    biot = Circle(
        coilset.subframe, coilset.subframe, turns=[True, True], reduce=[True, True]
    )
    Mc_ddd = [
        [7.076e-01, 1.348e-01, 6.021e-02],  # referance
        [1.348e-01, 7.954e-01, 2.471e-01],
        [6.021e-02, 2.471e-01, 7.954e-01],
    ]
    assert allclose(Mc_ddd, biot.compute("Psi")[0], atol=5e-3)


def test_biot_inductance():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(
        3.9431, 7.5641, 0.9590, 0.9841, nturn=248.64, name="PF1", part="PF"
    )
    coilset.coil.insert(1.722, 5.313, 0.719, 2.075, nturn=554, name="CS3U", part="CS")
    coilset.inductance.solve(0)
    Mc_ddd = [[7.076e-01, 1.348e-01], [1.348e-01, 7.954e-01]]  # referance
    assert allclose(Mc_ddd, coilset.inductance.Psi, atol=5e-3)


def test_inductance_number_none():
    coilset = CoilSet()
    coilset.coil.insert(1, 0, 0.1, 0.1)
    coilset.inductance.solve()
    assert len(coilset.inductance.data) == 0


def test_solenoid_grid():
    """verify solenoid vertical field using grid biot instance."""
    nturn, height, current = 500, 30, 1e3
    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(1.5, 0, 0.01, height, nturn=nturn, section="rect")
    coilset.sloc["Ic"] = current
    grid = Grid(*coilset.frames)
    grid.solve(4, [1e-9, 1.5, 0, 1])
    Bz_theory = Matrix.mu_0 * nturn * current / height
    Bz_grid = np.dot(grid.data.Bz, coilset.sloc["Ic"])
    assert allclose(Bz_grid[0], Bz_theory, atol=5e-3)


@pytest.mark.parametrize("segment", segments)
def test_solenoid_probe(segment):
    """Verify solenoid vertical field using probe biot instance."""
    nturn, height, current = 500, 30, 1e3
    coilset = CoilSet(dcoil=0.5)
    coilset.coil.insert(
        1.5, 0, 0.01, height, nturn=nturn, section="rectangle", segment=segment
    )
    coilset.sloc["Ic"] = current
    point = Point(*coilset.frames)
    point.solve((1e-9, 0))
    Bz_theory = Matrix.mu_0 * nturn * current / height
    Bz_point = np.dot(point.data.Bz, coilset.sloc["Ic"])
    assert allclose(Bz_point, Bz_theory, atol=5e-3)


def test_circle_circle_coil_pair():
    coilset = CoilSet(dcoil=-10)
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=-15e6, segment="circle")
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=15e6, segment="circle")
    coilset.point.solve([[8, 0]])
    assert np.isclose(coilset.point.psi, 0)


def test_cyliner_cylinder_coil_pair():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=-15e6, segment="cylinder")
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=15e6, segment="cylinder", delta=-10)
    coilset.point.solve([[8, 0]])
    assert np.isclose(coilset.point.psi, 0)


def test_cylinder_circle_coil_pair():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(6.6, 0, 0.2, 0.2, Ic=-15e6, segment="cylinder")
    coilset.coil.insert(6.6, 0, 0.2, 0.2, Ic=15e6, segment="circle", delta=-10)
    coilset.point.solve([[7, 0]])
    assert np.isclose(coilset.point.psi, 0, atol=1e-3)


@pytest.mark.parametrize("segment", segments)
def test_hemholtz_flux(segment):
    coilset = CoilSet(dcoil=-2)
    coilset.coil.insert(1, [-0.5, 0.5], 0.01, 0.01, Ic=1, segment=segment)
    point_radius = 0.1
    coilset.point.solve([[point_radius, 0]])
    Bz = (4 / 5) ** (3 / 2) * Matrix.mu_0
    psi = Bz * np.pi * point_radius**2
    assert np.isclose(coilset.point.psi[0], psi)


@pytest.mark.parametrize("segment", segments)
def test_hemholtz_field(segment):
    coilset = CoilSet(dcoil=-2)
    coilset.coil.insert(1, [-0.5, 0.5], 0.01, 0.01, Ic=1, segment=segment)
    coilset.point.solve([[1e-3, 0]])
    bz = (4 / 5) ** (3 / 2) * Matrix.mu_0
    assert np.isclose(coilset.point.bz[0], bz)


@pytest.mark.parametrize("section", ["rectangle", "circle", "c", "disc", "r"])
def test_coil_segment(section):
    coilset = CoilSet()
    coilset.coil.insert({section: [1, 0.5, 0.01, 0.01]}, Ic=1)
    section = PolyShape(section).shape
    assert coilset.frame.section.iloc[0] == section
    assert (
        coilset.subframe.segment.iloc[0]
        == {
            "disc": "circle",
            "rectangle": "cylinder",
        }[section]
    )


@pytest.mark.parametrize(
    "section,radius,height",
    product(["disc", "rectangle"], [2.1, 7.3, 12], [-3.2, 0, 7.3]),
)
def test_axial_vertical_field(section, radius, height):
    current = 5.3e4
    coilset = CoilSet()
    coilset.coil.insert({section: [radius, height, 0.01, 0.01]}, Ic=current)
    coilset.point.solve([[1e-6, 0]])
    assert np.isclose(
        coilset.point.bz[0], axial_vertical_field(radius, height, current)
    )


def test_coil_cylinder_isfinite_farfield():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(6.5, [-1, 0, 1], 0.4, 0.4, Ic=-15e6, segment="cylinder")
    coilset.grid.solve(60, [6, 7.0, -0.8, 0.8])
    assert np.isfinite(coilset.grid.psi).all()


def test_coil_cylinder_isfinite_coil():
    coilset = CoilSet(dcoil=-(2**3))
    coilset.coil.insert(0.3, 0, 0.15, 0.15, segment="cylinder", Ic=5e3)
    coilset.grid.solve(10**2, 0)
    assert np.isfinite(coilset.grid.psi).all()


@pytest.mark.parametrize(
    "section,radius,height,current",
    product(["disc", "rectangle"], [2.1, 7.3, 12], [-3.2, 0, 7.3], [-1e4, 5.3e4]),
)
def test_magnetic_field_analytic_poloidal_plane(section, radius, height, current):
    coilset = CoilSet()
    coilset.coil.insert({section: [radius, height, 0.01, 0.01]}, Ic=current)
    coilset.grid.solve(1e3, [1, 5, -3.2 + height, 4.1 + height])

    x = coilset.grid.data.x2d.data
    y = np.zeros_like(x)
    z = coilset.grid.data.z2d.data
    analytic = AnalyticField(radius, height, current, x, y, z)

    assert np.allclose(coilset.grid.br_, analytic.br, atol=1e-4)
    assert np.allclose(coilset.grid.bz_, analytic.bz, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
