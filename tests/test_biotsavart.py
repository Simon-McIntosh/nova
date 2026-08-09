from dataclasses import dataclass
from functools import cached_property
from itertools import product
from numpy import allclose
import numpy as np
import pytest
import scipy.special


from nova.biot.biotframe import BiotFrame
from nova.biot.circle import Circle
from nova.biot.constants import Constants
from nova.biot.grid import Grid
from nova.biot.matrix import Matrix
from nova.biot.point import Point
from nova.biot.solve import Solve
from nova.frame.coilset import CoilSet
from nova.geometry.polyline import PolyLine
from nova.geometry.polyshape import PolyShape

segments = ["circle", "cylinder"]


def axial_vertical_field(radius, height, current):
    """Return analytic axial vertical field."""
    return Matrix.mu_0 * current * radius**2 / (2 * (radius**2 + height**2) ** (3 / 2))


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
        return self.bx * np.cos(self.phi) + self.by * np.sin(self.phi)

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
    assert list(biotframe.biotreduce.indices) == [0, 1, 2, 3, 6, 7, 8, 9, 11]
    assert list(biotframe.biotreduce.link) == [2, 6, 8]
    assert list(biotframe.biotreduce.index) == [f"Coil{i}" for i in [0, 1, 3, 6, 7, 9]]


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
    point.solve(np.array([1e-9, 0]))
    Bz_theory = Matrix.mu_0 * nturn * current / height
    Bz_point = np.dot(point.data.Bz, coilset.sloc["Ic"])
    assert allclose(Bz_point, Bz_theory, atol=5e-3)


def test_circle_circle_coil_pair():
    coilset = CoilSet(dcoil=-10)
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=-15e6, segment="circle")
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=15e6, segment="circle")
    coilset.point.solve(np.array([[8, 0]]))
    assert np.isclose(coilset.point.psi, 0)


def test_cyliner_cylinder_coil_pair():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=-15e6, segment="cylinder")
    coilset.coil.insert(6.6, 0.1, 0.2, 0.2, Ic=15e6, segment="cylinder", delta=-10)
    coilset.point.solve(np.array([8, 0]))
    assert np.isclose(coilset.point.psi, 0)


def test_cylinder_circle_coil_pair():
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(6.6, 0, 0.2, 0.2, Ic=-15e6, segment="cylinder")
    coilset.coil.insert(6.6, 0, 0.2, 0.2, Ic=15e6, segment="circle", delta=-10)
    coilset.point.solve(np.array([7, 0]))
    assert np.isclose(coilset.point.psi, 0, atol=1e-3)


@pytest.mark.parametrize("segment", segments)
def test_hemholtz_flux(segment):
    coilset = CoilSet(dcoil=-2)
    coilset.coil.insert(1, [-0.5, 0.5], 0.01, 0.01, Ic=1, segment=segment)
    point_radius = 0.1
    coilset.point.solve(np.array([point_radius, 0]))
    Bz = (4 / 5) ** (3 / 2) * Matrix.mu_0
    psi = Bz * np.pi * point_radius**2
    assert np.isclose(coilset.point.psi[0], psi)


@pytest.mark.parametrize("segment", segments)
def test_hemholtz_field(segment):
    coilset = CoilSet(dcoil=-2)
    coilset.coil.insert(1, [-0.5, 0.5], 0.01, 0.01, Ic=1, segment=segment)
    coilset.point.solve(np.array([1e-3, 0]))
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
            "disc": "polysection",
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
    coilset.point.solve(np.array([1e-6, 0]))
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


@pytest.mark.parametrize(
    "segment,radius,height,current",
    product(["arc", "line", "bow"], [2.1, 7.3], [-3.2, 0, 7.3], [-1e4, 5.3e4]),
)
def test_magnetic_field_analytic_non_axisymmetric(segment, radius, height, current):
    segment_number = 51
    theta = np.linspace(0, 2 * np.pi, 1 + 2 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )
    if segment == "line":
        minimum_arc_nodes = len(points) + 1
    else:
        minimum_arc_nodes = 3
    coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Br"])
    coilset.winding.insert(
        points, {"c": (0, 0, 0.25)}, minimum_arc_nodes=minimum_arc_nodes, Ic=current
    )
    grid = np.meshgrid(
        np.linspace(-3.1, 5.1, 9), -2.5, np.linspace(-4.1, 1.7, 3), indexing="ij"
    )

    coilset.point.solve(np.stack(grid, axis=-1))
    analytic = AnalyticField(radius, height, current, *grid)

    for attr in coilset.field_attrs:
        assert np.allclose(
            getattr(coilset.point, attr.lower()),
            getattr(analytic, attr.lower()).flatten(),
            atol=1e-5,
        )


@pytest.mark.parametrize("rng_sead,current", product([2015, 2025, 2038], [-1e3, 68e3]))
def test_3d_line_arc(rng_sead, current):
    rng = np.random.default_rng(rng_sead)
    segment = PolyLine(rng.uniform(-1, 1, (3, 3)), minimum_arc_nodes=3).segments[0]

    coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Br"])
    coilset.winding.insert(
        segment.sample(3),
        {"c": (0, 0, 0.25)},
        minimum_arc_nodes=3,
        Ic=current,
        name="arc",
    )
    segment_number = 12
    coilset.winding.insert(
        segment.sample(segment_number),
        {"c": (0, 0, 0.25)},
        minimum_arc_nodes=segment_number,
        Ic=current,
        name="line",
    )
    coilset.linkframe(["arc", "line"], -1)
    coilset.point.solve(np.array([-0.3, 0.1, -1.1]))
    for attr in ["Bx", "By", "Bz", "Br"]:
        assert np.allclose(getattr(coilset.point, attr.lower()), 0, atol=1e-6)


@pytest.mark.skip("pending development of singularity skip methods")
def test_line_singularity():
    segment_number = 30
    x = np.linspace(0, 5, segment_number)
    points = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)
    coilset = CoilSet(field_attrs=["Bx", "By", "Bz"])
    coilset.winding.insert(points, {"c": (0, 0, 0.25)}, Ic=1)
    coilset.point.solve(np.c_[[0, 0, 0], [0, 0.25 / 2, 0], [0, 0.25, 0]].T)
    assert coilset.point.bz[0] < coilset.point.bz[1]
    assert coilset.point.bz[0] < coilset.point.bz[2]
    assert coilset.point.bz[2] < coilset.point.bz[1]


def test_multifilament_3d_vector():
    coilset = CoilSet(field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz"])
    coilset.coil.insert(5, [-1, 1], 0.1, 0.1, Ic=1e3, delta=-2, segment="circle")
    coilset.coil.insert(5, [-1, 1], 0.1, 0.1, Ic=-1e3, delta=-2, segment="circle")
    coilset.point.solve(np.array([[5, 0, 0], [6, 0, 0]]))
    assert np.allclose(coilset.point.vector_potential, 0)
    assert np.allclose(coilset.point.magnetic_field, 0)


@pytest.mark.skip("pending development of singularity skip methods")
def test_arc_singularity():
    segment_number = 501
    theta, dtheta = np.linspace(0, 2 * np.pi, segment_number, retstep=True)
    radius = 5.3
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=-1
    )
    coilset = CoilSet(field_attrs=["Ax", "Ay", "Az", "Bx", "By", "Bz"])
    coilset.coil.insert(radius, 0, 0.1, 0.1, Ic=1e6, ifttt=False, segment="cylinder")
    # coilset.coil.insert(radius, 0, 0.1, 0.1, Ic=-1e6, ifttt=False, segment="circle")

    coilset.winding.insert(points, {"c": (0, 0, 0.1)}, Ic=-1e6, minimum_arc_nodes=0)

    print(coilset.frame.segment)

    number = 200
    grid = np.stack(
        [
            np.linspace(-0.5, 0.5, number),
            radius * np.ones(number),
            np.zeros(number),
        ],
        axis=-1,
    )
    coilset.point.solve(grid)
    coilset.grid.solve(5e3, 1)

    print(coilset.grid.ay.max())

    coilset.grid.plot("ay")
    coilset.plot()

    # coilset.point.set_axes("1d")
    # coilset.point.axes.plot(grid[:, 0], coilset.point.ax)
    assert False


def test_ellipf():
    m = np.array([0.0, 0.2, 0.4, 0.8, 0.5])
    phi = np.array([-2.5, 0.0, 0.5, 5.5, 7.2])
    p_scipy = Constants.ellipkinc(phi, m)
    p_mpmath = np.array([-2.5, 0.0, 0.50827893, 8.17487571, 8.39751980])
    assert np.allclose(p_scipy, p_mpmath)


def test_ellipe():
    m = np.array([0.0, 0.2, 0.4, 0.8, 0.5])
    phi = np.array([-2.5, 0.0, 0.5, 5.5, 7.2])
    p_scipy = Constants.ellipeinc(phi, m)
    p_mpmath = np.array([-2.5, 0.0, 0.49195874, 3.99157413, 6.26205652])
    assert np.allclose(p_scipy, p_mpmath)


def test_ellipp():
    """Test ellippi against mpmath implementation.

    Copied from https://github.com/scipy/scipy/pull/15787."""
    n = np.array([-0.5, 0.0, 0.3, 1.3, -0.7])
    m = np.array([0.0, 0.2, 0.4, 0.8, 0.5])
    p_scipy = Constants.ellipp(n, m)
    p_mpmath = np.array([1.2825498, 1.6596236, 2.14879542, -1.7390616, 1.3902519])
    assert np.allclose(p_scipy, p_mpmath)


def test_ellippinc():
    n = np.array([-0.5, 0.0, 0.3, 1.3, -0.7])
    m = np.array([0.0, 0.2, 0.4, 0.8, 0.5])
    phi = np.array([-2.5, 0.0, 0.5, 5.5, 7.2])
    p_scipy = Constants.ellippinc(n, phi, m)
    p_mpmath = np.array([-1.96008157, 0.0, 0.521063078, -8.20462585, 6.408549098])
    assert np.allclose(p_scipy, p_mpmath)


#: Edges a generated curved profile carries. ``PolyGen.disc`` buffers a circle
#: with ``quadrant_segments`` edges per quadrant, so a ``disc`` and the two discs a
#: ``skin`` differences are regular polygons INSCRIBED in their circles -- not the
#: circles themselves. Stated here rather than folded into an area constant so that
#: raising the facet count moves the reference instead of silently breaking it.
DISC_EDGES = 4 * 16


def inscribed_area(diameter, edges=DISC_EDGES):
    """Return the area of the regular polygon a generated disc actually is.

    ``(n/2) R^2 sin(2 pi / n)``, which approaches ``pi R^2`` from below: at 64
    edges it under-fills its circle by 1.6e-03, and that shortfall is the whole
    difference between a hollow circular section and the square one a bounding-box
    reference would stand in for.
    """
    return edges / 2 * (diameter / 2) ** 2 * np.sin(2 * np.pi / edges)


#: Section geometry both hollow-section tests are built on: a ring at this major
#: radius and height, whose section has these outer and inner widths.
SHELL = dict(
    radius=3.945, height=2.0, outer_width=0.05, inner_width=0.04, current=5.3e5
)


def swept_shell(section, *, nturn=1, **shell):
    """Return a full ring swept from three arcs, carrying ``section``.

    ``filament=False`` is what thickens the segments so the section is evaluated at
    all; the three arcs close on themselves, so the result is one complete ring and
    may be compared against an axisymmetric coil.

    Four points per insert against ``minimum_arc_nodes=4`` is what makes each of the
    three a single 120-degree ARC, and the pairing is load-bearing rather than
    incidental.  :meth:`nova.geometry.polyline.PolyLine.append` fits an arc only to a
    run of at least ``minimum_arc_nodes`` points and emits chords otherwise, so
    raising the node count and ``minimum_arc_nodes`` together does not refine the
    path -- it drops every segment to ``beam`` and inscribes the circle in a polygon.
    Measured against the quadrature reference at a standoff of six section widths,
    the three arcs land at 8.8e-10 while the eighteen chords ``minimum_arc_nodes=8``
    produces land at 2.2e-02 and the thirty-six of ``minimum_arc_nodes=16`` at
    7.4e-03: a chord path does converge, at the second order its own count supplies,
    but from seven decades away.  Refinement here means MORE INSERTS of four points
    each, never a larger ``minimum_arc_nodes``.
    """
    shell = {**SHELL, **shell}
    segments = 3
    theta = np.linspace(0, 2 * np.pi, 1 + 3 * segments)
    points = np.stack(
        [
            shell["radius"] * np.cos(theta),
            shell["radius"] * np.sin(theta),
            shell["height"] * np.ones_like(theta),
        ],
        axis=-1,
    )
    coilset = CoilSet(field_attrs=["Ay", "Br", "Bz"])
    for i in range(segments):
        coilset.winding.insert(
            points[3 * i : 1 + 3 * (i + 1)],
            {
                section: (
                    0,
                    0,
                    shell["outer_width"],
                    1 - shell["inner_width"] / shell["outer_width"],
                )
            },
            nturn=nturn,
            minimum_arc_nodes=4,
            Ic=shell["current"],
            filament=False,
            ifttt=False,
        )
    return coilset


#: Panels per section half-width, and Gauss-Legendre order inside each panel, of
#: the annulus quadrature reference.
ANNULUS_PANELS = 32
ANNULUS_ORDER = 12

#: What panel doubling is allowed to move the reference by, which is the floor every
#: assertion measured against it stands on. Measured 1.1e-12 for a target beside the
#: section and 1.2e-15 for one metres away, so this carries a decade of reserve and
#: sits two decades under the tightest element assertion below.
ANNULUS_FLOOR = 1e-11


def filament_ring_rows(source_radius, source_height, radius, height):
    """Return a filament ring's ``(A_phi / mu_0, Br, Bz)`` per ampere, target by source.

    Legendre-form complete integrals taken straight from scipy, so the reference
    below shares no code, no argument reduction and no series with either element
    it measures.  The potential row carries no vacuum permeability and the two
    field rows do, which is the convention the operator store uses; the
    axisymmetric arm of the gate is what pins that, since one reference cannot
    reproduce all three rows of an element under two different conventions.

    Result is ``(3, source, target)``.
    """
    source_radius = np.asarray(source_radius, dtype=float)[:, np.newaxis]
    source_height = np.asarray(source_height, dtype=float)[:, np.newaxis]
    radius = np.asarray(radius, dtype=float)[np.newaxis, :]
    height = np.asarray(height, dtype=float)[np.newaxis, :]
    gap = height - source_height
    span = (source_radius + radius) ** 2 + gap**2
    modulus = 4 * source_radius * radius / span
    ellipk = scipy.special.ellipk(modulus)
    ellipe = scipy.special.ellipe(modulus)
    near = (source_radius - radius) ** 2 + gap**2
    return np.stack(
        [
            np.sqrt(source_radius / radius)
            / (np.pi * np.sqrt(modulus))
            * ((1 - modulus / 2) * ellipk - ellipe),
            Matrix.mu_0
            / (2 * np.pi)
            * gap
            / (radius * np.sqrt(span))
            * (-ellipk + (source_radius**2 + radius**2 + gap**2) / near * ellipe),
            Matrix.mu_0
            / (2 * np.pi)
            / np.sqrt(span)
            * (ellipk + (source_radius**2 - radius**2 - gap**2) / near * ellipe),
        ]
    )


def _gauss_panels(lower, upper, panels, order=ANNULUS_ORDER):
    """Return nodes and weights of ``panels`` Gauss-Legendre panels on an interval."""
    node, weight = np.polynomial.legendre.leggauss(order)
    edge = np.linspace(lower, upper, panels + 1)
    centre, half = 0.5 * (edge[1:] + edge[:-1]), 0.5 * np.diff(edge)
    return (
        (centre[:, np.newaxis] + half[:, np.newaxis] * node).ravel(),
        (half[:, np.newaxis] * weight).ravel(),
    )


def square_annulus_rows(targets, panels=ANNULUS_PANELS, **shell):
    """Return ``(A_phi / mu_0, Br, Bz)`` of a square-annulus-section ring by quadrature.

    The section is the region between two concentric squares, tiled by the four
    strips that make it up and integrated at the uniform current density the total
    current and the annulus area set.  Every target must lie OUTSIDE that region:
    that is what leaves the filament-ring integrand analytic at every node, so a
    panelled Gauss-Legendre rule converges spectrally and its own panel doubling
    reports the floor rather than an endpoint singularity's algebraic rate.
    """
    shell = {**SHELL, **shell}
    radius, height = shell["radius"], shell["height"]
    outer, inner = shell["outer_width"] / 2, shell["inner_width"] / 2
    density = shell["current"] / (4 * (outer**2 - inner**2))
    target_radius, target_height = targets[:, 0], targets[:, 2]
    rows = np.zeros((3, len(target_radius)))
    for lower_radius, upper_radius, lower_height, upper_height in (
        (radius - outer, radius - inner, height - outer, height + outer),
        (radius + inner, radius + outer, height - outer, height + outer),
        (radius - inner, radius + inner, height - outer, height - inner),
        (radius - inner, radius + inner, height + inner, height + outer),
    ):
        node_radius, weight_radius = _gauss_panels(
            lower_radius,
            upper_radius,
            max(1, round(panels * (upper_radius - lower_radius) / outer)),
        )
        node_height, weight_height = _gauss_panels(
            lower_height,
            upper_height,
            max(1, round(panels * (upper_height - lower_height) / outer)),
        )
        source_radius = np.repeat(node_radius, len(node_height))
        source_height = np.tile(node_height, len(node_radius))
        weight = np.outer(weight_radius, weight_height).ravel() * density
        for start in range(0, len(source_radius), 4096):
            block = slice(start, start + 4096)
            rows += (
                filament_ring_rows(
                    source_radius[block],
                    source_height[block],
                    target_radius,
                    target_height,
                )
                * weight[block][np.newaxis, :, np.newaxis]
            ).sum(axis=1)
    return rows


def axisymmetric_shell(**shell):
    """Return the square annulus as two solid axisymmetric coils at opposite density."""
    shell = {**SHELL, **shell}
    outer, inner = shell["outer_width"], shell["inner_width"]
    coilset = CoilSet(field_attrs=["Ay", "Br", "Bz"])
    coilset.coil.insert({"rect": (shell["radius"], shell["height"], outer, outer)})
    coilset.coil.insert({"rect": (shell["radius"], shell["height"], inner, inner)})
    density = shell["current"] / (outer**2 - inner**2)
    coilset.saloc["Ic"] = density * outer**2, -density * inner**2
    return coilset


def element_rows(instance, coilset):
    """Return ``(A_phi / mu_0, Br, Bz)`` from an element's own float64 operators.

    Contract the coupling rows directly so each element is compared before the
    higher-level source reduction combines it with the other element.
    """
    current = np.asarray(coilset.sloc["Ic"], dtype=float)
    return np.stack(
        [
            np.asarray(instance.data[attr], dtype=float) @ current
            for attr in ("Ay", "Br", "Bz")
        ]
    )


def row_deviation(rows, reference):
    """Return each row's largest departure from ``reference`` over its own scale.

    Normalised by the row's peak over the target set rather than point by point:
    ``Br`` passes through zero on the section's midplane, where a pointwise
    relative measure divides by nothing and reports a full-scale failure for a
    reference and an element that agree to round-off.
    """
    return np.max(np.abs(rows - reference), axis=-1) / np.max(
        np.abs(reference), axis=-1
    )


#: The axisymmetric element against the independent quadrature, all three rows, at
#: every target -- inside the cavity as much as outside the section. Measured 7.4e-12
#: beside the section and 1.5e-11 metres away, so this stands two decades clear of
#: the element and two above the reference's own floor: scaling any axisymmetric row
#: by a relative 1e-09 fails this arm.
AXISYMMETRIC_TOL = 1e-09

#: The swept arc against the same reference at six section widths of standoff and
#: beyond, where none of the near-face conditioning loss below is left. Measured
#: 8.8e-10, one decade of reserve: scaling the swept rows by a relative 1e-08 fails
#: this arm, 1e-09 passes it. A comparison of the two ELEMENTS against each other at
#: 1e-04 relative admits a 1e-04 scaling of the swept rows, four decades looser.
SWEPT_FAR_TOL = 1e-08


@pytest.mark.parametrize("section", ["box"])
def test_box_section(section):
    """A swept hollow square against a quadrature reference, in three standoff bands.

    The path is three 120-degree arcs, so the swept body is a TRUE RING of square
    annular section and the comparison carries no discretisation term at all: the
    axisymmetric arm below reproduces the reference to 7.4e-12, and the swept arc to
    8.8e-10 away from its own section, against the 2.2e-02 a chord path of the same
    node count returns (:func:`swept_shell`).  What the comparison does carry is the
    accuracy the thickened-arc antiderivative reaches NEAR its own section, and that
    is what the bands separate.

    The reference is an independent panelled Gauss-Legendre integral of the
    filament-ring Green's functions over the annulus, sharing no code with either
    element; every target sits outside the conducting material, which is asserted
    below, so the integrand is analytic and panel doubling puts the reference at
    1.1e-12.  An element-against-element comparison cannot say which of the two
    moved, and the answer matters here: the axisymmetric element reproduces the
    reference at every target, including the ones inside the cavity, so the whole of
    the departure belongs to the swept arc.

    That departure grows as a target approaches the section's own face: 8.8e-10 at
    six section widths of clearance, 2.5e-06 at 0.0075 m (a sixth of a width), and
    1.3e-04 at 0.0005 m, which is where the cavity targets sit against the core's
    face.  One tolerance over all three bands would be vacuous in the far one and
    unreachable in the near one, so the two near bands are pinned TWO-SIDED at the
    size measured in each: the gate fails if the arc gets worse, and equally if it
    gets better, which is what keeps a recorded accuracy limit a statement about the
    code rather than a ceiling nobody revisits.

    ``skin`` is not covered here.  Its section is the annulus between two inscribed
    64-gons, which no pair of ``rect`` coils and no square reference expresses;
    :func:`test_skin_section_area_is_the_inscribed_annulus` and
    :func:`test_a_hollow_square_carries_four_thirds_the_inscribed_moment` are where
    that section is held.
    """
    swept, axisymmetric = swept_shell(section), axisymmetric_shell()
    swept.grid.solve(30, 0.3)
    axisymmetric.grid.solve(30, 0.3)

    radius = np.asarray(swept.grid.data["x2d"], dtype=float).ravel()
    height = np.asarray(swept.grid.data["z2d"], dtype=float).ravel()
    assert np.allclose(radius, np.asarray(axisymmetric.grid.data["x2d"]).ravel())
    assert np.allclose(height, np.asarray(axisymmetric.grid.data["z2d"]).ravel())
    targets = np.stack([radius, np.zeros_like(radius), height], axis=-1)

    outer, inner = SHELL["outer_width"] / 2, SHELL["inner_width"] / 2
    offset = np.abs(np.stack([radius - SHELL["radius"], height - SHELL["height"]]))
    cavity = np.all(offset < inner, axis=0)
    outside = np.any(offset > outer, axis=0)
    # the reference integrates the material itself, so no target may sit in it
    assert np.all(cavity | outside)
    # the 6 x 5 grid the request resolves to straddles the 5 mm wall without landing
    # in it; a count that moves means the bands below no longer hold what they name
    assert cavity.sum() == 12 and outside.sum() == 18

    reference = square_annulus_rows(targets)
    assert np.all(
        row_deviation(square_annulus_rows(targets, 2 * ANNULUS_PANELS), reference)
        < ANNULUS_FLOOR
    )

    swept_rows = element_rows(swept.grid, swept)
    axisymmetric_rows = element_rows(axisymmetric.grid, axisymmetric)
    for band in (cavity, outside):
        assert np.all(
            row_deviation(axisymmetric_rows[:, band], reference[:, band])
            < AXISYMMETRIC_TOL
        )

    # the swept arc, pinned two-sided in each band at the size measured there
    near_face = row_deviation(swept_rows[:, cavity], reference[:, cavity]).max()
    clear_face = row_deviation(swept_rows[:, outside], reference[:, outside]).max()
    assert 6e-05 < near_face < 3e-04  # measured 1.3e-04 at 0.0005 m of clearance
    assert 1e-06 < clear_face < 6e-06  # measured 2.5e-06 at 0.0075 m
    assert clear_face < near_face / 10  # the growth toward the face, not a constant

    standoff = np.array([0.3, 0.5, 1.0, 2.0, 3.0])
    far = np.stack(
        [
            SHELL["radius"] + standoff,
            np.zeros_like(standoff),
            SHELL["height"] + standoff / 2,
        ],
        axis=-1,
    )
    reference = square_annulus_rows(far)
    for coilset in (swept, axisymmetric):
        coilset.point.solve(far)
        assert np.all(
            row_deviation(element_rows(coilset.point, coilset), reference)
            < SWEPT_FAR_TOL
        )


def test_skin_section_area_is_the_inscribed_annulus():
    """A hollow CIRCULAR section encloses its two inscribed polygons, not their box.

    The area a square annulus of the same widths would have is 4/pi larger, and a
    ``skin`` returning that figure means the section was taken as its bounding box.
    Stated against the geometry's own closed form so the check does not depend on
    any element, and so a change of facet count lands here rather than as a field
    disagreement thousands of grid points wide.
    """
    shell = inscribed_area(SHELL["outer_width"]) - inscribed_area(SHELL["inner_width"])
    square = SHELL["outer_width"] ** 2 - SHELL["inner_width"] ** 2
    assert np.isclose(square / shell, 4 / np.pi, rtol=2e-3)  # the 64-gon under-fill

    area = np.asarray(swept_shell("skin").subframe["area"], dtype=float)
    assert np.allclose(area, shell, rtol=1e-6)  # every swept member carries the shell
    assert np.asarray(swept_shell("box").subframe["area"], dtype=float)[0] > shell


def test_a_hollow_square_carries_four_thirds_the_inscribed_moment():
    """The two hollow profiles differ in the far field by a pure number, 4/3.

    Away from the section only the total current and the section's mean square
    radius survive, and for a square of side ``s`` against the circle INSCRIBED in
    it the ratio of those is ``(s^2/6) / (s^2/8) = 4/3`` -- for a shell, ``(a^2 +
    b^2)/6`` against ``((a/2)^2 + (b/2)^2)/2``, which is 4/3 for ANY pair of widths,
    so the constant is geometry and carries no fitted figure.

    The excess over a near-filament ring of the same current isolates that moment,
    and the ratio of the two excesses cancels the arc sweep the two share.  This is
    the statement a bounding-box evaluation cannot make: taking a circular section
    as its enclosing square returns the SQUARE's moment for both profiles and the
    ratio collapses to 1, which is what the ratio has to be resolved against.

    Section size and standoff are pinned by competing resolution limits.  The
    section must be wide enough that subtracting the reference ring leaves a stable
    moment, while the next multipole grows with the section's own moment over the
    squared standoff.  At a fifth of a metre that term costs 66% at 0.3 m of
    standoff and 0.3% beyond 1.5 m.  The window below is where neither term
    dominates, and the 64-gon's moment differs from its circle's by about the same
    1.6e-03 as its area does.
    """
    widths = dict(outer_width=0.2, inner_width=0.16)
    offset = np.array([1.5, 2.0, 3.0])
    targets = np.stack(
        [
            SHELL["radius"] + offset,
            np.zeros_like(offset),
            np.full_like(offset, SHELL["height"]),
        ],
        axis=-1,
    )

    def flux(coilset):
        coilset.point.solve(targets)
        return np.asarray(coilset.point.ay, dtype=float)

    # two thousand times narrower than the shells, so a millionth of their moment:
    # the filament limit rather than a fourth section to be accounted for
    filament = CoilSet(field_attrs=["Ay", "Br", "Bz"])
    filament.coil.insert({"rect": (SHELL["radius"], SHELL["height"], 1e-4, 1e-4)})
    filament.saloc["Ic"] = (SHELL["current"],)

    reference = flux(filament)
    circular = flux(swept_shell("skin", **widths)) / reference - 1.0
    square = flux(swept_shell("box", **widths)) / reference - 1.0

    assert np.all(circular > 0.0)  # a finite section always adds flux outside itself
    assert np.allclose(square / circular, 4 / 3, rtol=1.5e-2)  # measured within 0.4%


if __name__ == "__main__":
    pytest.main([__file__])
