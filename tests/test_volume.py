import numpy as np
import pytest
import shapely.geometry
from scipy.spatial.transform import Rotation

from nova.geometry.polygen import PolyGen
from nova.geometry.polygeom import Polygon
from nova.geometry.rotate import to_axes
from nova.utilities.importmanager import skip_import

with skip_import("vtk"):
    from nova.geometry.section import (
        Section,
        collapse_collinear,
        poloidal_footprint,
    )
    from nova.geometry.vtkgen import VtkFrame
    from nova.geometry.volume import Cell, Sweep


def swept(loop: np.ndarray, radius=3.0, elevation=0.2, stations=21) -> Sweep:
    """Return a cross-section swept at constant radius about the vertical."""
    angle = np.linspace(0.3, 1.9, stations)
    path = np.stack(
        [
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.full_like(angle, elevation),
        ],
        axis=-1,
    )
    return Sweep(np.c_[loop[:, 0], np.zeros(len(loop)), loop[:, 1]], path)


@pytest.fixture
def boundary():
    return np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]], float)


def test_section_translate(boundary):
    section = Section(boundary)
    section._append()
    for i in range(5):
        section.to_point((i * 5, 0, 2))
        section._append()
    assert len(section) == 6
    assert np.isclose(section.origin, (i * 5, 0, 2)).all()


def test_section_rotate_triad(boundary):
    section = Section(boundary)
    section.to_vector((0, -2, 0), 0)
    assert np.isclose(section.triad[0], (0, -1, 0)).all()


def test_section_rotate_pi(boundary):
    section = Section(boundary)
    section.to_vector((-1, 0, 0), 0)
    assert np.isclose(section.triad[0], (-1, 0, 0)).all()


def test_section_rotate_rotate(boundary):
    section = Section(boundary.copy())
    section.to_vector((0.5, 0.5, 66.7), 0)
    section.to_vector((1, 0, 0), 0)
    assert all([np.isclose(p, b).all() for p, b in zip(section.points, boundary)])


def test_rotate_to_axes(boundary):
    section = Section(boundary)
    target = Rotation.from_euler("x", np.pi).apply(section.triad)
    section.to_axes(target)
    assert np.allclose(section.triad, target)


def test_cell_volume(boundary):
    top = boundary + (0, 0, 3)
    mesh = Cell([boundary, top])
    mesh.triangulate()
    assert np.isclose(mesh.volume(), 6)


def test_cell_closed(boundary):
    base = boundary
    top = base + (0, 0, 3)
    mesh = Cell([base, top])
    assert not mesh.is_closed()
    mesh = Cell([boundary, top], cap=True)
    assert mesh.is_closed()


def test_cell_type(boundary):
    base = boundary
    top = base + (0, 0, 3)
    mesh = Cell([base, top])
    assert isinstance(mesh, VtkFrame)


def test_to_axes_to_axes():
    section = Section(np.identity(3))
    target = Rotation.from_euler("xyz", [0.3, -0.9, 1.4]).apply(section.triad.T).T
    section.to_axes(target)
    triad = section.triad.copy()
    section.to_axes(target)
    assert np.allclose(triad, section.triad)


def test_sweep():
    n_points, radius = 60, 5
    width, depth = 0.6, 0.9
    points = np.zeros((n_points, 3))
    theta = np.linspace(0, 2 * np.pi, n_points)
    points[:, 0] = radius * np.cos(theta)
    points[:, 2] = radius * np.sin(theta)
    boundary = Polygon({"r": [0, 0, width, depth]}).points
    coil = Sweep(boundary, points, align="axes")
    coil.triangulate()
    volume = 2 * np.pi * radius * width * depth
    assert np.isclose(coil.volume(), volume, rtol=1e-2)


@pytest.mark.parametrize("align", ["vector", "axes"])
def test_sweep_keeps_the_sections_width_radial_and_its_height_vertical(align):
    """A swept conductor must have the dimensions its cross-section declares.

    Volume alone cannot see this -- it is the same for a section and for its
    transpose -- so a sweep that rotated every section a quarter turn passed
    every test there was.  The two alignments have to agree for a path in a
    plane of constant height, because there is only one sensible answer there.
    """
    radius, width, height = 5.0, 0.6, 0.9
    angle = np.linspace(0.3, 1.9, 40)
    path = np.stack(
        [radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)], axis=-1
    )
    solid = Sweep(Polygon({"r": [0, 0, width, height]}).points, path, align=align)
    vertices = np.asarray(solid.triangulate().vertices)
    span = np.hypot(vertices[:, 0], vertices[:, 1])
    assert np.isclose(np.ptp(span), width, atol=1e-6)
    assert np.isclose(np.ptp(vertices[:, 2]), height, atol=1e-6)


@pytest.mark.parametrize(
    "sequence,angles",
    [
        ("x", np.pi / 3),
        ("xyz", [np.pi / 8, 0.133, -1.4]),
        ("xz", [-0.4, 2 * np.pi]),
        ("zy", [-3, 0.04]),
    ],
)
def test_to_axes(sequence, angles):
    theta = np.linspace(0, 2 * np.pi, 30)
    boundary = np.zeros((len(theta), 3))
    boundary[:, 0] = np.cos(theta)
    boundary[:, 2] = np.sin(theta)
    section = Section(boundary)
    target = Rotation.from_euler("x", np.pi / 2).apply(section.triad.T).T
    Rmat = to_axes(target, section.triad)
    assert np.allclose(Rmat.as_matrix() @ section.triad, target)
    assert np.allclose(Rmat.apply(section.triad.T).T, target)


# ---------------------------------------------------------------------------
# The section a sweep carries, and the poloidal footprint it reduces to.


def test_a_sweep_keeps_its_corner_loops_at_double_precision():
    """The mesh cannot be the geometry of record; the loops it was built from can.

    VTK's points default to ``VTK_FLOAT``, so a corner authored at 2.97 reads back
    from the mesh at 2.96999979.  That is 8e-09 relative -- four orders above a
    closed-form section reduction's own round-off -- and it is why the sweep keeps
    the transformed loops beside the mesh instead of measuring the mesh.  Forcing
    VTK to double precision would double every mesh in the frame and fix nothing
    else.
    """
    solid = swept(
        np.array([[2.97, 0.18], [3.03, 0.18], [3.03, 0.22], [2.97, 0.22]])
        - np.array([3.0, 0.2])
    )
    loops = solid.section_loops
    assert loops.dtype == np.float64
    radial = np.hypot(loops[..., 0], loops[..., 1])
    assert abs(float(np.max(radial)) - 3.03) <= 1e-15
    assert abs(float(np.min(radial)) - 2.97) <= 1e-15
    assert abs(float(np.max(np.abs(loops[..., 2] - 0.2))) - 0.02) <= 1e-15
    mesh = np.hypot(*np.asarray(solid.clone().triangulate().vertices)[:, :2].T)
    assert abs(float(np.max(mesh)) - 3.03) > 1e-9  # measured 2.1e-08


def test_collapse_collinear_keeps_the_corners_and_drops_the_rest():
    """Zero-length edges, the closing repeat, and points part way along an edge."""
    square = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    split = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.5],  # part way along an edge
            [1.0, 1.0],
            [1.0, 1.0],  # a zero-length edge
            [0.5, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],  # the closing repeat
        ]
    )
    collapsed = collapse_collinear(split)
    assert len(collapsed) == 4
    assert np.allclose(np.sort(collapsed, axis=0), np.sort(square, axis=0))


def test_collapse_collinear_leaves_a_finely_resolved_curve_alone():
    """A generated disc turns 2 pi / 64 per corner, nowhere near collinear."""
    disc = np.asarray(PolyGen("disc")(0.0, 0.0, 0.06).exterior.coords, dtype=float)
    assert len(collapse_collinear(disc)) == 64


def test_the_footprint_of_a_constant_radius_sweep_is_the_section():
    """The case every arc element in an axisymmetric frame reads.

    Every station projects onto the same ring, so the union of the projections is
    the cross-section itself -- exactly, not to a hull's tolerance.  Measured on a
    hexagon, whose footprint arrives with more corners than the section has and has
    to come back with six.
    """
    section = PolyGen("hexagon")(3.0, 0.2, 0.06, 0.04)
    exact = collapse_collinear(np.asarray(section.exterior.coords, dtype=float))
    loop = np.asarray(PolyGen("hexagon")(0.0, 0.0, 0.06, 0.04).exterior.coords, float)
    footprint = poloidal_footprint(swept(loop).section_loops)
    corners = collapse_collinear(np.asarray(footprint.exterior.coords, dtype=float))
    assert len(corners) == 6
    assert np.isclose(footprint.area, section.area, rtol=1e-12)
    order = [
        int(np.argmin(np.linalg.norm(exact - corner, axis=1))) for corner in corners
    ]
    assert sorted(order) == list(range(6))
    assert np.max(np.abs(corners - exact[order])) <= 1e-15  # measured 4.4e-16


def test_the_footprint_of_a_concave_section_stays_concave():
    """A notch has to survive the projection, or an L becomes a rectangle.

    The stations themselves are unioned rather than the solid bands between them
    for exactly this reason: a band's projection is only cheap if it is convexified,
    and a hull across two stations fills the notch in.  What survives is measured as
    the footprint being strictly smaller than its own convex hull, and as the
    reflex corner still being there.
    """
    loop = np.array(
        [[-0.03, -0.02], [0.03, -0.02], [0.02, 0.0], [0.03, 0.02], [-0.01, 0.015]]
    )
    exact = shapely.geometry.Polygon(loop + np.array([3.0, 0.2]))
    footprint = poloidal_footprint(swept(loop).section_loops)
    hull = shapely.geometry.Polygon(footprint.exterior).convex_hull
    assert footprint.area < 0.99 * hull.area
    assert np.isclose(footprint.area, exact.area, rtol=1e-03)
    corners = collapse_collinear(np.asarray(footprint.exterior.coords, dtype=float))
    assert np.min(np.linalg.norm(corners - np.array([3.02, 0.2]), axis=1)) <= 1e-04


# def test_to_axes_to_axes():
#    section = Section(np.identity(3))
#    section.to_axes(np.c_[])


# test_to_axes_to_axes()
# assert False


if __name__ == "__main__":
    pytest.main([__file__])
