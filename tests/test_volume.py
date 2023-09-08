from itertools import product
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nova.geometry.polygeom import Polygon
from nova.geometry.rotate import to_axes
from nova.utilities.importmanager import skip_import

with skip_import("vtk"):
    from nova.geometry.vtkgen import VtkFrame
    from nova.geometry.volume import Section, Cell, Sweep


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
    n_points, radius = 30, 5
    width, depth = 0.6, 0.9
    points = np.zeros((n_points, 3))
    theta = np.linspace(0, 2 * np.pi, n_points)
    points[:, 0] = radius * np.cos(theta)
    points[:, 2] = radius * np.sin(theta)
    boundary = Polygon({"r": [0, 0, width, depth]}).points
    coil = Sweep(boundary, points)
    coil.triangulate()
    volume = 2 * np.pi * radius * width * depth
    assert np.isclose(coil.volume(), volume, rtol=1e-2)


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


# def test_to_axes_to_axes():
#    section = Section(np.identity(3))
#    section.to_axes(np.c_[])


# test_to_axes_to_axes()
# assert False


if __name__ == "__main__":
    pytest.main([__file__])
