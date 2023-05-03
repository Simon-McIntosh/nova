
import numpy as np
import pytest

try:
    from nova.geometry.vtkgen import VtkFrame
    from nova.geometry.volume import Section, Cell, Sweep
except ModuleNotFoundError:
    pytest.skip("vtk modules not available\n"
                "try pip install -e .['vtk']",
                allow_module_level=True)


def test_section_translate():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    section = Section(base)
    section.append()
    for i in range(5):
        section.to_point((i*5, 0, 2))
        section.append()
    assert len(section) == 6
    assert np.isclose(section.origin, (i*5, 0, 2)).all()


def test_section_rotate_triad():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    section = Section(base)
    section.to_vector((0, -2, 0), 0)
    assert np.isclose(section.triad[0], (0, -1, 0)).all()


def test_section_rotate_pi():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    section = Section(base)
    section.to_vector((-1, 0, 0), 0)
    assert np.isclose(section.triad[0], (-1, 0, 0)).all()


def test_section_rotate_rotate():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    section = Section(base.copy())
    section.to_vector((0.5, 0.5, 66.7), 0)
    section.to_vector((1, 0, 0), 0)
    assert all([np.isclose(p, b).all()
                for p, b in zip(section.points, base)])


def test_cell_volume():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    top = base + (0, 0, 3)
    mesh = Cell([base, top])
    mesh.triangulate()
    assert np.isclose(mesh.volume(), 6)


def test_cell_closed():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    top = base + (0, 0, 3)
    mesh = Cell([base, top])
    assert mesh.is_closed()


def test_cell_type():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    top = base + (0, 0, 3)
    mesh = Cell([base, top])
    assert isinstance(mesh, VtkFrame)


def test_sweep():
    n_points, radius = 30, 5
    width, depth = 0.6, 0.9
    points = np.zeros((n_points, 3))
    theta = np.linspace(0, 2*np.pi, n_points)
    points[:, 0] = radius * np.cos(theta)
    points[:, 2] = radius * np.sin(theta)
    coil = Sweep(dict(r=(0, 0, width, depth)), points)
    coil.triangulate()
    volume = 2*np.pi * radius * width * depth
    assert np.isclose(coil.volume(), volume, rtol=1e-2)


if __name__ == '__main__':

    pytest.main([__file__])
