
import numpy as np
import pytest


from nova.geometry.vtkgen import VtkFrame
from nova.geometry.volume import Section, Cell


def test_section_translate():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    section = Section(base)
    for i in range(5):
        section.to_point((i*5, 0, 2))
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
    mesh = Cell(base, top, cap_base=True, cap_top=True)
    mesh.triangulate()
    assert np.isclose(mesh.volume(), 6)


def test_cell_closed():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    top = base + (0, 0, 3)
    mesh = Cell(base, top, cap_base=True, cap_top=True)
    assert mesh.isClosed()


def test_cell_type():
    base = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    top = base + (0, 0, 3)
    mesh = Cell(base, top, cap_base=True, cap_top=True)
    assert isinstance(mesh, VtkFrame)


if __name__ == '__main__':

    pytest.main([__file__])
