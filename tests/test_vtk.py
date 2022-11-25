
import numpy as np
import pytest
import tempfile
import vedo

from nova.frame.framespace import FrameSpace
from nova.geometry.volume import VtkFrame


def test_add_single_volume_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3)
    frame = FrameSpace(part='vtk')
    frame += dict(vtk=box)
    assert np.isclose(frame.volume[0], 6)


def test_add_single_area_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=1e-3, height=2)
    frame = FrameSpace(part='vtk')
    frame += dict(vtk=box)
    assert np.isclose(frame.area[0], 4)


def test_add_single_framearea_rotate_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=1e-3, height=2)
    frame = FrameSpace(part='vtk')
    frame += dict(vtk=box.rotate_z(45))
    assert np.isclose(frame.area[0], 4)


def test_add_single_polyarea_rotate_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=1e-3, height=2)
    frame = FrameSpace(part='vtk') + dict(vtk=box.rotate_z(45))
    assert np.isclose(frame.poly[0].area, 4)


def test_save_load_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=1e-3, height=2)
    frame = FrameSpace(part='vtk')
    frame += dict(vtk=box)
    frame.vtk[0].triangulate()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        frame.store(tmp.name, 'frame', vtk=True)
        new_frame = FrameSpace().load(tmp.name, 'frame')
    assert np.isclose(new_frame.vtk[0].volume(), frame.vtk[0].volume())


def test_box_bounds():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=8.3, height=2)
    frame = FrameSpace(part='vtk') + dict(vtk=box, name='box')
    assert np.isclose(np.array(frame.loc['box', ['dx', 'dy', 'dz']].values,
                               float), np.array([2, 8.3, 2])).all()


def test_poly_vtk():
    frame = FrameSpace(part='vtk') + dict(x=3, z=2, section='hex')
    frame += dict(vtk=vedo.shapes.Box(pos=(3.2, 0, 2),
                                      length=0.1, width=0.01, height=0.2))
    assert np.isclose(frame.area[1], 0.1*0.2)


def test_vtk_poly():
    frame = FrameSpace(part='vtk') + dict(vtk=vedo.shapes.Box(
        pos=(3.2, 0, 2), length=0.1, width=0.01, height=0.2))
    frame += dict(x=3, z=2, dl=0.2, dt=0.6, section='square')
    assert np.isclose(frame.area[1], 0.2**2)


def test_vtk_skin():
    frame = FrameSpace(required=['x', 'z', 'dl', 'dt'], available=['vtk'])
    frame.insert(5, 2, 4, 0.2, section='skin')
    frame.vtkgeo.generate_vtk()
    assert isinstance(frame.vtk[0], VtkFrame)


if __name__ == '__main__':

    pytest.main([__file__])
