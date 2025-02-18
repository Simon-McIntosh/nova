import os

import numpy as np
import pytest
import tempfile

from nova.utilities.importmanager import skip_import

with skip_import("vtk"):
    import vedo

from nova.frame.coilset import CoilSet
from nova.frame.framespace import FrameSpace
from nova.geometry.volume import Ring
from nova.geometry.vtkgen import VtkFrame


def test_add_single_volume_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3)
    frame = FrameSpace(segment="vtk")
    frame += dict(vtk=box)
    assert np.isclose(frame.volume.iloc[0], 6)


def test_add_single_area_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=1e-3, height=2)
    frame = FrameSpace(segment="vtk")
    frame += dict(vtk=box)
    assert np.isclose(frame.area.iloc[0], 4)


def test_add_single_framearea_rotate_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=1e-3, height=2)
    frame = FrameSpace(segment="vtk")
    frame += dict(vtk=box.rotate_z(45))
    assert np.isclose(frame.area.iloc[0], 4)


def test_add_single_polyarea_rotate_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=1e-3, height=2)
    frame = FrameSpace(segment="vtk") + dict(vtk=box.rotate_z(45))
    assert np.isclose(frame.poly.iloc[0].area, 4)


def test_save_load_vtk():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=1e-3, height=2)
    frame = FrameSpace(segment="vtk")
    frame += dict(vtk=box)
    frame.vtk.iloc[0].triangulate()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        frame.store(tmp.name, "frame", vtk=True)
        new_frame = FrameSpace().load(tmp.name, "frame")
    os.unlink(tmp.name)
    assert np.isclose(new_frame.vtk.iloc[0].volume(), frame.vtk.iloc[0].volume())


def test_box_bounds():
    box = vedo.shapes.Box(pos=(5, 0, 0), length=2, width=8.3, height=2)
    frame = FrameSpace() + dict(vtk=box, name="box", segment="vtk")
    assert np.isclose(
        np.array(frame.loc["box", ["dx", "dy", "dz"]].values, float),
        np.array([2, 8.3, 2]),
    ).all()


def test_poly_vtk():
    frame = FrameSpace(segment="vtk") + dict(x=3, z=2, section="hex")
    frame += dict(
        vtk=vedo.shapes.Box(pos=(3.2, 0, 2), length=0.1, width=0.01, height=0.2)
    )
    assert np.isclose(frame.area.iloc[1], 0.1 * 0.2)


def test_vtk_poly():
    frame = FrameSpace(segment="vtk") + dict(
        vtk=vedo.shapes.Box(pos=(3.2, 0, 2), length=0.1, width=0.01, height=0.2)
    )
    frame += dict(x=3, z=2, dl=0.2, dt=0.6, section="square")
    assert np.isclose(frame.area.iloc[1], 0.2**2)


def test_vtk_skin():
    frame = FrameSpace(required=["x", "z", "dl", "dt"], available=["vtk"])
    frame.insert(5, 2, 4, 0.2, section="skin")
    frame.vtkgeo.generate_vtk()
    assert isinstance(frame.vtk.iloc[0], VtkFrame)


def test_vtk_from_poly():
    frame = FrameSpace(required=["x", "z", "dl", "dt"], available=["vtk"])
    frame.insert(3, 2.2, 0.1, 0.1)
    assert frame.at["Coil0", "vtk"] is None
    frame.vtkgeo.generate_vtk()
    assert isinstance(frame.at["Coil0", "vtk"], Ring)


def test_coilset_default_vtk():
    coilset = CoilSet()
    coilset.coil.insert(6, 7.5, 0.5, 0.5)
    assert coilset.frame.at["Coil0", "vtk"] is None
    coilset.frame.vtkgeo.generate_vtk()
    assert isinstance(coilset.frame.at["Coil0", "vtk"], Ring)


if __name__ == "__main__":
    pytest.main([__file__])
