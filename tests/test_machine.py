"""Cover the machine geometry reader seam.

CrossSection dispatches poloidal geometry through a
:class:`~nova.imas.machine.MachineGeometryReader` provider so an IMAS IDS node
and a (FAIR-MAST) zarr source are interchangeable. These tests exercise the
default IMAS provider on a synthetic IDS-shaped node and a dict-backed provider
standing in for a columnar/zarr source, and assert both yield the same section.
"""

import numpy as np

from nova.imas.machine import CrossSection, GeomData, MachineGeometryReader
from nova.imas.test_utilities import mark


class _Value:
    """Mimic an IMAS scalar node exposing a ``.value`` attribute."""

    def __init__(self, value):
        self.value = value


class _Node:
    """Mimic an IMAS structure node with attribute access."""

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)


def _imas_rectangle():
    """Return a synthetic IMAS geometry node for a rectangular element."""
    return _Node(
        geometry_type=_Value(2),
        rectangle=_Node(
            r=_Value(4.0), z=_Value(0.5), width=_Value(0.2), height=_Value(0.3)
        ),
    )


class DictGeometryReader(MachineGeometryReader):
    """Present a plain dict (a columnar/zarr-style source) through the seam."""

    @property
    def geometry_type(self) -> int:
        return self.source["geometry_type"]

    def section(self, geometry: type[GeomData]) -> GeomData:
        data = {attr: self.source[attr] for attr in geometry.attrs}
        return geometry(None, data)


class DictCrossSection(CrossSection):
    """CrossSection wired to the dict-backed provider instead of IMAS."""

    reader = DictGeometryReader


@mark["imas"]
def test_imas_reader_dispatches_rectangle():
    section = CrossSection(_imas_rectangle())
    assert section.name == "rectangle"
    assert section.data.data["width"] == 0.2
    assert np.isclose(section.area, 0.2 * 0.3)


@mark["imas"]
def test_reader_seam_is_interchangeable():
    imas_section = CrossSection(_imas_rectangle())
    dict_source = {
        "geometry_type": 2,
        "r": 4.0,
        "z": 0.5,
        "width": 0.2,
        "height": 0.3,
    }
    dict_section = DictCrossSection(dict_source)
    assert dict_section.name == imas_section.name == "rectangle"
    assert np.isclose(dict_section.area, imas_section.area)
