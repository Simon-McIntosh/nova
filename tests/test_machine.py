"""Cover the machine geometry reader seam.

CrossSection dispatches poloidal geometry through a
:class:`~nova.imas.machine.MachineGeometryReader` provider so an IMAS IDS node
and a flat tabular source (a FAIR-MAST zarr row, a columnar record) are
interchangeable. These tests exercise the default IMAS provider on a synthetic
IDS-shaped node and the packaged :class:`~nova.imas.machine.
TabularGeometryReader`, and assert both yield the same section.
"""

import numpy as np

from nova.imas.machine import CrossSection, TabularGeometryReader
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


def _imas_annulus():
    """Return a synthetic IMAS geometry node for an annular element."""
    return _Node(
        geometry_type=_Value(5),
        annulus=_Node(
            r=_Value(1.5),
            z=_Value(-0.4),
            radius_inner=_Value(0.05),
            radius_outer=_Value(0.1),
        ),
    )


class TabularCrossSection(CrossSection):
    """CrossSection wired to the tabular provider instead of IMAS."""

    reader = TabularGeometryReader


@mark["imas"]
def test_imas_reader_dispatches_rectangle():
    section = CrossSection(_imas_rectangle())
    assert section.name == "rectangle"
    assert section.data.data["width"] == 0.2
    assert np.isclose(section.area, 0.2 * 0.3)


@mark["imas"]
def test_reader_seam_is_interchangeable():
    imas_section = CrossSection(_imas_rectangle())
    tabular_section = TabularCrossSection(
        {"geometry_type": 2, "r": 4.0, "z": 0.5, "width": 0.2, "height": 0.3}
    )
    assert tabular_section.name == imas_section.name == "rectangle"
    assert np.isclose(tabular_section.area, imas_section.area)


@mark["imas"]
def test_tabular_annulus_matches_imas_normalisation():
    """The annulus record normalises identically through either provider."""
    imas_section = CrossSection(_imas_annulus())
    tabular_section = TabularCrossSection(
        {
            "geometry_type": 5,
            "r": 1.5,
            "z": -0.4,
            "radius_inner": 0.05,
            "radius_outer": 0.1,
        }
    )
    assert tabular_section.name == imas_section.name == "annulus"
    assert tabular_section.data.data["width"] == imas_section.data.data["width"]
    assert np.isclose(tabular_section.area, imas_section.area)


@mark["imas"]
def test_tabular_reader_reports_missing_attribute():
    """A record missing a required attribute fails loudly, never fabricates."""
    record = {"geometry_type": 2, "r": 4.0, "z": 0.5, "width": 0.2}
    try:
        TabularCrossSection(record)
    except KeyError as error:
        assert "height" in str(error)
    else:
        raise AssertionError("missing attribute must raise KeyError")
