"""Cover the machine geometry reader seam.

CrossSection dispatches poloidal geometry through a
:class:`~nova.imas.machine.MachineGeometryReader` provider so an IMAS IDS node
and a flat tabular source (a FAIR-MAST zarr row, a columnar record) are
interchangeable. These tests exercise the default IMAS provider on a synthetic
IDS-shaped node and the packaged :class:`~nova.imas.machine.
TabularGeometryReader`, and assert both yield the same section.
"""

import numpy as np

from nova.catalog.mast_geometry import EvidenceState, MachineGeometryRegistry
from nova.imas.machine import CrossSection, Outline, TabularGeometryReader
from nova.imas.mast_geometry import (
    DD_VERSION,
    RegistryGeometryReader,
    author_catalog_ids,
    write_and_reopen,
)
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


@mark["imas"]
def test_registry_reader_selects_geometry_and_exposes_evidence():
    readers = RegistryGeometryReader.for_component(11766, "p2_inner_lower")

    assert len(readers) == 1
    assert readers[0].evidence == EvidenceState.OBSERVED
    assert (
        readers[0].physical_digest in MachineGeometryRegistry.default().configurations
    )
    assert readers[0].section(Outline).area > 0


@mark["imas"]
def test_registry_reader_preserves_missing_evidence_state():
    readers = RegistryGeometryReader.for_component(
        26963,
        "botcol",
        passive=True,
    )

    assert readers
    assert all(reader.evidence == EvidenceState.MISSING for reader in readers)


@mark["imas"]
def test_catalog_ids_validate_reopen_and_preserve_diagnostic_geometry(tmp_path):
    selection = MachineGeometryRegistry.default().select(11766)
    bundle = author_catalog_ids(selection)
    reopened = write_and_reopen(bundle, tmp_path / "mast_geometry")

    assert set(reopened) == {"pf_active", "pf_passive", "wall", "magnetics"}
    assert all(
        str(ids.ids_properties.version_put.data_dictionary) == DD_VERSION
        for ids in reopened.values()
    )

    geometry = selection.configuration.geometry
    pf_active = reopened["pf_active"]
    pf_passive = reopened["pf_passive"]
    wall = reopened["wall"]
    magnetics = reopened["magnetics"]

    assert len(pf_active.coil) == len(geometry["active_components"]) == 13
    assert len(pf_passive.loop) == len(geometry["passive_components"]) == 16
    limiter = np.column_stack(
        [
            wall.description_2d[0].limiter.unit[0].outline.r,
            wall.description_2d[0].limiter.unit[0].outline.z,
        ]
    )
    assert np.allclose(limiter[:-1], geometry["limiter"])

    source_probe = geometry["magnetics"]["poloidal_probes"][0]["pose"]
    probe = magnetics.b_field_pol_probe[0]
    assert np.isclose(probe.position.r, source_probe[0])
    assert np.isclose(probe.position.z, source_probe[1])
    assert not probe.position.phi.has_value
    assert np.isclose(probe.poloidal_angle, source_probe[2])
    assert not probe.toroidal_angle.has_value
    assert np.isclose(probe.length, source_probe[3])

    source_poloidal_points = [
        point
        for family, points in sorted(geometry["magnetics"]["additional_points"].items())
        if family.startswith("poloidal_")
        for point in points
    ]
    assert len(source_poloidal_points) == 61
    assert len(magnetics.b_field_pol_probe) == len(
        geometry["magnetics"]["poloidal_probes"]
    ) + len(source_poloidal_points)
    additional_probe = magnetics.b_field_pol_probe[
        len(geometry["magnetics"]["poloidal_probes"])
    ]
    assert np.allclose(
        [
            additional_probe.position.r,
            additional_probe.position.z,
            additional_probe.position.phi,
        ],
        source_poloidal_points[0],
    )
    assert not additional_probe.poloidal_angle.has_value
    assert not additional_probe.toroidal_angle.has_value
    assert not additional_probe.length.has_value
    assert not additional_probe.turns.has_value

    source_toroidal_points = geometry["magnetics"]["additional_points"]["toroidal_cc"]
    assert len(source_toroidal_points) == len(magnetics.b_field_phi_probe) == 36
    phi_probe = magnetics.b_field_phi_probe[0]
    assert np.allclose(
        [phi_probe.position.r, phi_probe.position.z, phi_probe.position.phi],
        source_toroidal_points[0],
    )
    assert not phi_probe.poloidal_angle.has_value
    assert not phi_probe.toroidal_angle.has_value
    assert not phi_probe.length.has_value
    assert not phi_probe.turns.has_value

    flux_loop = magnetics.flux_loop[0]
    source_loop = geometry["magnetics"]["flux_loops"][0]
    assert flux_loop.type.index == 1
    assert np.allclose(
        [[point.r, point.z, point.phi] for point in flux_loop.position],
        [[source_loop[0], source_loop[1], 0.0], source_loop],
    )

    first_saddle = magnetics.flux_loop[len(geometry["magnetics"]["flux_loops"])]
    source_saddle = geometry["magnetics"]["saddle_paths"]["l"][0]
    assert first_saddle.type.index == 2
    assert np.allclose(
        [
            [
                first_saddle.position[index].r,
                first_saddle.position[index].z,
                first_saddle.position[index].phi,
            ]
            for index in range(len(first_saddle.position) - 1)
        ],
        source_saddle,
    )
    assert not first_saddle.flux.data.has_value
    assert not first_saddle.voltage.data.has_value
