"""ITER winding placement and conductor circuit contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from nova.circuit.conductor import (
    ConductorSet,
    PolygonSection,
    subdivide_passive_thickness,
)
from nova.circuit.passive import terminal_impedance_spectrum
from nova.imas.machine import MachineCoil, MachineSection
from nova.imas.pulsedesign import ITER


def _rectangle(name: str, turns: float) -> MachineSection:
    return MachineSection.from_record(
        {
            "name": name,
            "geometry_type": 2,
            "r": 4.0,
            "z": 1.0,
            "width": 0.6,
            "height": 0.8,
            "turns_with_sign": turns,
        }
    )


def test_winding_sections_conserve_signed_turns_and_pack_geometry():
    coil = MachineCoil("PF1", "PF1", (_rectangle("PF1", 248.6),))
    sections = coil.winding_sections(8)
    assert len(sections) == 8
    assert sum(section.turns for section in sections) == pytest.approx(248.6)
    assert sum(section.height for section in sections) == pytest.approx(0.8)
    assert np.mean([section.centre[1] for section in sections]) == pytest.approx(1.0)
    assert sections[0].centre[1] < sections[-1].centre[1]


def test_winding_sections_reject_unknown_layout():
    coil = MachineCoil("PF1", "PF1", (_rectangle("a", 1), _rectangle("b", 1)))
    with pytest.raises(ValueError, match="layout"):
        coil.winding_sections(8)


def test_iter_layout_uses_pancakes_and_retains_in_vessel_turns():
    rectangle = SimpleNamespace(r=4.0, z=1.0, width=0.6, height=0.8)
    active = SimpleNamespace(
        coil=[
            SimpleNamespace(
                name="Central Solenoid 3U (CS3U)",
                element=[
                    SimpleNamespace(
                        geometry=SimpleNamespace(geometry_type=2, rectangle=rectangle),
                        turns_with_sign=554.0,
                    )
                ],
            ),
            SimpleNamespace(
                name="Poloidal Field Coil 2 (PF2)",
                element=[
                    SimpleNamespace(
                        geometry=SimpleNamespace(geometry_type=2, rectangle=rectangle),
                        turns_with_sign=115.2,
                    )
                ],
            ),
        ]
    )
    cs, pf = ITER.winding_sections(active)
    assert (len(cs.elements), len(pf.elements)) == (40, 6)
    assert sum(section.turns for section in cs.elements) == pytest.approx(554.0)
    assert sum(section.turns for section in pf.elements) == pytest.approx(115.2)


def test_passive_wall_layers_preserve_area_and_source_resistance():
    wall = np.array([[2.0, -0.05], [2.5, -0.05], [2.5, 0.05], [2.0, 0.05]])
    original = ConductorSet(
        r=[1.0, 2.25],
        z=[0.0, 0.0],
        dr=[0.1, 0.5],
        dz=[0.1, 0.1],
        current_share=[8.0, 1.0],
        circuit=[0, 1],
        polygon_sections=(PolygonSection(1, wall),),
    )
    refined = subdivide_passive_thickness(original, [1], np.array([0.1]), 0.025)
    assert refined.conductors.n_filaments == 5
    assert refined.conductors.current_share[0] == 8.0
    assert len(refined.circuits) == 4
    assert np.all(refined.parent == 1)
    assert np.all(refined.resistance_factor == 4)
    assert sum(
        abs(np.linalg.det(np.diff(section.vertices[:3], axis=0)))
        for section in refined.conductors.polygon_sections
    ) == pytest.approx(0.05)


def test_passive_screening_changes_terminal_impedance_and_removal_is_bare():
    frequency = np.geomspace(0.1, 5000.0, 20)
    coil = np.array([[0.7, 0.3], [0.3, 0.5]])
    series = np.ones((2, 1))
    coupling = np.array([[0.1], [0.05]])
    passive = np.array([[0.04]])
    resistance = np.array([0.001])
    screened = terminal_impedance_spectrum(
        coil, coupling, passive, resistance, series, frequency
    )[:, 0, 0]
    bare = 2j * np.pi * frequency * float(series[:, 0] @ coil @ series[:, 0])
    assert np.max(np.abs(screened / bare - 1)) > 0.01
    removed = terminal_impedance_spectrum(
        coil, np.empty((2, 0)), np.empty((0, 0)), np.empty(0), series, frequency
    )[:, 0, 0]
    np.testing.assert_allclose(removed, bare, rtol=1e-14)
