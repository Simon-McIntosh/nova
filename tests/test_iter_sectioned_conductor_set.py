"""ITER winding placement and conductor circuit contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from nova.circuit.conductor import (
    ConductorSet,
    PolygonSection,
    subdivide_passive_thickness,
)
from nova.circuit.linkage import (
    circuit_linkage_matrix,
    preserve_series_terminal_linkage,
)
from nova.circuit.passive import terminal_impedance_spectrum
from nova.imas.machine import MachineCoil, MachineSection
from nova.imas.pulsedesign import ITER
from scripts.iter_sectioned_conductor_set.build import calculate


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


def test_surface_graded_wall_preserves_parallel_conductance():
    source = ConductorSet(
        r=[2.0],
        z=[0.0],
        dr=[0.08],
        dz=[0.2],
        current_share=[1.0],
        circuit=[0],
    )
    refined = subdivide_passive_thickness(
        source,
        [0],
        np.array([0.08]),
        0.04,
        surface_layer_m=0.006,
        surface_growth=2.0,
    )
    widths = refined.conductors.dr
    assert widths[0] == pytest.approx(0.006)
    assert widths[-1] == pytest.approx(0.006)
    assert widths.max() <= 0.04
    assert widths.sum() == pytest.approx(0.08)
    assert np.sum(1 / refined.resistance_factor) == pytest.approx(1.0)


def test_parallel_linkage_columns_equal_serial_result():
    conductors = ConductorSet(
        r=[1.0, 1.5],
        z=[0.0, 0.2],
        dr=[0.04, 0.03],
        dz=[0.04, 0.03],
        current_share=[2.0, 3.0],
        circuit=[0, 1],
    )
    serial = circuit_linkage_matrix(conductors, [0, 1], workers=1)
    parallel = circuit_linkage_matrix(conductors, [0, 1], workers=2)
    np.testing.assert_allclose(parallel, serial, rtol=0, atol=0)


def test_frequency_grid_rejects_single_sample():
    with pytest.raises(ValueError, match="two frequency samples"):
        calculate(0.04, 3, frequency_samples=1)


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
    resistive = terminal_impedance_spectrum(
        coil,
        np.empty((2, 0)),
        np.empty((0, 0)),
        np.empty(0),
        series,
        frequency,
        terminal_resistance=np.array([0.0057]),
    )[:, 0, 0]
    np.testing.assert_allclose(resistive, bare + 0.0057, rtol=1e-14)


def test_terminal_reconciliation_preserves_differential_section_energy():
    linkage = np.array([[1.0, 0.4, 0.1], [0.4, 1.2, 0.2], [0.1, 0.2, 0.7]])
    series = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    reference = np.array([[3.3, 0.33], [0.33, 0.71]])
    corrected = preserve_series_terminal_linkage(linkage, series, reference)
    np.testing.assert_allclose(series.T @ corrected @ series, reference, rtol=1e-14)
    differential = np.array([1.0, -1.0, 0.0])
    np.testing.assert_allclose(
        differential @ corrected @ differential,
        differential @ linkage @ differential,
        rtol=1e-14,
    )


@pytest.mark.parametrize(
    "major,width,section_radius,offset,tolerance",
    [
        (1.5, 0.19, np.sqrt(0.19**2 / np.pi), -1.75, 0.05),
        (1.0, 0.05, 0.447049 * 0.05, -2.0, 0.01),
    ],
)
def test_single_ring_and_rectangular_coil_match_analytic_inductance(
    major, width, section_radius, offset, tolerance
):
    """Round equivalent and Grover square-section thin-ring limits."""
    turns = 1.0 if width > 0.1 else 12.0
    conductor = ConductorSet(
        r=[major],
        z=[0.0],
        dr=[width],
        dz=[width],
        current_share=[turns],
        circuit=[0],
    )
    measured = circuit_linkage_matrix(
        conductor, [0], section_scale_frac=0.05, section_n_max=24
    )[0, 0]
    expected = (
        4e-7 * np.pi * major * (np.log(8 * major / section_radius) + offset) * turns**2
    )
    assert measured == pytest.approx(expected, rel=tolerance)
