"""ITER winding placement and conductor circuit contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

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
