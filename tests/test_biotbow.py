"""Conditioning contracts for rectangular finite-arc sections."""

import numpy as np
import pytest

from benchmarks.bow_grid_solve import bow_arbiter_record
from nova.biot.biotframe import Source, Target
from nova.biot.bow import Bow
from nova.biot.polybeam import _oriented_loop
from nova.biot.polybow import section_area
from nova.frame.coilset import CoilSet


def _bow_inputs() -> tuple[Source, Target]:
    """Return one ordinary rectangular finite arc and an exterior target."""
    angle = np.array([-0.2, 0.0, 0.2])
    radius = 2.0
    coilset = CoilSet()
    coilset.winding.insert(
        np.column_stack(
            [radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)]
        ),
        {"rect": (0.0, 0.0, 0.08, 0.04)},
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
    )
    frame = coilset.subframe
    assert len(frame) == 1
    source = Source(
        {column: np.asarray(frame[column]) for column in frame.columns},
        index=list(frame.index),
    )
    target = Target({"x": [2.5], "y": [0.1], "z": [0.3]})
    return source, target


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("area", 0.0, "positive area"),
        ("area", np.nan, "positive area"),
        ("width", 0.0, "positive section width"),
        ("height", np.inf, "positive section height"),
    ],
)
def test_bow_rejects_nonphysical_section_normalisation(field, value, expected):
    """Zero or nonfinite material dimensions have no finite-volume topology."""
    source, target = _bow_inputs()
    source.loc[:, field] = value
    if field in {"width", "height"}:
        source.loc[:, "poly"] = None
    with pytest.raises(ValueError, match=expected):
        Bow(source, target, turns=[False, False], reduce=[False, False])


@pytest.mark.parametrize("offset", [1.0e8, 1.0e9])
def test_section_area_and_orientation_are_translation_stable(offset):
    """Small-section shoelace arithmetic is independent of global position."""
    local = np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.02], [0.0, 0.02]])
    translated = local + np.array([offset, -offset])
    assert section_area(translated) == pytest.approx(0.008, rel=2e-6)
    oriented = _oriented_loop(translated[::-1])
    assert section_area(oriented) == pytest.approx(0.008, rel=2e-6)
    first = oriented[1] - oriented[0]
    second = oriented[2] - oriented[1]
    assert first[0] * second[1] - first[1] * second[0] > 0.0


def test_bow_matches_the_direct_volume_integral_in_its_worst_sampled_regime():
    """The complete reduction against the defining three-dimensional integral.

    One target is in the ordinary far field and one is just outside an end-corner
    of the swept section, where every integration direction varies on the section
    scale.  The direct Gauss volume integral shares no elliptic or zeta code with
    Bow.  At 64 nodes per direction its last refinement moves 4.9e-12 relative;
    Bow's worst row is 1.84e-10 relative, on the far target's vector potential.
    """
    record = bow_arbiter_record()
    assert record["worst_production_relative_error"] == pytest.approx(
        1.84e-10, rel=0.08
    )
    assert record["production_relative_error"][1] < 1.0e-11
    assert record["worst_arbiter_relative_change_48_to_64"] < 5.0e-12
