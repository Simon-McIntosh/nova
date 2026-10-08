"""Pin the measured producer and registry descriptions of the same MAST shot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
from shapely.affinity import translate

from benchmarks.mast_description_equivalence import (
    GEOMETRY_BOUND_M,
    POSITION_BOUND_M,
    _coil_differences,
)

RECEIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/figures/machine-description-retirement/mdr-equivalence.json"
)


def _receipt() -> dict:
    return json.loads(RECEIPT.read_text())


def test_registry_and_producer_have_forty_four_unique_loop_positions():
    receipt = _receipt()
    assert receipt["registry"]["unique_loop_positions"] == 44
    assert receipt["producer"]["unique_loop_positions"] == 44, (
        receipt["loop_positions_only_in_registry_m"],
        receipt["loop_positions_only_in_producer_m"],
    )
    assert not receipt["loop_positions_only_in_registry_m"]
    assert not receipt["loop_positions_only_in_producer_m"]


def test_registry_and_producer_join_forty_three_reconstruction_loops():
    receipt = _receipt()
    assert receipt["registry"]["reconstruction_loops"] == 46
    assert receipt["producer"]["reconstruction_loops"] == 46
    assert receipt["registry"]["joined_reconstruction_loops"] == 43
    assert receipt["producer"]["joined_reconstruction_loops"] == 43


def test_every_coil_element_has_equal_geometry_within_the_declared_bound():
    receipt = _receipt()
    assert receipt["bounds"]["coil_vertex_coordinate_abs_m"] == GEOMETRY_BOUND_M
    assert len(receipt["coil_elements"]) == 13
    assert all(
        row["vertex_coordinate_abs_m"] is not None
        and row["vertex_coordinate_abs_m"] <= GEOMETRY_BOUND_M
        and row["symmetric_area_m2"] == 0
        for row in receipt["coil_elements"].values()
    )
    assert receipt["maximum_coil_vertex_coordinate_abs_m"] <= GEOMETRY_BOUND_M


def test_a_matching_loop_position_is_measured_to_be_coincident():
    receipt = _receipt()
    registry = np.asarray(receipt["registry_loop_positions_m"], dtype=float)
    producer = np.asarray(receipt["producer_loop_positions_m"], dtype=float)
    # The first emitted loop is a known present signal for this instrument.
    distance = float(np.linalg.norm(registry - producer[0], axis=1).min())
    assert distance <= POSITION_BOUND_M, distance


def test_geometry_instrument_sees_a_millimetre_displacement():
    outline = shapely.box(0.0, 0.0, 1.0, 1.0)
    moved = translate(outline, xoff=0.001)
    original_wkb = shapely.to_wkb(outline, byte_order=1).hex()
    moved_wkb = shapely.to_wkb(moved, byte_order=1).hex()
    difference = _coil_differences({"probe": original_wkb}, {"probe": moved_wkb})
    assert difference["probe"]["vertex_coordinate_abs_m"] >= 0.001 - 1e-12
    assert difference["probe"]["symmetric_area_m2"] > 0
