"""Tests for the circuit-coupling generator (nova.biot.coupling).

The golden fixture ``tests/data/coupling_classify_golden.json`` was produced
by the reference implementation operating on the MAST machine description
(23 pfCircuit / 24 pfCoil / 23 pfSupply): each fixture records the filament
set, the available measurement channels, and the reference classification.
The machine description itself travels inside the fixture as data — the
mechanism under test is machine-agnostic.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from nova.biot.coupling import (
    CaseChannel,
    CircuitTable,
    CoilChannel,
    classify_circuits,
    couple_circuits,
)

GOLDEN = Path(__file__).parent / "data" / "coupling_classify_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def table(golden) -> CircuitTable:
    """Build the machine circuit table from the golden fixture's data."""
    centroids = golden["coil_centroids"]
    preferred = golden["coil_channels"]
    coils = tuple(
        CoilChannel(
            label=label,
            centroid=(centroids[label][0], centroids[label][1]),
            channels=(preferred[label], f"{label}_current"),
        )
        for label in centroids
    )
    cases = tuple(
        CaseChannel(
            circuit=case["circuit_id"],
            coil_label=case["geometry_confusable_with"],
            channel=case["l1_case_channel"],
            constrained_zero=case["constrained_zero"],
        )
        for case in golden["machine"]["case_circuits"]
    )
    return CircuitTable(coils=coils, cases=cases, match_radius=golden["match_radius_m"])


def _arrays(fixture):
    filaments = fixture["filaments"]
    circuit = np.array([f["circuit"] for f in filaments], dtype=int)
    radius = np.array([f["r"] for f in filaments], dtype=float)
    height = np.array([f["z"] for f in filaments], dtype=float)
    weight = np.array([f["xmult"] for f in filaments], dtype=float)
    return circuit, radius, height, weight


def test_golden_classification_parity(golden, table):
    """Every golden fixture classifies identically to the reference."""
    for name, fixture in golden["fixtures"].items():
        circuit, radius, height, weight = _arrays(fixture)
        classes = classify_circuits(
            circuit, radius, height, weight, fixture["channels"], table
        )
        reference = fixture["classes"]
        assert len(classes) == len(reference), name
        for ours, ref in zip(classes, reference, strict=True):
            assert ours.circuit == ref["circuit"], name
            assert ours.role == ref["role"], (name, ours.circuit)
            assert ours.coil_label == ref["coil_label"], (name, ours.circuit)
            assert ours.channel == ref["amc_channel"], (name, ours.circuit)
            assert ours.centroid_radius == pytest.approx(ref["centroid_r"])
            assert ours.centroid_height == pytest.approx(ref["centroid_z"])
            assert ours.filament_count == ref["n_filament"], name
            assert ours.weight_sum == pytest.approx(ref["sum_xmult"])
            assert bool(ours.flag) == bool(ref["flag"]), (name, ours.circuit)


def test_case_circuit_gets_its_own_channel(golden, table):
    """A case circuit 2 cm from its active coil keeps its own channel."""
    fixture = golden["fixtures"]["p4u_active_and_case"]
    classes = classify_circuits(*_arrays(fixture), fixture["channels"], table)
    by_circ = {c.circuit: c for c in classes}
    assert by_circ[8].role == "known_pf"
    assert by_circ[8].channel == "p4u_coil_current"
    assert by_circ[18].role == "known_case"
    assert by_circ[18].coil_label == "p4u_case"
    assert by_circ[18].channel == "p4u_case_current"


def test_case_without_channel_is_inferred_with_flag(golden, table):
    """Missing case channel → inferred passive, never the active channel."""
    fixture = golden["fixtures"]["p4u_case_channel_absent"]
    classes = classify_circuits(*_arrays(fixture), fixture["channels"], table)
    by_circ = {c.circuit: c for c in classes}
    assert by_circ[18].role == "inferred_passive"
    assert by_circ[18].channel == ""
    assert by_circ[18].flag
    assert by_circ[8].role == "known_pf"


def test_constrained_zero_case_ignores_injected_channel(golden, table):
    """A constrained-to-zero case stays passive even if a channel appears."""
    fixture = golden["fixtures"]["p6u_case_injected_channel"]
    classes = classify_circuits(*_arrays(fixture), fixture["channels"], table)
    by_circ = {c.circuit: c for c in classes}
    assert by_circ[12].role == "known_pf"
    assert by_circ[22].role == "inferred_passive"
    assert by_circ[22].flag


def test_couple_merges_redundant_discretisations(golden, table):
    """Two circuits sharing one drive channel merge into one averaged column."""
    fixture = golden["fixtures"]["redundant_merge"]
    circuit, radius, height, weight = _arrays(fixture)
    classes = classify_circuits(
        circuit, radius, height, weight, fixture["channels"], table
    )
    plan = couple_circuits(classes)
    assert len(plan.columns) == 1
    column = plan.columns[0]
    assert column.channel == "p4u_coil_current"
    assert column.circuits == (8, 88)
    assert plan.passive == ()


def test_couple_keeps_case_column_separate(golden, table):
    fixture = golden["fixtures"]["p4u_active_and_case"]
    classes = classify_circuits(*_arrays(fixture), fixture["channels"], table)
    plan = couple_circuits(classes)
    assert [c.channel for c in plan.columns] == [
        "p4u_case_current",
        "p4u_coil_current",
    ]
    assert all(len(c.circuits) == 1 for c in plan.columns)


def test_couple_routes_zero_case_to_passive(golden, table):
    fixture = golden["fixtures"]["p6u_case_injected_channel"]
    classes = classify_circuits(*_arrays(fixture), fixture["channels"], table)
    plan = couple_circuits(classes)
    assert [c.channel for c in plan.columns] == ["p6u_current"]
    assert plan.passive == (22,)


def test_weight_matrix_averages_merged_columns(golden, table):
    """The per-source weight matrix reproduces merge-by-averaging exactly.

    For a column merging n circuits, each member filament carries
    ``xmult / n`` so a coupling matrix ``G_sources`` collapses to the merged
    column via one matmul: ``G_sources @ weights``.
    """
    fixture = golden["fixtures"]["redundant_merge"]
    circuit, radius, height, weight = _arrays(fixture)
    classes = classify_circuits(
        circuit, radius, height, weight, fixture["channels"], table
    )
    plan = couple_circuits(classes)
    weights = plan.weight_matrix(circuit, weight)
    assert weights.shape == (2, 1)
    np.testing.assert_allclose(weights[:, 0], [0.5, 0.5])

    # synthetic per-source couplings: the merged column is the mean of the
    # per-circuit (weight-summed) responses — the double-count fix.
    g_sources = np.array([[2.0, 4.0]])  # one sensor, two filaments
    np.testing.assert_allclose(g_sources @ weights, [[3.0]])


def test_weight_matrix_passive_columns_carry_raw_weight(golden, table):
    fixture = golden["fixtures"]["p6u_case_injected_channel"]
    circuit, radius, height, weight = _arrays(fixture)
    classes = classify_circuits(
        circuit, radius, height, weight, fixture["channels"], table
    )
    plan = couple_circuits(classes)
    weights = plan.weight_matrix(circuit, weight, include_passive=True)
    # columns: [p6u_current, passive circuit 22]
    assert weights.shape == (2, 2)
    np.testing.assert_allclose(weights, [[1.0, 0.0], [0.0, 1.0]])


def test_all_active_circuits_classify_known(golden, table):
    fixture = golden["fixtures"]["all_13_active"]
    classes = classify_circuits(*_arrays(fixture), fixture["channels"], table)
    assert len(classes) == 13
    assert all(c.role == "known_pf" for c in classes)
    plan = couple_circuits(classes)
    assert len(plan.columns) == 13


def test_unmatched_circuit_is_inferred_passive(golden, table):
    fixture = golden["fixtures"]["far_passive"]
    classes = classify_circuits(*_arrays(fixture), fixture["channels"], table)
    assert classes[0].role == "inferred_passive"
    assert classes[0].coil_label == ""
    assert classes[0].channel == ""
