from __future__ import annotations

import copy
import json

import pytest

from benchmarks import derived_criterion_family as family


@pytest.fixture(scope="module")
def receipt() -> dict:
    return family.build_receipt()


def test_receipt_covers_three_consumers_without_solves(receipt):
    summary = receipt["receipt"]

    assert summary["registered_consumer_count"] == 3
    assert summary["derived_bound_group_count"] == 3
    assert summary["coupled_configuration_count"] == 2
    assert summary["terminal_observable_bound_count"] == 69
    assert summary["equilibrium_solves_run"] == 0
    assert receipt["claim_bounds"]["registered_source_constants_changed"] is False


def test_coupled_bound_is_configuration_and_channel_dependent(receipt):
    criterion = receipt["criterion_family"]["coupled_response"]
    rows = {row["configuration"]: row for row in criterion["configuration_table"]}

    assert criterion["formula"] == "B(c, k_c) = D_ref * G(c, k_c)"
    assert criterion["formula_terms"]["D_ref_points"] == pytest.approx(0.098)
    assert criterion["single_scalar_bound_allowed"] is False
    assert rows["elongated_reference"]["gain"] == pytest.approx(6.5998576353591725)
    assert rows["elongated_reference"]["derived_bound_points"] == pytest.approx(
        0.6467860486651989
    )
    assert "radial-dominant" in rows["elongated_reference"]["channel_identity"]
    assert rows["stable_near_circular_control"]["gain"] == pytest.approx(
        106.99077030932676
    )
    assert rows["stable_near_circular_control"][
        "derived_bound_points"
    ] == pytest.approx(10.485095490114023)
    assert "vertical" in rows["stable_near_circular_control"]["channel_identity"]
    assert rows["stable_near_circular_control"]["mode_rate_reading"][
        "value"
    ] == pytest.approx(0.9874290999339198)
    assert all(row["measured_response_within_derived_bound"] for row in rows.values())


def test_coupled_relation_refuses_unmeasured_interpolation(receipt):
    criterion = receipt["criterion_family"]["coupled_response"]

    assert "not an interpolation" in criterion["domain_of_validity"]
    assert "must not be extrapolated" in criterion["domain_of_validity"]
    assert all(row["receipt_inputs"] for row in criterion["configuration_table"])


def test_diiid_anchor_is_an_achieved_residual_not_an_error_estimate(receipt):
    criterion = receipt["criterion_family"]["diiid_forward_gate"]
    anchor = criterion["anchor_object_adjudication"]
    candidates = {row["candidate"]: row for row in criterion["candidate_table"]}

    assert criterion["criterion_status"] == "NO_DEFENSIBLE_DIIID_TOLERANCE_DERIVED"
    assert criterion["selected_relative_residual_bound"] is None
    assert criterion["formula"] is None
    assert anchor["object"] == "achieved terminal relative fixed-point residual floor"
    assert anchor["source_field"] == (
        "rungs[reference_native].solver.terminal_relative_residual"
    )
    assert anchor["is_independent_discretisation_error_estimate"] is False
    assert anchor["value"] == pytest.approx(7.930534999195602e-5)
    assert (
        "uses the gated outcome as its own derivation input"
        in anchor["circularity_verdict"]
    )
    assert criterion["mesh_evidence"]["fine_relative_residual"] == pytest.approx(
        7.930534999195602e-5
    )
    assert candidates[1.0e-5]["achieved_residual_floor_over_candidate"] == (
        pytest.approx(7.930534999195602)
    )
    assert all(
        row["adjudication"] == "UNTRACED_AND_UNPASSABLE_ON_BANKED_DIIID_ROUTE"
        for row in candidates.values()
    )


def test_diiid_mesh_law_describes_the_stall_not_a_gate(receipt):
    criterion = receipt["criterion_family"]["diiid_forward_gate"]
    law = criterion["empirical_stall_mesh_law"]
    correction = criterion["required_correction"]

    assert law["formula"] == "R_stall(h) = R_fine * (h / h_fine)**p_stall"
    assert law["object"] == "achieved-residual stall, not a convergence criterion"
    assert law["reference_mesh"] == "65 by 65 lattice with 4,225 interior cells"
    assert law["p_stall"] == pytest.approx(1.9255759546587183)
    assert correction["numeric_value_available_now"] is False
    assert correction["admissible_future_form"] == (
        "tau(h) = q * E_disc(h), with 0 < q < 1"
    )


def test_diiid_passability_is_reported_from_banked_measurements(receipt):
    diagnostic = receipt["criterion_family"]["diiid_forward_gate"][
        "passability_diagnostic"
    ]
    controls = {row["arm"]: row for row in diagnostic["analytic_control_roots"]}
    frames = diagnostic["current_five_frame_remeasure"]
    rerun = diagnostic["gate_rerun_now"]

    assert controls["declared_passive_currents"]["terminal_relative_residual"] == (
        pytest.approx(1.7571246480565615e-12)
    )
    assert controls["passive_currents_zeroed"]["terminal_relative_residual"] == (
        pytest.approx(1.7414268631766974e-16)
    )
    assert all(row["clears_1e_5"] for row in controls.values())
    assert diagnostic["diiid_reference_native_route"]["clears_1e_5"] is False
    assert diagnostic["diiid_reference_native_route"]["residual_over_bound"] == (
        pytest.approx(7.930534999195602)
    )
    assert frames["frame_count"] == 5
    assert frames["pass_count_at_1e_5"] == frames["pass_count_at_1e_6"] == 0
    assert frames["terminal_residual_minimum"] == pytest.approx(0.08540058568223574)
    assert frames["terminal_residual_maximum"] == pytest.approx(0.16437432961354986)
    assert frames["all_terminal_diverted"] is True
    assert frames["all_krylov_actions_accepted"] is True
    assert rerun["execution"] == "banked re-score only; no equilibrium solve run"
    assert rerun["verdict"] == "FAIL"
    assert (rerun["passing_frames"], rerun["total_frames"]) == (0, 5)


def test_diiid_reference_accuracy_is_traced_without_norm_equation(receipt):
    criterion = receipt["criterion_family"]["diiid_forward_gate"]
    accuracy = criterion["reference_accuracy_trace"]

    assert accuracy["measured_label_representability_median_r_squared"] == 0.949
    assert accuracy["retained_fraction"] == 0.95
    assert accuracy["registered_median_interior_r_squared_bar"] == pytest.approx(
        0.90155
    )
    assert "different normalisations" in accuracy["normalisation_adjudication"]
    assert (
        receipt["claim_bounds"]["diiid_reference_accuracy_equated_to_solver_residual"]
        is False
    )
    assert receipt["claim_bounds"]["diiid_derived_tolerance_available"] is False


def test_unchallenged_family_members_are_byte_unchanged(receipt):
    integrity = receipt["unchallenged_member_integrity"]

    assert integrity["byte_unchanged"] is True
    assert integrity["indented_json_sha256"] == family.UNCHALLENGED_MEMBER_DIGESTS


def test_one_map_parity_uses_the_banked_float64_roundoff_band(receipt):
    parity = receipt["criterion_family"]["terminal_compiled_parity"]
    one_map = parity["one_map_bound"]

    assert one_map["relative_bound"] == pytest.approx(16.0 * one_map["float64_epsilon"])
    assert one_map["relative_bound"] == pytest.approx(3.552713678800501e-15)
    assert one_map["banked_maximum_relative_difference"] < one_map["relative_bound"]
    assert one_map["machine_precision_demonstrated"] is True


def test_terminal_parity_requires_exact_seed_alignment(receipt):
    parity = receipt["criterion_family"]["terminal_compiled_parity"]
    alignment = parity["seed_alignment_criterion"]
    measurement = alignment["direction_dependence_measurement"]

    assert alignment["formula"] == "||s_compiled - s_eager||_2 = 0"
    assert alignment["numeric_tolerance"] == 0.0
    assert alignment["conditional_terminal_parity"] is True
    assert measurement["seed_direction_l2"] == pytest.approx(0.7665327844967637)
    assert measurement["baseline_burst_updates"] == [3, 8, 12]
    assert measurement["alternate_burst_updates"] == []
    assert measurement["baseline_cumulative_separation_growth"] == pytest.approx(
        5379792627.282051
    )
    assert measurement["alternate_cumulative_separation_growth"] == pytest.approx(
        0.061860940695296525
    )
    assert "no calibrated nonzero safe radius" in alignment["derivation"]


def test_all_terminal_observables_have_separate_derived_bounds(receipt):
    registration = receipt["criterion_family"]["terminal_compiled_parity"][
        "terminal_observable_registration"
    ]
    rows = registration["bounds"]

    assert registration["observable_count"] == len(rows) == 69
    assert (
        registration["exact_equality_count"] + registration["dual_envelope_count"] == 69
    )
    assert len({row["observable"] for row in rows}) == 69
    assert all(
        row["receipt_inputs"] == [str(family.PARITY_GATE_SOURCE)] for row in rows
    )
    for row in rows:
        if row["criterion_kind"] == "exact_equality":
            assert "absolute_bound" not in row
        else:
            assert (
                row["absolute_bound"] == row["calibration_maximum_absolute_difference"]
            )
            assert (
                row["relative_bound"] == row["calibration_maximum_relative_difference"]
            )
    assert "not physics tolerances" in registration["calibration_limit"]


def test_every_criterion_names_its_receipt_inputs(receipt):
    criteria = receipt["criterion_family"]

    assert all(criteria[name]["receipt_inputs"] for name in criteria)
    assert all(
        row["receipt_inputs"]
        for row in criteria["coupled_response"]["configuration_table"]
    )
    assert criteria["terminal_compiled_parity"]["one_map_bound"]["receipt_inputs"]
    assert criteria["terminal_compiled_parity"]["seed_alignment_criterion"][
        "receipt_inputs"
    ]


def test_consumer_constants_are_replaced_not_inherited(receipt):
    replacements = receipt["inherited_constants_replaced"]

    assert len(replacements) == 4
    assert any("0.15" in row["inherited_reading"] for row in replacements)
    assert any("1e-6" in row["inherited_reading"] for row in replacements)
    assert any("1e-5" in row["inherited_reading"] for row in replacements)
    assert any("1e-10" in row["inherited_reading"] for row in replacements)
    assert all(row["replacement_reading"] for row in replacements)


def test_source_registration_changes_fail_closed():
    receipts, consumers = family._load_inputs()
    changed = copy.deepcopy(consumers)
    changed[family.DIIID_CONSUMER] = changed[family.DIIID_CONSUMER].replace(
        "GATE_RESIDUAL_TOLERANCE = 1.0e-6",
        "GATE_RESIDUAL_TOLERANCE = 2.0e-6",
    )

    with pytest.raises(RuntimeError, match="hard-coded gate reading changed"):
        family.build_receipt_from_data(receipts, changed)


def test_banked_receipt_matches_regeneration(tmp_path, receipt):
    checked = json.loads(family.OUTPUT_PATH.read_text())
    regenerated = family.write_receipt(tmp_path / "receipt.json")

    assert checked == regenerated == receipt
    assert set(receipt["sources"]) == {
        str(path) for path in (*family.JSON_SOURCES, *family.CONSUMER_SOURCES)
    }
