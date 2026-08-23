from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.efit_parity_fixed_boundary_seed import (
    FIXED_BOUNDARY_TOLERANCE,
    OUTPUT_FIGURE,
    OUTPUT_RECEIPT,
    PICARD_FINDING,
    TARGET_SHOT,
    TARGET_SLICE,
    _dirichlet_operator,
    run_fixed_boundary_only,
)


@pytest.fixture(scope="module")
def fixed_result(tmp_path_factory):
    output_directory = tmp_path_factory.mktemp("fixed-boundary-seed")
    return run_fixed_boundary_only(
        output_path=output_directory / OUTPUT_RECEIPT.name,
        figure_path=output_directory / OUTPUT_FIGURE.name,
    )


def test_declared_support_dirichlet_retains_every_exterior_node():
    radius = np.linspace(1.0, 2.0, 5)
    height = np.linspace(-1.0, 1.0, 5)
    support = np.zeros((5, 5), dtype=bool)
    support[2, 2] = True
    lattice = type(
        "Lattice",
        (),
        {
            "radius": radius,
            "height": height,
            "interior": lambda self: np.ones(25, dtype=bool),
        },
    )()
    operator_state = type("Operator", (), {"declared_support": support.ravel()})()
    profile = type("Profile", (), {"lattice": lattice, "operator": operator_state})()
    operator = _dirichlet_operator(profile)
    boundary = np.arange(25, dtype=float)
    solved = operator.solve(np.zeros(25), boundary)
    exterior = np.ones(25, dtype=bool)
    exterior[operator.unknown] = False
    assert np.array_equal(solved[exterior], boundary[exterior])


def test_fixed_boundary_measurement_is_single_reference_and_two_meshes(fixed_result):
    receipt, runtime = fixed_result
    assert receipt["receipt"]["reference"] == {
        "shot": TARGET_SHOT,
        "slice_index": TARGET_SLICE,
    }
    assert receipt["receipt"]["single_reference_only"] is True
    assert len(receipt["mesh_results"]) == len(runtime) == 2
    assert {item["mesh_name"] for item in runtime} == {"coarse", "fine"}


def test_fixed_boundary_current_distribution_is_quantitative(fixed_result):
    receipt, _runtime = fixed_result
    for row in receipt["mesh_results"]:
        fixed = row["fixed_boundary_solve"]
        assert fixed["nova_fixed_boundary_entry_point_available"] is False
        assert fixed["route"] == "newton_krylov"
        assert fixed["diagnostic_solver_scope"].startswith("benchmark-local")
        assert np.isfinite(fixed["terminal_relative_sup_residual"])
        assert np.isfinite(fixed["best_relative_sup_residual"])
        assert (
            fixed["best_relative_sup_residual"]
            <= fixed["terminal_relative_sup_residual"]
        )
        assert fixed["converged"] is (
            fixed["best_relative_sup_residual"] <= FIXED_BOUNDARY_TOLERANCE
        )
        distribution = fixed["current_distribution_comparison"]
        assert distribution["cell_count"] > 0
        assert 0.0 <= distribution["minimum"] <= distribution["median"]
        assert distribution["median"] <= distribution["p90"]
        assert distribution["p90"] <= distribution["supremum"]
        integrals = fixed["current_integrals_on_same_support_a"]
        assert np.isfinite(integrals["fixed_boundary"])
        assert np.isfinite(integrals["delta_star_implied"])


def test_homotopy_paths_and_protected_bank_are_kept_distinct(fixed_result):
    receipt, _runtime = fixed_result
    distinction = receipt["homotopy_distinction"]
    assert "source-strength homotopy" in distinction["banked_negative"]
    assert "boundary-condition homotopy" in distinction["this_measurement"]
    for position in ("before", "after"):
        protected = receipt["protected_banked_artifacts"][position]
        assert protected["verified_digest_count"] == 23
        assert protected["all_digests_match"] is True


def test_picard_stalls_are_retained_beside_newton_result(fixed_result):
    receipt, _runtime = fixed_result
    assert receipt["picard_iteration_finding"] == PICARD_FINDING
    assert PICARD_FINDING["constant_relaxation"]["iterations"] == 400
    adaptive = PICARD_FINDING["adaptive_relaxation"]
    assert adaptive["iterations"] == 2000
    assert adaptive["terminal_unrelaxed_relative_sup_residual"] == pytest.approx(
        7.659068e-4
    )
    assert adaptive["best_unrelaxed_relative_sup_residual"] == pytest.approx(
        7.645682e-4
    )


def test_stalled_newton_refuses_the_free_seed(fixed_result):
    receipt, _runtime = fixed_result
    outcome = receipt["fixed_boundary_outcome"]
    if outcome["converged_mesh_count"] == 0:
        assert outcome["verdict"] == "FIXED_BOUNDARY_NEWTON_STALLS"
        assert receipt["seeded_stage_eligible_meshes"] == []
        assert receipt["seeded_stage"] == {
            "status": "unrun_no_converged_fixed_boundary_state",
            "passive_inclusive_circuit_background_built": False,
            "reason": (
                "No fixed-boundary state reached the unchanged criterion, so "
                "neither mesh supplied an admissible state and current-moment seed."
            ),
        }
    assert outcome["criterion_unchanged_after_observing_stalls"] is True


def test_banked_artifact_records_the_fixed_boundary_stage():
    receipt = json.loads(OUTPUT_RECEIPT.read_text())
    assert receipt["receipt"]["status"] in {
        "fixed_boundary_complete_free_seed_pending",
        "fixed_boundary_complete_free_seed_unavailable",
        "complete",
    }
    assert receipt["figure"]["path"] == str(OUTPUT_FIGURE)
    assert OUTPUT_FIGURE.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
