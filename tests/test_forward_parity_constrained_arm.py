"""Contracts for the MAST declared-current parity arm."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import efit_forward_parity_slice as parity


def _banked_row(shot: int) -> dict:
    return {
        "reference": {"shot": shot},
        "solve_outcome": {
            "outcome_class": "bounded_non_convergence",
            "converged": False,
            "terminal_plasma_current_a": 8.0e5,
        },
        "accepted_residual_trajectory": [0.1],
    }


def test_banked_comparator_requires_zero_converged_plasma_roots() -> None:
    receipt = {"per_shot": [_banked_row(shot) for shot in range(6)]}

    indexed = parity._baseline_by_shot(receipt)

    assert len(indexed) == 6
    receipt["per_shot"][0]["solve_outcome"]["converged"] = True
    with pytest.raises(RuntimeError, match="zero plasma roots"):
        parity._baseline_by_shot(receipt)


def test_public_branch_solve_receives_the_declared_current() -> None:
    captured = {}
    equilibrium = SimpleNamespace(
        fixed_point=SimpleNamespace(trace=np.asarray([1.0])),
        cell_current=np.asarray([801_493.25]),
        finite=SimpleNamespace(passed=True),
        normalisation=SimpleNamespace(
            policy_name="declared_scalar_current", amplitude=1.125
        ),
        topology=SimpleNamespace(
            axis=np.asarray([0.9, 0.0]),
            x_point=np.asarray([0.7, -0.5]),
            diverted=True,
            axis_flux=0.3,
            boundary_flux=-0.1,
        ),
    )

    def solve_branch(state, requested_class, **options):
        captured.update(options)
        return SimpleNamespace(
            equilibrium=equilibrium,
            requested_class=requested_class,
            achieved_class=requested_class,
            topology_consistent=True,
            converged=False,
            residual=1.0e-3,
            iterations=12,
        )

    profile = SimpleNamespace(solve_branch=solve_branch)
    case = {"state": np.zeros(4)}
    context = {"group": {"plasma_current_c": np.asarray([801_493.25])}, "row": 0}

    record, _trace, _branch = parity._passive_inclusive_solve(
        case,
        context,
        profile,
        target_current=801_493.25,
    )

    assert captured["target_current"] == 801_493.25
    assert captured["tolerance"] == parity.FIXED_POINT_CRITERION
    assert record["target_current_a"] == 801_493.25
    assert record["terminal_state"]["normalisation_policy"] == (
        "declared_scalar_current"
    )
    assert record["terminal_state"]["normalisation_amplitude"] == 1.125


def test_constrained_artifacts_cannot_overwrite_the_banked_directory(
    tmp_path, monkeypatch
) -> None:
    banked = tmp_path / "banked"
    banked.mkdir()
    monkeypatch.setattr(parity, "configure_dtypes", lambda: None)

    with pytest.raises(ValueError, match="outside the banked directory"):
        parity.run_current_constrained(
            tmp_path,
            tmp_path / "selection.json",
            banked / "constrained",
            banked,
        )


def test_artifact_digest_detects_content_changes(tmp_path) -> None:
    artifact = tmp_path / "receipt.json"
    artifact.write_text("first\n")
    before = parity._artifact_digests(tmp_path)

    artifact.write_text("second\n")

    assert parity._artifact_digests(tmp_path) != before


def test_reference_native_grid_interpolates_the_stored_map() -> None:
    assert parity.REFERENCE_NATIVE_GRID_POINTS == 95
    radius = np.linspace(0.2, 1.8, 65)
    height = np.linspace(-1.4, 1.4, 65)
    reference = radius[:, None] ** 2 - 0.5 * height[None, :] ** 2

    refined_r, refined_z, refined, selection = parity._benchmark_spatial_grid(
        radius,
        height,
        reference,
        parity.REFERENCE_NATIVE_GRID_POINTS,
    )

    assert refined.shape == (95, 95)
    assert len(refined_r) == len(refined_z) == parity.REFERENCE_NATIVE_GRID_POINTS
    assert selection["mode"] == "fixed_uniform_axis_count"
    np.testing.assert_allclose(
        refined,
        refined_r[:, None] ** 2 - 0.5 * refined_z[None, :] ** 2,
        atol=2.0e-14,
    )
