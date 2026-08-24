from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from nova.equilibrium.conductor_current import (
    InferenceLikelihood,
    LikelihoodValue,
    UnknownCurrentPrior,
    solve_conductor_currents,
)
from nova.imas import diiid_current, diiid_description
from nova.imas.diiid_description import (
    CIRCUIT_DRIVEN_CONDUCTORS,
    PF_ACTIVE_CIRCUIT,
    POLOIDAL_CONDUCTORS,
)


def _circuit_bypass_priors() -> dict[str, UnknownCurrentPrior]:
    return {
        name: UnknownCurrentPrior(
            mean_a=10_000.0 + index * 1000.0,
            standard_deviation_a=3000.0,
            lower_a=-20_000.0,
            upper_a=40_000.0,
            provenance="declared demonstration prior",
        )
        for index, name in enumerate(diiid_current.CIRCUIT_BYPASS_PRIOR_CONDUCTORS)
    }


def _shipped() -> dict[str, float]:
    result = {
        name: float(index + 1) * 1000.0
        for index, name in enumerate(POLOIDAL_CONDUCTORS)
    }
    result["ECOILA"] = 10_000.0
    return result


def test_diiid_tiers_carry_fit_once_values_and_relation_uncertainty() -> None:
    resolution = diiid_current.resolve_diiid_currents(POLOIDAL_CONDUCTORS, _shipped())
    by_name = {name: index for index, name in enumerate(resolution.names)}

    assert len(resolution.names) == 24
    assert resolution.names[:19] == POLOIDAL_CONDUCTORS
    assert resolution.unknown_names == ()
    expected = {
        "ECOILB": 20_000.0,
        "E567UP": 10_000.0,
        "E567DN": 10_000.0,
        "E89UP": 10_456.947569496173,
        "E89DN": 10_456.240764323717,
    }
    for drive in PF_ACTIVE_CIRCUIT.drives:
        index = by_name[drive.conductor]
        assert resolution.template_a[index] == pytest.approx(expected[drive.conductor])
        assert resolution.prescribed_standard_deviation_a[index] == pytest.approx(
            abs(expected[drive.conductor]) * drive.uncertainty.residual_rms_fraction
        )
        declaration = resolution.declarations[index]
        assert declaration.relation is diiid_current.CIRCUIT_RELATIONS[drive.conductor]
        assert "leave-one-shot-out R-squared" in declaration.relation.provenance
        assert "label flux" in declaration.relation.transfer_caveat


def test_circuit_map_drives_all_registered_conductors_from_ecoila() -> None:
    shipped = _shipped()

    reconstructed = diiid_current.circuit_current_map(shipped)
    resolution = diiid_current.resolve_diiid_currents(POLOIDAL_CONDUCTORS, shipped)

    assert tuple(reconstructed) == CIRCUIT_DRIVEN_CONDUCTORS
    np.testing.assert_allclose(
        list(reconstructed.values()),
        [20_000.0, 10_000.0, 10_000.0, 10_456.947569496173, 10_456.240764323717],
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        resolution.current(())[: len(POLOIDAL_CONDUCTORS)],
        list(shipped.values()),
    )
    np.testing.assert_allclose(
        resolution.current(())[len(POLOIDAL_CONDUCTORS) :],
        list(reconstructed.values()),
    )


def test_banked_circuit_receipt_matches_runtime_calibration() -> None:
    root = Path(__file__).parents[2]
    receipt = json.loads(
        (
            root / "docs/figures/coil-circuit-discovery/pf_active_circuit_receipt.json"
        ).read_text()
    )
    ensemble = json.loads(
        (
            root
            / "docs/figures/coil-circuit-discovery"
            / "ensemble_relation_assessment_receipt.json"
        ).read_text()
    )
    frame_receipt = json.loads(
        (root / ensemble["inputs"]["fitted_current"]["path"]).read_text()
    )
    reproduced = diiid_description.adjudicate_circuit_wiring(
        frame_receipt["ensemble_ready_fitted_current_table"], ensemble
    )
    banked = receipt["adjudication"]["drives"]

    assert receipt["reconstruction_fixture"]["complete_response_current_count"] == 24
    assert receipt["reconstruction_fixture"]["free_current_count_with_circuit"] == 0
    assert reproduced == tuple(drive.wiring for drive in PF_ACTIVE_CIRCUIT.drives)
    assert [row["effective_gain"] for row in banked] == [
        drive.gain for drive in PF_ACTIVE_CIRCUIT.drives
    ]
    for row, drive in zip(banked, PF_ACTIVE_CIRCUIT.drives, strict=True):
        assert row["wiring"] == drive.wiring.as_record()
        assert row["uncertainty"] == drive.uncertainty.as_record()
    assert "nearly degenerate" in " ".join(receipt["caveats"])
    assert "1 of 60" in " ".join(receipt["caveats"])


def test_circuit_bypass_exposes_three_prior_driven_diagnostic_slots() -> None:
    resolution = diiid_current.resolve_diiid_currents(
        POLOIDAL_CONDUCTORS,
        _shipped(),
        _circuit_bypass_priors(),
        use_circuit=False,
    )
    by_name = {name: index for index, name in enumerate(resolution.names)}

    assert resolution.unknown_names == ("ECOILB", "E567DN", "E89UP")
    assert resolution.template_a[by_name["E567UP"]] == pytest.approx(9929.0)
    assert resolution.template_a[by_name["E89DN"]] == pytest.approx(10_165.0)
    assert all(
        np.isnan(resolution.template_a[by_name[name]])
        for name in diiid_current.CIRCUIT_BYPASS_PRIOR_CONDUCTORS
    )


class LabelRejectingRow(dict):
    def __getitem__(self, key):
        if "efit" in str(key).lower() or "recovered" in str(key).lower():
            raise AssertionError(f"label artifact read: {key}")
        return super().__getitem__(key)


@dataclass(frozen=True)
class Turns:
    applied_multiplier: float = 1.0


@dataclass(frozen=True)
class Conductor:
    name: str
    input_column: str
    turns: Turns = Turns()


@dataclass(frozen=True)
class Description:
    conductors: tuple[Conductor, ...]


def test_shipped_interpolation_never_reads_label_artifacts() -> None:
    row = LabelRejectingRow(
        magnetics_time=np.asarray([0.0, 1.0]),
        **{f"magnetics_{name}": np.asarray([1.0, 3.0]) for name in POLOIDAL_CONDUCTORS},
    )
    description = Description(
        tuple(Conductor(name, f"magnetics_{name}") for name in POLOIDAL_CONDUCTORS)
    )

    result = diiid_current.shipped_current_at(
        row, description, POLOIDAL_CONDUCTORS, 0.5
    )

    assert tuple(result) == POLOIDAL_CONDUCTORS
    assert set(result.values()) == {2000.0}


@dataclass(frozen=True)
class Target:
    source_target: np.ndarray
    coordinate: np.ndarray


@dataclass(frozen=True)
class Operator:
    grid: Target
    wall: Target
    external_current: np.ndarray


@dataclass(frozen=True)
class Profile:
    operator: Operator


def test_complete_adapter_preserves_response_order(monkeypatch) -> None:
    shipped_count = len(POLOIDAL_CONDUCTORS)
    profile = Profile(
        Operator(
            grid=Target(
                np.arange(2 * shipped_count).reshape(2, shipped_count),
                np.asarray([[1.0, 0.0], [2.0, 0.0]]),
            ),
            wall=Target(
                np.arange(3 * shipped_count).reshape(3, shipped_count),
                np.asarray([[1.0, -1.0], [2.0, 0.0], [1.0, 1.0]]),
            ),
            external_current=np.arange(shipped_count),
        )
    )

    def response(entry, dd_version, names, target_r, target_z):
        del entry, dd_version, target_z
        values = np.asarray(
            [np.full(len(target_r), 100.0 + index) for index in range(len(names))]
        )
        return tuple(names), values, {"target_points": len(target_r)}

    monkeypatch.setattr(diiid_current, "active_coil_response_from_imas", response)
    adapter = diiid_current.complete_profile_current_adapter(
        profile,
        shipped_names=POLOIDAL_CONDUCTORS,
        shipped_current_a=_shipped(),
        active_coil_entry="unused.nc",
    )

    assert adapter.resolution.names == (
        *POLOIDAL_CONDUCTORS,
        *CIRCUIT_DRIVEN_CONDUCTORS,
    )
    np.testing.assert_array_equal(
        np.asarray(adapter.profile.operator.grid.source_target)[:, :shipped_count],
        profile.operator.grid.source_target,
    )
    np.testing.assert_allclose(
        np.asarray(adapter.profile.operator.grid.source_target)[:, shipped_count:],
        [[100.0, 101.0, 102.0, 103.0, 104.0]] * 2,
    )
    assert adapter.response_receipt["complete_count"] == 24
    assert adapter.response_receipt["current_authority"] == "pf_active circuit"
    assert adapter.resolution.unknown_names == ()
    assert len(adapter.response_receipt["pf_active"]["circuit"]["drives"]) == 5


class RecordingProfile:
    def __init__(self):
        self.currents = []

    def solve(self, initial_flux, *, current, **options):
        del initial_flux, options
        self.currents.append(np.asarray(current, dtype=float))
        return self.currents[-1]


def test_outer_loop_updates_only_three_circuit_bypass_prior_slots() -> None:
    resolution = diiid_current.resolve_diiid_currents(
        POLOIDAL_CONDUCTORS,
        _shipped(),
        _circuit_bypass_priors(),
        use_circuit=False,
    )
    profile = RecordingProfile()
    target = np.asarray([11_500.0, 13_500.0, 15_500.0])
    likelihood = InferenceLikelihood(
        name="inference-available three-channel demonstration",
        provenance="banked demonstration observation",
        evaluator=lambda equilibrium, current: LikelihoodValue(
            residual=current[resolution.unknown_indices] - target,
            covariance=np.eye(3) * 100.0**2,
        ),
    )

    result = solve_conductor_currents(
        profile, np.zeros(1), resolution, likelihood=likelihood
    )

    assert result.receipt["likelihood_rank"] == 3
    assert result.receipt["posterior_status"] == "recovered"
    assert len(result.receipt["conductors"]) == 24
    assert {row["name"] for row in result.receipt["conductors"]} == set(
        resolution.names
    )
    assert {row["disposition"] for row in result.receipt["conductors"]} == {
        "prescribed",
        "predicted",
        "solved",
    }
    assert len(profile.currents) > 1
    fixed_indices = np.setdiff1d(
        np.arange(len(resolution.names)), resolution.unknown_indices
    )
    for current in profile.currents:
        np.testing.assert_allclose(
            current[fixed_indices],
            resolution.current(resolution.prior_mean_a)[fixed_indices],
        )
    assert not np.allclose(
        result.current_a[resolution.unknown_indices],
        resolution.prior_mean_a,
    )
