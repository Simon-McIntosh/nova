from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from nova.equilibrium.conductor_current import (
    InferenceLikelihood,
    LikelihoodValue,
    UnknownCurrentPrior,
    solve_conductor_currents,
)
from nova.imas import diiid_current
from nova.imas.diiid_description import POLOIDAL_CONDUCTORS


def _priors() -> dict[str, UnknownCurrentPrior]:
    return {
        name: UnknownCurrentPrior(
            mean_a=10_000.0 + index * 1000.0,
            standard_deviation_a=3000.0,
            lower_a=-20_000.0,
            upper_a=40_000.0,
            provenance="declared demonstration prior",
        )
        for index, name in enumerate(diiid_current.UNKNOWN_POLOIDAL_CONDUCTORS)
    }


def _shipped() -> dict[str, float]:
    result = {
        name: float(index + 1) * 1000.0
        for index, name in enumerate(POLOIDAL_CONDUCTORS)
    }
    result["ECOILA"] = 10_000.0
    return result


def test_diiid_tiers_carry_fit_once_values_and_relation_uncertainty() -> None:
    resolution = diiid_current.resolve_diiid_currents(
        POLOIDAL_CONDUCTORS, _shipped(), _priors()
    )
    by_name = {name: index for index, name in enumerate(resolution.names)}

    assert len(resolution.names) == 24
    assert resolution.names[:19] == POLOIDAL_CONDUCTORS
    assert resolution.unknown_names == ("ECOILB", "E567DN", "E89UP")
    assert resolution.template_a[by_name["E567UP"]] == pytest.approx(9929.0)
    assert resolution.template_a[by_name["E89DN"]] == pytest.approx(10_165.0)
    assert resolution.prescribed_standard_deviation_a[
        by_name["E567UP"]
    ] == pytest.approx(56.5953)
    assert resolution.prescribed_standard_deviation_a[
        by_name["E89DN"]
    ] == pytest.approx(71.7649)
    assert all(
        np.isnan(resolution.template_a[by_name[name]])
        for name in diiid_current.UNKNOWN_POLOIDAL_CONDUCTORS
    )
    assert all(
        "different pulses" in diiid_current.KNOWABLE_RELATIONS[name].transfer_caveat
        for name in diiid_current.KNOWABLE_RELATIONS
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
        unknown_priors=_priors(),
        active_coil_entry="unused.nc",
    )

    assert adapter.resolution.names == (
        *POLOIDAL_CONDUCTORS,
        *diiid_current.OMITTED_POLOIDAL_CONDUCTORS,
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


class RecordingProfile:
    def __init__(self):
        self.currents = []

    def solve(self, initial_flux, *, current, **options):
        del initial_flux, options
        self.currents.append(np.asarray(current, dtype=float))
        return self.currents[-1]


def test_outer_loop_drives_only_three_diiid_unknown_slots() -> None:
    resolution = diiid_current.resolve_diiid_currents(
        POLOIDAL_CONDUCTORS, _shipped(), _priors()
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
