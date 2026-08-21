from __future__ import annotations

import json

import numpy as np
import pytest

from nova.equilibrium.conductor_current import (
    ConductorCurrentDeclaration,
    ConductorCurrentInfeasible,
    CurrentTier,
    InferenceLikelihood,
    LikelihoodValue,
    StaticCurrentRelation,
    UnknownCurrentPrior,
    resolve_conductor_currents,
    solve_conductor_currents,
)


def _resolution():
    declarations = (
        ConductorCurrentDeclaration("known", CurrentTier.KNOWN, "shipped"),
        ConductorCurrentDeclaration(
            "predicted",
            CurrentTier.KNOWABLE,
            "fit once",
            relation=StaticCurrentRelation(
                source="known",
                scale=2.0,
                relative_residual=0.1,
                provenance="independent trace",
                transfer_caveat="different pulse",
            ),
        ),
        ConductorCurrentDeclaration(
            "free",
            CurrentTier.UNKNOWN,
            "declared prior",
            prior=UnknownCurrentPrior(5.0, 2.0, 1.0, 9.0, "declared prior"),
        ),
    )
    return resolve_conductor_currents(
        ("known", "predicted", "free"), declarations, {"known": 10.0}
    )


class RecordingProfile:
    def __init__(self, *, fail: bool = False):
        self.currents = []
        self.fail = fail

    def solve(self, initial_flux, *, current, **options):
        del initial_flux, options
        self.currents.append(np.asarray(current, dtype=float))
        if self.fail:
            raise RuntimeError("inner solve failed")
        return {"current": self.currents[-1]}


def test_resolver_preserves_order_and_never_fills_unknown_with_zero() -> None:
    resolution = _resolution()

    np.testing.assert_allclose(resolution.template_a[:2], [10.0, 20.0])
    assert np.isnan(resolution.template_a[2])
    assert resolution.unknown_names == ("free",)
    np.testing.assert_allclose(resolution.current([7.0]), [10.0, 20.0, 7.0])
    assert resolution.prescribed_standard_deviation_a[1] == pytest.approx(2.0)


def test_absent_likelihood_reports_prior_dominated_complete_receipt() -> None:
    profile = RecordingProfile()
    result = solve_conductor_currents(profile, np.zeros(2), _resolution())

    assert len(profile.currents) == 1
    np.testing.assert_allclose(profile.currents[0], [10.0, 20.0, 5.0])
    assert result.receipt["posterior_status"] == "prior-dominated"
    assert result.receipt["likelihood_rank"] == 0
    assert result.receipt["response_order"] == ["known", "predicted", "free"]
    rows = {row["name"]: row for row in result.receipt["conductors"]}
    assert rows["known"]["disposition"] == "prescribed"
    assert rows["predicted"]["disposition"] == "predicted"
    assert rows["predicted"]["transfer_caveat"] == "different pulse"
    assert rows["free"]["disposition"] == "solved"
    assert rows["free"]["posterior_status"] == "prior-dominated"
    assert rows["free"]["value_a"] == pytest.approx(5.0)
    json.dumps(result.receipt, allow_nan=False)


def test_sensitive_likelihood_recovers_free_current_through_profile_solve() -> None:
    profile = RecordingProfile()
    likelihood = InferenceLikelihood(
        name="inference sensor",
        provenance="banked inference input",
        evaluator=lambda equilibrium, current: LikelihoodValue(
            residual=np.asarray([current[-1] - 7.0]),
            covariance=np.asarray([[0.01]]),
        ),
    )

    result = solve_conductor_currents(
        profile, np.zeros(2), _resolution(), likelihood=likelihood
    )

    assert result.receipt["posterior_status"] == "recovered"
    assert result.receipt["likelihood_rank"] == 1
    assert result.unknown_posterior_mean_a[0] == pytest.approx(7.0, abs=0.02)
    assert all(current.shape == (3,) for current in profile.currents)
    assert all(np.isfinite(current).all() for current in profile.currents)


def test_insensitive_likelihood_cannot_upgrade_prior_to_recovery() -> None:
    result = solve_conductor_currents(
        RecordingProfile(),
        np.zeros(2),
        _resolution(),
        likelihood=InferenceLikelihood(
            name="insensitive sensor",
            provenance="banked inference input",
            evaluator=lambda equilibrium, current: LikelihoodValue(
                residual=np.asarray([3.0]), covariance=np.asarray([[1.0]])
            ),
        ),
    )

    assert result.receipt["posterior_status"] == "prior-dominated"
    assert result.receipt["likelihood_rank"] == 0
    assert result.unknown_posterior_mean_a[0] == pytest.approx(5.0)


def test_label_likelihood_is_rejected_before_any_solve() -> None:
    with pytest.raises(ValueError, match="label-free"):
        InferenceLikelihood(
            name="label flux",
            provenance="EFIT output",
            evaluator=lambda equilibrium, current: LikelihoodValue(
                residual=np.zeros(1), covariance=np.eye(1)
            ),
            uses_label_artifact=True,
        )


def test_inner_failure_is_loud_and_does_not_return_a_current() -> None:
    with pytest.raises(ConductorCurrentInfeasible, match="explicit candidate"):
        solve_conductor_currents(RecordingProfile(fail=True), None, _resolution())
