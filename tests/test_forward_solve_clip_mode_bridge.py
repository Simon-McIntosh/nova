"""An omitted request clip mode follows the process mode at construction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nova.equilibrium.forward_operator import (
    ForwardFluxOperator,
    set_support_clip_mode,
    support_clip_mode,
)
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    ForwardSolveRequest,
    ResolvedForwardSolveDefaults,
    declared_forward_solve_policy,
)
from tests.test_prescribed_current_solve import _profile


def test_omitted_request_clip_mode_follows_the_process_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request built with no mode inherits the process mode for solve+receipt."""

    profile, _ordinary_response, _prescribed_response = _profile()
    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        omitted = ForwardSolveRequest.from_defaults(
            carrier_identity="clip-mode-bridge-carrier",
            source_profile=profile.source,
            seed_policy=ExplicitSolveSeed(np.zeros(4)),
            policy_overrides={
                "route": "picard",
                "newton_steps": 1,
                "relaxation": 1.0,
            },
        )
        explicit_chord = ForwardSolveRequest.from_defaults(
            carrier_identity="clip-mode-bridge-carrier",
            source_profile=profile.source,
            seed_policy=ExplicitSolveSeed(np.zeros(4)),
            policy_overrides={
                "route": "picard",
                "newton_steps": 1,
                "relaxation": 1.0,
            },
            clip_mode="chord",
        )
        observed_modes: list[str] = []
        build_operator = ForwardFluxOperator.with_clip_mode

        def record_operator_mode(self, clip_mode: str):
            observed_modes.append(clip_mode)
            return build_operator(self, clip_mode)

        monkeypatch.setattr(ForwardFluxOperator, "with_clip_mode", record_operator_mode)
        monkeypatch.setattr(
            ForwardFluxOperator,
            "program_identity",
            property(lambda _operator: "clip-mode-bridge-program"),
        )
        monkeypatch.setattr(
            profile,
            "_solve_accelerated",
            lambda *_args, **_kwargs: SimpleNamespace(
                flux=np.zeros(4),
                fixed_point=SimpleNamespace(
                    residual=np.asarray(0.0),
                    termination_reason=0,
                    trace=np.zeros(1),
                    shadow_mask_changes=np.zeros((1, 4), dtype=bool),
                    inner_iteration_decisions=np.zeros(1),
                    inner_iteration_applied_factors=np.zeros(1),
                ),
                finite=SimpleNamespace(passed=True),
                constraints=(),
            ),
        )
        receipts = (profile.solve(omitted), profile.solve(explicit_chord))
    finally:
        set_support_clip_mode(previous_mode)

    assert omitted.clip_mode == "exact"
    assert explicit_chord.clip_mode == "chord"  # an explicit mode still wins
    assert receipts[0].clip_mode == "exact"
    assert receipts[1].clip_mode == "chord"
    assert observed_modes == ["exact", "chord"]


def test_directly_constructed_request_carries_the_process_mode_when_underfilled() -> (
    None
):
    """A request built field-by-field takes the process mode; an explicit mode wins."""

    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        policy = declared_forward_solve_policy()
        omitted = ForwardSolveRequest(
            carrier_identity="clip-mode-bridge-direct",
            source_profile=object(),
            seed_policy=ExplicitSolveSeed(np.zeros(4)),
            policy=policy,
            route=policy.route,
        )
        explicit_chord = ForwardSolveRequest(
            carrier_identity="clip-mode-bridge-direct",
            source_profile=object(),
            seed_policy=ExplicitSolveSeed(np.zeros(4)),
            policy=policy,
            route=policy.route,
            clip_mode="chord",
        )
    finally:
        set_support_clip_mode(previous_mode)

    assert omitted.clip_mode == "exact"
    assert explicit_chord.clip_mode == "chord"


def test_from_policy_records_the_process_mode_when_underfilled() -> None:
    """An omitted from_policy mode records the process mode; an explicit one wins."""

    policy = declared_forward_solve_policy()
    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        omitted = ResolvedForwardSolveDefaults.from_policy(policy)
        explicit_chord = ResolvedForwardSolveDefaults.from_policy(
            policy, clip_mode="chord"
        )
    finally:
        set_support_clip_mode(previous_mode)

    assert omitted.clip_mode == "exact"
    assert explicit_chord.clip_mode == "chord"
