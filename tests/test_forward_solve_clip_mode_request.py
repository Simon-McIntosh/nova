"""Request-owned support clip modes for forward equilibrium solves."""

from __future__ import annotations

import json
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
    ForwardSolvePolicy,
    ForwardSolveRequest,
)
from tests.test_prescribed_current_solve import _profile


def test_request_clip_mode_builds_each_operator_and_records_each_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two request modes remain independent of the legacy process default."""

    profile, _ordinary_response, _prescribed_response = _profile()
    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        requests = tuple(
            ForwardSolveRequest.from_defaults(
                carrier_identity="clip-mode-request-carrier",
                source_profile=profile.source,
                seed_policy=ExplicitSolveSeed(np.zeros(4)),
                policy_overrides={
                    "route": "picard",
                    "newton_steps": 1,
                    "relaxation": 1.0,
                },
                clip_mode=clip_mode,
            )
            for clip_mode in ("chord", "chord_cells")
        )
        observed_modes: list[str] = []
        build_operator = ForwardFluxOperator.with_clip_mode

        def record_operator_mode(self, clip_mode: str):
            observed_modes.append(clip_mode)
            return build_operator(self, clip_mode)

        monkeypatch.setattr(ForwardFluxOperator, "with_clip_mode", record_operator_mode)
        monkeypatch.setattr(
            profile,
            "_request_compilation_cache_key",
            lambda request, _initial_flux: request.clip_mode,
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
        receipts = tuple(profile.solve(request) for request in requests)
    finally:
        set_support_clip_mode(previous_mode)

    assert tuple(request.clip_mode for request in requests) == (
        "chord",
        "chord_cells",
    )
    assert tuple(receipt.clip_mode for receipt in receipts) == (
        "chord",
        "chord_cells",
    )
    assert observed_modes == ["chord", "chord_cells"]

    payload = json.loads(json.dumps(receipts[1].resolved_defaults.to_dict()))
    assert ForwardSolvePolicy.from_dict(payload["policy"]) == requests[1].policy
    assert payload["clip_mode"] == "chord_cells"
