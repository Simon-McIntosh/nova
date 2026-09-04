"""Public request and receipt contract for forward equilibrium solves."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import json

import numpy as np
import pytest

from benchmarks.efit_forward_parity_slice import _parity_solve_request
from nova import __version__
from nova.equilibrium.forward import PerturbedSeedPolicy
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    FORWARD_SOLVE_DEFAULTS,
    ForwardSolvePolicy,
    ForwardSolveReceipt,
    ForwardSolveRequest,
    ResolvedForwardSolveDefaults,
)
from tests.test_prescribed_current_solve import _profile


POLICY_FIELDS = (
    "route",
    "newton_steps",
    "gmres_iterations",
    "warmup",
    "relaxation",
    "step_cap",
    "active_set_steps",
    "kernel_tolerance",
    "qualification_tolerance",
    "current_pin",
    "settled_exit",
    "own_mask_acceptance",
    "continuation",
    "best_iterate_retention",
    "stagnation_stop",
    "exact_kernels",
    "cached_machine",
    "compilation_cache",
)
RECEIPT_FIELDS = (
    "terminal_state",
    "qualified",
    "termination_reason",
    "residual_history",
    "mask_history",
    "globalisation_decisions",
    "amplitude_history",
    "topology_read",
    "polish_receipt",
    "compilation_cache_hit",
    "wall_seconds",
    "resolved_defaults",
)


def test_default_request_schema_resolves_from_the_installed_version_table():
    profile, _ordinary_response, _prescribed_response = _profile()
    seed = np.zeros(4)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="cpu-linear-carrier",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(seed),
    )

    assert tuple(item.name for item in fields(ForwardSolvePolicy)) == POLICY_FIELDS
    assert tuple(item.name for item in fields(ForwardSolveReceipt)) == RECEIPT_FIELDS
    assert request.policy == FORWARD_SOLVE_DEFAULTS[__version__]
    assert request.policy.gmres_iterations == PerturbedSeedPolicy().gmres_iterations
    assert request.route == request.policy.route
    assert request.constraint_pairs == ()
    with pytest.raises(FrozenInstanceError):
        request.route = "picard"


def test_request_path_is_bit_identical_and_defaults_round_trip_through_json():
    profile, _ordinary_response, _prescribed_response = _profile()
    seed = np.zeros(4)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="cpu-linear-carrier",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(seed),
        policy_overrides={
            "route": "picard",
            "newton_steps": 1,
            "relaxation": 1.0,
        },
    )

    keyword_result = profile.solve(
        seed,
        route="picard",
        evaluations=1,
        relaxation=1.0,
    )
    request_receipt = profile.solve(request)

    assert isinstance(request_receipt, ForwardSolveReceipt)
    np.testing.assert_array_equal(
        request_receipt.equilibrium.flux,
        keyword_result.flux,
    )
    payload = json.loads(json.dumps(request_receipt.resolved_defaults.to_dict()))
    restored = ResolvedForwardSolveDefaults.from_dict(payload)
    assert restored == request_receipt.resolved_defaults
    assert restored.nova_version == __version__


def test_parity_request_records_the_gmres_budget_as_a_declared_deviation():
    profile, _ordinary_response, _prescribed_response = _profile()
    request = _parity_solve_request(
        profile,
        np.zeros(4),
        shot=22086,
        row=43,
        current_field="fcoil",
    )
    resolved = ResolvedForwardSolveDefaults.from_policy(request.policy)

    assert FORWARD_SOLVE_DEFAULTS[__version__].gmres_iterations == 30
    assert request.policy.gmres_iterations == 12
    assert dict(resolved.deviations)["gmres_iterations"] == 12
    assert resolved.to_dict()["deviations"]["gmres_iterations"] == 12
