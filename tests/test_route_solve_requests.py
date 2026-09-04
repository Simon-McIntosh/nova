"""Typed solve-policy provenance for production benchmark routes."""

from __future__ import annotations

import inspect
import json

import jax.numpy as jnp
import numpy as np

from benchmarks import (
    bank_revision_reproduction as bank_replay,
    diiid_forward_gs_match as diiid_match,
    solovev_certificate as solovev,
    solovev_certificate_current_pin as current_pin,
    topology_mask_replay as topology_replay,
)
from nova import __version__
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    FORWARD_SOLVE_DEFAULTS,
    ForwardSolveReceipt,
    ForwardSolveRequest,
    ResolvedForwardSolveDefaults,
)
from scripts.oracle_rebaseline import measure as oracle_recovery
from tests.test_prescribed_current_solve import _profile


def _deviations(request: ForwardSolveRequest) -> dict[str, object]:
    return dict(ResolvedForwardSolveDefaults.from_policy(request.policy).deviations)


def test_exact_bank_replay_request_names_every_policy_deviation():
    profile, _ordinary_response, _prescribed_response = _profile()
    request = bank_replay._bank_solve_request(
        profile,
        jnp.zeros((2, 4)),
        1.0,
        carrier_identity="cpu-bank-replay",
    )

    assert request.policy == FORWARD_SOLVE_DEFAULTS[__version__].__class__(
        **{
            **FORWARD_SOLVE_DEFAULTS[__version__].to_dict(),
            "newton_steps": 12,
            "gmres_iterations": 12,
            "qualification_tolerance": 1.0e-8,
        }
    )
    assert _deviations(request) == {
        "newton_steps": 12,
        "gmres_iterations": 12,
        "qualification_tolerance": 1.0e-8,
    }


def test_topology_replay_request_is_the_bare_public_policy():
    profile, _ordinary_response, _prescribed_response = _profile()
    request = topology_replay._topology_replay_request(
        profile.source,
        jnp.zeros(4),
        source_identity="sha256:cpu-topology",
    )

    assert request.policy == FORWARD_SOLVE_DEFAULTS[__version__]
    assert _deviations(request) == {}


def test_diiid_match_request_names_every_registered_budget_deviation():
    profile, _ordinary_response, _prescribed_response = _profile()
    request = diiid_match._registered_solve_request(
        profile,
        jnp.zeros((2, 4)),
        jnp.zeros(2),
        1.0,
        carrier_identity="cpu-diiid-match",
    )

    assert _deviations(request) == {
        "newton_steps": diiid_match.REGISTERED_ACCELERATED_NEWTON_STEPS,
        "gmres_iterations": diiid_match.REGISTERED_ACCELERATED_GMRES_ITERATIONS,
        "warmup": diiid_match.REGISTERED_ACCELERATED_WARMUP,
        "qualification_tolerance": diiid_match.GATE_RESIDUAL_TOLERANCE,
    }


def test_solovev_certificate_request_names_the_strict_tolerance_deviations():
    profile, _ordinary_response, _prescribed_response = _profile()
    request = solovev._certificate_solve_request(
        profile,
        jnp.zeros(4),
        1.0,
        carrier_identity="cpu-solovev-certificate",
    )

    assert _deviations(request) == {
        "kernel_tolerance": solovev.TERMINAL_RESIDUAL_BOUND,
        "qualification_tolerance": solovev.TERMINAL_RESIDUAL_BOUND,
    }


def test_solovev_current_pin_request_records_only_an_opt_out():
    profile, _ordinary_response, _prescribed_response = _profile()
    pinned = current_pin._current_pin_solve_request(
        profile,
        jnp.zeros(4),
        1.0,
        pinned=True,
        carrier_identity="cpu-current-pinned",
    )
    unpinned = current_pin._current_pin_solve_request(
        profile,
        jnp.zeros(4),
        1.0,
        pinned=False,
        carrier_identity="cpu-current-unpinned",
    )

    assert _deviations(pinned) == {}
    assert _deviations(unpinned) == {"current_pin": False}
    assert pinned.target_current == 1.0
    assert unpinned.target_current is None


def test_request_execution_is_bit_identical_to_the_keyword_fixture():
    profile, _ordinary_response, _prescribed_response = _profile()
    seed = np.zeros(4)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="cpu-route-fixture",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(seed),
        policy_overrides={
            "route": "picard",
            "newton_steps": 1,
            "relaxation": 1.0,
        },
    )

    keyword = profile.solve(seed, route="picard", evaluations=1, relaxation=1.0)
    receipt = profile.solve(request)

    assert isinstance(receipt, ForwardSolveReceipt)
    np.testing.assert_array_equal(receipt.equilibrium.flux, keyword.flux)
    payload = json.loads(json.dumps(receipt.resolved_defaults.to_dict()))
    assert ResolvedForwardSolveDefaults.from_dict(payload) == receipt.resolved_defaults


def test_direct_certificate_request_is_bit_identical_to_the_keyword_fixture():
    profile, _ordinary_response, _prescribed_response = _profile()
    seed = jnp.zeros(4)
    request = current_pin._current_pin_solve_request(
        profile,
        seed,
        1.0,
        pinned=False,
        carrier_identity="cpu-direct-certificate",
    )

    keyword = oracle_recovery._solve(profile.operator.flux_map(), seed)
    requested = current_pin._run_fixed_point_request(profile.operator, request)

    np.testing.assert_array_equal(requested.state, keyword.state)
    np.testing.assert_array_equal(requested.trace, keyword.trace)
    np.testing.assert_array_equal(requested.residual, keyword.residual)


def test_topology_replay_emits_the_request_policy_block():
    source = inspect.getsource(topology_replay.run)
    compact = "".join(source.split())

    assert '"resolved_defaults"' in source
    assert "ResolvedForwardSolveDefaults.from_policy(request.policy)" in compact
