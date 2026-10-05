"""Public request and receipt contract for forward equilibrium solves."""

from __future__ import annotations

from dataclasses import dataclass, FrozenInstanceError, fields
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.efit_forward_parity_slice import _parity_solve_request
from nova import __version__
from nova.biot.null import Null1D
from nova.equilibrium.forward import PerturbedSeedPolicy
from nova.equilibrium.solve_request import (
    ColdSeedPortfolio,
    ExplicitSolveSeed,
    FORWARD_SOLVE_DEFAULTS,
    ForwardSolvePolicy,
    ForwardSolveReceipt,
    ForwardSolveRequest,
    ResolvedForwardSolveDefaults,
)
from nova.jax.config import configure_dtypes
from tests import test_prescribed_current_solve as _prescribed_current_solve
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
    "topology",
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
    "seed_provenance",
)


@dataclass(frozen=True)
class _NullTarget:
    """Linear conductor target carrying a stationary-point locator.

    The prescribing fixture builds the operator with ``object.__new__``, so its
    targets expose only the response matrix the arithmetic touches.  The
    production ``FluxTarget`` also owns a ``null`` locator that the operator's
    geometry identity reads, so this stand-in carries one of the same type.
    """

    response: jax.Array
    null: object = None

    def __post_init__(self) -> None:
        if self.null is None:
            object.__setattr__(
                self,
                "null",
                Null1D(coordinate=jnp.zeros((self.response.shape[0], 2))),
            )

    @property
    def node_number(self) -> int:
        return self.response.shape[0]

    def external(self, current) -> jax.Array:
        return self.response @ current


jax.tree_util.register_pytree_node(
    _NullTarget,
    lambda target: ((target.response, target.null), None),
    lambda _aux, children: _NullTarget(response=children[0], null=children[1]),
)


def _completed_profile(monkeypatch: pytest.MonkeyPatch):
    """Return the shared linear fixture with the host geometry it omits.

    The lightweight operator carries only the members the arithmetic touches,
    so the area entry the geometry identity reads is absent.  Supply it
    alongside the stationary-point locator each target now owns.
    """
    monkeypatch.setattr(_prescribed_current_solve, "_LinearTarget", _NullTarget)
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile, ordinary_response, prescribed_response = _profile()
    operator = profile.operator
    operator.area = jnp.zeros(operator.grid.node_number)
    operator.cell_average_stencil = None
    operator.cell_average_weight = None
    operator.inside_material = None
    operator.moment_geometry = None
    operator.use_linear_moments = False
    operator.wall_unit_offsets = None
    operator.wall_unit_closed = None
    operator.wall_unit_kinds = None
    return profile, ordinary_response, prescribed_response


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


def test_request_path_is_bit_identical_and_defaults_round_trip_through_json(
    monkeypatch: pytest.MonkeyPatch,
):
    profile, _ordinary_response, _prescribed_response = _completed_profile(monkeypatch)
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


def test_cold_portfolio_seed_policy_solves_and_records_its_selected_branch(
    monkeypatch: pytest.MonkeyPatch,
):
    profile, _ordinary_response, _prescribed_response = _completed_profile(monkeypatch)
    portfolio_calls: list[tuple[object, ...]] = []

    def cold_seed_portfolio(*args, **kwargs):
        portfolio_calls.append((args, kwargs))
        return type(
            "Portfolio",
            (),
            {
                "branches": type(
                    "Branches",
                    (),
                    {"flux": np.asarray(((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)))},
                )()
            },
        )()

    monkeypatch.setattr(profile, "cold_seed_portfolio", cold_seed_portfolio)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="cpu-cold-seed-carrier",
        source_profile=profile.source,
        seed_policy=ColdSeedPortfolio(
            plasma_current=12_000.0,
            centroid=(1.0, 0.0),
            requested_class="diverted",
        ),
        policy_overrides={
            "route": "picard",
            "newton_steps": 1,
            "relaxation": 1.0,
        },
    )

    receipt = profile.solve(request)

    assert len(portfolio_calls) == 1
    assert receipt.seed_provenance is not None
    assert receipt.seed_provenance.to_dict() == {
        "kind": "cold_seed_portfolio",
        "requested_class": "diverted",
        "plasma_current": 12_000.0,
        "centroid": (1.0, 0.0),
    }


def test_compilation_cache_receipt_observes_reuse_not_the_request_field(
    monkeypatch: pytest.MonkeyPatch,
):
    profile, _ordinary_response, _prescribed_response = _completed_profile(monkeypatch)
    monkeypatch.setattr(
        profile,
        "_configure_solve_compilation_cache",
        staticmethod(lambda enabled: "/test-cache" if enabled else None),
    )
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="cpu-cache-carrier",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(np.zeros(4)),
        compilation_cache_hit=True,
        policy_overrides={
            "route": "picard",
            "newton_steps": 1,
            "relaxation": 1.0,
        },
    )

    first = profile.solve(request)
    second = profile.solve(request)

    assert first.compilation_cache_hit is False
    assert second.compilation_cache_hit is True
    payload = json.loads(json.dumps(second.resolved_defaults.to_dict()))
    assert ResolvedForwardSolveDefaults.from_dict(payload) == second.resolved_defaults


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
