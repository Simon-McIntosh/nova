"""Receipt-level guarantees for declared production solve defaults."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
import inspect
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import textwrap
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from benchmarks import (
    bank_revision_reproduction as bank_replay,
    diiid_forward_gs_match as diiid_match,
    efit_forward_parity_slice as efit_parity,
    solovev_certificate as solovev_certificate,
)
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    ForwardSolveRequest,
    ResolvedForwardSolveDefaults,
)
from scripts.oracle_rebaseline import measure as oracle_rebaseline


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROUTE = (
    REPOSITORY_ROOT
    / "docs/figures/primary-xpoint-evidence/real_equilibria_reachability.py"
)
VISUAL_SPEC = spec_from_file_location("real_equilibria_reachability", VISUAL_ROUTE)
if VISUAL_SPEC is None or VISUAL_SPEC.loader is None:
    raise RuntimeError("the real-equilibria reachability route cannot be imported")
real_equilibria_reachability = module_from_spec(VISUAL_SPEC)
VISUAL_SPEC.loader.exec_module(real_equilibria_reachability)

REQUIRED_DEFAULTS = (
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


class RecordingProfile:
    """Record requests crossing the public solve seam and return their receipts."""

    def __init__(self) -> None:
        self.source = object()
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def solve(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        request = args[0]
        return SimpleNamespace(
            equilibrium=SimpleNamespace(flux=jnp.zeros(1)),
            resolved_defaults=ResolvedForwardSolveDefaults.from_policy(request.policy),
        )


ReceiptFactory = Callable[[RecordingProfile], tuple[object, ...]]


@dataclass(frozen=True)
class ProductionEntryPoint:
    """One production request route and the function crossing its public seam."""

    name: str
    receipts: ReceiptFactory
    launcher: Callable[..., object] | None


def _solve_request(profile: RecordingProfile, request: ForwardSolveRequest) -> object:
    """Exercise a typed production request at the receipt-producing boundary."""

    return profile.solve(request)


def _public_solve_receipts(profile: RecordingProfile) -> tuple[object, ...]:
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="public-solve:cpu-fixture",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(jnp.zeros(4)),
        target_current=1.0,
    )
    return (_solve_request(profile, request),)


def _bank_replay_receipts(profile: RecordingProfile) -> tuple[object, ...]:
    request = bank_replay._bank_solve_request(
        profile,
        jnp.zeros((2, 4)),
        1.0,
        carrier_identity="bank-replay:cpu-fixture",
    )
    return (_solve_request(profile, request),)


def _topology_visual_receipts(profile: RecordingProfile) -> tuple[object, ...]:
    seed = jnp.zeros(4)
    return (
        real_equilibria_reachability._solve_with_defaults(
            profile,
            seed,
            carrier_identity="mast-visual:cpu-fixture",
            target_current=1.0,
        ),
        real_equilibria_reachability._solve_with_defaults(
            profile,
            seed,
            carrier_identity="diiid-visual:cpu-fixture",
            target_current=1.0,
            current=jnp.zeros(2),
        ),
    )


def _efit_parity_receipts(profile: RecordingProfile) -> tuple[object, ...]:
    request = efit_parity._parity_solve_request(
        profile,
        jnp.zeros(4),
        shot=21978,
        row=0,
        current_field="current",
    )
    return (_solve_request(profile, request),)


def _diiid_match_receipts(profile: RecordingProfile) -> tuple[object, ...]:
    request = diiid_match._registered_solve_request(
        profile,
        jnp.zeros((2, 4)),
        jnp.zeros(2),
        1.0,
        carrier_identity="diiid-match:cpu-fixture",
    )
    return (_solve_request(profile, request),)


def _solovev_oracle_receipts(profile: RecordingProfile) -> tuple[object, ...]:
    return (
        oracle_rebaseline._solve_with_defaults(
            profile,
            jnp.zeros(4),
            carrier_identity="solovev-oracle:cpu-fixture",
            target_current=1.0,
        ),
    )


def _solovev_certificate_receipts(profile: RecordingProfile) -> tuple[object, ...]:
    request = solovev_certificate._certificate_solve_request(
        profile,
        jnp.zeros(4),
        1.0,
        carrier_identity="solovev-certificate:cpu-fixture",
    )
    return (_solve_request(profile, request),)


PRODUCTION_ENTRY_POINTS = (
    ProductionEntryPoint("public-solve-and-routes", _public_solve_receipts, None),
    ProductionEntryPoint(
        "exact-bank-replay",
        _bank_replay_receipts,
        bank_replay._solve_pure_arm,
    ),
    ProductionEntryPoint(
        "topology-visuals",
        _topology_visual_receipts,
        real_equilibria_reachability._solve_with_defaults,
    ),
    ProductionEntryPoint(
        "efit-parity-slice",
        _efit_parity_receipts,
        efit_parity.solve_arm,
    ),
    ProductionEntryPoint(
        "diiid-grad-shafranov-match",
        _diiid_match_receipts,
        diiid_match._solve_registered,
    ),
    ProductionEntryPoint(
        "solovev-oracle-and-rebaseline",
        _solovev_oracle_receipts,
        oracle_rebaseline._solve_with_defaults,
    ),
    ProductionEntryPoint(
        "solovev-certificate",
        _solovev_certificate_receipts,
        solovev_certificate._measure,
    ),
)


@pytest.fixture(params=PRODUCTION_ENTRY_POINTS, ids=lambda route: route.name)
def production_receipts(request):
    """Return receipt blocks from a production request with no default overrides."""

    entry_point = request.param
    profile = RecordingProfile()
    receipts = entry_point.receipts(profile)

    assert receipts
    assert len(profile.calls) == len(receipts)
    for positional, keywords in profile.calls:
        assert keywords == {}
        assert len(positional) == 1
        assert isinstance(positional[0], ForwardSolveRequest)
    for receipt in receipts:
        assert not (
            set(dict(receipt.resolved_defaults.deviations)) & set(REQUIRED_DEFAULTS)
        )
    return entry_point.name, receipts


@pytest.mark.parametrize("default_name", REQUIRED_DEFAULTS)
def test_every_production_receipt_resolves_declared_default_on(
    production_receipts, default_name
):
    entry_point, receipts = production_receipts
    for receipt in receipts:
        resolved_defaults = receipt.resolved_defaults.to_dict()
        assert resolved_defaults["policy"][default_name] is True, (
            f"{entry_point} did not resolve {default_name} on"
        )
        assert default_name not in resolved_defaults["deviations"]


@pytest.mark.parametrize(
    "entry_point",
    tuple(route for route in PRODUCTION_ENTRY_POINTS if route.launcher is not None),
    ids=lambda route: route.name,
)
def test_launchers_leave_declared_defaults_to_the_public_seam(entry_point) -> None:
    """Reject launchers that bypass the declared table with boolean controls."""

    source = textwrap.dedent(inspect.getsource(entry_point.launcher))
    tree = ast.parse(source)
    solve_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {"solve", "solve_branch", "solve_portfolio", "solve_batch"}
    ]

    assert solve_calls, f"{entry_point.name} does not cross a public solve seam"
    for call in solve_calls:
        keyword_names = {keyword.arg for keyword in call.keywords}
        assert None not in keyword_names
        assert not (keyword_names & set(REQUIRED_DEFAULTS))
