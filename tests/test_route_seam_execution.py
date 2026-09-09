"""Route-level guards for request receipt ownership at the public solve seam."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import jax.numpy as jnp
import pytest

from benchmarks import bank_revision_reproduction as bank_replay
from benchmarks import diiid_forward_gs_match as diiid_match
from benchmarks import efit_forward_parity_slice as efit_parity
from nova import __version__
from nova.equilibrium.solve_request import (
    ForwardSolveRequest,
    ResolvedForwardSolveDefaults,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REAL_EQUILIBRIA_ROUTE = (
    REPOSITORY_ROOT
    / "docs/figures/primary-xpoint-evidence/real_equilibria_reachability.py"
)
REAL_EQUILIBRIA_SPEC = spec_from_file_location(
    "real_equilibria_reachability", REAL_EQUILIBRIA_ROUTE
)
if REAL_EQUILIBRIA_SPEC is None or REAL_EQUILIBRIA_SPEC.loader is None:
    raise RuntimeError("the real-equilibria reachability route cannot be imported")
real_equilibria_reachability = module_from_spec(REAL_EQUILIBRIA_SPEC)
REAL_EQUILIBRIA_SPEC.loader.exec_module(real_equilibria_reachability)


@dataclass(frozen=True)
class RouteBody:
    """One route body and the receipt variable it must serialize."""

    name: str
    path: Path
    function: str
    receipt_name: str
    consumer: str | None = None


ROUTE_BODIES = (
    RouteBody(
        "bank revision reproduction",
        Path(inspect.getfile(bank_replay)),
        "_solve_pure_arm",
        "receipt",
    ),
    RouteBody(
        "real equilibria reachability",
        REAL_EQUILIBRIA_ROUTE,
        "_solve_with_defaults",
        "receipt",
        "_mast_states",
    ),
    RouteBody(
        "DIII-D forward GS match",
        Path(inspect.getfile(diiid_match)),
        "_solve_registered",
        "diverted",
    ),
    RouteBody(
        "EFIT forward parity slice",
        Path(inspect.getfile(efit_parity)),
        "_passive_inclusive_solve",
        "receipt",
    ),
)


def _function_node(route: RouteBody, name: str | None = None) -> ast.FunctionDef:
    """Read one target function without importing its data-bearing module."""

    module = ast.parse(route.path.read_text(encoding="utf-8"))
    name = route.function if name is None else name
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _attribute_calls(function: ast.FunctionDef) -> list[ast.Call]:
    """Collect direct public solve invocations in one production route body."""

    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"solve", "solve_branch", "solve_portfolio"}
    ]


@pytest.mark.parametrize("route", ROUTE_BODIES, ids=lambda route: route.name)
def test_route_body_passes_its_request_to_the_public_solve(route: RouteBody) -> None:
    """Reject a request-shaped helper that leaves the production solve bypassed."""

    function = _function_node(route)
    calls = _attribute_calls(function)

    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call.func, ast.Attribute)
    assert call.func.attr == "solve"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "request"
    assert not call.keywords


@pytest.mark.parametrize("route", ROUTE_BODIES, ids=lambda route: route.name)
def test_route_body_serializes_defaults_from_the_returned_receipt(
    route: RouteBody,
) -> None:
    """Keep route provenance tied to the receipt returned by its invoked solve."""

    function = _function_node(route)
    consumer = (
        _function_node(route, route.consumer)
        if route.consumer is not None
        else function
    )
    attributes = [
        node
        for node in ast.walk(consumer)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    ]

    assert any(
        node.attr == "resolved_defaults"
        and isinstance(node.value, ast.Name)
        and (node.value.id.endswith("receipt") or node.value.id == route.receipt_name)
        for node in attributes
    )
    source = ast.unparse(consumer)
    assert "ResolvedForwardSolveDefaults.from_policy" not in source


def _block_until_ready(value):
    """Match the JAX synchronization method used by a production receipt."""

    return value


class ReceiptReturningProfile:
    """Return a deliberately distinct receipt block for one typed request."""

    def __init__(self) -> None:
        self.source = object()
        self.request: ForwardSolveRequest | None = None
        flux = type("Flux", (), {"block_until_ready": _block_until_ready})()
        self.receipt = type(
            "Receipt",
            (),
            {
                "equilibrium": type("Equilibrium", (), {"flux": flux})(),
                "resolved_defaults": None,
            },
        )()

    def solve(self, request: ForwardSolveRequest) -> object:
        self.request = request
        self.receipt.resolved_defaults = ResolvedForwardSolveDefaults.from_policy(
            request.policy,
            nova_version=__version__,
            compilation_cache_directory="receipt-owned-cache",
        )
        return self.receipt


def test_resolved_defaults_are_owned_by_the_invoked_solve_receipt() -> None:
    """Prove the production helper returns the profile receipt without rebuilding it."""

    profile = ReceiptReturningProfile()
    returned = real_equilibria_reachability._solve_with_defaults(
        profile,
        jnp.zeros(1),
        carrier_identity="route-seam-fixture",
        target_current=1.0,
    )

    assert isinstance(profile.request, ForwardSolveRequest)
    assert returned is profile.receipt
    assert (
        returned.resolved_defaults.compilation_cache_directory == "receipt-owned-cache"
    )
    assert returned.resolved_defaults.nova_version == __version__
