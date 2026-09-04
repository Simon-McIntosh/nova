"""Receipt-level wiring checks for production forward-solve routes."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from nova.equilibrium.solve_request import (
    ForwardSolveRequest,
    ResolvedForwardSolveDefaults,
    declared_forward_solve_policy,
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
    """Expose the request handed to the public seam and return its receipt block."""

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


@pytest.fixture(params=("solovev-oracle", "mast-visual", "diiid-visual"))
def production_route_receipt(request):
    """Construct each migrated route without numerical policy overrides."""

    profile = RecordingProfile()
    seed = jnp.zeros(1)
    target_current = 1.25e6
    current = jnp.asarray((1.0, -2.0))
    if request.param == "solovev-oracle":
        receipt = oracle_rebaseline._solve_with_defaults(
            profile,
            seed,
            carrier_identity="solovev:cpu-fixture",
            target_current=target_current,
        )
        expected_current = None
    else:
        expected_current = None if request.param == "mast-visual" else current
        receipt = real_equilibria_reachability._solve_with_defaults(
            profile,
            seed,
            carrier_identity=f"{request.param}:cpu-fixture",
            target_current=target_current,
            current=expected_current,
        )

    assert len(profile.calls) == 1
    positional, keywords = profile.calls[0]
    assert keywords == {}
    assert len(positional) == 1
    solve_request = positional[0]
    assert isinstance(solve_request, ForwardSolveRequest)
    assert solve_request.policy == declared_forward_solve_policy()
    assert solve_request.target_current == target_current
    if expected_current is None:
        assert solve_request.current is None
    else:
        assert jnp.array_equal(solve_request.current, expected_current)
    assert receipt.resolved_defaults.deviations == ()
    return request.param, receipt


@pytest.mark.parametrize("default_name", REQUIRED_DEFAULTS)
def test_production_route_receipt_resolves_each_declared_default(
    production_route_receipt, default_name
):
    route_name, receipt = production_route_receipt
    assert route_name
    assert getattr(receipt.resolved_defaults.policy, default_name) is True


def test_production_routes_do_not_call_private_solver_kernels():
    oracle_source = Path(oracle_rebaseline.__file__).read_text(encoding="utf-8")
    visual_source = VISUAL_ROUTE.read_text(encoding="utf-8")

    fixture_source = oracle_source.split("def measure_fixture", 1)[1].split(
        "def _numeric_gate", 1
    )[0]
    assert "_solve(operator.flux_map(), seed)" not in fixture_source
    assert "_solve_with_defaults(" in fixture_source
    assert "_margin_graded_newton_krylov" not in visual_source
