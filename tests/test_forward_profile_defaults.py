"""Default policy wiring through the public forward solve."""

from __future__ import annotations

from inspect import signature

import jax
import numpy as np
import pytest

from nova.equilibrium import fixed_point
from nova.equilibrium.forward import ForwardProfile, PerturbedSeedPolicy
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    ForwardSolveRequest,
    declared_forward_solve_policy,
)
from tests.test_prescribed_current_solve import _profile


DEFAULT_ON_FIELDS = (
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


def _newton_ready_profile() -> ForwardProfile:
    """Return the small CPU fixture with the promoted-mask map attached."""

    profile, _ordinary_response, _prescribed_response = _profile()

    def flux_map_with_shadow(
        current=None,
        requested_class=None,
        target_current=None,
        prescribed_current=None,
    ):
        mapped = profile.operator.flux_map(
            current,
            requested_class,
            target_current,
            prescribed_current,
        )
        return lambda state, _shadow: mapped(state)

    profile.operator.flux_map_with_shadow = flux_map_with_shadow
    return profile


def _assert_trees_bit_identical(left: object, right: object) -> None:
    """Require equal structure, dtype, shape, and bytes for every array leaf."""

    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    assert left_structure == right_structure
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_no_option_keyword_solve_matches_the_bare_request_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _newton_ready_profile()
    seed = np.zeros(4)
    cache_requests: list[bool] = []
    kernel_options: list[dict[str, object]] = []
    original_newton_krylov = fixed_point.newton_krylov

    def record_cache(enabled: bool) -> str | None:
        cache_requests.append(enabled)
        if enabled:
            return "/temporary-runtime/nova/jax-compilation/runtime-cpu"
        return None

    def record_newton_krylov(*args: object, **options: object):
        kernel_options.append(options)
        return original_newton_krylov(*args, **options)

    monkeypatch.setattr(
        ForwardProfile,
        "_configure_solve_compilation_cache",
        staticmethod(record_cache),
    )
    monkeypatch.setattr(fixed_point, "newton_krylov", record_newton_krylov)

    keyword_equilibrium = profile.solve(seed)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="cpu-linear-carrier",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(seed),
    )
    request_receipt = profile.solve(request)

    _assert_trees_bit_identical(
        keyword_equilibrium.fixed_point,
        request_receipt.equilibrium.fixed_point,
    )
    np.testing.assert_array_equal(
        keyword_equilibrium.flux,
        request_receipt.equilibrium.flux,
    )
    assert kernel_options[0].keys() == kernel_options[1].keys()
    for name in kernel_options[0]:
        if callable(kernel_options[0][name]):
            assert callable(kernel_options[1][name])
        else:
            assert kernel_options[0][name] == kernel_options[1][name]
    assert cache_requests == [True, True, False]

    resolved = request_receipt.resolved_defaults
    assert resolved.policy == declared_forward_solve_policy()
    assert resolved.deviations == ()
    assert resolved.compilation_cache_directory == (
        "/temporary-runtime/nova/jax-compilation/runtime-cpu"
    )
    assert all(getattr(resolved.policy, name) for name in DEFAULT_ON_FIELDS)
    assert sum(bool(getattr(resolved.policy, name)) for name in DEFAULT_ON_FIELDS) == 9


def test_public_defaults_have_one_version_keyed_authority() -> None:
    policy = declared_forward_solve_policy()

    assert PerturbedSeedPolicy().newton_steps == policy.newton_steps
    assert PerturbedSeedPolicy().gmres_iterations == policy.gmres_iterations
    assert PerturbedSeedPolicy().tolerance == policy.qualification_tolerance
    assert ForwardProfile.__dataclass_fields__["newton_steps"].default_factory() == (
        policy.newton_steps
    )
    assert ForwardProfile.__dataclass_fields__["relaxation"].default_factory() == (
        policy.relaxation
    )
    for method_name in ("solve", "solve_branch", "solve_portfolio", "solve_batch"):
        assert (
            signature(getattr(ForwardProfile, method_name)).parameters["route"].default
            is None
        )


def test_a_diagnostic_absolute_source_declares_the_current_pin_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _newton_ready_profile()
    monkeypatch.setattr(
        "nova.equilibrium.forward.configure_persistent_compilation_cache",
        lambda _root: None,
    )

    absolute = profile.solve(
        np.zeros(4),
        route="picard",
        evaluations=1,
        current_pin=False,
    )

    assert absolute.flux.shape == (4,)
    with pytest.raises(ValueError, match="target_current requires"):
        profile.solve(
            np.zeros(4),
            route="picard",
            evaluations=1,
            target_current=1.0,
            current_pin=False,
        )
