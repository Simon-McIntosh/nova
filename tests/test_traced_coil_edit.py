"""The compiled forward solve accepts conductor edits as runtime data."""

import jax
import jax.numpy as jnp
import numpy as np
import re
import pytest

from nova.equilibrium.fixed_point import FixedPointResult
from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from tests.test_forward_solve_request import _completed_profile


def test_coil_edits_share_one_program_without_current_constants(monkeypatch):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile, ordinary_response, prescribed_response = _completed_profile(monkeypatch)
    profile.operator.external_current = jnp.asarray((127_021.0, 131_027.0))
    program = profile._accelerated_history_program(
        "picard", evaluations=1, relaxation=1.0
    )
    seed = jnp.zeros(4)
    ordinary = jnp.asarray((101_003.0, 103_009.0))
    prescribed = jnp.asarray((107_011.0, 109_013.0, 113_017.0))
    lowered = program.lower(seed, ordinary, prescribed, profile.operator, None)
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    hlo_numbers = (
        float(token)
        for token in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", hlo)
    )
    hlo_values = set(hlo_numbers)
    for value in (*ordinary, *prescribed, *profile.operator.external_current):
        assert float(value) not in hlo_values

    first = program(seed, ordinary, prescribed, profile.operator, None)
    second = program(seed, ordinary + 1.0, prescribed + 1.0, profile.operator, None)
    np.testing.assert_allclose(
        first.state, ordinary_response @ ordinary + prescribed_response @ prescribed
    )
    np.testing.assert_allclose(
        second.state,
        ordinary_response @ (ordinary + 1.0) + prescribed_response @ (prescribed + 1.0),
    )
    assert program._cache_size() == 1


def test_typed_requests_reuse_the_compiled_solve_across_coil_edits(monkeypatch):
    configure_dtypes()
    profile, ordinary_response, prescribed_response = _completed_profile(monkeypatch)
    seed = jnp.zeros(4)

    def request(ordinary, prescribed):
        return ForwardSolveRequest.from_defaults(
            carrier_identity="linear-carrier",
            source_profile=profile.source,
            seed_policy=ExplicitSolveSeed(seed),
            policy_overrides={"route": "picard", "relaxation": 1.0},
            requested_class=TopologyClass.DIVERTED,
            current=jnp.asarray(ordinary),
            prescribed_current=jnp.asarray(prescribed),
        )

    first = profile.solve(request((3.0, 5.0), (7.0, 11.0, 13.0)))
    first_program = next(iter(profile._accelerated_program_cache.values()))
    second = profile.solve(request((17.0, 19.0), (23.0, 29.0, 31.0)))
    np.testing.assert_allclose(
        second.equilibrium.flux,
        ordinary_response @ jnp.asarray((17.0, 19.0))
        + prescribed_response @ jnp.asarray((23.0, 29.0, 31.0)),
    )
    assert first.compilation_cache_hit is False
    assert second.compilation_cache_hit is True
    assert next(iter(profile._accelerated_program_cache.values())) is first_program


def test_explicit_pin_omission_uses_unconstrained_krylov_route(monkeypatch):
    configure_dtypes()
    profile, _ordinary_response, _prescribed_response = _completed_profile(monkeypatch)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="linear-carrier",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(jnp.zeros(4)),
        vertical_pin="omit",
        prescribed_current=jnp.asarray((7.0, 11.0, 13.0)),
        policy_overrides={"newton_steps": 1, "gmres_iterations": 1, "warmup": 0},
    )

    observed_routes = []

    def history_program(route, **_options):
        observed_routes.append(route)
        return lambda seed, *_arguments: FixedPointResult(
            state=seed,
            residual=jnp.asarray(0.0),
            trace=jnp.asarray((0.0,)),
            converged=jnp.asarray(True),
        )

    monkeypatch.setattr(profile, "_accelerated_history_program", history_program)
    monkeypatch.setattr(
        profile,
        "_solve_augmented_constraints",
        lambda *_args, **_kwargs: pytest.fail("augmented route selected"),
    )

    receipt = profile.solve(request)

    assert observed_routes == ["newton_krylov"]
    assert bool(receipt.qualified)
