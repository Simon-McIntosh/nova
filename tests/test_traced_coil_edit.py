"""The compiled forward solve accepts conductor edits as runtime data."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from nova.jax.config import configure_dtypes
from tests.test_forward_solve_request import _completed_profile


def test_coil_edits_share_one_program_without_current_constants(monkeypatch):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile, ordinary_response, prescribed_response = _completed_profile(monkeypatch)
    program = profile._accelerated_history_program(
        "picard", evaluations=1, relaxation=1.0
    )
    seed = jnp.zeros(4)
    ordinary = jnp.asarray((101_003.0, 103_009.0))
    prescribed = jnp.asarray((107_011.0, 109_013.0, 113_017.0))
    lowered = program.lower(seed, ordinary, prescribed, profile.operator, None)
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    for value in (*ordinary, *prescribed):
        assert str(int(value)) not in hlo

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
            current=jnp.asarray(ordinary),
            prescribed_current=jnp.asarray(prescribed),
        )

    first = profile.solve(request((3.0, 5.0), (7.0, 11.0, 13.0)))
    second = profile.solve(request((17.0, 19.0), (23.0, 29.0, 31.0)))
    np.testing.assert_allclose(
        second.equilibrium.flux,
        ordinary_response @ jnp.asarray((17.0, 19.0))
        + prescribed_response @ jnp.asarray((23.0, 29.0, 31.0)),
    )
    assert first.compilation_cache_hit is False
    assert second.compilation_cache_hit is True
