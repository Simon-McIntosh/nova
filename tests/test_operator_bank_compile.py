"""The operator bank compiles once per shape bucket, not once per geometry."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import support_read
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from tests.test_prescribed_current_solve import _profile

assert jax.config.jax_enable_x64 is True

MOMENTS = (jnp.ones(2), jnp.ones(2), jnp.ones(2))


def _bank(active, bucket, seed):
    """One machine's operators padded to a shape bucket."""
    rows = np.zeros((bucket, 2, 3, 6))
    exterior = np.zeros((bucket, 6))
    rows[:active] = seed * np.arange(1, 1 + active * 2 * 3 * 6).reshape(active, 2, 3, 6)
    exterior[:active] = seed * np.arange(1, 1 + active * 6).reshape(active, 6)
    return support_read.SupportOperands(
        None,
        None,
        jnp.zeros((bucket, 2)),
        jnp.zeros((1, 1), dtype=int),
        jnp.asarray(rows),
        jnp.asarray(exterior),
        bucket,
        jnp.asarray(active),
    )


def _program():
    return jax.jit(support_read.bank_program)


def test_a_second_geometry_in_one_bucket_compiles_nothing_new():
    program = _program()
    first = _bank(60, 64, 1.0)
    second = _bank(60, 64, 2.0)
    jax.block_until_ready(program(first, MOMENTS))
    assert program._cache_size() == 1
    jax.block_until_ready(program(second, MOMENTS))
    assert program._cache_size() == 1
    assert not np.array_equal(np.asarray(first.rows), np.asarray(second.rows))


def test_a_geometry_in_another_bucket_compiles_exactly_once_more():
    program = _program()
    jax.block_until_ready(program(_bank(60, 64, 1.0), MOMENTS))
    assert program._cache_size() == 1
    jax.block_until_ready(program(_bank(60, 128, 1.0), MOMENTS))
    assert program._cache_size() == 2
    jax.block_until_ready(program(_bank(60, 128, 3.0), MOMENTS))
    assert program._cache_size() == 2


def test_the_padded_rows_do_not_change_the_active_result():
    bank = _bank(60, 64, 1.0)
    plain = _bank(60, 60, 1.0)
    padded = np.asarray(support_read.bank_program(bank, MOMENTS))
    bare = np.asarray(support_read.bank_program(plain, MOMENTS))
    assert np.array_equal(padded[:60], bare)
    assert np.all(padded[60:] == 0.0)


def test_the_compile_key_carries_the_bank_bucket(monkeypatch):
    """A bank's bucket selects its program inside the one compilation cache."""
    monkeypatch.setattr(
        ForwardFluxOperator,
        "program_identity",
        property(lambda _operator: "operator-bank-key-test"),
    )
    profile, _ordinary, _prescribed = _profile()
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="operator-bank-key-carrier",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(np.zeros(4)),
        policy_overrides={"route": "picard", "newton_steps": 1, "relaxation": 1.0},
    )
    seed = np.zeros(4)

    def key():
        return profile._request_compilation_cache_key(request, seed)

    unattached = key()
    assert unattached[1] is None

    profile.support_bank = _bank(60, 64, 1.0)
    narrow = key()
    assert narrow[1] == 64

    profile.support_bank = _bank(60, 128, 1.0)
    wide = key()
    assert wide[1] == 128

    profile.support_bank = _bank(60, 64, 2.0)
    assert key() == narrow

    assert len({unattached, narrow, wide}) == 3
