"""Per-iteration receipts and resumable fixed-point checkpoints."""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.fixed_point import (
    InnerIterationDecision,
    load_fixed_point_checkpoint,
    newton_krylov,
)


def _contracting_map(state):
    return 0.5 * state + 1.0


def _limited_solve(
    initial,
    *,
    iterations: int,
    stream: bool = False,
    checkpoint: Path | None = None,
):
    return newton_krylov(
        _contracting_map,
        initial,
        newton_steps=iterations,
        gmres_iterations=2,
        warmup=0,
        step_cap=0.05,
        convergence_tolerance=1.0e-14,
        stream_inner_iterations=stream,
        checkpoint_path=checkpoint,
    )


def _run(transform: Callable, **options):
    result = transform(lambda: _limited_solve(jnp.zeros(1), **options))()
    result.state.block_until_ready()
    jax.effects_barrier()
    return result


def _assert_result_equal(left, right):
    for field in left._fields:
        left_value = np.asarray(getattr(left, field))
        right_value = np.asarray(getattr(right, field))
        if np.issubdtype(left_value.dtype, np.inexact):
            assert np.array_equal(left_value, right_value, equal_nan=True), field
        else:
            np.testing.assert_array_equal(left_value, right_value, err_msg=field)


@pytest.mark.parametrize("transform", [lambda function: function, jax.jit])
def test_streaming_and_checkpointing_leave_numerics_bit_identical(
    transform, tmp_path, capsys
):
    quiet = _run(transform, iterations=3)
    assert capsys.readouterr().out == ""

    checkpoint = tmp_path / "nonconverged-state.npz"
    observed = _run(
        transform,
        iterations=3,
        stream=True,
        checkpoint=checkpoint,
    )
    lines = capsys.readouterr().out.splitlines()

    _assert_result_equal(observed, quiet)
    assert len(lines) == int(observed.attempted_newton_promotions) == 3
    for iteration, line in enumerate(lines):
        assert f"iteration={iteration}" in line
        assert "residual_before=" in line
        assert "residual_after=" in line
        assert "proposed_step_norm=" in line
        assert "accepted=" in line
        assert "decision=" in line
        assert "krylov_qualification=" in line
        assert "applied_factor=" in line
        assert "krylov_reduction=" in line
        assert "krylov_tolerance=" in line

    persisted = load_fixed_point_checkpoint(checkpoint)
    np.testing.assert_array_equal(persisted["state"], observed.state)
    np.testing.assert_array_equal(persisted["residual"], observed.residual)
    np.testing.assert_array_equal(
        persisted["inner_iteration_residuals_after"],
        observed.inner_iteration_residuals_after,
    )
    assert checkpoint.with_suffix(".npz.sha256").is_file()


@pytest.mark.parametrize("transform", [lambda function: function, jax.jit])
def test_checkpoint_state_resumes_to_the_uninterrupted_iterate(transform, tmp_path):
    uninterrupted = _run(transform, iterations=3)
    checkpoint = tmp_path / "resume-state.npz"
    partial = _run(transform, iterations=1, checkpoint=checkpoint)
    resumed_state = jnp.asarray(load_fixed_point_checkpoint(checkpoint)["state"])

    resumed = transform(lambda: _limited_solve(resumed_state, iterations=2))()
    resumed.state.block_until_ready()

    np.testing.assert_array_equal(partial.state, resumed_state)
    np.testing.assert_array_equal(resumed.state, uninterrupted.state)
    np.testing.assert_array_equal(resumed.residual, uninterrupted.residual)


def test_short_inner_solve_retains_distinguishable_padding():
    result = newton_krylov(
        lambda state: jnp.ones_like(state),
        jnp.zeros(1),
        newton_steps=4,
        gmres_iterations=1,
        warmup=0,
    )

    assert int(result.attempted_newton_promotions) == 1
    assert int(result.inner_iteration_accepted[0]) == 1
    assert (
        int(result.inner_iteration_decisions[0])
        == InnerIterationDecision.NEWTON_LADDER_ACCEPTED
    )
    assert np.all(np.isnan(result.inner_iteration_residuals_before[1:]))
    assert np.all(np.isnan(result.inner_iteration_residuals_after[1:]))
    np.testing.assert_array_equal(result.inner_iteration_accepted[1:], -1)
    np.testing.assert_array_equal(
        result.inner_iteration_decisions[1:], InnerIterationDecision.NOT_EXECUTED
    )
    np.testing.assert_array_equal(result.inner_iteration_krylov_qualifications[1:], 0)
