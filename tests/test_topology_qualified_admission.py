"""Fail-closed admission of topology-qualified Newton trials."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import fixed_point
    from nova.equilibrium.fixed_point import (
        KrylovActionQualification,
        kink_aware_newton_krylov,
    )


def _solve(map_fn, admissibility_fn):
    return jax.jit(
        lambda: kink_aware_newton_krylov(
            map_fn,
            jnp.zeros(1),
            strategy="nonmonotone",
            newton_steps=1,
            gmres_iterations=1,
            warmup=0,
            admissibility_fn=admissibility_fn,
        )
    )()


def test_an_offered_limited_candidate_is_refused():
    """The longest trial is skipped when its achieved class is limited."""

    def map_fn(state):
        return jnp.full_like(state, 2.0)

    def remains_diverted(candidate):
        return candidate[0] < 1.5

    result = _solve(map_fn, remains_diverted)

    np.testing.assert_array_equal(
        np.asarray(result.candidate_admissibility[0]),
        np.asarray([False, True, True, True]),
    )
    assert float(result.accepted_factors[0]) == 0.5
    np.testing.assert_allclose(np.asarray(result.state), [1.0])


def test_an_offered_nonfinite_candidate_is_refused():
    """A finite state whose mapped residual is non-finite cannot be selected."""

    def map_fn(state):
        return jnp.where(state[0] > 1.5, jnp.full_like(state, jnp.nan), 2.0)

    result = _solve(map_fn, lambda _candidate: jnp.asarray(True))

    np.testing.assert_array_equal(
        np.asarray(result.candidate_admissibility[0]),
        np.asarray([False, True, True, True]),
    )
    assert float(result.accepted_factors[0]) == 0.5
    np.testing.assert_allclose(np.asarray(result.state), [1.0])


def test_an_admissible_diverted_candidate_is_accepted():
    """The full Newton proposal remains preferred when its class is admitted."""

    def map_fn(state):
        return jnp.full_like(state, 2.0)

    result = _solve(map_fn, lambda _candidate: jnp.asarray(True))

    assert np.all(np.asarray(result.candidate_admissibility[0]))
    assert float(result.accepted_factors[0]) == 1.0
    np.testing.assert_allclose(np.asarray(result.state), [2.0])
    assert float(result.residual) == 0.0
    assert (
        KrylovActionQualification(int(result.krylov_action_qualification))
        is KrylovActionQualification.ACCEPTED
    )


def test_an_unqualified_krylov_action_selects_no_backtracking_trial(monkeypatch):
    """A solver-status refusal leaves the state and every trial unpromoted."""

    def unsuccessful_gmres(_operator, right_hand_side, **_options):
        return right_hand_side, jnp.asarray(1)

    monkeypatch.setattr(
        fixed_point.jax.scipy.sparse.linalg, "gmres", unsuccessful_gmres
    )
    result = _solve(
        lambda state: jnp.full_like(state, 2.0),
        lambda _candidate: jnp.asarray(True),
    )

    assert not np.any(np.asarray(result.candidate_admissibility[0]))
    assert float(result.accepted_factors[0]) == 0.0
    np.testing.assert_array_equal(np.asarray(result.state), [0.0])
    assert (
        KrylovActionQualification(int(result.krylov_action_qualification))
        is KrylovActionQualification.NONSUCCESSFUL_GMRES_STATUS
    )


if __name__ == "__main__":
    pytest.main([__file__])
