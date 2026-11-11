"""Stable candidate identity at roundoff-scale extrema in the support read."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, support_read, topology

assert jax.config.jax_enable_x64 is True


@pytest.mark.parametrize("minimize", [False, True])
def test_exact_tie_preserves_identity_across_candidate_and_sum_orders(minimize):
    # Canonical indices travel with candidates; array positions may change.
    identities = jnp.asarray([8, 2, 5])
    signs = -1.0 if minimize else 1.0
    choose = jax.jit(
        jax.vmap(
            lambda score, ids: ids[
                topology.extremum_index(score, minimize=minimize, candidate_index=ids)
            ]
        )
    )
    orders = jnp.asarray([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    exact = signs * jnp.ones(3, dtype=jnp.float64)
    np.testing.assert_array_equal(choose(exact[orders], identities[orders]), 2)
    terms = jnp.asarray([[0.1, 0.2, 0.7], [0.7, 0.1, 0.2], [0.2, 0.7, 0.1]])
    for columns in ([0, 1, 2], [2, 1, 0]):
        scores = signs * jnp.sum(terms[:, jnp.asarray(columns)], axis=1)
        np.testing.assert_array_equal(choose(scores[orders], identities[orders]), 2)


@pytest.mark.parametrize("minimize", [False, True])
def test_clear_leader_and_margin_boundary(minimize):
    sign = -1.0 if minimize else 1.0
    for scale in (1e-20, 1.0, 1e20):
        for delta, expected in ((0.5e-12, 0), (2e-12, 1)):
            score = sign * scale * jnp.asarray([1.0 - delta, 1.0, 0.0])
            actual = jax.jit(lambda s: topology.extremum_index(s, minimize=minimize))(
                score
            )
            assert int(actual) == expected
    score = sign * jnp.asarray([1.0, 2.0, 0.0])
    assert int(topology.extremum_index(score, minimize=minimize)) == 1


def test_masks_nan_zero_and_batched_axes():
    score = jnp.asarray([[jnp.nan, 1.0, 1.0], [-jnp.inf, 0.0, 1e-30]])
    selected = jax.jit(lambda s: topology.extremum_index(s, axis=1))(score)
    np.testing.assert_array_equal(selected, [1, 2])
    assert int(topology.extremum_index(jnp.asarray([-jnp.inf, -jnp.inf]))) == 0
    assert int(topology.extremum_index(jnp.asarray([1.0, jnp.inf]))) == 1
    assert int(topology.extremum_index(jnp.asarray([jnp.nan, jnp.nan]))) == -1


@jax.jit
def _wall_scores(operands, moments, sigma, policy):
    field = fixed_points.field_from_jets(
        support_read.nodal_jets(operands, moments),
        operands.sample_index,
        operands.mesh,
        bank_rows=operands.rows,
        bank_moments=jnp.stack(tuple(moments), axis=-1),
        bank_exterior=operands.exterior,
    )
    _, flux = topology._wall_events(field, operands.geometry, policy, sigma)
    return sigma * flux


@pytest.mark.parametrize("preceded", [False, True], ids=["isolated", "after-diverted"])
def test_original_limited_seed_tie_has_stable_winner(preceded):
    from tests.test_tracked_read import _case, _read, _shifted

    if preceded:
        operands, cache, roots, convention, policy, moments, _ = _case("diverted")
        prior, _ = _read(operands, cache, roots, convention, policy, moments)
        jax.block_until_ready(prior)
        assert bool(prior.valid)
    operands, cache, _, convention, policy, moments, _ = _case("limited")
    operands, _ = _shifted(operands, cache, 0.005 * jnp.asarray((-1.0, 0.0)))
    scores = np.asarray(_wall_scores(operands, moments, convention.sigma, policy))
    order = np.argsort(-scores, kind="stable")
    gap = scores[order[0]] - scores[order[1]]
    assert int(order[0]) == 1327
    assert gap == pytest.approx(1.7763568394002505e-15, abs=2e-16)
    tied = np.flatnonzero(scores.max() - scores <= 1e-12 * abs(scores.max()))
    expected = int(tied.min())
    print("LIMITED_SEED_TIE", preceded, "raw_gap", gap, "canonical_winner", expected)
    assert expected != int(order[0])
    choose = jax.jit(
        lambda score, ids: ids[topology.extremum_index(score, candidate_index=ids)]
    )
    for permutation in (np.arange(scores.size), np.arange(scores.size)[::-1]):
        assert (
            int(choose(jnp.asarray(scores[permutation]), jnp.asarray(permutation)))
            == expected
        )
