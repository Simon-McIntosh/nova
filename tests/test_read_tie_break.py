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
    assert int(topology.extremum_index(jnp.asarray([jnp.nan, jnp.nan]))) == 0


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


@pytest.mark.parametrize("outside", [False, True])
def test_x_point_data_refuses_all_nan_admitted_scores(outside):
    from tests.test_topology_o_qualification import _topology_fixture

    reader, _ = _topology_fixture()
    candidates = jnp.asarray([[1.0, 0.0, 2.0], [1.0, 1.0, 3.0], [3.0, 0.0, 4.0]])
    np.testing.assert_array_equal(reader.contained_x_candidates(candidates), [1, 1, 0])
    np.testing.assert_array_equal(
        reader.x_point_data(candidates, 1.0, 0.0), candidates[1]
    )
    refused = reader.x_point_data(candidates, 1.0, jnp.nan)
    print("X_POINT_REFUSAL", np.asarray(refused))
    assert np.all(np.isnan(np.asarray(refused)))
    assert int(reader.x_point_index(candidates, 1.0, jnp.nan)) == -1


def _wall_read_fixture(monkeypatch, flux):
    """Hold component geometry unambiguous while exercising read selections."""
    from types import SimpleNamespace

    wall_levels = []
    points = jnp.asarray([[1.0, -1.0], [1.0, 1.0]])
    geometry = SimpleNamespace(
        contour_mesh=SimpleNamespace(overflow=False),
        pitch=jnp.ones(1),
        full_area=jnp.ones(1),
        fit_inverse=jnp.eye(2)[None],
    )
    monkeypatch.setattr(
        topology, "_wall_events", lambda values, *args: (points, values)
    )
    monkeypatch.setattr(topology, "_inside_cells", lambda *args: jnp.ones(1, bool))
    monkeypatch.setattr(
        topology,
        "_curvature_derivatives",
        lambda *args: (jnp.zeros((2, 2, 2)), jnp.zeros((2, 2, 2, 2))),
    )
    monkeypatch.setattr(
        topology, "represented_saddle_cells", lambda *args: jnp.zeros(1, bool)
    )
    monkeypatch.setattr(
        topology, "_membership_roundoff_bound", lambda *args: jnp.asarray(1e-12)
    )

    def support(
        _geometry, _coefficient, _sigma, level, _axis, _form, saddle_live, _policy
    ):
        def observe(value, saddle):
            if not saddle:
                wall_levels.append(float(value))

        jax.debug.callback(observe, level, saddle_live, ordered=True)
        return (
            jnp.asarray([[1.0, 0.0]]),
            jnp.zeros((1, 2, 1, 2)),
            jnp.asarray([[0, 1]]),
            jnp.ones((1, 2), bool),
            jnp.asarray([[True, False]]),
            jnp.ones(1, bool),
            jnp.asarray(True),
            jnp.asarray(1),
        )

    monkeypatch.setattr(topology, "_support_at_level", support)
    nulls = topology.StationaryRead(
        jnp.asarray([[1.0, 0.0]]),
        topology.FieldJet(jnp.asarray([2.0]), jnp.zeros((1, 2)), -jnp.eye(2)[None]),
        jnp.ones(1, bool),
        jnp.ones(1, bool),
        jnp.zeros(1, jnp.int32),
    )
    policy = topology.TopologyPolicy(null_capacity=1)
    run = jax.jit(
        lambda values, wall_flux: topology._read_at_nulls(
            wall_flux,
            geometry,
            topology.TopologyConvention.from_cocos(17, 1.0),
            policy,
            jnp.zeros((1, 6)),
            values,
            jnp.ones(1, bool),
            jnp.asarray(1),
            jnp.asarray(True),
            None,
        )
    )
    return (
        lambda values: run(values, flux),
        nulls,
        geometry,
        policy,
        points,
        wall_levels,
    )


def test_read_at_nulls_uses_tie_rule_for_both_wall_rankings(monkeypatch):
    flux = jnp.asarray([1.0, 1.0 + 5e-13])
    run, nulls, _, _, points, wall_levels = _wall_read_fixture(monkeypatch, flux)
    result = run(nulls)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert bool(result.valid)
    print("READ_WALL_LEVELS", wall_levels, "BOUNDARY", np.asarray(result.boundary))
    np.testing.assert_array_equal(result.boundary, points[0])
    np.testing.assert_array_equal(wall_levels, [float(flux[0]), float(flux[0])])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_shared_exact_owner_preserves_input_precision(dtype):
    from nova.equilibrium import connectivity_boundary

    values = jnp.asarray([1.0, -2.0, 3.0, 3.0], dtype=dtype)
    assert int(jax.jit(connectivity_boundary._argmax_exact)(values)) == 2
    assert int(jax.jit(connectivity_boundary._argmin_exact)(values)) == 1
    assert int(jax.jit(topology.extremum_index)(values)) == 2
    missing = jnp.full(2, jnp.nan, dtype=dtype)
    assert int(jax.jit(connectivity_boundary._argmax_exact)(missing)) == 0
    assert int(jax.jit(connectivity_boundary._argmin_exact)(missing)) == 0


@pytest.mark.parametrize("site", ["mode", "initial_wall", "confined_wall", "contact"])
def test_all_nan_call_sites_select_first_candidate(monkeypatch, site):
    """Observe each production selection; all-NaN scores must not wrap to last."""
    from types import SimpleNamespace

    indices = []
    select = topology.extremum_index

    def observed(score, **kwargs):
        index = select(score, **kwargs)

        def record(values, selected):
            if np.all(np.isnan(values)):
                indices.append(int(selected))

        jax.debug.callback(record, score, index, ordered=True)
        return index

    monkeypatch.setattr(topology, "extremum_index", observed)
    if site == "mode":
        from tests.test_topology_o_qualification import _topology_fixture

        reader, _ = _topology_fixture()
        result = reader.boundary(
            jnp.asarray([1.0, 0.0, 2.0]),
            jnp.full((3, 3), jnp.nan),
            jnp.asarray([1.0, 0.0, jnp.nan]),
            1.0,
        )
        jax.block_until_ready(result)
        jax.effects_barrier()
        print("ALL_NAN_SELECTION", site, indices, np.asarray(result))
        assert indices == [0]
        assert np.all(np.isnan(np.asarray(result)))
        return

    flux = jnp.asarray([1.0, 2.0]) if site == "contact" else jnp.full(2, jnp.nan)
    run, nulls, geometry, policy, points, _ = _wall_read_fixture(monkeypatch, flux)
    decision = run(nulls)
    jax.block_until_ready(decision)
    jax.effects_barrier()
    if site != "contact":
        print("ALL_NAN_SELECTION", site, indices, np.asarray(decision.boundary))
        assert len(indices) == 2
        assert indices[0 if site == "initial_wall" else 1] == 0
        np.testing.assert_array_equal(decision.boundary, points[0])
        return

    indices.clear()
    monkeypatch.setattr(topology, "_wall_events", lambda *args: (points, flux))
    decision = decision._replace(boundary=jnp.full(2, jnp.nan))
    coefficient = jnp.zeros((1, 6))
    field = SimpleNamespace(sampled_coefficient=lambda _: coefficient)
    monkeypatch.setattr(
        topology,
        "_fragments_at_level",
        lambda *args: (
            jnp.asarray([[1.0, 0.0]]),
            jnp.zeros((1, 2, 1, 2)),
            jnp.asarray(True),
            jnp.asarray(1),
        ),
    )
    result = jax.jit(
        lambda solved, carried: topology.refresh_at_stationary_points(
            field,
            geometry,
            topology.TopologyConvention.from_cocos(17, 1.0),
            policy,
            solved,
            carried,
        )
    )(nulls, decision)
    jax.block_until_ready(result)
    jax.effects_barrier()
    print("ALL_NAN_SELECTION", site, indices, np.asarray(result.boundary))
    assert indices == [0]
    np.testing.assert_array_equal(result.boundary, points[0])
