"""Admitted separatrices survive finite proposals onto another branch."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.fixed_point import newton_krylov
from nova.equilibrium.constraint import ConstraintMultiplier, assemble_augmented_system
from nova.equilibrium.topology import TopologyState
from nova.jax.config import Precision, configure_dtypes
from tests.test_equilibrium_constraint_protocol import (
    _CoordinateFunctional,
    _pair,
    _profile,
)

configure_dtypes()
assert jax.config.jax_enable_x64


class Partition(NamedTuple):
    label: jax.Array
    topology: TopologyState
    profile_support: None
    residual_shadow: jax.Array


def _solve(initial, *, switch="limited", warmup=0, newton_steps=2):
    def mapped(state):
        return 2.0 - 0.5 * state

    def mask(state):
        return jnp.ones_like(state, dtype=bool)

    def read(state, previous=None):
        del previous
        changed = state[0] >= 0.8
        axis = jnp.array([1.5, 0.0])
        saddle = jnp.array([1.0, jnp.where(changed & (switch == "upper"), 1.0, -1.0)])
        diverted = ~changed if switch == "limited" else jnp.asarray(True)
        topology = TopologyState(
            axis, 0.0, saddle, 1.0, saddle, 1.0, axis, 2.0, diverted
        )
        return Partition(jnp.ones(1, dtype=jnp.int32), topology, None, mask(state))

    def shadowed(state, shadow):
        del shadow
        return mapped(state)

    shadowed._read_frozen_partition = read
    shadowed._map_frozen_partition = lambda state, partition: mapped(state)
    shadowed._frozen_partition_shadow = lambda partition: partition.residual_shadow
    return newton_krylov(
        mapped,
        initial,
        newton_steps=newton_steps,
        gmres_iterations=1,
        warmup=warmup,
        relaxation=1.0,
        shadow_mask_fn=mask,
        promoted_shadow_mask_fn=lambda state, previous: mask(state),
        shadowed_map_fn=shadowed,
        active_set_steps=2,
        preserve_diverted_branch=True,
        precision=Precision.DOUBLE,
    )


@pytest.mark.parametrize("switch", ["limited", "upper"])
def test_newton_refuses_boundary_loss_and_upper_saddle_switch(switch):
    result = jax.jit(lambda initial: _solve(initial, switch=switch))(jnp.zeros(1))
    np.testing.assert_array_equal(result.state, [0.0])
    np.testing.assert_array_equal(result.trajectory_state, [0.0])
    assert int(result.accepted_newton_promotions) == 0
    assert int(jnp.sum(result.newton_branch_refused_proposals)) > 0
    assert int(result.newton_first_branch_refusal_index[0]) == 0


@pytest.mark.parametrize("switch", ["limited", "upper"])
def test_warmup_refuses_boundary_loss_and_upper_saddle_switch(switch):
    result = jax.jit(lambda initial: _solve(initial, switch=switch, warmup=2))(
        jnp.zeros(1)
    )
    np.testing.assert_array_equal(result.state, [0.0])
    np.testing.assert_array_equal(result.trajectory_state, [0.0])
    assert int(jnp.sum(result.warmup_branch_refused_proposals)) == 2
    assert int(result.warmup_first_branch_refusal_index[0]) == 0
    assert int(jnp.sum(result.warmup_refused_proposals)) == 0


def test_limited_seed_remains_unconstrained():
    result = jax.jit(_solve)(jnp.ones(1))
    np.testing.assert_allclose(result.state, [4.0 / 3.0], rtol=0.0, atol=1e-14)
    assert bool(result.converged)
    assert int(jnp.sum(result.newton_branch_refused_proposals)) == 0


def test_warmup_relaxation_stays_damped_after_finite_refusal():
    def mapped(state):
        return jnp.where(state > 1.5, jnp.nan, 2.0 - 0.5 * state)

    result = jax.jit(
        lambda initial: newton_krylov(
            mapped,
            initial,
            warmup=3,
            newton_steps=0,
            relaxation=1.0,
            precision=Precision.DOUBLE,
        )
    )(jnp.zeros(1))
    # Refuse 2, then retain relaxation 1/2 for both accepted proposals.
    np.testing.assert_array_equal(result.trajectory_state, [1.25])
    np.testing.assert_array_equal(result.warmup_refused_proposals, [1])


@pytest.mark.parametrize("warmup", [0, 2])
def test_augmented_route_qualifies_physical_flux_with_runtime_rows(warmup):
    profile = _profile()

    def read(flux, requested_class=None):
        assert requested_class is None
        axis, saddle = jnp.array([1.5, 0.0]), jnp.array([1.0, -1.0])
        topology = TopologyState(
            axis, 0.0, saddle, 1.0, saddle, 1.0, axis, 2.0, flux[0] < 0.8
        )
        return None, topology, None, None

    profile.operator._fixed_design_read = read
    pair = _pair(
        _CoordinateFunctional(1),
        ConstraintMultiplier(jnp.asarray([1.0])),
        target=2.0,
        payload=0.0,
    )

    def mapped(flux):
        return flux.at[0].set(2.0 - 0.5 * flux[0])

    def solve(initial):
        system = assemble_augmented_system(
            profile,
            initial,
            (pair,),
            base_map=mapped,
            base_shadow_mask=lambda flux: jnp.zeros_like(flux, dtype=bool),
            base_promoted_shadow_mask=lambda flux, previous: previous,
            base_shadowed_map=lambda flux, shadow: mapped(flux),
            requested_class=None,
            target_current=None,
        )
        result = newton_krylov(
            system.map_fn,
            system.initial,
            map_arguments=((pair,),),
            shadow_mask_fn=system.shadow_mask_fn,
            promoted_shadow_mask_fn=system.promoted_shadow_mask_fn,
            shadowed_map_fn=system.shadowed_map_fn,
            warmup=warmup,
            newton_steps=2,
            gmres_iterations=1,
            active_set_steps=2,
            preserve_diverted_branch=True,
            relaxation=1.0,
            precision=Precision.DOUBLE,
        )
        flux, _unknown = system.split(result.state)
        return flux, result

    compiled = jax.jit(solve)
    flux, result = compiled(jnp.array([0.0, 2.0]))
    np.testing.assert_array_equal(flux, [0.0, 2.0])
    counters = (
        result.warmup_branch_refused_proposals
        if warmup
        else result.newton_branch_refused_proposals
    )
    assert int(jnp.sum(counters)) > 0
    # A second scale and seed use the same compiled program and remain protected.
    other, _result = compiled(jnp.array([0.0, 3.0]))
    np.testing.assert_array_equal(other, [0.0, 3.0])
    assert compiled._cache_size() == 1


@pytest.mark.parametrize(
    "diverted", [False, True], ids=["limited", "saddle-transition"]
)
@pytest.mark.parametrize("active_set", [False, True], ids=["smooth", "active-set"])
def test_unrequested_branch_reader_preserves_program_and_vector_state(
    diverted, active_set
):
    coefficient = jnp.array([[0.2, -0.3, 0.1], [0.4, 0.1, -0.2], [0.1, 0.3, 0.2]])
    forcing = jnp.array([1.1, 0.8, 1.3])
    reads = []

    def mapped(state):
        return forcing + coefficient @ jnp.sin(state)

    def decorated(state):
        return mapped(state)

    def read(state):
        reads.append(True)
        axis = jnp.array([1.5, 0.0])
        saddle = jnp.array([1.0, jnp.where(state[0] < 0.5, -1.0, 1.0)])
        return TopologyState(
            axis, 0.0, saddle, 1.0, saddle, 1.0, axis, 2.0, jnp.asarray(diverted)
        )

    decorated._read_branch_topology = read

    def solve(fn, initial):
        options = {}
        if active_set:

            def shadowed(state, shadow):
                return fn(state)

            if hasattr(fn, "_read_branch_topology"):
                shadowed._read_branch_topology = fn._read_branch_topology
            options = dict(
                shadow_mask_fn=lambda state: jnp.zeros_like(state, dtype=bool),
                promoted_shadow_mask_fn=lambda state, previous: previous,
                shadowed_map_fn=shadowed,
                active_set_steps=2,
            )
        return newton_krylov(
            fn,
            initial,
            newton_steps=5,
            warmup=3,
            gmres_iterations=3,
            precision=Precision.DOUBLE,
            **options,
        )

    initial = jnp.zeros(3)
    plain = jax.jit(lambda value: solve(mapped, value))(initial)
    observed = jax.jit(lambda value: solve(decorated, value))(initial)
    np.testing.assert_array_equal(observed.state, plain.state)
    assert bool(observed.converged) == bool(plain.converged)
    assert int(jnp.sum(observed.warmup_branch_refused_proposals)) == 0
    assert int(jnp.sum(observed.newton_branch_refused_proposals)) == 0
    assert not reads, "unrequested topology qualification was traced"
    plain_program = str(jax.make_jaxpr(lambda value: solve(mapped, value))(initial))
    decorated_program = str(
        jax.make_jaxpr(lambda value: solve(decorated, value))(initial)
    )
    assert decorated_program == plain_program


def test_saved_state_refresh_reuses_measurement_lower_root_reader(
    tmp_path, monkeypatch
):
    import json
    from scripts.prototypes.diiid_diverted_branch import measure, refresh

    radius, height = np.linspace(1.0, 2.0, 5), np.linspace(-1.0, 1.0, 5)
    flux = (radius[:, None] - 1.5) ** 2 - (height[None, :] + 0.5) ** 2
    state = flux.ravel()
    axis, saddle = [1.5, 0.5], [1.5, -0.5]
    data = dict(
        radius=radius,
        height=height,
        flux=flux,
        state=state,
        axis=axis,
        reference_axis=axis,
        limiter=np.array(
            [[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
        ),
    )
    np.savez_compressed(tmp_path / "terminal.npz", **data)
    np.savez_compressed(tmp_path / "inner-0.npz", state=state, initial=state)
    topology = dict(selected_saddle_rz_m=saddle, axis_rz_m=axis)
    record = dict(
        efit_x_point_rz_m=saddle,
        inner=[dict(index=0)],
        recovery={},
        trip_topology=[dict(trip=0, initial=dict(topology), returned=dict(topology))],
        terminal_topology=dict(topology),
    )
    (tmp_path / "receipt.json").write_text(json.dumps(record))
    calls = []
    owner = measure.lower_saddle_reader

    def observed(data, reference):
        read = owner(data, reference)

        def sample(state, selected, axis):
            calls.append(np.asarray(axis))
            return read(state, selected, axis)

        return sample

    monkeypatch.setattr(measure, "lower_saddle_reader", observed)
    monkeypatch.setattr("sys.argv", ["refresh", "--row", str(tmp_path)])
    refresh.main()
    assert len(calls) == 3, "refresh bypassed the measurement's lower-root owner"
    refreshed = json.loads((tmp_path / "receipt.json").read_text())
    terminal = refreshed["terminal_topology"]
    assert terminal["lower_saddle_qualified"]
    assert terminal["lower_saddle_distance_m"] < 1e-12
    np.testing.assert_allclose(terminal["lower_saddle_rz_m"], saddle, atol=1e-12)
