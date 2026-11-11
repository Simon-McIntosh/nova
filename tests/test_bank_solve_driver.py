"""The staged bank driver skips the Jacobian when its batch reuses a factor."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.equilibrium import support_read  # noqa: E402

pytestmark = pytest.mark.timeout(1800)


class _Reading(NamedTuple):
    membership: jax.Array
    normal_form_cells: jax.Array
    fragment_selected: jax.Array
    boundary: jax.Array | None = None


class _Stationary(NamedTuple):
    position: jax.Array


class _Support(NamedTuple):
    stationary: _Stationary
    reading: _Reading
    valid: jax.Array
    cells: jax.Array
    saddle_live: jax.Array
    rebuilds: jax.Array
    rebuilt: jax.Array


def _toy_read(_operands, _cache, states, seeds, previous, *_args, **_kwargs):
    lanes = states[0].shape[0]
    return _Support(
        _Stationary(previous),
        _Reading(
            jnp.full((lanes, 1), 0.5),
            jnp.zeros((lanes, 1), bool),
            jnp.ones((lanes, 1, 1), bool),
        ),
        jnp.ones(lanes, bool),
        jnp.zeros((lanes, 1), jnp.int32),
        jnp.zeros(lanes, bool),
        jnp.zeros(lanes, jnp.int32),
        jnp.zeros(lanes, bool),
    )


def _toy_residual(_operands, base, state, support, _profile, _quadrature):
    return support_read.StepResidual(
        state,
        (*state, jnp.asarray(1.0)),
        support.reading,
        support.valid,
    )


def _moving_support(states, membership, fragments):
    lanes = states[0].shape[0]
    return _Support(
        _Stationary(jnp.zeros((lanes, 1, 2))),
        _Reading(
            membership[:, None],
            jnp.zeros((lanes, 1), bool),
            fragments[:, None, None],
            states[0],
        ),
        jnp.ones(lanes, bool),
        jnp.zeros((lanes, 1), jnp.int32),
        jnp.zeros(lanes, bool),
        jnp.zeros(lanes, jnp.int32),
        jnp.zeros(lanes, bool),
    )


def _unit_update(_operands, moments, _support, *_args, **_kwargs):
    update = (jnp.ones_like(moments[0]),) + tuple(
        jnp.zeros_like(value) for value in moments[1:]
    )
    size = sum(value.size for value in moments)
    matrix = jnp.eye(size, dtype=moments[0].dtype)
    factor = support_read.NewtonFactor(
        matrix, jnp.ones(size), jnp.ones(size), matrix, jnp.zeros(size, jnp.int32)
    )
    return support_read.NewtonStep(
        update, 1.0, 1.0, 0.0, jnp.asarray(True), factor, jnp.asarray(True)
    )


def test_frozen_set_keeps_fresh_continuous_area(monkeypatch):
    """Moving a boundary within fixed cells must not freeze its area fraction."""
    observed = []

    def read(_operands, _cache, states, *_args, **_kwargs):
        position = states[0][:, 0]
        return _moving_support(
            states, 0.4 + 0.1 * position, jnp.ones_like(position, bool)
        )

    def residual(_operands, _base, state, support, *_args):
        area = support.reading.membership * 1e-7
        return support_read.StepResidual(
            (area, jnp.zeros_like(state[1]), jnp.zeros_like(state[2])),
            (*state, jnp.asarray(1.0)),
            support.reading,
            support.valid,
        )

    def step(operands, moments, support, *args, **kwargs):
        fresh = jax.tree.map(
            lambda value: value[0],
            read(None, None, tuple(value[None] for value in moments)),
        )
        frozen_residual = residual(None, moments, moments, support).residual[0][0]
        fresh_residual = residual(None, moments, moments, fresh).residual[0][0]
        jax.debug.callback(
            lambda boundary, frozen, unfrozen: observed.append(
                (float(boundary), float(frozen), float(unfrozen))
            ),
            support.reading.boundary[0],
            frozen_residual,
            fresh_residual,
        )
        result = _unit_update(operands, moments, support, *args, **kwargs)
        return result._replace(update=(0.1 * result.update[0], *result.update[1:]))

    monkeypatch.setattr(support_read, "read_batch", read)
    monkeypatch.setattr(support_read, "step_residual", residual)
    monkeypatch.setattr(support_read, "newton_step", step)
    state = tuple(jnp.zeros((1, 1)) for _ in range(3))
    seeds = jnp.zeros((1, 1, 2))
    run = jax.jit(
        lambda values: support_read.solve_batch(
            None,
            None,
            values,
            seeds,
            seeds,
            None,
            None,
            None,
            None,
            None,
            max_steps=2,
            tolerance=1e-10,
        )
    )
    jax.block_until_ready(run(state))
    assert len(observed) == 2
    boundary, frozen_residual, fresh_residual = observed[1]
    relative_residual_gap = abs(frozen_residual - fresh_residual) / abs(fresh_residual)
    print(
        "FROZEN_RESIDUAL",
        boundary,
        frozen_residual,
        fresh_residual,
        relative_residual_gap,
    )
    assert boundary > 0 and relative_residual_gap <= 1e-12


def test_fragment_switch_forces_fresh_extra_step(monkeypatch):
    """A changed fragment within the same cut cell releases the frozen set."""

    def read(_operands, _cache, states, *_args, **_kwargs):
        position = states[0][:, 0]
        return _moving_support(states, jnp.full_like(position, 0.5), position >= 1)

    def residual(_operands, _base, state, support, *_args):
        value = jnp.where(state[0] >= 1, 1e-12, 1e-7)
        return support_read.StepResidual(
            (value, jnp.zeros_like(state[1]), jnp.zeros_like(state[2])),
            (*state, jnp.asarray(1.0)),
            support.reading,
            support.valid,
        )

    monkeypatch.setattr(support_read, "read_batch", read)
    monkeypatch.setattr(support_read, "step_residual", residual)
    monkeypatch.setattr(support_read, "newton_step", _unit_update)
    state = tuple(jnp.zeros((1, 1)) for _ in range(3))
    seeds = jnp.zeros((1, 1, 2))
    result = jax.jit(
        lambda values: support_read.solve_batch(
            None,
            None,
            values,
            seeds,
            seeds,
            None,
            None,
            None,
            None,
            None,
            max_steps=3,
            tolerance=1e-10,
        )
    )(state)
    jax.block_until_ready(result)
    print("FRAGMENT_SWITCH", int(result.steps[0]), int(result.unfreezes[0]))
    assert bool(result.converged[0])
    assert int(result.steps[0]) == 2
    assert int(result.unfreezes[0]) == 1


def test_first_batched_read_has_one_rebuild_flag_per_lane(monkeypatch):
    """The first read must match the shape of subsequent tracked reads."""
    monkeypatch.setattr(
        support_read,
        "stationary_stencil",
        lambda _bank, _cache, _state, _seed, warm, *_args, **_kwargs: (
            support_read.Stencil(
                _Stationary(warm), jnp.zeros((1,), jnp.int32), jnp.asarray(False)
            )
        ),
    )
    monkeypatch.setattr(
        support_read,
        "_rebuilt_reading",
        lambda *_args: _Reading(
            jnp.asarray([0.5]), jnp.asarray([False]), jnp.asarray([[True]])
        ),
    )
    monkeypatch.setattr(
        support_read,
        "_verdict",
        lambda stencil, reading, rebuilt, rebuilds: _Support(
            stencil.solved,
            reading,
            jnp.asarray(True),
            stencil.cells,
            stencil.saddle_live,
            rebuilds,
            rebuilt,
        ),
    )
    states = tuple(jnp.ones((3, 1)) for _ in range(3))
    seeds = jnp.zeros((3, 1, 2))
    first = support_read.read_batch(None, None, states, seeds, seeds, None, None)
    assert first.rebuilt.shape == (3,)
    assert first.rebuilds.shape == (3,)


def test_batch_reuses_factor_without_rebuilding_jacobians(monkeypatch):
    """A batch-level branch skips every Jacobian on contracting steps."""
    builds = []

    def jacobian(_operands, moments, *_args, **_kwargs):
        jax.debug.callback(lambda value: builds.append(int(value)), moments[0][0])
        return 1.5 * jnp.eye(3, dtype=moments[0].dtype)

    monkeypatch.setattr(support_read, "read_batch", _toy_read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(support_read, "step_jacobian", jacobian)
    states = tuple(jnp.asarray([[1.0], [2.0], [3.0]]) for _ in range(3))
    seeds = jnp.zeros((3, 1, 2))
    run = jax.jit(
        lambda values: support_read.solve_batch(
            None,
            None,
            values,
            seeds,
            seeds,
            None,
            None,
            None,
            None,
            None,
            max_steps=8,
            tolerance=0.1,
            reuse_factor=True,
        )
    )
    result = run(states)
    jax.block_until_ready(result)
    assert bool(jnp.all(result.converged))
    assert np.array_equal(np.asarray(result.refactors), [1, 1, 1])
    assert len(builds) == 3, f"Jacobian rebuilt {len(builds)} times for three lanes"
    assert np.array_equal(
        np.asarray(result.reads), np.full(3, int(jnp.max(result.steps)) + 1)
    )
    assert len(set(np.asarray(result.steps))) > 1


def test_step_cap_reports_nonconvergence(monkeypatch):
    monkeypatch.setattr(support_read, "read_batch", _toy_read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(
        support_read,
        "step_jacobian",
        lambda _operands, moments, *_args, **_kwargs: (
            2 * jnp.eye(3, dtype=moments[0].dtype)
        ),
    )
    state = tuple(jnp.asarray([1.0]) for _ in range(3))
    seeds = jnp.zeros((1, 2))
    result = jax.jit(
        lambda values: support_read.solve(
            None,
            None,
            values,
            seeds,
            seeds,
            None,
            None,
            None,
            None,
            None,
            max_steps=1,
            tolerance=1e-10,
        )
    )(state)
    jax.block_until_ready(result)
    assert not bool(result.converged)
    assert int(result.steps) == 1
    assert float(result.residual) > 1e-10


def test_reads_include_companion_lanes_after_two_halvings(monkeypatch):
    """Every trial reads even the lanes accepted earlier or already at a root."""
    observed = []

    def read(*args, **kwargs):
        jax.debug.callback(
            lambda values: observed.append(np.asarray(values)), args[2][0]
        )
        return _toy_read(*args, **kwargs)

    def jacobian(_operands, moments, *_args, **_kwargs):
        slope = jnp.where(moments[0][0] == 1.0, 0.25, 1.0)
        return slope * jnp.eye(3, dtype=moments[0].dtype)

    monkeypatch.setattr(support_read, "read_batch", read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(support_read, "step_jacobian", jacobian)
    states = tuple(
        jnp.array([[1.0], [2.0], [0.0]], dtype=jnp.float64) for _ in range(3)
    )
    points = jnp.zeros((3, 1, 2), dtype=jnp.float64)
    result = jax.jit(
        lambda values: support_read.solve_batch(
            None,
            None,
            values,
            points,
            points,
            None,
            None,
            None,
            None,
            None,
            max_steps=1,
        )
    )(states)
    jax.block_until_ready(result)
    jax.effects_barrier()
    print("READ_ACCOUNTING", len(observed), np.asarray(result.reads).tolist())
    np.testing.assert_array_equal(np.asarray(result.steps), [1, 1, 0])
    np.testing.assert_array_equal(np.asarray(observed)[:, 0, 0], [1, -3, -1, 0])
    assert bool(jnp.all(result.converged))
    assert len(observed) == 4
    np.testing.assert_array_equal(np.asarray(result.reads), [4, 4, 4])


def test_dyadic_search_keeps_incumbent_and_rejects_invalid_trials():
    """A masked lane stays still; invalid improvements cannot displace its read."""
    from nova.equilibrium.solve import _damped_scale

    def search(value, change, active):
        def evaluate(candidate):
            return jnp.abs(candidate[0]), candidate[0] != 0, candidate

        return _damped_scale(
            evaluate,
            value,
            change,
            jnp.abs(value[0]),
            value,
            active=active,
            max_steps=3,
        )

    result = jax.jit(jax.vmap(search))(
        jnp.ones((3, 1), dtype=jnp.float64),
        jnp.array([[-1.0], [1.0], [-1.0]], dtype=jnp.float64),
        jnp.array([True, True, False]),
    )
    np.testing.assert_array_equal(result.state[:, 0], [0.5, 1.0, 1.0])
    np.testing.assert_array_equal(result.auxiliary, result.state)
    np.testing.assert_array_equal(result.scale, [0.5, 0.0, 0.0])
    np.testing.assert_array_equal(result.evaluations, [2, 3, 0])

    uphill = _damped_scale(
        lambda candidate: (candidate[0], jnp.asarray(True), candidate),
        jnp.array([1.0]),
        jnp.array([1.0]),
        jnp.asarray(1.0),
        jnp.array([1.0]),
        max_steps=3,
        keep_incumbent=False,
    )
    np.testing.assert_array_equal(uphill.state, [1.25])
    np.testing.assert_array_equal(uphill.auxiliary, uphill.state)
    assert float(uphill.residual) == 1.25


@pytest.fixture(scope="module")
def diverted_case():
    from scripts.prototypes.bank_newton_step.measure import machine

    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    return bank, cache, roots, seeds, convention, policy, profile, quadrature, moments


def _device_solve(case, *, reuse_factor, freeze_active_set, contraction_limit=0.5):
    bank, cache, roots, seeds, convention, policy, profile, quadrature, moments = case

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    run = jax.jit(
        lambda state: support_read.solve(
            bank,
            cache,
            state,
            seeds,
            roots,
            convention,
            policy,
            profile,
            gradient,
            quadrature,
            max_steps=60,
            reuse_factor=reuse_factor,
            contraction_limit=contraction_limit,
            freeze_active_set=freeze_active_set,
        )
    )
    result = run(moments)
    jax.block_until_ready(result)
    return result


@pytest.fixture(scope="module")
def prototype_solution(diverted_case):
    bank, cache, roots, seeds, convention, policy, profile, quadrature, moments = (
        diverted_case
    )

    @jax.jit
    def read(state):
        return support_read.read(bank, cache, state, seeds, roots, convention, policy)

    @jax.jit
    def step(state, support):
        return support_read.newton_step(
            bank,
            state,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
        )

    @jax.jit
    def norm(state, support):
        residual = support_read.step_residual(
            bank, state, state, support, profile, quadrature
        )
        return jnp.max(jnp.abs(jnp.concatenate(residual.residual)))

    residuals, updates = [], []
    state = moments
    for _ in range(60):
        support = read(state)
        fresh = float(norm(state, support))
        residuals.append(fresh)
        if bool(support.valid) and fresh <= 1e-10:
            break
        correction = step(state, support)
        updates.append(float(jnp.max(jnp.abs(jnp.concatenate(correction.update)))))
        state = tuple(
            value + change
            for value, change in zip(state, correction.update, strict=True)
        )
    assert bool(support.valid) and residuals[-1] <= 1e-10
    return state, support, residuals, updates


@pytest.fixture(scope="module")
def fresh_solution(diverted_case):
    return _device_solve(diverted_case, reuse_factor=False, freeze_active_set=False)


@pytest.fixture(scope="module")
def reused_solution(diverted_case):
    return _device_solve(diverted_case, reuse_factor=True, freeze_active_set=False)


@pytest.fixture(scope="module")
def strict_reused_solution(diverted_case):
    return _device_solve(
        diverted_case,
        reuse_factor=True,
        freeze_active_set=False,
        contraction_limit=0.1,
    )


@pytest.fixture(scope="module")
def frozen_solution(diverted_case):
    return _device_solve(diverted_case, reuse_factor=False, freeze_active_set=True)


def _precision_gaps(bank, one_state, one_support, other_state, other_support, radius):
    flux_one = support_read.nodal_jets(bank, one_state)[:, 0]
    flux_other = support_read.nodal_jets(bank, other_state)[:, 0]
    span = abs(float(one_support.reading.boundary_flux - one_support.reading.axis_flux))
    return dict(
        flux=float(jnp.max(jnp.abs(flux_one - flux_other))) / span,
        axis=float(
            jnp.linalg.norm(one_support.reading.axis - other_support.reading.axis)
        )
        / radius,
        saddle=float(
            jnp.linalg.norm(
                one_support.reading.boundary - other_support.reading.boundary
            )
        )
        / radius,
        boundary=abs(
            float(
                one_support.reading.boundary_flux - other_support.reading.boundary_flux
            )
        )
        / span,
    )


@pytest.mark.slow
def test_converged_state_matches_prototype_budget(
    diverted_case, prototype_solution, fresh_solution
):
    from tests.equilibrium import test_topology_read as fixture

    bank = diverted_case[0]
    state, support, residuals, updates = prototype_solution
    staged = fresh_solution
    _, _, wall, _, _ = fixture._analytic_inputs("diverted")
    radius = 0.5 * float(np.ptp(np.asarray(wall)[:, 0]))
    gaps = _precision_gaps(bank, state, support, staged.moments, staged.support, radius)
    print(
        "TABLE step prototype_residual prototype_update staged_residual staged_update"
    )
    for index in range(max(len(residuals), int(staged.steps) + 1)):
        prototype_residual = (
            residuals[index] if index < len(residuals) else float("nan")
        )
        prototype_update = updates[index] if index < len(updates) else float("nan")
        staged_residual = (
            float(staged.residual_path[index])
            if index <= int(staged.steps)
            else float("nan")
        )
        staged_update = (
            float(staged.update_path[index])
            if index < int(staged.steps)
            else float("nan")
        )
        print(
            f"ROW {index} {prototype_residual:.12e} {prototype_update:.12e} "
            f"{staged_residual:.12e} {staged_update:.12e}"
        )
    print(
        "PRECISION",
        gaps,
        "prototype_steps",
        len(updates),
        "staged_steps",
        int(staged.steps),
    )
    assert bool(staged.converged) and float(staged.residual) <= 1e-10
    assert residuals[-1] <= 1e-10
    assert gaps["flux"] <= 1e-8
    assert gaps["axis"] <= 5e-4 and gaps["saddle"] <= 5e-4
    assert gaps["boundary"] <= 1e-4


@pytest.mark.slow
def test_factor_reuse_costs(fresh_solution, reused_solution, strict_reused_solution):
    for label, result in (
        ("fresh", fresh_solution),
        ("half", reused_solution),
        ("tenth", strict_reused_solution),
    ):
        steps, refactors = int(result.steps), int(result.refactors)
        wall_estimate = 8 * steps + 20 * refactors
        print(
            f"COST {label} steps={steps} refactors={refactors} "
            f"reads={int(result.reads)} estimated_ms={wall_estimate} "
            f"residual={float(result.residual):.12e}"
        )
        assert bool(result.converged) and float(result.residual) <= 1e-10


@pytest.mark.slow
def test_active_set_freeze_trajectory_converges(fresh_solution, frozen_solution):
    print(
        "UNFROZEN",
        np.asarray(fresh_solution.residual_path[: int(fresh_solution.steps) + 1]),
    )
    print(
        "FROZEN",
        np.asarray(frozen_solution.residual_path[: int(frozen_solution.steps) + 1]),
    )
    assert bool(fresh_solution.converged) and bool(frozen_solution.converged)
    assert int(frozen_solution.freezes) >= 1
    assert float(frozen_solution.residual) <= 1e-10


@pytest.mark.parametrize(
    "slope, expected_refactors", [(1.0, 0), (4.0, 1), (float("nan"), 1)]
)
def test_carried_factor_first_update_is_guarded(monkeypatch, slope, expected_refactors):
    """A carried trial must contract its fresh residual before it is retained."""
    builds = []

    def jacobian(_operands, moments, *_args, **_kwargs):
        jax.debug.callback(lambda _: builds.append(1), moments[0][0])
        return jnp.eye(3, dtype=moments[0].dtype)

    monkeypatch.setattr(support_read, "read_batch", _toy_read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(support_read, "step_jacobian", jacobian)
    states = tuple(jnp.ones((1, 1), dtype=jnp.float64) for _ in range(3))
    points = jnp.zeros((1, 1, 2), dtype=jnp.float64)
    matrix = slope * jnp.eye(3, dtype=jnp.float64)
    inverse, indices = support_read._block_preconditioner(matrix)
    carried = support_read.NewtonFactor(
        matrix[None], jnp.ones((1, 3)), jnp.ones((1, 3)), inverse[None], indices[None]
    )
    result = jax.jit(
        lambda state, factor: support_read.solve_batch(
            None,
            None,
            state,
            points,
            points,
            None,
            None,
            None,
            None,
            None,
            max_steps=1,
            reuse_factor=True,
            factor_cache=factor,
            freeze_active_set=False,
        )
    )(states, carried)
    jax.block_until_ready(result)
    jax.effects_barrier()
    print(
        "CARRIED_FACTOR",
        slope,
        np.asarray(result.refactors),
        len(builds),
        np.asarray(result.residual),
        np.asarray(result.reads),
    )
    assert bool(result.converged[0])
    assert float(result.residual[0]) <= 1e-10
    assert float(result.residual_path[0, 1]) == float(result.residual[0])
    assert int(result.refactors[0]) == expected_refactors
    assert len(builds) == expected_refactors
    assert bool(result.factor_valid[0])
    np.testing.assert_allclose(result.factor.matrix[0], jnp.eye(3), rtol=0, atol=0)


def test_factor_is_returned_for_the_next_slice(monkeypatch):
    monkeypatch.setattr(support_read, "read_batch", _toy_read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(
        support_read,
        "step_jacobian",
        lambda *_args, **_kwargs: jnp.eye(3, dtype=jnp.float64),
    )
    points = jnp.zeros((1, 1, 2), dtype=jnp.float64)

    def run(state, factor=None):
        return support_read.solve_batch(
            None,
            None,
            state,
            points,
            points,
            None,
            None,
            None,
            None,
            None,
            max_steps=1,
            reuse_factor=True,
            factor_cache=factor,
        )

    state = tuple(jnp.ones((1, 1), dtype=jnp.float64) for _ in range(3))
    first = jax.jit(run)(state)
    second = jax.jit(run)(state, first.factor)
    jax.block_until_ready(second)
    assert int(first.refactors[0]) == 1
    assert int(second.refactors[0]) == 0
    assert bool(second.converged[0]) and float(second.residual[0]) <= 1e-10
