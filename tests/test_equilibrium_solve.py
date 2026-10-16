"""Chord-Newton fixed-point solve: convergence, chord reuse, batching, tracing."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import solve

TOLERANCE = 1e-12
CORRECTIONS_TO_TOLERANCE = 4
NEIGHBOUR_OFFSET = 1e-5


def _map(state, operands):
    """Analytic nonlinear map with a known root at ``state == operands[0]``."""
    root, coupling = operands
    error = state - root
    return error + coupling @ (error**2)


ROOT = jnp.asarray([1.0, 1.0])
COUPLING = jnp.asarray([[0.5, 0.1], [0.1, 0.5]])
OPERANDS = (ROOT, COUPLING)
START = ROOT + jnp.asarray([0.2, 0.3])


def _hlo_constants(lowered):
    """Shapes of every literal-producing HLO operation in a lowered program."""
    source = lowered.compiler_ir("hlo").as_hlo_text(print_large_constants=False)
    expression = re.compile(
        r"^\s*([^\s=]+)\s*=\s*([a-z0-9]+)\[([0-9,]*)\].*\bconstant\("
    )
    shapes = []
    for line in source.splitlines():
        found = expression.match(line)
        if found is not None:
            shapes.append(tuple(int(v) for v in found.group(3).split(",") if v))
    return shapes


@pytest.mark.slow
def test_cold_converges_to_tolerance_in_four_corrections():
    result = solve.fixed_point(_map, START, OPERANDS)
    assert bool(result.converged)
    assert int(result.corrections) == CORRECTIONS_TO_TOLERANCE
    assert float(result.residual) <= TOLERANCE


@pytest.mark.slow
def test_cold_residual_reduction_is_quadratic():
    residuals = [
        float(solve.fixed_point(_map, START, OPERANDS, max_corrections=step).residual)
        for step in range(1, CORRECTIONS_TO_TOLERANCE + 1)
    ]
    for previous, following in zip(residuals, residuals[1:]):
        assert following < previous
        assert following <= previous**2 * 10
    assert residuals[-1] <= TOLERANCE


@pytest.mark.slow
def test_chord_reuses_a_neighbouring_factor_in_two_corrections():
    predecessor = ROOT + NEIGHBOUR_OFFSET
    factored = solve.factor(_map, predecessor, OPERANDS)
    start = ROOT + jnp.asarray([NEIGHBOUR_OFFSET, -NEIGHBOUR_OFFSET])

    single = solve.fixed_point(
        _map, start, OPERANDS, factor=factored, max_corrections=1
    )
    assert not bool(single.converged)

    result = solve.fixed_point(
        _map, start, OPERANDS, factor=factored, max_corrections=2
    )
    assert int(result.corrections) == 2
    assert bool(result.converged)
    assert float(result.residual) <= TOLERANCE


@pytest.mark.slow
def test_factor_perturbed_by_two_percent_fails_chord_convergence():
    predecessor = ROOT + NEIGHBOUR_OFFSET
    factored = solve.factor(_map, predecessor, OPERANDS)
    perturbed = solve.Factor(lu=factored.lu * 1.02, pivots=factored.pivots)
    start = ROOT + jnp.asarray([NEIGHBOUR_OFFSET, -NEIGHBOUR_OFFSET])

    result = solve.fixed_point(
        _map, start, OPERANDS, factor=perturbed, max_corrections=2
    )
    assert float(result.residual) > TOLERANCE
    assert not bool(result.converged)


@pytest.mark.slow
def test_vmap_over_sixteen_states_all_converge():
    count = 16
    roots = jnp.stack(
        [jnp.asarray([1.0 + 0.1 * index, 0.5 - 0.05 * index]) for index in range(count)]
    )
    starts = roots + jnp.asarray([0.15, -0.12])
    couplings = jnp.broadcast_to(COUPLING, (count, 2, 2))

    solved = jax.jit(jax.vmap(lambda state, ops: solve.fixed_point(_map, state, ops)))(
        starts, (roots, couplings)
    )
    assert int(jnp.sum(solved.converged)) == count
    assert float(jnp.max(solved.residual)) <= TOLERANCE


@pytest.mark.slow
def test_compiled_chord_carries_no_state_scaled_constant():
    def constants_for(dimension):
        root = jnp.arange(dimension, dtype=jnp.float64) + 1.0
        coupling = jnp.eye(dimension, dtype=jnp.float64) * 0.5
        operands = (root, coupling)
        factored = solve.factor(_map, root, operands)
        program = jax.jit(
            lambda state, ops, fac: solve.fixed_point(
                _map, state, ops, factor=fac, max_corrections=2
            )
        )
        lowered = program.lower(root, operands, factored)
        return _hlo_constants(lowered)

    small = constants_for(2)
    large = constants_for(6)
    assert small and large
    for shape in small + large:
        assert shape == ()
    assert sum(np.prod(shape) for shape in small) == sum(
        np.prod(shape) for shape in large
    )
