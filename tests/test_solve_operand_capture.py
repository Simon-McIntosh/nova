"""The solve's linearized programs carry their operands, never as literals.

A linearization taken over concrete operands embeds every operand array as a
constant in the compiled program, so each Newton step's factor program carries
its own copy of the window rows and the interior jets. The solve compiles one
such program per step, so the copies multiply and the lowering reports tens of
gigabytes of captured constants. The tangent map built by ``operand_tangent``
takes the operands as arguments, which the lowering reads as program parameters
rather than literals.
"""

import ast
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.prototypes.fast_map_production.solver import operand_tangent

# The operand is a window-sized block: larger than any plausible literal, so a
# program that embeds it is separated from one that carries it by orders of
# magnitude rather than by a tolerance.
OPERAND_SHAPE = (1024, 1024)
STATE_SIZE = 1024
# Literals the program legitimately carries are the few scalars in its body.
LITERAL_CEILING_BYTES = 1 << 20


class _Equation:
    """A tangent-linear stand-in for the solve's moment equation."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, state, operands):
        (rows,) = operands
        return jnp.tanh(rows @ state) + state * jnp.sum(rows[:, 0])


def _constant_bytes(closed):
    total = 0
    for constant in closed.consts:
        array = np.asarray(constant)
        total += array.size * array.dtype.itemsize
    return total


def _operand_bytes(operand):
    array = np.asarray(operand)
    return array.size * array.dtype.itemsize


@pytest.fixture(scope="module")
def operands():
    rows = np.linspace(-1.0, 1.0, int(np.prod(OPERAND_SHAPE)), dtype=np.float64)
    return (jnp.asarray(rows.reshape(OPERAND_SHAPE)),)


def test_tangent_program_carries_no_operand_literal(operands):
    equation = _Equation(operands[0])
    point = jnp.zeros(STATE_SIZE)
    direction = jnp.ones(STATE_SIZE)
    tangent = operand_tangent(equation, point)

    closed = jax.make_jaxpr(tangent)(direction, *operands)

    captured = _constant_bytes(closed)
    assert captured < LITERAL_CEILING_BYTES, (
        f"tangent program captured {captured} constant bytes"
    )


def test_closing_over_the_operand_is_what_captures_it(operands):
    (rows,) = operands
    equation = _Equation(rows)
    point = jnp.zeros(STATE_SIZE)
    direction = jnp.ones(STATE_SIZE)

    def closed_equation(state):
        return equation(state, (rows,))

    factor = jax.linearize(closed_equation, point)[1]
    closed = jax.make_jaxpr(factor)(direction)

    captured = _constant_bytes(closed)
    assert captured >= _operand_bytes(rows) // 2, (
        f"a linearization over concrete operands captured only {captured} bytes, "
        "so this instrument cannot show a capture when one is present"
    )


DENSE_ARM_DRIVERS = [
    "scripts/prototypes/fast_map_production/solver.py",
    "scripts/prototypes/diverted_stall/measure.py",
]


def _bound_names(statements):
    """The names a block binds at its top level, by assignment or by def."""
    names = set()
    for statement in statements:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    names.update([target.id])
        elif isinstance(statement, ast.FunctionDef):
            names.update([statement.name])
    return names


def _dense_arm_bindings(path):
    """The names bound directly in the fall-through arm of ``linear_step``.

    The step row and the diagnose block read a single-direction tangent program
    after the method branch, so every arm must bind one; the dense arm binds a
    program batched over a block of directions, and it is the arm that supplied
    none.
    """
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "linear_step":
            for statement in node.body:
                if isinstance(statement, ast.If) and statement.orelse:
                    chain = statement
                    while len(chain.orelse) == 1 and isinstance(
                        chain.orelse[0], ast.If
                    ):
                        chain = chain.orelse[0]
                    if any(isinstance(item, ast.Assign) for item in chain.orelse):
                        return _bound_names(chain.orelse)
    raise AssertionError(f"linear_step's dense arm not found in {path}")


@pytest.mark.parametrize("path", DENSE_ARM_DRIVERS)
def test_the_dense_arm_binds_a_single_direction_program(path):
    bound = _dense_arm_bindings(path)
    assert "product" in bound, (
        f"{path}: the dense arm binds {sorted(bound)}, but the step row and the "
        "diagnose block read a single-direction 'product' after the branch"
    )
