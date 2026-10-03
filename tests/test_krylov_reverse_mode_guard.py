"""Reverse-mode census for the qualified Krylov step.

``_qualified_krylov_step`` (``nova/equilibrium/fixed_point.py``) is built on a
``jax.custom_batching.custom_vmap`` stream: it carries a batching rule but no
derivative rule. A caller that reverse-differentiates through it does not get
the step's derivative -- linearisation fails. Every call site must therefore
either sit under an explicit derivative rule (``custom_jvp``, ``custom_vjp`` or
``custom_linear_solve``) or belong to a forward-only route listed in
``_FORWARD_ONLY_CALLERS`` with a one-line reason. A new caller that does
neither fails :func:`test_every_krylov_caller_is_classified`, which names it,
until it is classified.
"""

from __future__ import annotations

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_point
from nova.equilibrium.fixed_point import _qualified_krylov_step, picard
from nova.jax.config import configure_dtypes

_KRYLOV_STEP_NAME = "_qualified_krylov_step"
_DERIVATIVE_RULE_DECORATORS = frozenset(
    {"custom_jvp", "custom_vjp", "custom_linear_solve"}
)

# Functions that call ``_qualified_krylov_step`` on a forward-only route: the
# enclosing solve never reverse- or forward-differentiates the step, so no
# derivative rule is required. A caller absent from this mapping must sit under
# a derivative rule, or the census fails and names it.
_FORWARD_ONLY_CALLERS = {
    "_manifold_newton_krylov.newton_body.attempt_step": (
        "forward-only: the manifold Newton route reports terminal states, never "
        "differentiates them"
    ),
    "_newton_krylov_inner.newton_body.attempt_step": (
        "forward-only: Newton-Krylov's exact-tangent route carries no reverse"
    ),
    "kink_aware_newton_krylov.krylov_step": (
        "forward-only: the kink-aware route reports the promoted state"
    ),
    "kink_aware_newton_krylov.newton_body.clarke_step": (
        "forward-only: the Clarke averaged-tangent step is promoted, not read "
        "for a derivative"
    ),
}


def _called_name(call: ast.Call) -> str | None:
    """The bare name of a call's callee, whether spelled ``f`` or ``a.f``."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_functions(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The function chain around ``node``, innermost first."""
    chain: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    cursor = parents.get(node)
    while cursor is not None:
        if isinstance(cursor, ast.FunctionDef | ast.AsyncFunctionDef):
            chain.append(cursor)
        cursor = parents.get(cursor)
    return chain


def _dotted_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
) -> str:
    parts = [node.name]
    cursor = parents.get(node)
    while cursor is not None:
        if isinstance(cursor, ast.FunctionDef | ast.AsyncFunctionDef):
            parts.append(cursor.name)
        cursor = parents.get(cursor)
    return ".".join(reversed(parts))


def _names_a_derivative_rule(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Whether ``node`` is decorated with, or contains, a derivative rule."""
    for decorator in node.decorator_list:
        cursor = decorator
        while isinstance(cursor, ast.Attribute):
            if cursor.attr in _DERIVATIVE_RULE_DECORATORS:
                return True
            cursor = cursor.value
        if isinstance(cursor, ast.Name) and cursor.id in _DERIVATIVE_RULE_DECORATORS:
            return True
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _called_name(child)
            if name in _DERIVATIVE_RULE_DECORATORS:
                return True
    return False


def _krylov_calls() -> dict[str, dict]:
    """Classify each caller of the Krylov step by its call sites.

    Returns ``{dotted name: {"lines": [...], "under_rule": bool}}`` for every
    function (or ``<module>``) that calls ``_qualified_krylov_step``, where
    ``under_rule`` records whether the call sits under a ``custom_jvp``,
    ``custom_vjp`` or ``custom_linear_solve`` rule.
    """
    source = inspect.getsource(fixed_point)
    tree = ast.parse(source)
    parents = _parent_map(tree)
    calls: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _called_name(node) != _KRYLOV_STEP_NAME:
            continue
        chain = _enclosing_functions(node, parents)
        if chain:
            name = _dotted_name(chain[0], parents)
            under_rule = any(_names_a_derivative_rule(member) for member in chain)
        else:
            name = "<module>"
            under_rule = False
        entry = calls.setdefault(name, {"lines": [], "under_rule": under_rule})
        entry["lines"].append(node.lineno)
        entry["under_rule"] = entry["under_rule"] or under_rule
    return calls


def _picard_solve(control, *, evaluations: int = 80):
    """Converge a scalar contraction while keeping the control explicit."""
    return picard(
        lambda state, value: 0.2 * state + value,
        jnp.zeros(1),
        evaluations=evaluations,
        relaxation=0.7,
        map_arguments=(control,),
        implicit_tolerance=1.0e-12,
    )


def test_picard_route_reverse_mode_returns_finite_gradients():
    """The implicit Picard adjoint is the reverse-mode positive case."""
    configure_dtypes()
    control = jnp.asarray([2.0])

    def response(value):
        return _picard_solve(value).state[0]

    gradient = jax.grad(response)(control)

    assert bool(_picard_solve(control).converged)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_allclose(gradient, jnp.asarray([1.25]), atol=1.0e-10)


def test_reverse_mode_through_the_raw_krylov_step_refuses():
    """A direct reverse-mode call has no rule to differentiate and fails."""

    def step_sum(vector):
        qualified = _qualified_krylov_step(
            lambda value: value - 0.5 * value,
            vector,
            jnp.max(jnp.abs(vector)),
            gmres_iterations=4,
            condition_ratio_limit=1.0e6,
            preceding_condition_baseline=jnp.asarray(1.0),
        )
        return jnp.sum(qualified.step)

    with pytest.raises(Exception) as refusal:
        jax.grad(step_sum)(jnp.asarray([1.0, 2.0]))

    message = str(refusal.value).lower()
    assert "lineariz" in message or "differentiat" in message


def test_every_krylov_caller_is_classified():
    """Every Krylov caller is under a derivative rule or on the forward list."""
    known = {
        "_manifold_newton_krylov.newton_body.attempt_step",
        "_newton_krylov_inner.newton_body.attempt_step",
        "kink_aware_newton_krylov.krylov_step",
        "kink_aware_newton_krylov.newton_body.clarke_step",
    }
    calls = _krylov_calls()

    # Liveness: the census must see the callers known to be present, so a
    # parser that silently finds nothing cannot pass an absence check.
    missed = known - set(calls)
    assert not missed, (
        f"census missed known {_KRYLOV_STEP_NAME} callers: {sorted(missed)}"
    )
    assert sum(len(entry["lines"]) for entry in calls.values()) >= 5

    # A forward-only entry that no longer calls the step is a stale allowance.
    stale = set(_FORWARD_ONLY_CALLERS) - set(calls)
    assert not stale, (
        f"_FORWARD_ONLY_CALLERS names callers that no longer call "
        f"{_KRYLOV_STEP_NAME}: {sorted(stale)}"
    )

    offenders = {
        name: entry["lines"]
        for name, entry in calls.items()
        if not entry["under_rule"] and name not in _FORWARD_ONLY_CALLERS
    }
    assert not offenders, (
        f"Unclassified caller(s) of {_KRYLOV_STEP_NAME}: "
        + ", ".join(
            f"{name} (lines {lines})" for name, lines in sorted(offenders.items())
        )
        + ". Each caller must sit under a custom_jvp, custom_vjp or "
        "custom_linear_solve rule, or be named in _FORWARD_ONLY_CALLERS with a "
        "one-line reason."
    )
