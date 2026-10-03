"""Reverse-mode census for the qualified Krylov step.

``_qualified_krylov_step`` (``nova/equilibrium/fixed_point.py``) is built on a
``jax.custom_batching.custom_vmap`` stream: it carries a batching rule but no
derivative rule, so a caller that reverse-differentiates through it does not get
the step's derivative -- linearisation fails. This module censuses every module
in the ``nova`` package and requires each call to the step to be *covered*.

A call is covered only when it sits lexically inside a function that is itself
the subject of a derivative rule: a function decorated with ``custom_jvp`` or
``custom_vjp``, a function registered as that rule's ``defjvp``/``defvjp`` (fwd
or bwd) function, or a function passed as the ``matvec``, ``solve`` or
``transpose_solve`` argument of ``jax.lax.custom_linear_solve``. A rule merely
called somewhere in an enclosing function's body does not cover a call, because
that call is still differentiated unless it sits inside the rule's own subject.
An uncovered caller must be named in ``_FORWARD_ONLY_CALLERS`` with a one-line
reason, or :func:`test_every_krylov_census_is_clean` fails and names it.

Tests and benchmarks are out of scope: this census reads the ``nova`` package
only.
"""

from __future__ import annotations

import ast
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nova
from nova.equilibrium.fixed_point import _qualified_krylov_step
from nova.jax.config import configure_dtypes
from tests.test_picard_implicit_derivative import _solve

_KRYLOV_STEP_NAME = "_qualified_krylov_step"
_RULE_DECORATORS = frozenset({"custom_jvp", "custom_vjp"})
_REGISTERING_DECORATORS = frozenset({"defjvp", "defvjp"})
_LINEAR_SOLVE_ARGS = frozenset({"matvec", "solve", "transpose_solve"})

# Functions that call ``_qualified_krylov_step`` on a forward-only route: the
# enclosing solve never reverse- or forward-differentiates the step, so no
# derivative rule is required. A caller absent from this mapping must be covered
# by a derivative rule, or the census fails and names it.
_FORWARD_ONLY_CALLERS = {
    "equilibrium/fixed_point.py::_manifold_newton_krylov.newton_body.attempt_step": (
        "forward-only: the manifold Newton route reports terminal states, never "
        "differentiates them"
    ),
    "equilibrium/fixed_point.py::_newton_krylov_inner.newton_body.attempt_step": (
        "forward-only: Newton-Krylov's exact-tangent route carries no reverse"
    ),
    "equilibrium/fixed_point.py::kink_aware_newton_krylov.krylov_step": (
        "forward-only: the kink-aware route reports the promoted state"
    ),
    "equilibrium/fixed_point.py::kink_aware_newton_krylov.newton_body.clarke_step": (
        "forward-only: the Clarke averaged-tangent step is promoted, not read "
        "for a derivative"
    ),
}

_KNOWN_CALLERS = frozenset(_FORWARD_ONLY_CALLERS)


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


def _is_function_node(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)


def _enclosing_functions(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[ast.AST]:
    """The function chain around ``node``, innermost first."""
    chain: list[ast.AST] = []
    cursor = parents.get(node)
    while cursor is not None and cursor is not node:
        if _is_function_node(cursor):
            chain.append(cursor)
        cursor = parents.get(cursor)
    return chain


def _dotted_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        parts = [node.name]
    else:
        parts = ["<lambda>"]
    cursor = parents.get(node)
    while cursor is not None and cursor is not node:
        if isinstance(cursor, ast.FunctionDef | ast.AsyncFunctionDef):
            parts.append(cursor.name)
        cursor = parents.get(cursor)
    return ".".join(reversed(parts))


def _decorator_terminals(node: ast.AST) -> set[str]:
    """Terminal names of a function's decorators (``@a.b`` yields ``b`` and ``a``)."""
    terminals: set[str] = set()
    for decorator in getattr(node, "decorator_list", []):
        cursor: ast.AST = decorator
        while isinstance(cursor, ast.Attribute):
            terminals.add(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            terminals.add(cursor.id)
    return terminals


def _rule_subject_ids(tree: ast.AST) -> set[int]:
    """Ids of functions that are themselves the subject of a derivative rule."""
    subjects: set[int] = set()
    defs_by_name: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            defs_by_name.setdefault(node.name, node)
            terminals = _decorator_terminals(node)
            if terminals & (_RULE_DECORATORS | _REGISTERING_DECORATORS):
                subjects.add(id(node))
    linear_solve_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _called_name(node) != "custom_linear_solve":
            continue
        arguments = list(node.args[:1]) + [
            keyword.value
            for keyword in node.keywords
            if keyword.arg in _LINEAR_SOLVE_ARGS
        ]
        for argument in arguments:
            if isinstance(argument, ast.Lambda):
                subjects.add(id(argument))
            elif isinstance(argument, ast.Name):
                linear_solve_names.add(argument.id)
            elif isinstance(argument, ast.Attribute):
                linear_solve_names.add(argument.attr)
    for name in linear_solve_names:
        subject = defs_by_name.get(name)
        if subject is not None:
            subjects.add(id(subject))
    return subjects


def _module_calls(relative_path: str, source: str) -> dict[str, dict]:
    """Classify each Krylov-step call in one module's source.

    Returns ``{module-qualified caller: {"lines": [...], "covered": bool}}``.
    """
    tree = ast.parse(source)
    parents = _parent_map(tree)
    subjects = _rule_subject_ids(tree)
    calls: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _called_name(node) != _KRYLOV_STEP_NAME:
            continue
        chain = _enclosing_functions(node, parents)
        if chain:
            name = f"{relative_path}::{_dotted_name(chain[0], parents)}"
        else:
            name = f"{relative_path}::<module>"
        covered = any(id(member) in subjects for member in chain)
        entry = calls.setdefault(name, {"lines": [], "covered": covered})
        entry["lines"].append(node.lineno)
        entry["covered"] = entry["covered"] or covered
    return calls


def _census_calls() -> dict[str, dict]:
    """Classify every Krylov-step call across the ``nova`` package."""
    root = pathlib.Path(nova.__file__).resolve().parent
    calls: dict[str, dict] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative_path = path.relative_to(root).as_posix()
        calls.update(_module_calls(relative_path, path.read_text()))
    return calls


def test_picard_route_reverse_mode_returns_finite_gradients():
    """The implicit Picard adjoint is the reverse-mode positive case."""
    configure_dtypes()
    control = jnp.asarray([2.0])

    def response(value):
        return _solve(value).state[0]

    gradient = jax.grad(response)(control)

    assert bool(_solve(control).converged)
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


def test_a_rule_call_elsewhere_in_the_body_does_not_cover_the_step():
    """A rule called in the body leaves a step outside it uncovered."""
    source = (
        "def _rule_elsewhere_but_uncovered(vector):\n"
        "    jax.lax.custom_linear_solve(\n"
        "        lambda value: value,\n"
        "        vector,\n"
        "        solve=lambda matvec, rhs: matvec(rhs),\n"
        "    )\n"
        "    return _qualified_krylov_step(lambda value: value, vector)\n"
    )
    calls = _module_calls("synthetic.py", source)
    entry = calls["synthetic.py::_rule_elsewhere_but_uncovered"]
    assert entry["covered"] is False


def test_a_rule_subject_covers_a_step_inside_it():
    """The classifier sees coverage so its absence checks mean something."""
    decorated = (
        "@jax.custom_jvp\n"
        "def _rule_subject(vector):\n"
        "    return _qualified_krylov_step(lambda value: value, vector)\n"
    )
    covered = _module_calls("synthetic.py", decorated)
    assert covered["synthetic.py::_rule_subject"]["covered"] is True

    registered = (
        "def _solve_step(matvec, rhs):\n"
        "    return _qualified_krylov_step(lambda value: value, rhs)\n"
        "\n"
        "def _route(vector):\n"
        "    return jax.lax.custom_linear_solve(\n"
        "        lambda value: value, vector, solve=_solve_step\n"
        "    )\n"
    )
    covered = _module_calls("synthetic.py", registered)
    assert covered["synthetic.py::_solve_step"]["covered"] is True


def test_every_krylov_census_is_clean():
    """Every Krylov caller is covered by a rule or on the forward-only list."""
    calls = _census_calls()

    # Liveness: the census must see the callers known to be present, so a parser
    # that silently finds nothing cannot pass an absence check.
    missed = _KNOWN_CALLERS - set(calls)
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
        if not entry["covered"] and name not in _FORWARD_ONLY_CALLERS
    }
    assert not offenders, (
        f"Unclassified caller(s) of {_KRYLOV_STEP_NAME}: "
        + ", ".join(
            f"{name} (lines {lines})" for name, lines in sorted(offenders.items())
        )
        + ". Each caller must be the subject of a custom_jvp, custom_vjp or "
        "custom_linear_solve rule, or be named in _FORWARD_ONLY_CALLERS with a "
        "one-line reason."
    )
