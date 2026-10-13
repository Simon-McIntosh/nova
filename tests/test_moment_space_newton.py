"""Moment residual derivatives through stationary points and moving membership."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nova.equilibrium import current, gs, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy
from tests.test_support_read import _case, UnitDensity

assert jax.config.jax_enable_x64 is True


def test_conductor_jets_keep_currents_as_operands():
    points = jnp.asarray(((2.0, 0.1), (2.2, -0.3)))
    sections = jnp.asarray(((1.2, 0.5, 0.0, 0.0), (3.0, -0.8, 0.2, 0.3)))
    rows = support_read.conductor_jet_rows(points, sections)
    assert rows.shape == (2, 2, 6)
    assert np.isfinite(np.asarray(rows)).all()
    currents = jnp.asarray((1e4, -2e4))
    direction = jnp.asarray((0.3, -0.2))

    def field(values):
        return jnp.einsum("ncq,c->nq", rows, values)

    _, tangent = jax.jvp(field, (currents,), (direction,))
    central = (field(currents + direction) - field(currents - direction)) / 2
    np.testing.assert_allclose(tangent, central, atol=1e-14, rtol=1e-9)


@pytest.mark.parametrize("kind", ["limited", "diverted"])
def test_moment_residual_jvp_moves_the_support(kind):
    operands, cache, roots, pitch, _step, reference, _axis = _case(kind)
    cells = len(operands.geometry.centre)
    source_cell = int(
        np.argmax(np.asarray(reference.membership * operands.geometry.full_area))
    )
    rows = jnp.zeros((len(operands.nodes), cells, 3, 6))
    rows = rows.at[:, source_cell, 0, 0].set(0.01 * operands.nodes[:, 1])
    rows = rows.at[:, source_cell, 0, 2].set(0.01)
    operands = replace(operands, rows=rows)
    cache = replace(cache, rows=rows)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)

    def residual(state):
        return gs.moment_residual(
            state,
            operands,
            cache,
            roots,
            convention,
            TopologyPolicy(),
            UnitDensity(),
            quadrature,
        )[0]

    compiled = jax.jit(residual)
    state = jnp.zeros((cells, 3))
    direction = jnp.zeros_like(state).at[source_cell, 0].set(1.0)
    for image in range(2):
        value, tangent = jax.jit(
            lambda state: jax.jvp(residual, (state,), (direction,))
        )(state)
        for step in (1e-3, 1e-4, 1e-5, 1e-6):
            estimate = (
                compiled(state + step * direction) - compiled(state - step * direction)
            ) / (2 * step)
            difference = np.abs(np.asarray(tangent - estimate))
            if step == 1e-5:
                central = estimate
            worst = np.unravel_index(np.argmax(difference), difference.shape)
            print(
                f"KIND={kind} IMAGE={image} STEP={step} "
                f"WORST={worst} ERROR={difference[worst]}",
                flush=True,
            )
        assert np.isfinite(np.asarray(value)).all()
        assert np.isfinite(np.asarray(tangent)).all()
        # The identity term alone would conceal a frozen booking derivative.
        response = np.asarray(direction - central)
        assert np.linalg.norm(response) > 1e-6
        error = float(np.max(np.abs(np.asarray(tangent - central))))
        print(
            f"KIND={kind} IMAGE={image} CELLS={cells} JVP_ERROR={error} "
            f"BOOKING_RESPONSE={np.linalg.norm(response)}"
        )
        np.testing.assert_allclose(tangent, central, rtol=2e-4, atol=2e-7)
        following = state - value
        before = support_read.nodal_jets(operands, tuple(state[:, k] for k in range(3)))
        after = support_read.nodal_jets(
            operands, tuple(following[:, k] for k in range(3))
        )
        if image == 0:
            assert np.max(np.abs(np.asarray(after - before))) > 1e-7
        state = following


@pytest.mark.parametrize("method", ["dense", "krylov"])
def test_device_newton_backtracks_and_solves_every_moment(monkeypatch, method):
    from types import SimpleNamespace

    target = jnp.asarray(((1.0, 0.25, 0.09), (4.0, 0.16, 0.36)))

    def equation(moments, *args):
        return moments * moments - target, None

    monkeypatch.setattr(gs, "moment_residual", equation)
    operands = SimpleNamespace(geometry=SimpleNamespace(pitch=jnp.ones(2)))

    def solve(initial):
        return gs.newton_moments(
            initial,
            operands,
            None,
            None,
            None,
            None,
            None,
            None,
            method=method,
            steps=20,
            krylov_size=6,
            tolerance=1e-12,
        )

    initial = jnp.full((2, 3), 0.05)
    result = jax.jit(solve)(initial)
    assert bool(result.converged)
    assert int(result.iterations) > 0
    np.testing.assert_allclose(result.moments, jnp.sqrt(target), rtol=1e-11, atol=1e-12)
    assert float(result.residual_history[-1]) == float(result.residual)
    selected = np.asarray(result.residual_history[1:])[np.asarray(result.accepted)]
    assert np.all(np.diff(selected) < 0)
    np.testing.assert_array_equal(result.moments, solve(initial).moments)
    batched = jax.jit(jax.vmap(solve))(jnp.stack((initial, initial)))
    np.testing.assert_array_equal(batched.moments[0], result.moments)


def test_moment_forward_path_uses_only_the_locked_read():
    import ast
    import inspect
    import textwrap
    from nova.equilibrium.forward import ForwardProfile

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(ForwardProfile._solve_moment_request))
    )
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"newton_moments", "moment_residual", "apply"} <= calls
    assert not calls & {
        "_support_partition",
        "_fixed_design_read",
        "_receipt",
        "_integral_state",
        "_solve_accelerated",
    }, calls
    entry = ast.parse(
        textwrap.dedent(inspect.getsource(ForwardProfile._solve_imposed_request))
    )
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "_solve_moment_request"
        for node in ast.walk(entry)
    )


@pytest.mark.parametrize("route", ["moment_dense", "moment_krylov"])
def test_moment_request_resolves_exact_support(route):
    from nova.equilibrium.solve_request import ForwardSolveRequest, ExplicitSolveSeed

    request = ForwardSolveRequest.from_defaults(
        carrier_identity="moment-control",
        source_profile=object(),
        seed_policy=ExplicitSolveSeed(jnp.ones((2, 3))),
        policy_overrides={"route": route},
    )
    assert request.clip_mode == "exact"
    with pytest.raises(ValueError, match="locked exact support"):
        replace(request, clip_mode="chord")


def test_operator_constructor_keeps_exterior_inputs_distinct(monkeypatch):
    """Dataclass construction delivers each exterior input to its own operand."""
    import inspect

    from nova.equilibrium.forward_operator import ForwardFluxOperator

    signature = inspect.signature(ForwardFluxOperator.__post_init__)
    sections, prescribed = object(), object()
    captured = {}

    def capture(*args):
        captured.update(signature.bind(*args).arguments)

    monkeypatch.setattr(ForwardFluxOperator, "__post_init__", capture)
    ForwardFluxOperator(
        grid=None,
        wall=None,
        source=None,
        external_current=None,
        area=None,
        conductor_sections=sections,
        prescribed_current_field=prescribed,
    )
    assert captured["conductor_sections"] is sections
    assert captured["prescribed_current_field"] is prescribed
