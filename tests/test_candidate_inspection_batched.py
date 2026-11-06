"""Candidate admission and normal forms agree with independent slot evaluation."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import ast
from dataclasses import dataclass
import inspect
from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy

assert jax.config.jax_enable_x64
pytestmark = pytest.mark.timeout(1800)


def candidate_program():
    """Expose the actual read's intermediates without adding a product test seam."""
    module = ast.parse(inspect.getsource(topology._read_at_nulls))
    function = module.body[0]
    end = next(
        (
            i
            for i, node in enumerate(function.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "saddle_index"
                for target in node.targets
            )
        ),
        None,
    )
    if end is None:
        raise ValueError("_read_at_nulls: expected top-level saddle_index assignment")
    function.body = function.body[: end + 1] + [
        ast.parse("return candidates_admitted, forms, saddle_index").body[0]
    ]
    namespace = dict(vars(topology))
    exec(
        compile(ast.fix_missing_locations(module), topology.__file__, "exec"), namespace
    )
    return namespace["_read_at_nulls"]


def slotwise(
    field,
    geometry,
    convention,
    policy,
    coefficient,
    nulls,
    live,
    required,
    census_resolved,
    tree,
):
    """Evaluate each slot alone, including the no-admission fallback form."""
    sigma = convention.sigma
    eigenvalues = jnp.linalg.eigvalsh(sigma * nulls.jet.hessian)
    axes = live & jnp.all(eigenvalues < 0, axis=1)
    saddles = live & (eigenvalues[:, 0] < 0) & (eigenvalues[:, 1] > 0)
    axis = nulls.position[jnp.argmax(axes)]

    def one(index):
        point = nulls.position[index]
        third, fourth = topology._curvature_derivatives(
            field, point, jnp.median(geometry.pitch)
        )
        form = topology.saddle_normal_form(
            point,
            sigma * nulls.jet.hessian[index],
            sigma * third,
            policy.hessian_tolerance,
            sigma * fourth,
        )
        form = jax.tree.map(
            lambda leaf: jnp.where(
                saddles[index] | (index == 0), leaf, jnp.zeros_like(leaf)
            ),
            form,
        )
        _, _, labels, wall, selected, owner, valid, _ = topology._support_at_level(
            geometry,
            coefficient,
            sigma,
            nulls.jet.value[index],
            axis,
            form,
            True,
            policy,
        )
        owner_index = jnp.argmax(owner)
        private_index = jnp.where(selected[owner_index, 0], 1, 0)
        private_label = labels[owner_index, private_index]
        admitted = (
            saddles[index]
            & valid
            & jnp.any(owner[:, None] & selected)
            & jnp.any(wall & (labels == private_label))
        )
        return admitted, form

    admitted, forms = jax.lax.map(one, jnp.arange(policy.null_capacity))
    selected = jnp.argmax(jnp.where(admitted, sigma * nulls.jet.value, -jnp.inf))
    return admitted, forms, selected


def stationary_adapter(function):
    """Use the production padding and live-point rules for both evaluations."""
    return FunctionType(
        topology.read_at_stationary_points.__code__,
        dict(vars(topology), _read_at_nulls=function),
    )


def compare(field, geometry, convention, policy, solved):
    arguments = (field, geometry, convention, policy, solved)
    actual = jax.jit(stationary_adapter(candidate_program()))(*arguments)
    expected = jax.jit(stationary_adapter(slotwise))(*arguments)
    assert_candidates_equal(actual, expected)
    return actual


def assert_candidates_equal(actual, expected):
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[2], expected[2])
    errors = []
    for got, want in zip(
        jax.tree.leaves(actual[1]), jax.tree.leaves(expected[1]), strict=True
    ):
        got, want = np.asarray(got), np.asarray(want)
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))
        if want.dtype.kind in "biu":
            np.testing.assert_array_equal(got, want)
            continue
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13, equal_nan=True)
        finite = np.isfinite(want)
        if finite.any():
            errors.append(float(np.max(np.abs(got[finite] - want[finite]))))
    print(
        "CANDIDATES",
        np.asarray(actual[0]).tolist(),
        "SELECTED",
        int(actual[2]),
        "FORM_MAX_ABSOLUTE",
        max(errors),
        flush=True,
    )


def test_candidate_program_names_missing_statement(monkeypatch):
    source = inspect.getsource(topology._read_at_nulls)
    monkeypatch.setattr(
        inspect, "getsource", lambda _: source.replace("saddle_index", "selected_index")
    )
    with pytest.raises(ValueError, match="expected top-level saddle_index assignment"):
        candidate_program()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SampledField:
    analytic: object
    coefficient: object

    def evaluate(self, point):
        return self.analytic.evaluate(point)

    def sampled_coefficient(self, geometry):
        return self.coefficient


def small_case():
    from tests.equilibrium.test_topology_read import (
        _analytic_inputs,
        _realised_hex_geometry,
    )

    _, field, wall, axis, saddle = _analytic_inputs("diverted")
    geometry = _realised_hex_geometry(wall, 30)
    points = jnp.asarray([axis, saddle])
    solved = topology.StationaryRead(
        points,
        jax.vmap(field.evaluate)(points),
        jnp.ones(2, bool),
        jnp.ones(2, bool),
        jnp.zeros(2, jnp.int32),
    )
    field = SampledField(field, topology._sampled_flux(field, geometry)[0])
    return (
        field,
        geometry,
        topology.TopologyConvention.from_cocos(17, 1.0),
        TopologyPolicy(),
        solved,
    )


def test_candidate_slots_match_individual_evaluation():
    actual = compare(*small_case())
    # A real admitted saddle is the positive control for the slot permutation.
    assert np.any(np.asarray(actual[0])[1:])
    assert int(actual[2]) == 1


def nested_equations(program):
    for equation in program.eqns:
        yield equation
        for value in equation.params.values():
            for child in value if isinstance(value, (tuple, list)) else (value,):
                child = getattr(child, "jaxpr", child)
                if hasattr(child, "eqns"):
                    yield from nested_equations(child)


def test_candidate_slots_share_one_program(monkeypatch):
    monkeypatch.setattr(
        topology,
        "_support_at_level",
        jax.named_call(topology._support_at_level, name="candidate_support"),
    )
    args = small_case()
    program = jax.make_jaxpr(stationary_adapter(candidate_program()))(*args).jaxpr
    scans = [
        equation
        for equation in program.eqns
        if equation.primitive.name == "scan"
        and equation.params["length"] == args[3].null_capacity
    ]
    assert len(scans) == 0, "candidate slots are evaluated by a sequential scan"
    supports = [
        equation
        for equation in nested_equations(program)
        if equation.primitive.name == "while"
        and "candidate_support" in str(equation.source_info.name_stack)
    ]
    # The connectivity flood is one while per support evaluation: one per branch.
    assert len(supports) == 2, f"expected two support programs, got {len(supports)}"


def test_candidate_overflow_matches_full_evaluation():
    field, geometry, convention, policy, solved = small_case()
    # Four live saddle slots exceed the compact width even without slot 0.
    solved = jax.tree.map(lambda leaf: leaf[jnp.array([0, 1, 1, 1, 1])], solved)
    sparse = solved._replace(valid=jnp.array([True, False, False, False, True]))
    batch = jax.tree.map(lambda left, right: jnp.stack((left, right)), sparse, solved)

    def mapped(function):
        return jax.vmap(
            lambda lane: stationary_adapter(function)(
                field, geometry, convention, policy, lane
            )
        )

    actual = jax.jit(mapped(candidate_program()))(batch)
    expected = jax.jit(mapped(slotwise))(batch)
    for lane in range(2):
        assert_candidates_equal(
            jax.tree.map(lambda leaf: leaf[lane], actual),
            jax.tree.map(lambda leaf: leaf[lane], expected),
        )
    assert bool(actual[0][0, 4]) and int(actual[2][0]) == 4
    assert np.all(np.asarray(actual[0][1, 1:5]))
    program = jax.make_jaxpr(mapped(candidate_program()))(batch).jaxpr
    # A lane-varying cond becomes selects under vmap; the overflow cond must survive.
    choices = [eq for eq in program.eqns if eq.primitive.name == "cond"]
    assert len(choices) == 1
    assert choices[0].invars[0].aval.shape == ()


@pytest.fixture(scope="module", params=["limited", "diverted"])
def physical_case(request):
    from tests.test_tracked_read_accuracy import _built

    case, bank, cache = _built(request.param)
    assert len(case["geometry"].full_area) == 550
    return case, bank, cache


@pytest.mark.slow
@pytest.mark.parametrize("state", ["trial", "converged"])
def test_certificate_candidates_match_slotwise(physical_case, state):
    case, bank, cache = physical_case
    moments = case["states"][state]
    roots = jnp.asarray(case["roots"])
    stencil = support_read.stationary_stencil(
        bank, cache, moments, roots, roots, case["convention"], iterations=24
    )
    field = fixed_points.field_from_jets(
        support_read.nodal_jets(bank, moments), bank.sample_index, bank.mesh
    )
    print("CERTIFICATE", case["kind"], state, flush=True)
    compare(field, case["geometry"], case["convention"], case["policy"], stencil.solved)
    eigenvalues = np.linalg.eigvalsh(
        case["convention"].sigma * stencil.solved.jet.hessian
    )
    saddles = (
        np.asarray(stencil.solved.valid)
        & (eigenvalues[:, 0] < 0)
        & (eigenvalues[:, 1] > 0)
    )
    print("SADDLE_COUNT", int(np.sum(saddles)), flush=True)
