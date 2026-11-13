"""Residual exclusions must not freeze the physical flux extension."""

import ast
import inspect
import textwrap
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_point
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium import topology as topology_module
from nova.equilibrium.topology import Topology, TopologyClass
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64


def operator():
    """A coupled affine source makes held and imaged shadow roots distinct."""
    result = object.__new__(ForwardFluxOperator)
    result.use_linear_moments = True
    result.moment_geometry = None
    result.external = lambda *_: jnp.array([1.0, 2.0, 3.0])
    result.internal = lambda state, *_: jnp.array(
        [0.2 * state[0] + 0.1 * state[1], 0.3 * state[0], 0.1 * state[2]]
    )
    result.residual_shadow_mask = lambda *_: jnp.array([False, True, False])
    return result


def image(op, state):
    return op.external() + op.internal(state)


@pytest.mark.parametrize("solver", ["warmup", "newton"])
def test_shadow_root_follows_physical_image_instead_of_seed(solver):
    op = operator()
    seed = jnp.array([0.0, -11.0, 0.0])
    mask = op.residual_shadow_mask(seed)
    mapped = op.flux_map_with_shadow()
    arguments = dict(
        shadow_mask_fn=op.residual_shadow_mask,
        promoted_shadow_mask_fn=lambda *_: mask,
        shadowed_map_fn=mapped,
    )

    def solve(state):
        if solver == "warmup":
            return fixed_point.picard(
                lambda value: mapped(value, mask),
                state,
                evaluations=100,
                relaxation=0.5,
                **arguments,
            )
        return fixed_point.newton_krylov(
            lambda value: mapped(value, mask),
            state,
            newton_steps=8,
            warmup=0,
            gmres_iterations=3,
            convergence_tolerance=1e-11,
            **arguments,
        )

    result = jax.jit(solve)(seed)
    np.testing.assert_allclose(result.state, image(op, result.state), atol=1e-9, rtol=0)
    assert abs(float(result.state[1] - seed[1])) > 10
    np.testing.assert_array_equal(op.residual_shadow_mask(result.state), mask)


def test_warmup_assigns_the_shadow_image_without_relaxing_it():
    op = operator()
    seed = jnp.array([0.0, -11.0, 0.0])
    mask = op.residual_shadow_mask(seed)
    mapped = op.flux_map_with_shadow()
    result = fixed_point._newton_krylov_inner(
        lambda value: mapped(value, mask),
        seed,
        newton_steps=0,
        warmup=1,
        relaxation=0.25,
        shadow_mask_fn=op.residual_shadow_mask,
        promoted_shadow_mask_fn=lambda *_: mask,
        shadowed_map_fn=mapped,
    )
    np.testing.assert_allclose(result.trajectory_state[1], image(op, seed)[1])


def test_residual_still_excludes_shadows_and_preserves_other_entries():
    op = operator()
    state = jnp.array([0.0, -11.0, 0.0])
    expected = jnp.where(op.residual_shadow_mask(state), 0.0, state - image(op, state))
    for residual in (op.residual, jax.jit(op.residual)):
        np.testing.assert_array_equal(residual(state), expected)
    _, tangent = jax.jvp(op.residual, (state,), (jnp.array([0.0, 1.0, 0.0]),))
    assert float(tangent[1]) == 0
    assert float(tangent[0]) == -0.1


def test_frozen_partition_map_and_dynamic_exterior_share_one_compilation():
    op = operator()
    op.use_linear_moments = False
    op._internal_on_partition = lambda state, *_: op.internal(state)
    traced = op.traced_flux_map_with_shadow()
    state = jnp.array([0.0, -11.0, 0.0])
    mask = op.residual_shadow_mask(state)
    compiled = jax.jit(lambda state, mask, external: traced(state, mask, external))
    for seed, shadow, external in (
        (state, mask, op.external()),
        (state + 1, ~mask, op.external() + 2),
    ):
        np.testing.assert_allclose(
            compiled(seed, shadow, external), external + op.internal(seed)
        )
    assert compiled._cache_size() == 1
    partition = SimpleNamespace(residual_shadow=mask)
    np.testing.assert_allclose(
        traced._map_frozen_partition(state, partition, op.external()), image(op, state)
    )


@pytest.fixture
def wall_contact_read(monkeypatch):
    """Isolate a moving wall anchor while exercising the complete topology read.

    The local field gradient and anchor velocity are measured operands from a
    limited equilibrium. Null selection is held fixed so the wall chain term
    is the only changing geometric contribution.
    """
    radius = jnp.linspace(2.1, 2.6, 5)
    height = jnp.linspace(-0.2, 0.3, 5)
    rr, zz = jnp.meshgrid(radius, height, indexing="ij")
    point = jnp.array([2.3485665682296073, 0.061132787458339645])
    velocity = jnp.array([-0.014427078741060997, 0.10847979989734477])
    gradient = jnp.array([-5.579951948407475, -0.6796977367107974])
    direct = 1.0635041339809386
    field = -1.301153965748883 + gradient[0] * (rr - point[0])
    field = field + gradient[1] * (zz - point[1])
    axis = jnp.array([2.2, 0.0, 0.6, 1.0])
    saddle = jnp.array([2.3, -0.1, -1.5, 0.0])

    class NullGrid:
        node_number = 25
        direct_sample_count = 0

        def __call__(self, _flux):
            return axis[None, :], saddle[None, :]

    class WallRead:
        connectivity_radius = radius
        connectivity_height = height
        connectivity_coordinate = jnp.c_[rr.ravel(), zz.ravel()]
        grid = NullGrid()
        wall = SimpleNamespace(coordinate=point[None, :], node_number=1)
        wall_unit_offsets = jnp.array([0, 1])

        def split_flux_map(self, state):
            return state[:25], state[25:]

        def wall_anchor_data(self, wall_flux, *_args):
            position = point + velocity * wall_flux[0]
            return jnp.r_[position, -1.3, 0.0]

        def qualified_o_candidates(self, *_args):
            return jnp.array([True])

        def o_point_qualification(self, *_args):
            return SimpleNamespace(data=axis, admitted=jnp.array(True))

        def x_point_data(self, *_args):
            return saddle

        def boundary(self, _axis, _saddles, wall, _polarity):
            return wall

        def pinned_boundary(self, _saddle, wall, _requested):
            return wall

        def boundary_interpolation_uncertainty(self, *_args):
            return jnp.array(0.0)

        def normalize(self, axis_flux, boundary_flux, values):
            return (values - axis_flux) / (boundary_flux - axis_flux)

        def psi_mask(self, polarity, values, boundary_flux, _uncertainty):
            return polarity * (values - boundary_flux) >= 0

        def axis_component(
            self, _flux, _boundary, _axis_flux, _axis, closed, *_args, **_kw
        ):
            return closed

    def fixed_nulls(
        _flux, _radius, _height, _boundary, _polarity, data_o, data_x, **_kw
    ):
        data = jnp.stack((data_o, data_x))
        return (
            data_o,
            data_x,
            dict(selected_position_rz=data[:, :2], selected_value=data[:, 2]),
        )

    monkeypatch.setattr(topology_module, "polish_census_stationary_points", fixed_nulls)
    reader = WallRead()
    inside = jnp.ones(25, dtype=bool)

    def state(parameter):
        return jnp.r_[field.ravel() + direct * parameter, parameter]

    def read(function, parameter):
        return function(reader, state(parameter), 1.0, inside, TopologyClass.LIMITED)

    return read, gradient @ velocity, direct


def stopped_wall_contact_read():
    """Retain the stopped-position forward reference without a second kernel."""
    body = Topology.read_qualification.__wrapped__
    tree = ast.parse(textwrap.dedent(inspect.getsource(body)))
    tree.body[0].decorator_list = []
    selected = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "boundary_position"
            for target in node.targets
        )
    ]
    assert len(selected) == 1
    selected[0].value = ast.parse("jax.lax.stop_gradient(data_b[:2])", mode="eval").body
    namespace = dict(body.__globals__)
    exec(
        compile(ast.fix_missing_locations(tree), "<stopped-wall-contact>", "exec"),
        namespace,
    )
    return namespace[body.__name__]


def test_wall_contact_forward_values_are_bit_identical(wall_contact_read):
    read, _chain, _direct = wall_contact_read
    reference = stopped_wall_contact_read()
    for parameter in (-0.02, 0.0, 0.02):
        actual = read(Topology.read_qualification.__wrapped__, parameter)
        stopped = read(reference, parameter)
        for value, expected in zip(
            jax.tree.leaves(actual), jax.tree.leaves(stopped), strict=True
        ):
            assert np.asarray(value).dtype == np.asarray(expected).dtype
            assert np.asarray(value).tobytes() == np.asarray(expected).tobytes()


def test_wall_contact_tangent_includes_the_measured_position_chain(wall_contact_read):
    read, chain, direct = wall_contact_read

    def boundary(parameter):
        return read(
            Topology.read_qualification.__wrapped__, parameter
        ).state.boundary_flux

    tangent = jax.jvp(boundary, (jnp.array(0.0),), (jnp.array(1.0),))[1]
    assert float(chain) == pytest.approx(0.006768931661945934, abs=1e-12)
    assert float(tangent - direct) == pytest.approx(float(chain), abs=1e-10)
    for epsilon in (1e-5, 1e-6, 1e-7):
        finite_difference = (boundary(epsilon) - boundary(-epsilon)) / (2 * epsilon)
        assert float(tangent) == pytest.approx(float(finite_difference), abs=1e-8)
