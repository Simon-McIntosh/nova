"""Traced-array ownership at the forward-operator program boundary."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium import fixed_point
from nova.equilibrium.forward import _lattice_cells
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


def _operator() -> ForwardFluxOperator:
    """Build a small operator carrying every mesh-array family."""
    lattice = FluxLattice(np.linspace(0.8, 1.2, 7), np.linspace(-0.2, 0.2, 7))
    stencil = hex_stencil(lattice.shape)
    angle = 2.0 * np.pi * np.arange(16) / 16
    wall_coordinate = np.c_[1.0 + 0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    cell_count = lattice.node_count
    wall_count = len(wall_coordinate)

    def zero(normalized_flux):
        return jnp.zeros_like(normalized_flux)

    mesh = StencilMesh(lattice.coordinate, stencil, lattice.cell_area)
    geometry = MomentGeometry.from_cells(mesh, _lattice_cells(lattice))
    return ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.arange(cell_count, dtype=jnp.float64)[:, None],
            plasma_target=jnp.arange(
                cell_count * cell_count, dtype=jnp.float64
            ).reshape((cell_count, cell_count)),
            null=Null2D.from_coordinates(lattice.coordinate, stencil),
        ),
        wall=FluxTarget(
            source_target=jnp.ones((wall_count, 1), dtype=jnp.float64),
            plasma_target=jnp.arange(
                wall_count * cell_count, dtype=jnp.float64
            ).reshape((wall_count, cell_count)),
            null=Null1D(jnp.asarray(wall_coordinate)),
        ),
        source=ForwardSource(core=DomainProfile(p_prime=zero, ff_prime=zero)),
        external_current=jnp.ones(1, dtype=jnp.float64),
        area=np.asarray(lattice.cell_area),
        moment_geometry=geometry,
        use_linear_moments=False,
    )


def test_every_mesh_array_family_is_a_pytree_leaf() -> None:
    """Kernel, wall, topology and moment arrays cross the JIT call boundary."""
    configure_dtypes()
    operator = _operator()
    leaves = jax.tree_util.tree_leaves(operator)
    identities = {id(value) for value in leaves}

    assert id(operator.grid.plasma_target) in identities
    assert id(operator.wall.plasma_target) in identities
    assert id(operator.area) in identities
    assert id(operator.moment_geometry.second_moment) in identities
    assert id(operator.moment_geometry.atomic_mesh.node_coordinates) in identities
    assert id(operator.topology.connectivity_coordinate) in identities


def test_jitted_coupling_reads_the_matrix_from_an_argument() -> None:
    """The known interaction block is an operand, never a closed-over literal."""
    configure_dtypes()
    operator = _operator()
    current = jnp.arange(operator.grid.node_number, dtype=jnp.float64)

    def image(active_operator, cell_current):
        return active_operator.grid.internal(cell_current)

    closed = jax.make_jaxpr(image)(operator, current)
    captured_shapes = {
        tuple(np.shape(value)) for value in closed.consts if hasattr(value, "shape")
    }
    assert tuple(operator.grid.plasma_target.shape) not in captured_shapes
    np.testing.assert_array_equal(
        image(operator, current), operator.grid.plasma_target @ current
    )


def test_fixed_point_program_carries_the_operator_as_an_argument() -> None:
    """One map iteration consumes exterior and operator through loop operands."""
    configure_dtypes()
    operator = _operator()
    external = operator.external()
    initial = jnp.linspace(-1.0, 1.0, operator.node_number)
    mapped = operator.traced_flux_map()

    @jax.jit
    def solve(state, exterior, active_operator):
        return fixed_point.picard(
            mapped,
            state,
            evaluations=1,
            map_arguments=(exterior, active_operator),
        )

    result = solve(initial, external, operator)
    image = mapped(initial, external, operator)
    expected = initial + 0.5 * (image - initial)
    np.testing.assert_array_equal(result.state, expected)


def test_the_bound_frozen_partition_hooks_take_the_map_operand_order() -> None:
    """A bound shadowed map reads its partition through the same operand list.

    The solver binds the exterior flux, the operator and the per-slice current
    target onto the map, so the hooks it lifts from that map receive all three
    after the state. A hook that declares fewer operands refuses the bound call
    before any partition is read.
    """
    configure_dtypes()
    operator = _operator()
    external = operator.external()
    initial = jnp.linspace(-1.0, 1.0, operator.node_number)
    shadowed = operator.traced_flux_map_with_shadow()
    assert shadowed._read_frozen_partition is not None

    bound = fixed_point._bind_traced_map_arguments(
        shadowed, (external, operator, None)
    )
    partition = bound._read_frozen_partition(initial, None)
    assert partition is not None
    mapped = bound._map_frozen_partition(initial, partition)
    assert mapped.shape == initial.shape
