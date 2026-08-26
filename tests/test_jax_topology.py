"""Batched-evaluation contract for the jax plasma-topology stack."""

import typing

import numpy as np
import pytest

from nova.geometry.hexstencil import hex_stencil
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.biot.null import Null1D, Null2D
    from nova.biot.target import FluxTarget
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.domain import PlasmaDomain, classify_domains
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.topology import Topology, TopologyState
    from nova.geometry import select
    from nova.jax.config import Precision, configure_dtypes


def _structured_grid(nx, nz, xlim=(0.5, 1.5), zlim=(-0.6, 0.6)):
    """Return flat coordinates, hexagonal stencil and stencil geometry."""
    x = np.linspace(*xlim, nx)
    z = np.linspace(*zlim, nz)
    x2d, z2d = np.meshgrid(x, z, indexing="ij")
    coordinate = np.c_[x2d.ravel(), z2d.ravel()]
    stencil = hex_stencil((nx, nz))
    return coordinate, stencil, coordinate[stencil]


def _flux_field(coordinate, xo=1.0, zo=0.0, xs=1.0, zs=-0.4, amp=1.0):
    """Return an analytic flux map with an o-point (max) and a saddle."""
    x, z = coordinate[:, 0], coordinate[:, 1]
    return -amp * ((x - xo) ** 2 + (z - zo) ** 2) + 0.3 * (
        (x - xs) ** 2 - (z - zs) ** 2
    )


@pytest.fixture(scope="module")
def topology():
    """Return fixed-shape topology on a synthetic structured grid and wall loop."""
    configure_dtypes()
    coordinate, stencil, coordinate_stencil = _structured_grid(40, 40)
    del coordinate_stencil
    grid = Null2D.from_coordinates(coordinate, stencil, maxsize=5)
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    wall_xy = np.c_[1.0 + 0.45 * np.cos(theta), 0.45 * np.sin(theta)]
    return Topology(grid, Null1D(jnp.asarray(wall_xy))), coordinate, wall_xy


def _flux_stack(coordinate, wall_xy, scales):
    """Return a (batch, node) flux map stack scaled per slice."""
    psi_grid = _flux_field(coordinate)
    psi_wall = _flux_field(wall_xy)
    psi = np.r_[psi_grid, psi_wall]
    return jnp.asarray(np.stack([psi * s for s in scales]))


def test_update_resolves_finite_topology(topology):
    """A single update returns a finite normalised map and a plasma region."""
    topo, coordinate, wall_xy = topology
    psi = _flux_stack(coordinate, wall_xy, [1.0])[0]
    psi_norm, ionize = topo.update(psi, 1)
    assert np.all(np.isfinite(np.asarray(psi_norm)))
    assert int(np.sum(np.asarray(ionize))) > 0


def test_physical_nulls_keep_double_outputs_with_single_selection_scores(topology):
    """Physical null data stay fp64 while topology ranking follows fit precision.

    The reconstruction carries the physical origin and scale in fp64 whatever
    the fit is normalised in, so narrowing the fit narrows the selection score
    and the flux ladder without narrowing the coordinates that come back. The
    narrow locator is built here rather than taken from the fixture because
    the unqualified fit is fp64.
    """
    topo, coordinate, wall_xy = topology
    narrow = Topology(
        Null2D.from_coordinates(
            np.asarray(topo.grid.coordinate),
            np.asarray(topo.grid.stencil),
            maxsize=topo.grid.maxsize,
            precision=Precision.SINGLE,
        ),
        topo.wall,
    )
    psi = _flux_stack(coordinate, wall_xy, [1.0])[0]
    psi_grid, psi_wall = narrow.split_flux_map(psi)
    vmap_o, vmap_x = narrow.grid(psi_grid)
    data_o = narrow.o_point_data(vmap_o, 1)
    data_w = narrow.wall(psi_wall, 1)
    data_b = narrow.boundary(data_o, vmap_x, data_w, 1)

    assert topo.grid.fit_dtype == jnp.float64
    assert narrow.grid.fit_dtype == jnp.float32
    assert vmap_o.dtype == jnp.float64
    assert vmap_x.dtype == jnp.float64
    assert data_b.dtype == jnp.float64
    assert np.all(np.isfinite(np.asarray(data_b)))


def test_batched_update_matches_per_slice(topology):
    """update_batch over a leading axis equals per-slice update calls."""
    topo, coordinate, wall_xy = topology
    scales = [1.0, 1.1, 0.9, 1.2, 0.75]
    psi_batch = _flux_stack(coordinate, wall_xy, scales)
    polarity = 1

    batch_norm, batch_ionize = topo.update_batch(psi_batch, polarity)
    assert batch_norm.shape[0] == len(scales)

    for i in range(len(scales)):
        norm_i, ionize_i = topo.update(psi_batch[i], polarity)
        assert np.allclose(
            np.asarray(batch_norm[i]), np.asarray(norm_i), equal_nan=True
        )
        assert np.array_equal(np.asarray(batch_ionize[i]), np.asarray(ionize_i))


def test_batched_primary_points_match_per_slice(diverted):
    """Batched primary X coordinates and flux equal per-slice reads.

    The shared grid carries two physical saddles.  Two additional slices move
    only the wall operand to either side of an almost-tangent wall/X flux tie,
    so the parity assertion covers both a double-null and the class hand-off.
    """
    topo, psi, inside = diverted
    polarity = 1
    psi_grid, psi_wall = topo.split_flux_map(psi)
    vmap_o, vmap_x = topo.grid(psi_grid)
    data_o = topo.o_point_data(vmap_o, polarity)
    data_x = topo.x_point_data(vmap_x, polarity, data_o[2])
    data_w = topo.wall(psi_wall, polarity)
    finite_x_count = int(np.sum(np.isfinite(np.asarray(vmap_x)[:, 0])))
    assert finite_x_count == 2

    flux_span = abs(float(data_x[2] - data_o[2]))
    tie_offset = float(data_x[2] - data_w[2])
    tie_epsilon = 1.0e-10 * flux_span
    wall_offsets = jnp.asarray(
        [0.0, tie_offset - tie_epsilon, tie_offset + tie_epsilon]
    )
    psi_batch = jnp.concatenate(
        (
            jnp.broadcast_to(psi_grid, (wall_offsets.size, psi_grid.size)),
            psi_wall[None, :] + wall_offsets[:, None],
        ),
        axis=1,
    )

    _batch_masks, batch_state = topo.read_batch(psi_batch, polarity, inside)
    np.testing.assert_allclose(
        np.abs(
            np.asarray(batch_state.wall_point_flux[1:] - batch_state.x_point_flux[1:])
        ),
        tie_epsilon,
        rtol=1.0e-6,
        atol=8.0 * np.finfo(float).eps * max(flux_span, 1.0),
    )
    for index in range(psi_batch.shape[0]):
        _slice_masks, slice_state = topo.read(psi_batch[index], polarity, inside)
        np.testing.assert_allclose(
            np.asarray(batch_state.x_point[index]),
            np.asarray(slice_state.x_point),
            rtol=0.0,
            atol=8.0 * np.finfo(float).eps,
        )
        assert float(batch_state.x_point_flux[index]) == float(slice_state.x_point_flux)


def test_flux_target_preserves_matrix_and_tree_contracts():
    """The domain target evaluates both matrices and reconstructs as a pytree."""
    null = Null1D(jnp.asarray([[1.0, -0.2], [1.1, 0.0], [1.0, 0.2]]))
    source_target = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    plasma_target = jnp.asarray([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]])
    target = FluxTarget(source_target, plasma_target, null)

    np.testing.assert_allclose(target.external(jnp.asarray([2.0, -1.0])), [0, 2, 4])
    np.testing.assert_allclose(
        target.internal(jnp.asarray([1.0, 2.0])), [3.5, 9.5, 15.5]
    )
    leaves, structure = jax.tree_util.tree_flatten(target)
    restored = jax.tree_util.tree_unflatten(structure, leaves)

    assert restored.node_number == 3
    np.testing.assert_array_equal(restored.coordinate, null.coordinate)
    np.testing.assert_array_equal(restored.source_target, source_target)
    np.testing.assert_array_equal(restored.plasma_target, plasma_target)


def test_forward_operator_uses_domain_flux_targets():
    """The differentiable consumer names its grid and wall capability."""
    hints = typing.get_type_hints(ForwardFluxOperator)
    assert hints["grid"] is FluxTarget
    assert hints["wall"] is FluxTarget


#: A main plasma ring with a weaker divertor ring above and below it. The two
#: outer rings each raise a saddle between themselves and the plasma, so the
#: flux map carries two X-points of opposite sense about the axis and a
#: private-flux pocket under each — the arrangement that exercises the
#: saddle-aware axis component with more than one finite X-point.
DOUBLE_NULL_RING = np.array([[1.0, 0.0], [1.0, -0.62], [1.0, 0.62]])
DOUBLE_NULL_CURRENT = np.array([1.0e6, 5.0e5, 4.0e5])


def _ring_flux(coordinate):
    """Return the poloidal flux the divertor ring set drives on a point set."""
    columns = np.stack(
        [
            hybrid_greens(
                coordinate[:, 0], coordinate[:, 1], radius, height, 0.06, 0.06
            )[0]
            for radius, height in DOUBLE_NULL_RING
        ],
        axis=1,
    )
    return columns @ DOUBLE_NULL_CURRENT


@pytest.fixture(scope="module")
def diverted():
    """Return a double-null topology, its flux vector and its material mask."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.55, 1.45, 45), np.linspace(-0.75, 0.75, 71))
    coordinate = lattice.coordinate
    angle = 2 * np.pi * np.arange(48) / 48
    wall = np.c_[1.0 + 0.42 * np.cos(angle), 0.62 * np.sin(angle)]
    topology = Topology(
        Null2D.from_coordinates(coordinate, hex_stencil(lattice.shape), maxsize=5),
        Null1D(jnp.asarray(wall)),
    )
    inside = jnp.asarray(
        ((coordinate[:, 0] - 1.0) / 0.42) ** 2 + (coordinate[:, 1] / 0.62) ** 2 <= 1.0
    )
    psi = jnp.asarray(np.r_[_ring_flux(coordinate), _ring_flux(wall)])
    return topology, psi, inside


def _traversed_categorize(null, psi_stencil):
    """Return the null clusters, counting and selecting by explicit traversal.

    The crossing counter is carried along each ring's vertices and the counted
    rings are walked in turn, which is the arrangement the kernel formulation
    is required to reproduce value for value rather than merely agree with.
    """
    psi_stencil = jnp.asarray(psi_stencil, dtype=null.fit_dtype)

    def zero_cross(carry, value):
        """Increment the crossing counter at one vertex of one ring."""
        count, sign, centre = carry
        step = value > centre
        change = step != sign
        count += change
        return (count, jnp.where(change, step, sign), centre), None

    def count_ring(number, patch):
        """Return the running null counts and this ring's crossing count."""
        extremum, saddle = number
        centre = patch[0]
        sign = patch[-1] > centre
        count = jax.lax.scan(zero_cross, (0, sign, centre), patch[1:])[0][0]
        return (extremum + (count == 0), saddle + (count == 4)), count

    number, count = jax.lax.scan(count_ring, (0, 0), psi_stencil)

    def select_type(_, null_type):
        """Return the padded cluster selection of one null type."""
        index = jnp.where(count == null_type, size=null.maxsize)[0]
        return _, (
            jnp.concatenate(
                (
                    null.local_coordinate_stencil[index],
                    psi_stencil[index, :, jnp.newaxis],
                ),
                axis=-1,
            ),
            null.physical_origin[index],
            null.physical_scale[index],
        )

    cluster, origin, scale = jax.lax.scan(select_type, (), jnp.array([0, 4]))[1]
    return jnp.array(number), cluster, origin, scale


def _traversed_interpolate(number, cluster, origin, scale):
    """Return one null type's sub-cell fits, masked by a carried position."""

    def subnull(carry, values):
        """Fit one cluster and drop it when the padding has been reached."""
        one, physical_origin, physical_scale = values
        carry += 1
        local = select.traced_subnull(one[:, 0], one[:, 1], one[:, 2])
        physical = physical_origin + local[:2].astype(jnp.float64) * physical_scale
        return carry, jnp.where(
            carry <= number,
            jnp.concatenate((physical, local[2:].astype(jnp.float64))),
            jnp.full(4, jnp.nan, dtype=jnp.float64),
        )

    return jax.lax.scan(subnull, 0, (cluster, origin, scale))[1]


def _traversed_nulls(null, psi_grid):
    """Return the extremum and saddle tables by the traversed formulation."""
    psi_stencil = jnp.asarray(psi_grid, dtype=null.fit_dtype)[null.stencil]
    number, cluster, origin, scale = _traversed_categorize(null, psi_stencil)
    return jax.vmap(_traversed_interpolate)(number, cluster, origin, scale)


def _vertical_cut_connectivity(topology, data_o, vmap_x):
    """Return vertical X-point cuts for regression contrast."""

    def narrow(mask, data_x):
        """Apply one X-point's half-plane cut to the surviving cells."""
        return jax.lax.select(
            mask & jnp.isfinite(data_x[0]),
            jax.lax.cond(
                data_x[1] < data_o[1],
                jnp.greater,
                jnp.less,
                topology.grid.coordinate[:, 1],
                data_x[1],
            ),
            mask,
        ), None

    mask = jnp.ones(topology.grid.node_number, dtype=bool)
    return jax.lax.scan(narrow, mask, vmap_x)[0]


def _traversed_read(topology, psi, polarity, inside_material):
    """Return the saddle-aware partition and state with traversed null fits."""
    psi_grid, psi_wall = topology.split_flux_map(psi)
    vmap_o, vmap_x = _traversed_nulls(topology.grid, psi_grid)
    data_o = topology.o_point_data(vmap_o, polarity)
    data_x = topology.x_point_data(vmap_x, polarity, data_o[2])
    data_w = topology.wall(psi_wall, polarity)
    data_b = topology.boundary(data_o, vmap_x, data_w, polarity)
    psi_norm = topology.normalize(data_o[2], data_b[2], psi_grid)
    closed = topology.psi_mask(polarity, psi_grid, data_b[2])
    connected = topology.axis_component(
        psi_grid,
        data_b[2],
        data_o[2],
        data_o[:2],
        closed,
        inside_material,
    )
    masks = classify_domains(
        psi_norm,
        closed,
        connected,
        inside_material,
    )
    half_plane_masks = classify_domains(
        psi_norm,
        closed,
        _vertical_cut_connectivity(topology, data_o, vmap_x),
        inside_material,
    )
    state = TopologyState(
        axis=data_o[:2],
        axis_flux=data_o[2],
        boundary=data_b[:2],
        boundary_flux=data_b[2],
        x_point=data_x[:2],
        x_point_flux=data_x[2],
        wall_point=data_w[:2],
        wall_point_flux=data_w[2],
        diverted=jnp.equal(data_b[2], data_x[2]),
    )
    return masks, state, half_plane_masks


def test_the_double_null_fixture_exercises_every_branch_of_the_read(diverted):
    """The fixture carries padded null tables, two X-points and all four labels.

    An identity read is only worth as much as the branches it drives, so the
    counts that make this map a demanding one are pinned rather than assumed:
    both null tables are selected below their capacity, the connectivity cut
    has more than one finite X-point to conjoin, and the label partition is
    total over four populated domains.
    """
    topology, psi, inside = diverted
    vmap_o, vmap_x = topology.grid(topology.split_flux_map(psi)[0])
    masks, state = topology.read(psi, 1, inside)
    label = np.asarray(masks.label)

    assert int(np.sum(np.isfinite(np.asarray(vmap_o)[:, 0]))) == 3
    assert int(np.sum(np.isfinite(np.asarray(vmap_x)[:, 0]))) == 2
    assert topology.grid.maxsize == 5
    assert bool(state.diverted)
    assert sorted(np.unique(label)) == [0, 1, 2, 3]
    assert int(np.sum(label == 3)) > 0


#: Quantities the read publishes that select or normalise rather than fit, and
#: that no rearrangement of the traversals may move by so much as a last bit.
DECIDING_STATE = (
    "axis_flux",
    "boundary_flux",
    "x_point_flux",
    "wall_point_flux",
    "diverted",
)
#: The fitted positions. These are the output of a least-squares solve, which
#: is the one place the backend rather than the formulation decides the last
#: bit, so they are read against the floor measured below.
FITTED_STATE = ("axis", "boundary", "x_point", "wall_point")


def _fit_reproducibility_floor(null, psi_grid):
    """Return how far the sub-cell fit moves when only its batching changes.

    The same least-squares fit is driven over the same clusters three ways —
    both null types mapped together, one type at a time, and one cluster at a
    time with no batch axis anywhere — so anything separating them is the
    backend's scheduling of identical arithmetic rather than a difference in
    what is computed. The unbatched arm matters most: mapping a fit and
    serialising it is exactly the rearrangement the assertions below are
    reading. A backend that schedules a batch the same way at every width
    returns exactly zero, and those assertions then read as bit-identity.
    """
    psi_stencil = jnp.asarray(psi_grid, dtype=null.fit_dtype)[null.stencil]
    number, cluster, origin, scale = null.categorize(psi_stencil)

    @jax.jit
    def one(single, single_origin, single_scale):
        """Fit exactly one cluster, with no batch axis anywhere."""
        local = select.traced_subnull(single[:, 0], single[:, 1], single[:, 2])
        physical = single_origin + local[:2].astype(jnp.float64) * single_scale
        return jnp.concatenate((physical, local[2:].astype(jnp.float64)))

    kinds = range(number.shape[0])
    mapped = np.asarray(
        jax.vmap(null.interpolate, (0, 0, 0, 0))(number, cluster, origin, scale)
    )
    arrangement = [
        np.stack(
            [
                np.asarray(
                    null.interpolate(
                        number[kind], cluster[kind], origin[kind], scale[kind]
                    )
                )
                for kind in kinds
            ]
        ),
        np.stack(
            [
                np.stack(
                    [
                        np.asarray(
                            one(cluster[kind, i], origin[kind, i], scale[kind, i])
                        )
                        for i in range(cluster.shape[1])
                    ]
                )
                for kind in kinds
            ]
        ),
    ]
    floor = 0.0
    for other in arrangement:
        finite = np.isfinite(mapped) & np.isfinite(other)
        floor = max(floor, float(np.max(np.where(finite, np.abs(mapped - other), 0.0))))
    return floor


def test_the_topology_read_matches_a_traversed_formulation(diverted):
    """Every quantity the read publishes matches the traversal it stands for.

    The crossing count and sub-cell fit selection are evaluated over the whole
    grid at once rather than walked, and both rearrangements are
    value-preserving by construction: a ring's crossing count depends on no
    other ring, and a cluster's position in the padded selection is known
    without carrying it. The reference partition independently applies the
    same saddle-aware axis-component authority as the production read. Labels,
    normalised flux and every published flux are therefore required to be
    bit-identical, with no tolerance anywhere.

    The vertical half-plane contrast misclassifies 51 cells as private flux
    that the saddle-aware hex connectivity identifies as part of the core.

    The fitted positions are the exception, and not because the claim is weaker
    there: a least-squares solve is the one step whose last bit the backend
    decides, and on an accelerator the same solve over the same clusters lands
    differently at two batch widths. They are therefore held to the floor that
    self-disagreement measures, which is zero wherever the backend is
    batch-reproducible.
    """
    topology, psi, inside = diverted
    polarity = 1
    floor = _fit_reproducibility_floor(topology.grid, topology.split_flux_map(psi)[0])

    masks, state = topology.read(psi, polarity, inside)
    reference_masks, reference_state, half_plane_masks = _traversed_read(
        topology, psi, polarity, inside
    )

    np.testing.assert_array_equal(
        np.asarray(masks.label), np.asarray(reference_masks.label)
    )
    np.testing.assert_array_equal(
        np.asarray(masks.psi_norm), np.asarray(reference_masks.psi_norm)
    )
    changed = np.asarray(masks.label) != np.asarray(half_plane_masks.label)
    assert masks.label.size == 3195
    assert int(np.sum(changed)) == 51
    assert np.all(np.asarray(masks.label)[changed] == int(PlasmaDomain.CORE))
    assert np.all(
        np.asarray(half_plane_masks.label)[changed] == int(PlasmaDomain.PRIVATE_FLUX)
    )
    for field in DECIDING_STATE:
        np.testing.assert_array_equal(
            np.asarray(getattr(state, field)),
            np.asarray(getattr(reference_state, field)),
            err_msg=field,
        )
    for field in FITTED_STATE:
        deviation = float(
            np.max(
                np.abs(
                    np.asarray(getattr(state, field))
                    - np.asarray(getattr(reference_state, field))
                )
            )
        )
        assert deviation <= floor, (field, deviation, floor)


def test_the_null_tables_match_a_traversed_formulation(diverted):
    """The located nulls themselves match, padding rows included.

    Which rings were selected, and the flux and classification the fit returns
    on them, are bit-identical; the fitted coordinates are held to the same
    backend floor the published positions are.
    """
    topology, psi, _ = diverted
    psi_grid = topology.split_flux_map(psi)[0]
    floor = _fit_reproducibility_floor(topology.grid, psi_grid)

    for located, reference in zip(
        topology.grid(psi_grid), _traversed_nulls(topology.grid, psi_grid)
    ):
        located = np.asarray(located)
        reference = np.asarray(reference)
        np.testing.assert_array_equal(
            np.isfinite(located), np.isfinite(reference), err_msg="selection"
        )
        np.testing.assert_array_equal(
            located[:, 2:], reference[:, 2:], err_msg="flux and classification"
        )
        difference = np.abs(located[:, :2] - reference[:, :2])
        assert np.nanmax(np.where(np.isnan(difference), 0.0, difference)) <= floor


def test_topology_tree_roundtrip_preserves_null_kernels(topology):
    """Topology reconstruction retains the fixed grid and wall capacities."""
    topo = topology[0]
    leaves, structure = jax.tree_util.tree_flatten(topo)
    restored = jax.tree_util.tree_unflatten(structure, leaves)

    assert restored.grid.maxsize == topo.grid.maxsize
    assert restored.grid.node_number == topo.grid.node_number
    assert restored.wall.node_number == topo.wall.node_number
    np.testing.assert_array_equal(restored.grid.coordinate, topo.grid.coordinate)
    np.testing.assert_array_equal(restored.wall.coordinate, topo.wall.coordinate)


if __name__ == "__main__":
    pytest.main([__file__])
