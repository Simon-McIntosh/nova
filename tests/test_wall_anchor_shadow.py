"""Wall-anchor selection against a carried private-wall shadow."""

from types import SimpleNamespace

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.null import Null1D
    from nova.equilibrium.domain import DomainMasks, PlasmaDomain
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.topology import Topology, TopologyClass, TopologyState
    from nova.jax.config import configure_dtypes


def _wall_topology() -> tuple[Topology, np.ndarray, np.ndarray]:
    """Return a wall-only topology and a smooth flux with one extreme node."""
    configure_dtypes()
    angle = 2.0 * np.pi * np.arange(12) / 12
    coordinate = np.column_stack((1.0 + 0.4 * np.cos(angle), 0.6 * np.sin(angle)))
    flux = np.cos(angle - angle[3])
    topology = object.__new__(Topology)
    topology.wall = Null1D(jnp.asarray(coordinate))
    return topology, coordinate, flux


def test_private_wall_node_cannot_win_limited_anchor() -> None:
    """A shadowed wall node carrying the extreme flux cannot bind the plasma."""
    topology, coordinate, flux = _wall_topology()
    private_index = int(np.argmax(flux))
    private_wall = np.arange(flux.size) == private_index

    unmasked = topology.wall_anchor_data(
        jnp.asarray(flux),
        1,
        TopologyClass.LIMITED,
    )
    masked = topology.wall_anchor_data(
        jnp.asarray(flux),
        1,
        TopologyClass.LIMITED,
        jnp.asarray(private_wall),
    )

    unmasked_distance = np.linalg.norm(
        np.asarray(unmasked[:2]) - coordinate[private_index]
    )
    masked_distance = np.linalg.norm(np.asarray(masked[:2]) - coordinate[private_index])
    assert unmasked_distance < 1.0e-12
    assert masked_distance > 0.1
    assert float(masked[2]) < float(unmasked[2])


def test_diverted_anchor_is_bit_identical_with_private_wall_mask() -> None:
    """A diverted read never consults the wall anchor selected for limiting."""
    topology, _coordinate, flux = _wall_topology()
    private_wall = jnp.asarray(np.arange(flux.size) == int(np.argmax(flux)))

    baseline = topology.wall_anchor_data(
        jnp.asarray(flux),
        1,
        TopologyClass.DIVERTED,
    )
    masked = topology.wall_anchor_data(
        jnp.asarray(flux),
        1,
        TopologyClass.DIVERTED,
        private_wall,
    )

    np.testing.assert_array_equal(np.asarray(masked), np.asarray(baseline))


def test_frozen_partition_supplies_previous_wall_shadow_to_topology_read() -> None:
    """Cold reads stay unmasked; later reads consume the promoted wall shadow."""
    operator = object.__new__(ForwardFluxOperator)
    operator.grid = SimpleNamespace(node_number=2)
    operator.wall = SimpleNamespace(node_number=3)
    operator.sample = None
    operator.use_linear_moments = False
    masks = DomainMasks(
        label=jnp.asarray([PlasmaDomain.CORE, PlasmaDomain.PRIVATE_FLUX]),
        psi_norm=jnp.asarray([0.0, 2.0]),
    )
    topology = TopologyState(
        axis=jnp.asarray([1.0, 0.0]),
        axis_flux=jnp.asarray(1.0),
        boundary=jnp.asarray([1.4, 0.0]),
        boundary_flux=jnp.asarray(0.5),
        x_point=jnp.asarray([1.0, -0.4]),
        x_point_flux=jnp.asarray(0.4),
        wall_point=jnp.asarray([1.4, 0.0]),
        wall_point_flux=jnp.asarray(0.5),
        diverted=jnp.asarray(False),
    )
    supplied_masks: list[object] = []

    def fixed_read(_physical, _requested=None, private_wall_node_mask=None):
        supplied_masks.append(private_wall_node_mask)
        return masks, topology, masks.core, jnp.asarray(True)

    operator._fixed_design_read = fixed_read
    operator._residual_shadow_components_from_read = (
        lambda _physical, current_masks, _topology, _previous=None: (
            current_masks.private_flux,
            jnp.asarray([False, True, False]),
        )
    )
    state = jnp.arange(5, dtype=jnp.float64)

    operator._frozen_topology_partition(state, TopologyClass.LIMITED)
    previous = jnp.asarray([False, True, True, False, True])
    operator._frozen_topology_partition(
        state,
        TopologyClass.LIMITED,
        previous_shadow=previous,
    )

    assert supplied_masks[0] is None
    np.testing.assert_array_equal(np.asarray(supplied_masks[1]), [True, False, True])
