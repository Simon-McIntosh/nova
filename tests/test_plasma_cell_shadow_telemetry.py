"""Carrier-owned wall shadows read from node flux, and active-set label telemetry."""

import json

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.fixed_point import newton_krylov
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.topology import TopologyClass
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


def _zero_profile(psi_norm):
    return jnp.zeros_like(psi_norm)


def _raster_operator(
    shape: tuple[int, int], *, wall_through_lower_lobe: bool = False
) -> ForwardFluxOperator:
    """Return a structured carrier with an independently sampled wall."""
    radius = np.linspace(1.05, 2.35, shape[0])
    height = np.linspace(-0.72, 0.72, shape[1])
    lattice = FluxLattice(radius, height)
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    wall_coordinate = np.c_[
        1.7 + 0.64 * np.cos(wall_angle),
        0.68 * np.sin(wall_angle),
    ]
    if wall_through_lower_lobe:
        lower_lobe_segment = np.asarray(
            (
                (1.55, -0.46),
                (1.55, -0.31),
                (1.70, -0.31),
                (1.82, -0.31),
                (1.82, -0.46),
            )
        )
        wall_coordinate = np.r_[
            wall_coordinate[:68], lower_lobe_segment, wall_coordinate[77:]
        ]
    return ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.zeros((lattice.node_count, 1)),
            plasma_target=jnp.zeros((lattice.node_count, 1)),
            null=Null2D.from_coordinates(
                lattice.coordinate, hex_stencil(lattice.shape), maxsize=5
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.zeros((len(wall_coordinate), 1)),
            plasma_target=jnp.zeros((len(wall_coordinate), 1)),
            null=Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64)),
        ),
        source=ForwardSource(
            core=DomainProfile(p_prime=_zero_profile, ff_prime=_zero_profile)
        ),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        polarity=-1,
        use_linear_moments=False,
    )


def _single_null_flux(radius, height):
    local = radius - 1.7
    offset = 0.31
    return (local**2 + (height - offset) ** 2) * (local**2 + (height + offset) ** 2)


def _limited_flux(radius, height):
    return (radius - 1.7) ** 2 + height**2


def _fixture_read(flux, requested_class):
    """Return one structured fixture read through the carrier authority."""
    operator = _raster_operator(
        (17, 19),
        wall_through_lower_lobe=requested_class == TopologyClass.DIVERTED,
    )
    grid = np.asarray(operator.grid.coordinate)
    wall = np.asarray(operator.wall.coordinate)
    physical = jnp.asarray(
        np.r_[
            flux(grid[:, 0], grid[:, 1]),
            flux(wall[:, 0], wall[:, 1]),
        ]
    )
    masks, topology, _connected, admitted = operator._fixed_design_read(
        physical, requested_class
    )
    assert bool(admitted)
    return operator, physical, masks, topology


def _band_limits(axis_height, saddle_height):
    """Bound the axis-facing height interval with the admitted saddle position.

    A single admitted saddle splits the axis-facing band into the side of the
    low saddle and the side of the high saddle: a saddle above the axis leaves
    only the band below it, and vice versa. This recomputes the limit the
    topology read carries without calling the production helper.
    """
    lower = float(saddle_height)
    upper = float(saddle_height)
    lower = -np.inf if lower > axis_height else lower
    upper = np.inf if upper < axis_height else upper
    return lower, upper


def _node_flux_private_mask(
    wall_flux, wall_height, axis_height, saddle_height, saddle_flux, polarity
):
    """Recompute the node-flux private rule independently of the production read.

    The flag is true when the node's own finite flux lies on the private side of
    the admitted saddle flux and the node height sits outside the axis-facing
    band bounded by the admitted saddle. Cell labels play no part.
    """
    flux = np.asarray(wall_flux, dtype=np.float64)
    height = np.asarray(wall_height, dtype=np.float64)
    qualified = bool(np.isfinite(saddle_flux)) and bool(np.isfinite(axis_height))
    if not qualified:
        return np.zeros_like(flux, dtype=bool)
    lower, upper = _band_limits(float(axis_height), float(saddle_height))
    flux_side = np.isfinite(flux) & (polarity * (flux - saddle_flux) >= 0.0)
    height_band = np.isfinite(height) & ((height < lower) | (height > upper))
    return flux_side & height_band


def test_carrier_private_wall_follows_the_node_flux_rule():
    """A wall node is private by its own flux and height band, not its nearest cell."""
    configure_dtypes()
    disagreeing = {}
    for fixture_name, flux, requested_class in (
        ("limited", _limited_flux, TopologyClass.LIMITED),
        ("single_null", _single_null_flux, TopologyClass.DIVERTED),
    ):
        operator, physical, masks, topology = _fixture_read(flux, requested_class)
        carrier = np.asarray(
            operator._carrier_shadow_read(physical, masks)["private_wall_node_mask"]
        )
        coordinate = np.asarray(operator.wall.coordinate)
        wall_flux = np.asarray(
            physical[operator.grid.node_number : operator.physical_node_number]
        )
        labels = np.asarray(masks.label)
        owner_label = labels[np.asarray(operator._wall_carrier_index)]
        label_owned = owner_label == int(PlasmaDomain.PRIVATE_FLUX)
        axis_height = float(np.asarray(topology.axis)[1])
        x_point = np.asarray(topology.x_point)
        saddle_height = float(x_point[1])
        saddle_flux = float(np.asarray(topology.x_point_flux))
        expected = _node_flux_private_mask(
            wall_flux,
            coordinate[:, 1],
            axis_height,
            saddle_height,
            saddle_flux,
            operator.polarity,
        )
        np.testing.assert_array_equal(carrier, expected)
        disagreeing[fixture_name] = int(np.count_nonzero(label_owned != expected))
        print(
            "node_flux_wall_census="
            + json.dumps(
                {
                    "fixture": fixture_name,
                    "wall_nodes": operator.wall.node_number,
                    "carrier_private": int(np.count_nonzero(carrier)),
                    "label_owned_private": int(np.count_nonzero(label_owned)),
                    "nearest_cell_disagreements": disagreeing[fixture_name],
                    "axis_m": np.asarray(topology.axis).tolist(),
                    "x_point_m": x_point.tolist(),
                    "saddle_flux": saddle_flux,
                },
                sort_keys=True,
            )
        )
    # The diverted fixture carries a wall node in the narrow private leg whose
    # nearest cell centre lies across the diverted read's leg, so the retired
    # label-owned rule reads the opposite flag there. This count is what makes
    # the oracle discriminate the two rules rather than agree with both.
    assert disagreeing["single_null"] > 0


def test_residual_shadow_uses_carrier_operands_without_raster_read():
    """Constructed operators never request the tensor-product boundary read."""
    configure_dtypes()
    operator, physical, masks, _topology = _fixture_read(
        _single_null_flux, TopologyClass.DIVERTED
    )

    def forbid_raster_read(*_args, **_kwargs):
        raise AssertionError("residual shadowing requested the raster boundary read")

    operator._connectivity_read = forbid_raster_read
    flood_shadow, wall_shadow = operator.residual_shadow_components(physical)
    np.testing.assert_array_equal(flood_shadow, masks.private_flux)
    assert wall_shadow.shape == (operator.wall.node_number,)


def test_active_set_receipts_equal_tripwise_cell_label_differences():
    """Each active-set receipt counts the symmetric difference of cell labels."""

    def labels(state):
        private = jnp.stack((state[0] >= 0.5, state[0] >= 1.5))
        return jnp.where(
            private,
            jnp.int8(PlasmaDomain.PRIVATE_FLUX),
            jnp.int8(PlasmaDomain.CORE),
        )

    def private_mask(state):
        return labels(state) == jnp.int8(PlasmaDomain.PRIVATE_FLUX)

    def shadowed_map(state, mask):
        target = 1.0 + jnp.sum(mask, dtype=state.dtype)
        return jnp.full_like(state, target)

    def solve():
        return newton_krylov(
            lambda state: shadowed_map(state, private_mask(state)),
            jnp.zeros(1),
            newton_steps=1,
            gmres_iterations=1,
            warmup=0,
            shadow_mask_fn=private_mask,
            promoted_shadow_mask_fn=lambda state, _previous: private_mask(state),
            shadowed_map_fn=shadowed_map,
            active_set_steps=2,
        )

    label_states = [
        np.asarray(labels(jnp.asarray([value]))) for value in (0.0, 1.0, 2.0)
    ]
    expected = np.asarray(
        [
            np.count_nonzero(left != right)
            for left, right in zip(label_states, label_states[1:])
        ]
    )
    for result in (solve(), jax.jit(solve)()):
        np.testing.assert_array_equal(result.active_set_mask_differences, expected)
