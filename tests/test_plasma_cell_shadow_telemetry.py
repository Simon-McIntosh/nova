"""Carrier-owned wall shadows and active-set label telemetry."""

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
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


def _zero_profile(psi_norm):
    return jnp.zeros_like(psi_norm)


def _raster_operator(shape: tuple[int, int]) -> ForwardFluxOperator:
    """Return a structured carrier with an independently sampled wall."""
    radius = np.linspace(1.05, 2.35, shape[0])
    height = np.linspace(-0.72, 0.72, shape[1])
    lattice = FluxLattice(radius, height)
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    wall_coordinate = np.c_[
        1.7 + 0.64 * np.cos(wall_angle),
        0.68 * np.sin(wall_angle),
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
        use_linear_moments=False,
    )


def _single_null_flux(radius, height):
    local = radius - 1.7
    offset = 0.31
    return (local**2 + (height - offset) ** 2) * (local**2 + (height + offset) ** 2)


def _limited_flux(radius, height):
    return (radius - 1.7) ** 2 + height**2


def _fixture_read(flux):
    """Return one structured fixture read through the carrier authority."""
    operator = _raster_operator((17, 19))
    grid = np.asarray(operator.grid.coordinate)
    wall = np.asarray(operator.wall.coordinate)
    physical = jnp.asarray(
        np.r_[
            flux(grid[:, 0], grid[:, 1]),
            flux(wall[:, 0], wall[:, 1]),
        ]
    )
    masks, topology, _connected, admitted = operator._fixed_design_read(physical)
    assert bool(admitted)
    return operator, physical, masks, topology


def test_carrier_private_wall_is_label_owned_with_raster_census():
    """Nearest-cell private labels own the mask; raster deltas are censused."""
    configure_dtypes()
    for fixture_name, flux in (
        ("limited", _limited_flux),
        ("single_null", _single_null_flux),
    ):
        operator, physical, masks, topology = _fixture_read(flux)
        retained = np.asarray(
            operator._connectivity_read(physical, topology, classify=False)[
                "private_wall_node_mask"
            ]
        )
        carrier = np.asarray(
            operator._carrier_shadow_read(physical, masks)["private_wall_node_mask"]
        )
        coordinate = np.asarray(operator.wall.coordinate)
        owner = np.asarray(operator._wall_carrier_index)
        labels = np.asarray(masks.label)
        owner_label = labels[owner]
        expected = owner_label == PlasmaDomain.PRIVATE_FLUX
        np.testing.assert_array_equal(carrier, expected)

        differing = np.flatnonzero(retained != carrier)
        rows = []
        contradictions = []
        for index in differing:
            label = PlasmaDomain(int(owner_label[index]))
            contradiction = label == PlasmaDomain.CORE and bool(retained[index])
            classification = (
                "contradiction"
                if contradiction
                else (
                    "binding-level difference"
                    if label == PlasmaDomain.PRIVATE_FLUX
                    else "touch-dilation"
                )
            )
            row = {
                "index": int(index),
                "position_m": coordinate[index].tolist(),
                "nearest_cell_label": label.name,
                "retained_private": bool(retained[index]),
                "classification": classification,
            }
            rows.append(row)
            if contradiction:
                contradictions.append(row)
        print(
            "wall_mask_census="
            + json.dumps(
                {
                    "fixture": fixture_name,
                    "wall_nodes": operator.wall.node_number,
                    "carrier_private": int(np.count_nonzero(carrier)),
                    "retained_private": int(np.count_nonzero(retained)),
                    "differing": int(differing.size),
                    "rows": rows,
                },
                sort_keys=True,
            )
        )
        assert not contradictions


def test_residual_shadow_uses_carrier_operands_without_raster_read():
    """Constructed operators never request the tensor-product boundary read."""
    configure_dtypes()
    operator, physical, masks, _topology = _fixture_read(_single_null_flux)

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
