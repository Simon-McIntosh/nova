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
                (1.58, -0.46),
                (1.58, -0.31),
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


def _inside_boundary(flux, boundary_flux, polarity):
    """Evaluate one flux against a boundary using the operator convention."""
    values = np.asarray(flux)
    return values >= boundary_flux if polarity > 0 else values < boundary_flux


def test_carrier_private_wall_is_label_owned_with_raster_census():
    """Nearest-cell private labels own the mask; raster deltas are censused."""
    configure_dtypes()
    for fixture_name, flux, requested_class in (
        ("limited", _limited_flux, TopologyClass.LIMITED),
        ("single_null", _single_null_flux, TopologyClass.DIVERTED),
    ):
        operator, physical, masks, topology = _fixture_read(flux, requested_class)
        retained_read = operator._connectivity_read(physical, topology, classify=False)
        retained = np.asarray(retained_read["private_wall_node_mask"])
        carrier = np.asarray(
            operator._carrier_shadow_read(physical, masks)["private_wall_node_mask"]
        )
        coordinate = np.asarray(operator.wall.coordinate)
        owner = np.asarray(operator._wall_carrier_index)
        labels = np.asarray(masks.label)
        owner_label = labels[owner]
        expected = owner_label == PlasmaDomain.PRIVATE_FLUX
        np.testing.assert_array_equal(carrier, expected)

        grid_coordinate = np.asarray(operator.grid.coordinate)
        radial_spacing = np.min(np.diff(np.unique(grid_coordinate[:, 0])))
        vertical_spacing = np.min(np.diff(np.unique(grid_coordinate[:, 1])))
        cell_spacing = max(radial_spacing, vertical_spacing)
        if fixture_name == "single_null":
            analytic_axes = np.asarray(((1.7, -0.31), (1.7, 0.31)))
            axis_error = np.min(
                np.linalg.norm(analytic_axes - np.asarray(topology.axis), axis=1)
            )
            saddle_error = np.linalg.norm(
                np.asarray(topology.x_point) - np.asarray((1.7, 0.0))
            )
            assert axis_error <= cell_spacing
            assert saddle_error <= cell_spacing
            assert bool(topology.boundary_is_xpoint)
            assert np.isclose(
                float(topology.boundary_flux),
                float(_single_null_flux(1.7, 0.0)),
                rtol=0.0,
                atol=cell_spacing**2,
            )
            domain_counts = {
                domain.name: int(np.count_nonzero(labels == domain))
                for domain in (
                    PlasmaDomain.CORE,
                    PlasmaDomain.PRIVATE_FLUX,
                    PlasmaDomain.COMMON_SOL,
                )
            }
            assert all(count > 0 for count in domain_counts.values())
            assert np.count_nonzero(carrier) > 0
        else:
            domain_counts = {
                domain.name: int(np.count_nonzero(labels == domain))
                for domain in PlasmaDomain
            }
            assert domain_counts[PlasmaDomain.CORE.name] > 0
            assert domain_counts[PlasmaDomain.COMMON_SOL.name] > 0
            assert domain_counts[PlasmaDomain.PRIVATE_FLUX.name] == 0
            assert (
                np.linalg.norm(np.asarray(topology.axis) - np.asarray((1.7, 0.0)))
                <= cell_spacing
            )

        differing = np.flatnonzero(retained != carrier)
        wall_flux = np.asarray(
            physical[operator.grid.node_number : operator.physical_node_number]
        )
        grid_flux = np.asarray(physical[: operator.grid.node_number])
        census_boundary_flux = (
            float(_single_null_flux(1.7, 0.0))
            if fixture_name == "single_null"
            else float(topology.boundary_flux)
        )
        raster_boundary_flux = float(retained_read["psi_bnd"])
        wall_census_inside = _inside_boundary(
            wall_flux, census_boundary_flux, operator.polarity
        )
        wall_raster_inside = _inside_boundary(
            wall_flux, raster_boundary_flux, operator.polarity
        )
        owner_census_inside = _inside_boundary(
            grid_flux[owner], census_boundary_flux, operator.polarity
        )
        owner_raster_inside = _inside_boundary(
            grid_flux[owner], raster_boundary_flux, operator.polarity
        )
        rows = []
        contradictions = []
        for index in differing:
            label = PlasmaDomain(int(owner_label[index]))
            level_difference = bool(
                wall_census_inside[index] != wall_raster_inside[index]
            )
            touch_difference = bool(
                (wall_census_inside[index] != owner_census_inside[index])
                or (wall_raster_inside[index] != owner_raster_inside[index])
            )
            contradiction = not (level_difference or touch_difference)
            classification = (
                "binding-level difference"
                if level_difference
                else "touch-dilation"
                if touch_difference
                else "contradiction"
            )
            row = {
                "index": int(index),
                "position_m": coordinate[index].tolist(),
                "wall_flux": float(wall_flux[index]),
                "census_boundary_flux": census_boundary_flux,
                "raster_boundary_flux": raster_boundary_flux,
                "wall_inside_census_boundary": bool(wall_census_inside[index]),
                "wall_inside_raster_boundary": bool(wall_raster_inside[index]),
                "nearest_cell_label": label.name,
                "retained_private": bool(retained[index]),
                "classification": classification,
            }
            rows.append(row)
            if contradiction:
                contradictions.append(row)
        classification_tally = {
            classification: sum(row["classification"] == classification for row in rows)
            for classification in (
                "binding-level difference",
                "touch-dilation",
                "contradiction",
            )
        }
        print(
            "wall_mask_census="
            + json.dumps(
                {
                    "fixture": fixture_name,
                    "wall_nodes": operator.wall.node_number,
                    "carrier_private": int(np.count_nonzero(carrier)),
                    "retained_private": int(np.count_nonzero(retained)),
                    "differing": int(differing.size),
                    "domain_counts": domain_counts,
                    "selected_axis_m": np.asarray(topology.axis).tolist(),
                    "selected_x_m": np.asarray(topology.x_point).tolist(),
                    "census_boundary_flux": census_boundary_flux,
                    "raster_boundary_flux": raster_boundary_flux,
                    "classification_tally": classification_tally,
                    "rows": rows,
                },
                sort_keys=True,
            )
        )
        assert not contradictions


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
