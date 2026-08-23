"""Exact forward-map tangents on symmetric null-fit geometry."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.topology import TopologyClass
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


def _flux_field(coordinate: np.ndarray) -> np.ndarray:
    """Return a smooth field with one magnetic axis and one saddle."""
    radial = coordinate[:, 0] - 1.0
    height = coordinate[:, 1]
    separation = 0.22
    return -(radial**3 / 3.0 - separation**2 * radial + height**2)


def _gradient(psi_norm):
    """Return a flux-dependent source so the topology tangent reaches the map."""
    return 1.0e5 * (1.0 - jnp.asarray(psi_norm))


def test_symmetric_null_fit_has_finite_exact_residual_action() -> None:
    """A fixed repeated-singular-value fit differentiates only sampled flux."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.5, 1.5, 9), np.linspace(-0.5, 0.5, 9))
    angle = 2.0 * np.pi * np.arange(64) / 64
    wall_coordinate = np.c_[1.0 + 0.7 * np.cos(angle), 0.7 * np.sin(angle)]
    node_count = lattice.node_count
    wall_count = len(wall_coordinate)
    grid = FluxTarget(
        source_target=jnp.zeros((node_count, 1)),
        plasma_target=1.0e-16 * jnp.eye(node_count),
        null=Null2D.from_coordinates(lattice.coordinate, hex_stencil(lattice.shape)),
    )
    wall = FluxTarget(
        source_target=jnp.zeros((wall_count, 1)),
        plasma_target=jnp.zeros((wall_count, node_count)),
        null=Null1D(jnp.asarray(wall_coordinate)),
    )
    operator = ForwardFluxOperator(
        grid=grid,
        wall=wall,
        source=ForwardSource(
            core=DomainProfile(
                p_prime=_gradient,
                ff_prime=lambda psi_norm: jnp.zeros_like(psi_norm),
            )
        ),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        inside_material=jnp.ones(node_count, dtype=bool),
        use_linear_moments=False,
    )
    state = jnp.asarray(
        np.r_[_flux_field(lattice.coordinate), _flux_field(wall_coordinate)]
    )
    mapped = operator.flux_map(requested_class=TopologyClass.DIVERTED)
    image, tangent = jax.linearize(mapped, state)
    residual = image - state
    direction = residual / jnp.linalg.norm(residual)
    exact = direction - tangent(direction)

    scale = float(jnp.max(jnp.abs(state)))
    delta = 1.0e-6 * scale
    plus = state + delta * direction
    minus = state - delta * direction
    central = (plus - mapped(plus) - minus + mapped(minus)) / (2.0 * delta)

    assert bool(jnp.all(jnp.isfinite(image)))
    assert bool(jnp.all(jnp.isfinite(exact)))
    relative_error = float(jnp.linalg.norm(exact - central) / jnp.linalg.norm(central))
    assert relative_error < 1.0e-6
