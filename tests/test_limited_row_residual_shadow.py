"""Private-flux shadows require a finite saddle-owned boundary."""

import jax.numpy as jnp
import numpy as np

from nova.equilibrium.domain import (
    DomainMasks,
    PlasmaDomain,
    saddle_qualified_domains,
)
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.topology import TopologyState
from nova.jax.config import configure_dtypes


configure_dtypes()


def _masks() -> DomainMasks:
    return DomainMasks(
        label=jnp.asarray(
            [PlasmaDomain.CORE, PlasmaDomain.PRIVATE_FLUX], dtype=jnp.int8
        ),
        psi_norm=jnp.asarray([0.2, 0.8]),
    )


def _topology(*, diverted: bool) -> TopologyState:
    x_point = jnp.asarray([1.5, -0.4]) if diverted else jnp.full(2, jnp.nan)
    x_flux = jnp.asarray(1.0) if diverted else jnp.asarray(jnp.nan)
    return TopologyState(
        axis=jnp.asarray([1.5, 0.0]),
        axis_flux=jnp.asarray(0.0),
        boundary=x_point if diverted else jnp.asarray([2.0, 0.0]),
        boundary_flux=jnp.asarray(1.0),
        x_point=x_point,
        x_point_flux=x_flux,
        wall_point=jnp.asarray([2.0, 0.0]),
        wall_point_flux=jnp.asarray(1.0),
        diverted=jnp.asarray(diverted),
    )


def test_limited_class_relabels_disconnected_closed_carriers_as_core():
    """A wall-cut connectivity split cannot create private flux without a saddle."""
    topology = _topology(diverted=False)
    qualified = saddle_qualified_domains(
        _masks(), ForwardFluxOperator._private_flux_saddle_admitted(topology)
    )

    np.testing.assert_array_equal(
        qualified.label,
        np.asarray([PlasmaDomain.CORE, PlasmaDomain.CORE], dtype=np.int8),
    )
    assert not bool(jnp.any(qualified.private_flux))


def test_limited_class_maps_every_physical_carrier():
    """No limited physical carrier is copied through the residual unchanged."""
    topology = _topology(diverted=False)
    qualified = saddle_qualified_domains(
        _masks(), ForwardFluxOperator._private_flux_saddle_admitted(topology)
    )
    trial = jnp.asarray([3.0, 5.0])
    mapped = jnp.asarray([7.0, 11.0])

    result = ForwardFluxOperator._exclude_shadow_residual(
        object.__new__(ForwardFluxOperator),
        trial,
        mapped,
        shadow=qualified.private_flux,
    )

    np.testing.assert_array_equal(result, mapped)


def test_diverted_class_preserves_private_shadow_behind_finite_saddle():
    """A finite saddle-owned boundary retains its disconnected closed branch."""
    topology = _topology(diverted=True)
    qualified = saddle_qualified_domains(
        _masks(), ForwardFluxOperator._private_flux_saddle_admitted(topology)
    )
    trial = jnp.asarray([3.0, 5.0])
    mapped = jnp.asarray([7.0, 11.0])

    result = ForwardFluxOperator._exclude_shadow_residual(
        object.__new__(ForwardFluxOperator),
        trial,
        mapped,
        shadow=qualified.private_flux,
    )

    np.testing.assert_array_equal(qualified.label, _masks().label)
    np.testing.assert_array_equal(result, np.asarray([7.0, 5.0]))
