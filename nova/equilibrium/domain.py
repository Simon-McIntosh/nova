r"""Topology-qualified plasma domain labels.

A normalised-flux range is not a domain. In a diverted configuration the
private-flux region carries :math:`\psi_N < 1` exactly like the core, and the
open scrape-off layer carries :math:`\psi_N > 1` exactly like every cell
outside the material boundary. Selecting a source domain from
:math:`\psi_N > 1` alone therefore either misses the private-flux branch or
sweeps in cells that no plasma closure owns.

The labels below split the flux map on the two properties that actually
distinguish the branches: whether a cell lies inside the material boundary,
and whether it is connected to the magnetic axis. With the closed test
:math:`\psi_N \le \psi_N^{\mathrm{lcfs}}` written ``closed`` and the
axis-connected test written ``connected``, the partition is exact and total:

===================  =========================================
label                selection
===================  =========================================
``CORE``             inside and closed and connected
``PRIVATE_FLUX``     inside and closed and not connected
``COMMON_SOL``       inside and not closed
``EXCLUDED_MATERIAL`` outside the material boundary
===================  =========================================

The closed test uses the last-closed-flux-surface cut rather than the
separatrix itself, matching the ionisation cut the topology read already
applies, so the thin shell between them is labelled with the open branch and
carries no core source.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["DomainMasks", "PlasmaDomain", "classify_domains"]


class PlasmaDomain(IntEnum):
    """Domain a source closure may be declared on."""

    EXCLUDED_MATERIAL = 0
    CORE = 1
    COMMON_SOL = 2
    PRIVATE_FLUX = 3


class DomainMasks(NamedTuple):
    """Fixed-shape domain labels and the normalised flux they were cut from."""

    label: jax.Array
    psi_norm: jax.Array

    @property
    def core(self) -> jax.Array:
        """Return the axis-connected closed-surface mask."""
        return self.label == PlasmaDomain.CORE

    @property
    def common_sol(self) -> jax.Array:
        """Return the open scrape-off mask inside the material boundary."""
        return self.label == PlasmaDomain.COMMON_SOL

    @property
    def private_flux(self) -> jax.Array:
        """Return the closed-surface mask disconnected from the axis."""
        return self.label == PlasmaDomain.PRIVATE_FLUX

    @property
    def excluded_material(self) -> jax.Array:
        """Return the mask of cells outside the material boundary."""
        return self.label == PlasmaDomain.EXCLUDED_MATERIAL

    def mask(self, domain: PlasmaDomain) -> jax.Array:
        """Return the selection mask of one named domain."""
        return self.label == domain

    def cell_count(self) -> jax.Array:
        """Return the cell count per domain in :class:`PlasmaDomain` order."""
        return jnp.stack(
            [jnp.sum(self.mask(domain)) for domain in PlasmaDomain],
        )


def classify_domains(
    psi_norm: jax.Array,
    closed: jax.Array,
    connected: jax.Array,
    inside_material: jax.Array,
) -> DomainMasks:
    """Return the domain partition of one flux map.

    ``closed`` marks cells inside the last-closed-flux-surface cut,
    ``connected`` marks cells on the axis side of every X-point, and
    ``inside_material`` marks cells the material boundary encloses. The three
    are combined into a single integer label so no downstream selection can
    reconstruct a domain from a flux range alone.
    """
    label = jnp.where(
        closed & connected,
        jnp.int8(PlasmaDomain.CORE),
        jnp.where(
            closed,
            jnp.int8(PlasmaDomain.PRIVATE_FLUX),
            jnp.int8(PlasmaDomain.COMMON_SOL),
        ),
    )
    label = jnp.where(inside_material, label, jnp.int8(PlasmaDomain.EXCLUDED_MATERIAL))
    return DomainMasks(label=label, psi_norm=psi_norm)
