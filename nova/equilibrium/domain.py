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
:math:`\psi_N \le 1` written ``closed`` and the axis-connected test written
``connected``, the partition is exact and total:

===================  =========================================
label                selection
===================  =========================================
``CORE``             inside and closed and connected
``PRIVATE_FLUX``     inside and closed and not connected
``COMMON_SOL``       inside and not closed
``EXCLUDED_MATERIAL`` outside the material boundary
===================  =========================================

The plasma reaches its own boundary
-----------------------------------
A cell whose centroid lies inside the plasma boundary curve — the separatrix
of a diverted configuration, the limiting surface of a wall-limited one — IS
plasma, and carries :math:`\psi_N \le 1` by definition. The closed test
therefore cuts at that boundary and nowhere short of it: the core mask extends
exactly to the boundary and no within-boundary shell may carry an open label.
A cut a declared fraction inside the surface belongs to a fitted current image,
where it guards cells straddling the edge; it is not a statement about where
the plasma ends, and using one to partition domains shaves the plasma's outer
edge into the scrape-off layer.

The open branch is consequently strict. ``COMMON_SOL`` is :math:`\psi_N > 1`
inside the material — a centroid outside the boundary curve. ``PRIVATE_FLUX``
is what the CONNECTIVITY test removes from the closed set: geometrically
outside the boundary curve although its flux value sits below one, which is
exactly why the connectivity test and not the flux value decides it.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nova.equilibrium.flux_surface_connectivity import (
    label_saddle_aware_hex_connected_components,
    private_flux_mask,
)

__all__ = [
    "DomainMasks",
    "PlasmaDomain",
    "ProfileDomainChange",
    "axis_connected_component",
    "classify_domains",
    "profile_domain_change",
]


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

    @property
    def profile_participation(self) -> jax.Array:
        """Return the boundary-free solve domain owned by the flux profile."""

        return ~(self.private_flux | self.excluded_material)

    def mask(self, domain: PlasmaDomain) -> jax.Array:
        """Return the selection mask of one named domain."""
        return self.label == domain

    def cell_count(self) -> jax.Array:
        """Return the cell count per domain in :class:`PlasmaDomain` order."""
        return jnp.stack(
            [jnp.sum(self.mask(domain)) for domain in PlasmaDomain],
        )


class ProfileDomainChange(NamedTuple):
    """Fixed-shape cause counts between two residual-domain reads."""

    shadow_entered: jax.Array
    shadow_left: jax.Array
    material_changed: jax.Array

    @property
    def shadow_changed(self) -> jax.Array:
        """Return the total number of cells whose shadow state changed."""

        return self.shadow_entered + self.shadow_left


@jax.jit
def profile_domain_change(
    previous: DomainMasks, current: DomainMasks
) -> ProfileDomainChange:
    """Attribute a profile-domain transition to shadow or material changes."""

    entered = current.private_flux & ~previous.private_flux
    left = previous.private_flux & ~current.private_flux
    material = previous.excluded_material != current.excluded_material
    return ProfileDomainChange(
        shadow_entered=jnp.sum(entered, dtype=jnp.int32),
        shadow_left=jnp.sum(left, dtype=jnp.int32),
        material_changed=jnp.sum(material, dtype=jnp.int32),
    )


@jax.jit
def axis_connected_component(
    confined: jax.Array,
    rings: jax.Array,
    link_admissible: jax.Array,
    axis_seed: jax.Array,
) -> jax.Array:
    """Return the saddle-aware hex component containing ``axis_seed``.

    Component propagation remains owned by the shared connectivity kernel. This
    adapter only selects its magnetic-axis component, preserving the exact
    centre-first six-neighbour graph and strict shared-edge admissibility used
    by the boundary read.
    """
    labels = label_saddle_aware_hex_connected_components(
        confined,
        rings,
        link_admissible,
        confined.size,
    )
    return (labels > 0) & ~private_flux_mask(labels, axis_seed)


def classify_domains(
    psi_norm: jax.Array,
    closed: jax.Array,
    connected: jax.Array,
    inside_material: jax.Array,
) -> DomainMasks:
    """Return the domain partition of one flux map.

    ``closed`` marks cells on the plasma side of the boundary flux itself,
    ``connected`` marks cells in the saddle-aware hex component of the axis, and
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
