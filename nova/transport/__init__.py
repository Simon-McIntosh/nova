"""Transport-tier solvers: how the plasma state evolves between equilibria.

A per-slice equilibrium fit treats successive reconstructions as independent.
Physically they are not: the poloidal flux obeys the resistive diffusion
equation, whose boundary conditions (total plasma current, surface loop voltage)
are measured and whose only genuine unknown is the parallel resistivity profile.
That equation is a temporal prior linking one slice's profile coefficients to the
next.

* :class:`~nova.transport.current_diffusion.CurrentDiffusion` -- the 1D
  flux-surface-averaged resistive current diffusion, its predicted current
  profiles, their projection back onto a profile-coefficient ladder, and the
  inductive/resistive flux-consumption ledger.
* :func:`~nova.transport.current_diffusion.flux_surface_geometry` -- the
  fixed-shape assembly of diffusion metrics from an equilibrium grid and its
  fitted profile ladder.

Conventions: total poloidal flux ``Phi = 2 pi R A_phi`` [Wb], explicit ``mu0``,
raw SI throughout.
"""

from nova.transport.current_diffusion import (
    CurrentDiffusion,
    EtaProfile,
    FluxSurfaceGeometry,
    traced_assemble_flux_surface_geometry,
    basis_projection_images,
    diffuse_psi,
    ejima_coefficient,
    flux_budget,
    flux_surface_geometry,
    traced_flux_surface_geometry,
    poloidal_field_energy_li,
    predicted_current,
    project_coefficients,
)

__all__ = [
    "CurrentDiffusion",
    "EtaProfile",
    "FluxSurfaceGeometry",
    "traced_assemble_flux_surface_geometry",
    "basis_projection_images",
    "diffuse_psi",
    "ejima_coefficient",
    "flux_budget",
    "flux_surface_geometry",
    "traced_flux_surface_geometry",
    "poloidal_field_energy_li",
    "predicted_current",
    "project_coefficients",
]
