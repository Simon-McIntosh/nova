r"""Pinned poloidal-flux, current-density and Grad-Shafranov conventions.

Every forward-equilibrium quantity in :mod:`nova.equilibrium` is raw SI and
carries the TOTAL poloidal flux :math:`\Phi = 2 \pi R A_\phi` in Wb rather
than the flux per radian used by most textbook statements of the
Grad-Shafranov equation. :math:`\mu_0` is always written explicitly.

Three statements fix the whole sign chain; nothing downstream may re-derive
them from a textbook equation, because the textbook equation is written in
flux per radian and in the opposite flux-function sense.

Poloidal field from the flux map
    .. math::
        B_R = -\frac{1}{2 \pi R} \frac{\partial \Phi}{\partial Z}, \qquad
        B_Z = \frac{1}{2 \pi R} \frac{\partial \Phi}{\partial R}.

Toroidal current density from the flux functions
    .. math::
        j_\phi = -2 \pi \left( R\, p'(\psi_N)
                 + \frac{FF'(\psi_N)}{\mu_0 R} \right).

Ampere's law then closes the loop with no further freedom
    .. math::
        \Delta^\star \Phi = -2 \pi \mu_0 R\, j_\phi
                          = 4 \pi^2 \left( \mu_0 R^2 p' + FF' \right),

where :math:`\Delta^\star \Phi = \partial_{RR}\Phi - R^{-1}\partial_R\Phi
+ \partial_{ZZ}\Phi`.

The last equality pins what ``p_prime`` and ``ff_prime`` mean dimensionally:
they are derivatives with respect to the NEGATED total flux,

.. math::
    p' = -\frac{\mathrm{d}p}{\mathrm{d}\Phi}, \qquad
    FF' = -F \frac{\mathrm{d}F}{\mathrm{d}\Phi},

so a profile primitive recovered from its gradient integrates from the
boundary inwards with a plus sign,

.. math::
    p(\psi_N) = p_b + (\Phi_b - \Phi_a) \int_{\psi_N}^{1} p'(s)\,\mathrm{d}s,
    \qquad
    F^2(\psi_N) = F_b^2
        + 2 (\Phi_b - \Phi_a) \int_{\psi_N}^{1} FF'(s)\,\mathrm{d}s .

The arithmetic here is namespace free: it evaluates unchanged on host arrays
and on traced device arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import mu_0

__all__ = [
    "TOTAL_FLUX_FACTOR",
    "delta_star_from_current_density",
    "flux_function_pressure",
    "flux_function_toroidal_field",
    "grad_shafranov_source",
    "toroidal_current_density",
]

#: Wb of total poloidal flux per Wb/rad of flux per radian.
TOTAL_FLUX_FACTOR = 2.0 * np.pi


def toroidal_current_density(radius, p_prime, ff_prime):
    """Return the toroidal current density [A/m^2] the flux functions drive."""
    return -TOTAL_FLUX_FACTOR * (radius * p_prime + ff_prime / (mu_0 * radius))


def delta_star_from_current_density(radius, current_density):
    """Return the elliptic operator value [Wb/m^2] Ampere's law demands."""
    return -TOTAL_FLUX_FACTOR * mu_0 * radius * current_density


def grad_shafranov_source(radius, p_prime, ff_prime):
    """Return the elliptic operator value [Wb/m^2] the flux functions demand."""
    return delta_star_from_current_density(
        radius, toroidal_current_density(radius, p_prime, ff_prime)
    )


def flux_function_pressure(boundary_pressure, flux_span, gradient_tail):
    """Return pressure [Pa] from its boundary value and inward gradient tail.

    ``gradient_tail`` is the integral of ``p_prime`` from the evaluation point
    out to the boundary in normalised flux and ``flux_span`` is
    :math:`\\Phi_b - \\Phi_a` in Wb.
    """
    return boundary_pressure + flux_span * gradient_tail


def flux_function_toroidal_field(boundary_field_function, flux_span, gradient_tail):
    """Return the squared toroidal-field function [T^2 m^2] from its tail.

    ``gradient_tail`` is the integral of ``ff_prime`` from the evaluation
    point out to the boundary in normalised flux.
    """
    return boundary_field_function**2 + 2.0 * flux_span * gradient_tail
