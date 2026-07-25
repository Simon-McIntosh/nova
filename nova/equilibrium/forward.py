"""Forward free-boundary equilibrium solve on a flux-function profile ladder."""

from dataclasses import dataclass

from scipy.optimize import newton_krylov

from nova.biot.plasma import Plasma


@dataclass
class ForwardProfile:
    """Solve the free-boundary equilibrium for prescribed sources and profiles.

    Given the external conductor currents and the pair of flux functions
    :math:`p'(\\psi_N)` and :math:`FF'(\\psi_N)` that set the toroidal current
    density

    .. math::
        j_\\phi = -2 \\pi \\left( R\\, p'(\\psi_N)
                  + \\frac{FF'(\\psi_N)}{\\mu_0 R} \\right),

    find the poloidal flux map consistent with the plasma current it induces.

    Writing a trial flux map to the plasma component re-normalizes
    :math:`\\psi`, re-ionizes the filaments inside the resulting separatrix and
    redistributes the plasma current across them, so the equilibrium is the
    fixed point of that write-then-read cycle. The residual is driven to zero
    with a Jacobian-free Newton-Krylov iteration, which needs no explicit
    Jacobian of the free-boundary map.

    Fluxes are total poloidal fluxes, :math:`\\Phi = 2 \\pi R A_\\phi` in Wb,
    concatenated over the plasma grid nodes followed by the wall nodes.
    """

    plasma: Plasma

    def residual(self, psi):
        """Return the free-boundary flux residual for a trial flux map."""
        self.plasma.psi = psi
        return self.plasma.psi - psi

    def solve(self, **kwargs):
        """Return the equilibrium flux map across the plasma grid and boundary.

        Keyword arguments are passed to :func:`scipy.optimize.newton_krylov`.
        """
        self.plasma.psi = newton_krylov(self.residual, self.plasma.psi, **kwargs)
        return self.plasma.psi
