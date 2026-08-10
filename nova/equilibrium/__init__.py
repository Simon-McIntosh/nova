"""Axisymmetric equilibrium solvers.

Classes are named from a two-axis constructor, ``ProblemRepresentation``: the
problem token says what is given and what is sought (forward: sources to flux;
inverse: shape targets to currents; reconstruct: measurements to state) and the
representation token says how the plasma current is carried (profile: a
:math:`j_\\phi(\\psi_N)` ladder; moment: current moments; harmonic: vacuum ring
functions).

:class:`~nova.equilibrium.forward.ForwardProfile` maps a prescribed source
state to equilibrium flux and :class:`~nova.equilibrium.profile.ReconstructProfile`
maps magnetic measurements to a fitted profile state. They share no inference:
the forward solve consumes the source it is given.
:class:`~nova.equilibrium.forward_operator.ForwardFluxOperator` is the traced
map behind the forward solve, not a second public problem.

Optional physics reaches the forward solve as a typed force-balance closure
on the source rather than as a solver variant. The first is toroidal rotation
under isothermal flux surfaces,
:class:`~nova.equilibrium.rotation.RotatingDomainProfile`, which makes the
pressure source depend on major radius and publishes the thermodynamic
conventions it was formed under. The second is a bounded continuation of the
flux functions past the separatrix,
:class:`~nova.equilibrium.continuation.SeparatrixContinuation`, declared
independently on the common scrape-off layer and the private-flux branch.

:class:`~nova.equilibrium.flux_surface_geometry.FluxSurfaceGeometry` reads a
converged map for the flux-surface-averaged metric a one-dimensional
transport balance consumes.

All quantities are raw SI. Poloidal flux is the total flux
:math:`\\Phi = 2 \\pi R A_\\phi` in Wb and :math:`\\mu_0` is always written
explicitly; the full sign chain is pinned in
:mod:`nova.equilibrium.convention`.
"""

from typing import TYPE_CHECKING

from nova.equilibrium.diagnostics import (
    DECAY_INDEX_WINDOW,
    decay_index,
    shafranov_vertical_field,
    shafranov_vertical_field_elongated,
)

if TYPE_CHECKING:
    from nova.equilibrium.conservation import ConservationLedger, FluxLattice
    from nova.equilibrium.continuation import (
        ContinuedDomainProfile,
        SeparatrixContinuation,
        SeparatrixJumpError,
    )
    from nova.equilibrium.domain import DomainMasks, PlasmaDomain
    from nova.equilibrium.flux_surface_geometry import (
        FluxSurfaceGeometry,
        GridMotion,
        SurfaceGeometryError,
        source_field_function,
    )
    from nova.equilibrium.forward import ForwardEquilibrium, ForwardProfile
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.observation import (
        CurrentLedger,
        IntegralObservation,
        MomentEnforcementError,
        MomentTargets,
    )
    from nova.equilibrium.profile import (
        ProfileDegrees,
        ProfilePrior,
        ProfileResult,
        ReconstructProfile,
    )
    from nova.equilibrium.rotation import (
        IsothermalRotation,
        RotatingDomainProfile,
    )
    from nova.equilibrium.source import (
        ContinuationForm,
        ContinuationLedger,
        ContinuationRecord,
        DomainProfile,
        ForwardSource,
        NormalisationPolicy,
        RotationClosure,
        RotationRecord,
        SeparatrixContinuity,
    )

#: Names loaded from their module on first attribute access, so importing the
#: package boundary never pulls in the optional jax extra.
_DEFERRED_EXPORTS: dict[str, str] = {
    "ConservationLedger": "conservation",
    "ContinuationForm": "source",
    "ContinuationLedger": "source",
    "ContinuationRecord": "source",
    "ContinuedDomainProfile": "continuation",
    "CurrentLedger": "observation",
    "DomainMasks": "domain",
    "DomainProfile": "source",
    "FluxLattice": "conservation",
    "FluxSurfaceGeometry": "flux_surface_geometry",
    "ForwardEquilibrium": "forward",
    "ForwardFluxOperator": "forward_operator",
    "ForwardProfile": "forward",
    "ForwardSource": "source",
    "GridMotion": "flux_surface_geometry",
    "IntegralObservation": "observation",
    "IsothermalRotation": "rotation",
    "MomentEnforcementError": "observation",
    "MomentTargets": "observation",
    "NormalisationPolicy": "source",
    "PlasmaDomain": "domain",
    "ProfileDegrees": "profile",
    "ProfilePrior": "profile",
    "ProfileResult": "profile",
    "ReconstructProfile": "profile",
    "RotatingDomainProfile": "rotation",
    "RotationClosure": "source",
    "RotationRecord": "source",
    "SeparatrixContinuation": "continuation",
    "SeparatrixContinuity": "source",
    "SeparatrixJumpError": "continuation",
    "SurfaceGeometryError": "flux_surface_geometry",
    "source_field_function": "flux_surface_geometry",
}

__all__ = [
    "DECAY_INDEX_WINDOW",
    "ConservationLedger",
    "ContinuationForm",
    "ContinuationLedger",
    "ContinuationRecord",
    "ContinuedDomainProfile",
    "CurrentLedger",
    "DomainMasks",
    "DomainProfile",
    "FluxLattice",
    "FluxSurfaceGeometry",
    "ForwardEquilibrium",
    "ForwardFluxOperator",
    "ForwardProfile",
    "ForwardSource",
    "GridMotion",
    "IntegralObservation",
    "IsothermalRotation",
    "MomentEnforcementError",
    "MomentTargets",
    "NormalisationPolicy",
    "PlasmaDomain",
    "ProfileDegrees",
    "ProfilePrior",
    "ProfileResult",
    "ReconstructProfile",
    "RotatingDomainProfile",
    "RotationClosure",
    "RotationRecord",
    "SeparatrixContinuation",
    "SeparatrixContinuity",
    "SeparatrixJumpError",
    "SurfaceGeometryError",
    "decay_index",
    "shafranov_vertical_field",
    "shafranov_vertical_field_elongated",
    "source_field_function",
]


def __getattr__(name: str):
    """Load a deferred equilibrium export only when it is requested."""
    if name not in _DEFERRED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(
        import_module(f"{__name__}.{_DEFERRED_EXPORTS[name]}"),
        name,
    )
    globals()[name] = value
    return value
