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
    from nova.equilibrium.branch_selection import (
        AdmissibilityCriterion,
        BranchAdmissibility,
        BranchAvailability,
        ColdStartRule,
        DisappearanceCriterion,
        SelectionHistory,
        SelectionPolicy,
        SelectionReason,
        SelectionReceipt,
        select_forward_branch,
    )
    from nova.equilibrium.conservation import (
        ConservationLedger,
        FluxLattice,
        FluxMesh,
    )
    from nova.equilibrium.continuation import (
        ContinuedDomainProfile,
        SeparatrixContinuation,
        SeparatrixJumpError,
    )
    from nova.equilibrium.constraint import (
        CircuitCurrentUnknown,
        CompensatingUnknown,
        ConstraintBinding,
        ConstraintContext,
        ConstraintFunctional,
        ConstraintMultiplier,
        ConstraintPair,
        ConstraintRecord,
        CompensatorRule,
        CompensatorSelection,
        CurrentCentroidConstraint,
        IsofluxConstraint,
        IsofluxReference,
        ProfileAmplitudeUnknown,
        WallGapConstraint,
        WallGapTarget,
        XPointConstraint,
        compensator_rule_name,
        constraint_response_matrix,
        derive_circuit_compensators,
        miller_boundary_points,
        sample_lattice_flux,
        select_compensating_directions,
    )
    from nova.equilibrium.domain import DomainMasks, PlasmaDomain
    from nova.equilibrium.extraction_lattice import (
        GreenSourceRepresentation,
        evaluate_forward_equilibrium,
    )
    from nova.equilibrium.flux_surface_geometry import (
        FluxSurfaceGeometry,
        GridMotion,
        SurfaceGeometryError,
        source_field_function,
    )
    from nova.equilibrium.flux_surface_extraction import (
        extract_flux_surface_geometry,
        traced_assemble_flux_surface_geometry,
        traced_flux_surface_geometry,
    )
    from nova.equilibrium.forward import (
        ColdSeedConstruction,
        ForwardBranchReceipt,
        ForwardColdSeedPortfolio,
        ForwardColdSeedReceipt,
        ForwardEquilibrium,
        ForwardPerturbedSeedReceipt,
        ForwardPortfolio,
        ForwardProfile,
        PerturbedSeedPolicy,
        SaddleSeedGeometry,
    )
    from nova.equilibrium.forward_operator import (
        ForwardFluxOperator,
        PrescribedCurrentField,
    )
    from nova.equilibrium.map_extraction import (
        ChordSamplingReceipt,
        MapCurrentReceipt,
        SurfaceExtractionReceipt,
        VacuumRegionReceipt,
        apply_delta_star,
        extract_flux_functions,
        sample_chord_psi_norm,
        vacuum_region_receipt,
    )
    from nova.equilibrium.internal_inductance import (
        Li2Geometry,
        Li3Geometry,
        convert_li_2_to_li_3,
        convert_li_3_to_li_2,
        li_2_from_field_energy,
        li_2_normaliser,
        li_3_from_field_energy,
        li_3_normaliser,
    )
    from nova.equilibrium.moment import (
        CurrentIntegralSupport,
        MomentSeed,
        PredictedCurrentMoments,
        ReconstructMoment,
    )
    from nova.equilibrium.observation import (
        CurrentLedger,
        IntegralObservation,
        MomentEnforcementError,
        MomentTargets,
    )
    from nova.equilibrium.observation_kernels import (
        InterpolationSupportReceipt,
        ObservationKernelReceipt,
        ThomsonSignals,
        VirtualMagneticSignals,
        synthesize_thomson,
        virtual_flux_loops,
        virtual_poloidal_probes,
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
    from nova.equilibrium.solve_request import (
        ExplicitSolveSeed,
        FORWARD_SOLVE_DEFAULTS,
        ForwardSolvePolicy,
        ForwardSolveReceipt,
        ForwardSolveRequest,
        ResolvedForwardSolveDefaults,
        declared_forward_solve_policy,
    )
    from nova.equilibrium.stencil_mesh import StencilMesh
    from nova.equilibrium.topology import TopologyClass
    from nova.equilibrium.source import (
        ContinuationForm,
        ContinuationLedger,
        ContinuationRecord,
        CurrentNormalisationError,
        DomainProfile,
        ForwardSource,
        NormalisationRecord,
        NormalisationPolicy,
        RotationClosure,
        RotationRecord,
        SeparatrixContinuity,
    )

#: Names loaded from their module on first attribute access, so importing the
#: package boundary never pulls in the optional jax extra.
_DEFERRED_EXPORTS: dict[str, str] = {
    "AdmissibilityCriterion": "branch_selection",
    "BranchAdmissibility": "branch_selection",
    "BranchAvailability": "branch_selection",
    "ConservationLedger": "conservation",
    "CircuitCurrentUnknown": "constraint",
    "CompensatingUnknown": "constraint",
    "ConstraintBinding": "constraint",
    "ConstraintContext": "constraint",
    "ConstraintFunctional": "constraint",
    "ConstraintMultiplier": "constraint",
    "ConstraintPair": "constraint",
    "ConstraintRecord": "constraint",
    "CompensatorRule": "constraint",
    "CompensatorSelection": "constraint",
    "IsofluxConstraint": "constraint",
    "IsofluxReference": "constraint",
    "WallGapConstraint": "constraint",
    "WallGapTarget": "constraint",
    "XPointConstraint": "constraint",
    "compensator_rule_name": "constraint",
    "constraint_response_matrix": "constraint",
    "derive_circuit_compensators": "constraint",
    "miller_boundary_points": "constraint",
    "sample_lattice_flux": "constraint",
    "select_compensating_directions": "constraint",
    "ContinuationForm": "source",
    "ContinuationLedger": "source",
    "ContinuationRecord": "source",
    "CurrentNormalisationError": "source",
    "ColdStartRule": "branch_selection",
    "ColdSeedConstruction": "forward",
    "ContinuedDomainProfile": "continuation",
    "CurrentLedger": "observation",
    "CurrentCentroidConstraint": "constraint",
    "CurrentIntegralSupport": "moment",
    "DomainMasks": "domain",
    "DomainProfile": "source",
    "DisappearanceCriterion": "branch_selection",
    "ExplicitSolveSeed": "solve_request",
    "FORWARD_SOLVE_DEFAULTS": "solve_request",
    "FluxLattice": "conservation",
    "FluxMesh": "conservation",
    "FluxSurfaceGeometry": "flux_surface_geometry",
    "ForwardEquilibrium": "forward",
    "ForwardPerturbedSeedReceipt": "forward",
    "ForwardBranchReceipt": "forward",
    "ForwardColdSeedPortfolio": "forward",
    "ForwardColdSeedReceipt": "forward",
    "ForwardFluxOperator": "forward_operator",
    "ForwardPortfolio": "forward",
    "ForwardProfile": "forward",
    "ForwardSolvePolicy": "solve_request",
    "ForwardSolveReceipt": "solve_request",
    "ForwardSolveRequest": "solve_request",
    "ForwardSource": "source",
    "GridMotion": "flux_surface_geometry",
    "GreenSourceRepresentation": "extraction_lattice",
    "IntegralObservation": "observation",
    "InterpolationSupportReceipt": "observation_kernels",
    "IsothermalRotation": "rotation",
    "Li2Geometry": "internal_inductance",
    "Li3Geometry": "internal_inductance",
    "ChordSamplingReceipt": "map_extraction",
    "MapCurrentReceipt": "map_extraction",
    "MomentEnforcementError": "observation",
    "MomentSeed": "moment",
    "MomentTargets": "observation",
    "NormalisationPolicy": "source",
    "NormalisationRecord": "source",
    "ObservationKernelReceipt": "observation_kernels",
    "PlasmaDomain": "domain",
    "PrescribedCurrentField": "forward_operator",
    "PerturbedSeedPolicy": "forward",
    "ProfileDegrees": "profile",
    "ProfilePrior": "profile",
    "ProfileResult": "profile",
    "ProfileAmplitudeUnknown": "constraint",
    "PredictedCurrentMoments": "moment",
    "ReconstructMoment": "moment",
    "ReconstructProfile": "profile",
    "ResolvedForwardSolveDefaults": "solve_request",
    "RotatingDomainProfile": "rotation",
    "RotationClosure": "source",
    "RotationRecord": "source",
    "SaddleSeedGeometry": "forward",
    "SeparatrixContinuation": "continuation",
    "SeparatrixContinuity": "source",
    "SeparatrixJumpError": "continuation",
    "SelectionHistory": "branch_selection",
    "SelectionPolicy": "branch_selection",
    "SelectionReason": "branch_selection",
    "SelectionReceipt": "branch_selection",
    "StencilMesh": "stencil_mesh",
    "SurfaceGeometryError": "flux_surface_geometry",
    "SurfaceExtractionReceipt": "map_extraction",
    "ThomsonSignals": "observation_kernels",
    "VacuumRegionReceipt": "map_extraction",
    "VirtualMagneticSignals": "observation_kernels",
    "TopologyClass": "topology",
    "apply_delta_star": "map_extraction",
    "convert_li_2_to_li_3": "internal_inductance",
    "convert_li_3_to_li_2": "internal_inductance",
    "declared_forward_solve_policy": "solve_request",
    "extract_flux_functions": "map_extraction",
    "evaluate_forward_equilibrium": "extraction_lattice",
    "extract_flux_surface_geometry": "flux_surface_extraction",
    "li_2_from_field_energy": "internal_inductance",
    "li_2_normaliser": "internal_inductance",
    "li_3_from_field_energy": "internal_inductance",
    "li_3_normaliser": "internal_inductance",
    "sample_chord_psi_norm": "map_extraction",
    "source_field_function": "flux_surface_geometry",
    "select_forward_branch": "branch_selection",
    "traced_assemble_flux_surface_geometry": "flux_surface_extraction",
    "traced_flux_surface_geometry": "flux_surface_extraction",
    "synthesize_thomson": "observation_kernels",
    "vacuum_region_receipt": "map_extraction",
    "virtual_flux_loops": "observation_kernels",
    "virtual_poloidal_probes": "observation_kernels",
}

__all__ = [
    "AdmissibilityCriterion",
    "BranchAdmissibility",
    "BranchAvailability",
    "DECAY_INDEX_WINDOW",
    "ChordSamplingReceipt",
    "ConservationLedger",
    "CircuitCurrentUnknown",
    "CompensatingUnknown",
    "ConstraintBinding",
    "ConstraintContext",
    "ConstraintFunctional",
    "ConstraintMultiplier",
    "ConstraintPair",
    "ConstraintRecord",
    "CompensatorRule",
    "CompensatorSelection",
    "IsofluxConstraint",
    "IsofluxReference",
    "WallGapConstraint",
    "WallGapTarget",
    "XPointConstraint",
    "compensator_rule_name",
    "constraint_response_matrix",
    "derive_circuit_compensators",
    "miller_boundary_points",
    "sample_lattice_flux",
    "select_compensating_directions",
    "ContinuationForm",
    "ContinuationLedger",
    "ContinuationRecord",
    "CurrentNormalisationError",
    "ColdStartRule",
    "ColdSeedConstruction",
    "ContinuedDomainProfile",
    "CurrentLedger",
    "CurrentCentroidConstraint",
    "CurrentIntegralSupport",
    "DomainMasks",
    "DomainProfile",
    "DisappearanceCriterion",
    "ExplicitSolveSeed",
    "FORWARD_SOLVE_DEFAULTS",
    "FluxLattice",
    "FluxMesh",
    "FluxSurfaceGeometry",
    "ForwardEquilibrium",
    "ForwardPerturbedSeedReceipt",
    "ForwardBranchReceipt",
    "ForwardColdSeedPortfolio",
    "ForwardColdSeedReceipt",
    "ForwardFluxOperator",
    "ForwardPortfolio",
    "ForwardProfile",
    "ForwardSolvePolicy",
    "ForwardSolveReceipt",
    "ForwardSolveRequest",
    "ForwardSource",
    "GridMotion",
    "GreenSourceRepresentation",
    "IntegralObservation",
    "InterpolationSupportReceipt",
    "IsothermalRotation",
    "Li2Geometry",
    "Li3Geometry",
    "MapCurrentReceipt",
    "MomentEnforcementError",
    "MomentSeed",
    "MomentTargets",
    "NormalisationPolicy",
    "NormalisationRecord",
    "ObservationKernelReceipt",
    "PlasmaDomain",
    "PrescribedCurrentField",
    "PerturbedSeedPolicy",
    "ProfileDegrees",
    "ProfilePrior",
    "ProfileResult",
    "ProfileAmplitudeUnknown",
    "PredictedCurrentMoments",
    "ReconstructMoment",
    "ReconstructProfile",
    "ResolvedForwardSolveDefaults",
    "RotatingDomainProfile",
    "RotationClosure",
    "RotationRecord",
    "SaddleSeedGeometry",
    "SeparatrixContinuation",
    "SeparatrixContinuity",
    "SeparatrixJumpError",
    "SelectionHistory",
    "SelectionPolicy",
    "SelectionReason",
    "SelectionReceipt",
    "StencilMesh",
    "SurfaceExtractionReceipt",
    "SurfaceGeometryError",
    "ThomsonSignals",
    "TopologyClass",
    "VacuumRegionReceipt",
    "VirtualMagneticSignals",
    "apply_delta_star",
    "convert_li_2_to_li_3",
    "convert_li_3_to_li_2",
    "decay_index",
    "declared_forward_solve_policy",
    "extract_flux_functions",
    "evaluate_forward_equilibrium",
    "extract_flux_surface_geometry",
    "li_2_from_field_energy",
    "li_2_normaliser",
    "li_3_from_field_energy",
    "li_3_normaliser",
    "sample_chord_psi_norm",
    "shafranov_vertical_field",
    "shafranov_vertical_field_elongated",
    "source_field_function",
    "select_forward_branch",
    "traced_assemble_flux_surface_geometry",
    "traced_flux_surface_geometry",
    "synthesize_thomson",
    "vacuum_region_receipt",
    "virtual_flux_loops",
    "virtual_poloidal_probes",
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
