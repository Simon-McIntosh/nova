"""Public deterministic transport ladder and coupled-window contract.

The package exposes all three fidelity rungs through stable entry points:

* :class:`CurrentDiffusion` is the dependency-free, differentiable native
  poloidal-flux diffusion rung.
* :class:`ForwardTransport` is the typed facade shared by the native rung and
  the embedded TORAX multi-channel rung.  Engine selection is explicit in
  :class:`TransportModel`, and every result records the selected rung in its
  provenance.
* :func:`solve_window` is the coupled equilibrium--transport rung.  Its
  :class:`Waveform` exchange carries the evolving physical coordinate map and
  returns convergence and conservation receipts; failure to converge or close
  the ledgers raises instead of returning a degraded state.

The single-side :func:`equilibrium_sweep` and :func:`transport_sweep` functions
remain public for callers that own their composition.  :class:`EnsembleForwardTransport`
batches identified members across the typed facade, while
:func:`forward_source_from_receipt` maps an evolved receipt back onto Nova's
equilibrium source seam.

Conventions: total poloidal flux ``Phi = 2 pi R A_phi`` [Wb], explicit ``mu0``,
raw SI throughout.
"""

from nova.equilibrium.flux_surface_extraction import (
    traced_assemble_flux_surface_geometry,
    traced_flux_surface_geometry,
)
from nova.transport.coupled_window import (
    CouplingFieldSpec,
    CouplingState,
    EquilibriumSweepReceipt,
    ExchangeSweepResult,
    TransportSweepReceipt,
    Waveform,
    WaveformSample,
    WindowConfig,
    WindowConservationError,
    WindowConservationReceipt,
    WindowConvergenceError,
    WindowConvergenceReceipt,
    WindowReceipt,
    equilibrium_sweep,
    implicit_window_state,
    solve_window,
    transport_sweep,
)
from nova.transport.current_diffusion import (
    NONNEGATIVE_EXPONENTS,
    CurrentDiffusion,
    EtaProfile,
    FluxSurfaceGeometry,
    basis_projection_images,
    diffuse_psi,
    ejima_coefficient,
    flux_budget,
    flux_surface_geometry,
    poloidal_field_energy_li,
    predicted_current,
    profile_shapes,
    project_coefficients,
)
from nova.transport.ensemble import (
    EnsembleForwardTransport,
    EnsembleMemberReceipt,
    EnsembleTransportInput,
    EnsembleTransportReceipt,
    EnsembleTransportState,
)
from nova.transport.evolved_state import (
    EvolvedFluxFunction,
    forward_source_from_receipt,
)
from nova.transport.forward import (
    AchievedBoundaryValues,
    FluxConsumptionLedger,
    ForwardTransport,
    ForwardTransportInput,
    ForwardTransportReceipt,
    PlasmaCurrentLedger,
    SolverDiagnostics,
    TransportEngineError,
    TransportGeometry,
    TransportModel,
    TransportProvenance,
    TransportRung,
    TransportState,
    TransportWaveforms,
)
from nova.transport.torax_geometry import torax_geometry_from_fsa
from nova.transport.window_batch import (
    BatchedCouplingState,
    BatchedExchangeSweepResult,
    BatchedWaveform,
    MemberArrayBatch,
    WindowBatchError,
    WindowBatchInput,
    WindowBatchReceipt,
    WindowMemberReceipt,
    WindowMemberRefusal,
    WindowRefusalReason,
    solve_window_batch,
)

__all__ = [
    "AchievedBoundaryValues",
    "BatchedCouplingState",
    "BatchedExchangeSweepResult",
    "BatchedWaveform",
    "CouplingFieldSpec",
    "CouplingState",
    "CurrentDiffusion",
    "EnsembleForwardTransport",
    "EnsembleMemberReceipt",
    "EnsembleTransportInput",
    "EnsembleTransportReceipt",
    "EnsembleTransportState",
    "EquilibriumSweepReceipt",
    "EtaProfile",
    "EvolvedFluxFunction",
    "ExchangeSweepResult",
    "FluxConsumptionLedger",
    "FluxSurfaceGeometry",
    "ForwardTransport",
    "ForwardTransportInput",
    "ForwardTransportReceipt",
    "MemberArrayBatch",
    "NONNEGATIVE_EXPONENTS",
    "PlasmaCurrentLedger",
    "SolverDiagnostics",
    "TransportEngineError",
    "TransportGeometry",
    "TransportModel",
    "TransportProvenance",
    "TransportRung",
    "TransportState",
    "TransportSweepReceipt",
    "TransportWaveforms",
    "Waveform",
    "WaveformSample",
    "WindowConfig",
    "WindowBatchError",
    "WindowBatchInput",
    "WindowBatchReceipt",
    "WindowConservationError",
    "WindowConservationReceipt",
    "WindowConvergenceError",
    "WindowConvergenceReceipt",
    "WindowReceipt",
    "WindowMemberReceipt",
    "WindowMemberRefusal",
    "WindowRefusalReason",
    "basis_projection_images",
    "diffuse_psi",
    "ejima_coefficient",
    "equilibrium_sweep",
    "flux_budget",
    "flux_surface_geometry",
    "forward_source_from_receipt",
    "implicit_window_state",
    "poloidal_field_energy_li",
    "predicted_current",
    "profile_shapes",
    "project_coefficients",
    "solve_window",
    "solve_window_batch",
    "torax_geometry_from_fsa",
    "traced_assemble_flux_surface_geometry",
    "traced_flux_surface_geometry",
    "transport_sweep",
]
