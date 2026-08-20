"""Import contract for Nova's public deterministic transport surface."""

import nova.transport as transport
from nova.transport import CurrentDiffusion, ForwardTransport, solve_window


EXPECTED_PUBLIC_NAMES = {
    "AchievedBoundaryValues",
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
    "WindowConservationError",
    "WindowConservationReceipt",
    "WindowConvergenceError",
    "WindowConvergenceReceipt",
    "WindowReceipt",
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
    "torax_geometry_from_fsa",
    "traced_assemble_flux_surface_geometry",
    "traced_flux_surface_geometry",
    "transport_sweep",
}


def test_every_declared_public_name_imports_from_transport_package():
    """The declared surface is complete and contains no missing bindings."""
    assert set(transport.__all__) == EXPECTED_PUBLIC_NAMES
    assert all(getattr(transport, name, None) is not None for name in transport.__all__)


def test_all_ladder_entry_points_import_from_transport_package():
    """Native, embedded-facade, and coupled entry points need no submodule path."""
    assert CurrentDiffusion is transport.CurrentDiffusion
    assert ForwardTransport is transport.ForwardTransport
    assert solve_window is transport.solve_window
