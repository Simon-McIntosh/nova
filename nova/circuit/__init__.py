"""Circuit-representation solvers: conductors carried as L/R eigenmodes.

The circuit tier of the physics spine.  A conductor set (vessel shells, coil
cases, in-vessel structures) is reduced to a linear inductance/resistance
system whose eigenmodes carry the machine's magnetic memory: the flux swing a
coil or plasma history induces decays on the L/R times of those modes, which is
precisely the state a per-slice equilibrium fit cannot see.

Models here are inherently temporal, so they carry the domain word directly:

* :class:`~nova.circuit.passive.PassiveCircuit` -- the passive structure's
  L/R eigenmode system, its data-led resistance calibration, and exact
  zero-order-hold propagation of the mode state along a measured drive history.

Inductance comes from the axisymmetric finite-section kernels in
:mod:`nova.biot.greens` / :mod:`nova.biot.polygon`; resistance from the true
conductor cross-sections at a nominal resistivity, calibrated against
coil-only (vacuum) intervals.  Conventions: total poloidal flux
``Phi = 2 pi R A_phi`` [Wb], explicit ``mu0``, raw SI throughout.
"""

from nova.circuit.conductor import ConductorSet, PolygonSection, SensorSet
from nova.circuit.passive import (
    NOMINAL_STEEL_RESISTIVITY,
    PassiveCircuit,
    PassiveCircuitSystem,
    PassiveEigenbasis,
    build_passive_circuit_system,
    build_passive_eigenbasis,
    reduce_passive_system,
)
from nova.circuit.propagate import (
    integrate_eddy_ode,
    scan_eddy_modes,
    zoh_mode_response,
)

__all__ = [
    "NOMINAL_STEEL_RESISTIVITY",
    "ConductorSet",
    "PassiveCircuit",
    "PassiveCircuitSystem",
    "PassiveEigenbasis",
    "PolygonSection",
    "SensorSet",
    "build_passive_circuit_system",
    "build_passive_eigenbasis",
    "integrate_eddy_ode",
    "reduce_passive_system",
    "scan_eddy_modes",
    "zoh_mode_response",
]
