"""Public deterministic forward transport over Nova's fidelity ladder.

Both engines consume :class:`ForwardTransportInput` and return
:class:`ForwardTransportReceipt`.  Engine selection is explicit in the input
and repeated in the receipt provenance; engine failures are exceptions, never
requests to manufacture a replacement state.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.metadata import version
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from nova.transport.current_diffusion import (
    EtaProfile,
    FluxSurfaceGeometry,
    diffuse_psi,
    flux_budget,
)
from nova.transport.torax_geometry import torax_geometry_from_fsa


def _readonly_array(value: object) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            frozen[str(key)] = _freeze_mapping(item)
        elif isinstance(item, np.ndarray):
            frozen[str(key)] = _readonly_array(item)
        elif isinstance(item, list | tuple):
            frozen[str(key)] = tuple(copy.deepcopy(item))
        else:
            frozen[str(key)] = copy.deepcopy(item)
    return MappingProxyType(frozen)


def _thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    thawed: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            thawed[key] = _thaw_mapping(item)
        elif isinstance(item, np.ndarray):
            thawed[key] = np.array(item, copy=True)
        elif isinstance(item, tuple):
            thawed[key] = copy.deepcopy(list(item))
        else:
            thawed[key] = copy.deepcopy(item)
    return thawed


class TransportRung(StrEnum):
    """Declared physics fidelity selected for one forward interval."""

    NATIVE_PSI_DIFFUSION = "native-psi-diffusion"
    TORAX_MULTI_CHANNEL = "torax-multi-channel"


@dataclass(frozen=True)
class TransportGeometry:
    """Immutable flux-surface-average record shared by both engines."""

    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", _freeze_mapping(self.record))
        if not bool(self.record.get("valid", True)):
            raise ValueError("transport geometry record is not valid")

    def native_geometry(self) -> FluxSurfaceGeometry:
        """Project the shared record onto the native diffusion metrics."""
        array_fields = (
            "rho_face",
            "rho_cell",
            "psi_face",
            "psi_n_face",
            "psi_n_cell",
            "vpr_face",
            "vpr_cell",
            "g2_face",
            "g3_face",
            "g3_cell",
            "f_face",
            "f_cell",
            "b2_cell",
            "inv_r_cell",
            "q_face",
        )
        scalar_fields = (
            "phi_b",
            "r0",
            "ip_amperes",
            "axis_psi",
            "boundary_psi",
            "volume",
            "flux_sign",
        )
        return FluxSurfaceGeometry(
            **{
                name: np.array(self.record[name], dtype=np.float64, copy=True)
                for name in array_fields
            },
            **{name: float(self.record[name]) for name in scalar_fields},
        )


@dataclass(frozen=True)
class TransportState:
    """Flux-surface state at one instant, in raw SI except temperatures [keV]."""

    rho: np.ndarray
    psi: np.ndarray
    ion_temperature: np.ndarray
    electron_temperature: np.ndarray
    electron_density: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "rho",
            "psi",
            "ion_temperature",
            "electron_temperature",
            "electron_density",
        ):
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        shape = self.rho.shape
        if len(shape) != 1 or any(
            getattr(self, name).shape != shape
            for name in (
                "psi",
                "ion_temperature",
                "electron_temperature",
                "electron_density",
            )
        ):
            raise ValueError("all transport state channels must share one 1D grid")
        if self.rho.size < 3 or not np.all(np.diff(self.rho) > 0.0):
            raise ValueError("transport state rho must be strictly increasing")
        if self.rho[0] < 0.0 or self.rho[-1] > 1.0:
            raise ValueError("transport state rho must lie in [0, 1]")


@dataclass(frozen=True)
class TransportWaveforms:
    """Boundary waveforms prescribed across one transport interval."""

    time: np.ndarray
    plasma_current: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _readonly_array(self.time))
        object.__setattr__(self, "plasma_current", _readonly_array(self.plasma_current))
        if self.time.ndim != 1 or self.time.size < 2:
            raise ValueError("transport waveform needs at least two time samples")
        if self.plasma_current.shape != self.time.shape:
            raise ValueError("plasma-current waveform must match the time grid")
        if not np.all(np.diff(self.time) > 0.0):
            raise ValueError("transport waveform time must be strictly increasing")


@dataclass(frozen=True)
class TransportModel:
    """Declared engine and its closure configuration."""

    rung: TransportRung
    eta: EtaProfile = field(default_factory=EtaProfile)
    theta: float = 1.0
    torax_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rung", TransportRung(self.rung))
        object.__setattr__(self, "torax_config", _freeze_mapping(self.torax_config))
        if not 0.5 <= self.theta <= 1.0:
            raise ValueError("native theta must lie in [0.5, 1.0]")
        if self.rung is TransportRung.TORAX_MULTI_CHANNEL and not self.torax_config:
            raise ValueError("the TORAX rung requires an explicit TORAX configuration")


@dataclass(frozen=True)
class ForwardTransportInput:
    """Typed immutable input shared by every transport rung."""

    geometry: TransportGeometry
    initial_state: TransportState
    waveforms: TransportWaveforms
    model: TransportModel


@dataclass(frozen=True)
class FluxConsumptionLedger:
    """Surface-flux swing decomposed into resistive and internal channels."""

    boundary: float
    resistive: float
    internal: float
    mean_axis_voltage: float
    mean_boundary_voltage: float


@dataclass(frozen=True)
class PlasmaCurrentLedger:
    """Requested and achieved current at both ends of the interval."""

    requested_initial: float
    requested_final: float
    achieved_initial: float
    achieved_final: float


@dataclass(frozen=True)
class AchievedBoundaryValues:
    """Final values at the last radial sample."""

    psi: float
    plasma_current: float
    ion_temperature: float
    electron_temperature: float
    electron_density: float


@dataclass(frozen=True)
class SolverDiagnostics:
    """Uniform diagnostics carried by either engine."""

    engine_status: str
    steps: int
    outer_iterations: int
    inner_iterations: int


@dataclass(frozen=True)
class TransportProvenance:
    """Engine identity and selected ladder rung."""

    rung: TransportRung
    engine: str
    engine_version: str


@dataclass(frozen=True)
class ForwardTransportReceipt:
    """Evolved state plus conservation, boundary and numerical receipts."""

    state: TransportState
    flux_consumption: FluxConsumptionLedger
    plasma_current: PlasmaCurrentLedger
    boundary: AchievedBoundaryValues
    diagnostics: SolverDiagnostics
    provenance: TransportProvenance


class TransportEngineError(RuntimeError):
    """A selected transport engine failed to produce a valid state."""


class ForwardTransport:
    """One public deterministic forward solve over the transport ladder."""

    def solve(self, inputs: ForwardTransportInput) -> ForwardTransportReceipt:
        """Advance one interval with the explicitly selected fidelity rung."""
        if inputs.model.rung is TransportRung.NATIVE_PSI_DIFFUSION:
            return _solve_native(inputs)
        if inputs.model.rung is TransportRung.TORAX_MULTI_CHANNEL:
            return _solve_torax(inputs)
        raise ValueError(f"unsupported transport rung: {inputs.model.rung}")


def _solve_native(inputs: ForwardTransportInput) -> ForwardTransportReceipt:
    geometry = inputs.geometry.native_geometry()
    if inputs.initial_state.psi.shape != geometry.rho_face.shape:
        raise ValueError("native psi state must use the geometry face grid")
    step = diffuse_psi(
        geometry,
        inputs.model.eta,
        t_grid=inputs.waveforms.time,
        ip_of_t=inputs.waveforms.plasma_current,
        psi0_face=inputs.initial_state.psi,
        theta=inputs.model.theta,
    )
    final_psi = np.asarray(step["psi_face"][-1])
    state = TransportState(
        rho=inputs.initial_state.rho,
        psi=final_psi,
        ion_temperature=inputs.initial_state.ion_temperature,
        electron_temperature=inputs.initial_state.electron_temperature,
        electron_density=inputs.initial_state.electron_density,
    )
    budget = flux_budget(step, geometry)
    initial_current = float(geometry.enclosed_current(inputs.initial_state.psi)[-1])
    final_current = float(geometry.enclosed_current(final_psi)[-1])
    return ForwardTransportReceipt(
        state=state,
        flux_consumption=FluxConsumptionLedger(
            boundary=budget["d_psi_bdry"],
            resistive=budget["d_psi_axis"],
            internal=budget["d_psi_internal"],
            mean_axis_voltage=budget["v_axis_mean"],
            mean_boundary_voltage=budget["v_bdry_mean"],
        ),
        plasma_current=PlasmaCurrentLedger(
            requested_initial=float(inputs.waveforms.plasma_current[0]),
            requested_final=float(inputs.waveforms.plasma_current[-1]),
            achieved_initial=initial_current,
            achieved_final=final_current,
        ),
        boundary=AchievedBoundaryValues(
            psi=float(state.psi[-1]),
            plasma_current=final_current,
            ion_temperature=float(state.ion_temperature[-1]),
            electron_temperature=float(state.electron_temperature[-1]),
            electron_density=float(state.electron_density[-1]),
        ),
        diagnostics=SolverDiagnostics(
            engine_status="converged",
            steps=inputs.waveforms.time.size - 1,
            outer_iterations=inputs.waveforms.time.size - 1,
            inner_iterations=inputs.waveforms.time.size - 1,
        ),
        provenance=TransportProvenance(
            rung=TransportRung.NATIVE_PSI_DIFFUSION,
            engine="nova.current_diffusion",
            engine_version="1",
        ),
    )


def _profile(coordinate: np.ndarray, values: np.ndarray, time: float) -> dict:
    return {
        float(time): {
            float(position): float(value)
            for position, value in zip(coordinate, values, strict=True)
        }
    }


def _prepare_torax_config(inputs: ForwardTransportInput):
    from torax._src.geometry.geometry_provider import ConstantGeometryProvider
    from torax._src.torax_pydantic.model_config import ToraxConfig

    config_data = _thaw_mapping(inputs.model.torax_config)
    n_rho = int(np.asarray(inputs.geometry.record["rho_face"]).size - 1)
    config_data["geometry"] = {"geometry_type": "circular", "n_rho": n_rho}

    initial = inputs.initial_state
    start = float(inputs.waveforms.time[0])
    profile_conditions = dict(config_data.get("profile_conditions", {}))
    profile_conditions.update(
        {
            "Ip": {
                float(time): float(current)
                for time, current in zip(
                    inputs.waveforms.time,
                    inputs.waveforms.plasma_current,
                    strict=True,
                )
            },
            "T_i": _profile(initial.rho, initial.ion_temperature, start),
            "T_e": _profile(initial.rho, initial.electron_temperature, start),
            "n_e": _profile(initial.rho, initial.electron_density, start),
            "psi": _profile(initial.rho, initial.psi, start),
            "initial_psi_mode": "profile_conditions",
        }
    )
    config_data["profile_conditions"] = profile_conditions

    numerics = dict(config_data.get("numerics", {}))
    numerics.update(
        {
            "t_initial": start,
            "t_final": float(inputs.waveforms.time[-1]),
            "exact_t_final": True,
            "evolve_current": True,
            "evolve_density": True,
            "evolve_electron_heat": True,
            "evolve_ion_heat": True,
        }
    )
    config_data["numerics"] = numerics

    config = ToraxConfig.from_dict(config_data)
    geometry = dataclasses.replace(
        torax_geometry_from_fsa(inputs.geometry.record), Ip_from_parameters=True
    )
    config.geometry.__dict__["build_provider"] = ConstantGeometryProvider(geometry)
    return config


def _run_torax_simulation(config):
    from torax._src.orchestration.run_simulation import run_simulation

    return run_simulation(config, progress_bar=False)


def _solve_torax(inputs: ForwardTransportInput) -> ForwardTransportReceipt:
    output, history = _run_torax_simulation(_prepare_torax_config(inputs))
    if history.sim_error.name != "NO_ERROR":
        raise TransportEngineError(
            f"TORAX failed with simulation status {history.sim_error.name}"
        )

    profiles = output.children["profiles"].dataset
    scalars = output.children["scalars"].dataset
    numerics = output.children["numerics"].dataset
    psi_history = np.asarray(profiles["psi"], dtype=np.float64)
    rho = np.asarray(output.coords["rho_norm"], dtype=np.float64)
    ion_temperature = np.asarray(profiles["T_i"][-1], dtype=np.float64)
    electron_temperature = np.asarray(profiles["T_e"][-1], dtype=np.float64)
    electron_density = np.asarray(profiles["n_e"][-1], dtype=np.float64)
    current_history = np.asarray(scalars["Ip"], dtype=np.float64)
    state = TransportState(
        rho=rho,
        psi=psi_history[-1],
        ion_temperature=ion_temperature,
        electron_temperature=electron_temperature,
        electron_density=electron_density,
    )
    boundary_swing = float(psi_history[-1, -1] - psi_history[0, -1])
    axis_swing = float(psi_history[-1, 0] - psi_history[0, 0])
    elapsed = float(history.times[-1] - history.times[0])
    mean_axis_voltage = axis_swing / elapsed if elapsed else 0.0
    mean_boundary_voltage = boundary_swing / elapsed if elapsed else 0.0
    outer_iterations = int(np.asarray(numerics["outer_solver_iterations"]).sum())
    inner_iterations = int(np.asarray(numerics["inner_solver_iterations"]).sum())
    return ForwardTransportReceipt(
        state=state,
        flux_consumption=FluxConsumptionLedger(
            boundary=boundary_swing,
            resistive=axis_swing,
            internal=boundary_swing - axis_swing,
            mean_axis_voltage=mean_axis_voltage,
            mean_boundary_voltage=mean_boundary_voltage,
        ),
        plasma_current=PlasmaCurrentLedger(
            requested_initial=float(inputs.waveforms.plasma_current[0]),
            requested_final=float(inputs.waveforms.plasma_current[-1]),
            achieved_initial=float(current_history[0]),
            achieved_final=float(current_history[-1]),
        ),
        boundary=AchievedBoundaryValues(
            psi=float(state.psi[-1]),
            plasma_current=float(current_history[-1]),
            ion_temperature=float(state.ion_temperature[-1]),
            electron_temperature=float(state.electron_temperature[-1]),
            electron_density=float(state.electron_density[-1]),
        ),
        diagnostics=SolverDiagnostics(
            engine_status=history.sim_error.name,
            steps=len(history.times) - 1,
            outer_iterations=outer_iterations,
            inner_iterations=inner_iterations,
        ),
        provenance=TransportProvenance(
            rung=TransportRung.TORAX_MULTI_CHANNEL,
            engine="torax",
            engine_version=version("torax"),
        ),
    )


__all__ = [
    "AchievedBoundaryValues",
    "FluxConsumptionLedger",
    "ForwardTransport",
    "ForwardTransportInput",
    "ForwardTransportReceipt",
    "PlasmaCurrentLedger",
    "SolverDiagnostics",
    "TransportEngineError",
    "TransportGeometry",
    "TransportModel",
    "TransportProvenance",
    "TransportRung",
    "TransportState",
    "TransportWaveforms",
]
