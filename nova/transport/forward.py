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

from nova.biot.greens import MU0
from nova.transport.current_diffusion import (
    EtaProfile,
    FluxSurfaceGeometry,
    diffuse_psi,
    flux_budget,
)
from nova.transport.torax_geometry import torax_geometry_from_fsa


def _readonly_array(value: object) -> np.ndarray:
    if hasattr(value, "aval"):
        return value
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
        if not hasattr(self.rho, "aval"):
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
    resistivity_multiplier: Any = 1.0

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

    def __init__(self) -> None:
        self._native_batch_functions: dict[tuple[Any, ...], Any] = {}

    def solve(self, inputs: ForwardTransportInput) -> ForwardTransportReceipt:
        """Advance one interval with the explicitly selected fidelity rung."""
        if inputs.model.rung is TransportRung.NATIVE_PSI_DIFFUSION:
            return _solve_native(inputs)
        if inputs.model.rung is TransportRung.TORAX_MULTI_CHANNEL:
            return _solve_torax(inputs)
        raise ValueError(f"unsupported transport rung: {inputs.model.rung}")

    def solve_state_batch(
        self,
        geometry: TransportGeometry,
        waveforms: TransportWaveforms,
        model: TransportModel,
        state_channels: tuple[Any, Any, Any, Any, Any],
        *,
        jit: bool = False,
    ) -> tuple[Any, Any, Any]:
        """Advance a leading member axis inside one traced computation.

        The host-facing dataclasses stay outside this boundary.  The returned
        arrays carry evolved state channels, scalar receipt values, and integer
        diagnostics respectively, each with the member axis leading.
        """
        if model.rung is TransportRung.NATIVE_PSI_DIFFUSION:
            key = (
                id(geometry),
                id(waveforms),
                model.eta.eta0,
                model.eta.contrast,
                model.eta.shape,
                model.theta,
                jit,
            )
            mapped = self._native_batch_functions.get(key)
            if mapped is None:
                mapped = _native_batch_function(geometry, waveforms, model, jit=jit)
                self._native_batch_functions[key] = mapped
            return mapped(
                model.resistivity_multiplier,
                *(state_channels),
            )
        if model.rung is TransportRung.TORAX_MULTI_CHANNEL:
            return _solve_torax_batch(
                geometry, waveforms, model, state_channels, jit=jit
            )
        raise ValueError(f"unsupported transport rung: {model.rung}")


def _native_enclosed_current(geometry: TransportGeometry, psi):
    import jax.numpy as jnp

    record = geometry.record
    rho_face = jnp.asarray(record["rho_face"], dtype=jnp.float64)
    g2_face = jnp.asarray(record["g2_face"], dtype=jnp.float64)
    g3_face = jnp.asarray(record["g3_face"], dtype=jnp.float64)
    f_face = jnp.asarray(record["f_face"], dtype=jnp.float64)
    d_face = (
        jnp.zeros_like(rho_face).at[1:].set(g2_face[1:] * g3_face[1:] / rho_face[1:])
    )
    d_mid = 0.5 * (d_face[:-1] + d_face[1:])
    f_mid = 0.5 * (f_face[:-1] + f_face[1:])
    drho = rho_face[1] - rho_face[0]
    i_mid = (
        jnp.asarray(record["flux_sign"], dtype=jnp.float64)
        * d_mid
        * (jnp.diff(psi) / drho)
        * f_mid
        / (jnp.asarray(record["phi_b"], dtype=jnp.float64) * 16.0 * jnp.pi**3 * MU0)
    )
    return jnp.concatenate(
        [
            jnp.zeros(1, dtype=psi.dtype),
            0.5 * (i_mid[:-1] + i_mid[1:]),
            (1.5 * i_mid[-1] - 0.5 * i_mid[-2])[None],
        ]
    )


def _native_batch_function(geometry, waveforms, model, *, jit: bool):
    import jax
    import jax.numpy as jnp

    from nova.transport.current_diffusion import _diffuse_scan

    record = geometry.record
    rho_face = jnp.asarray(record["rho_face"], dtype=jnp.float64)
    rho_cell = jnp.asarray(record["rho_cell"], dtype=jnp.float64)
    psi_n_cell = jnp.asarray(record["psi_n_cell"], dtype=jnp.float64)
    g2_face = jnp.asarray(record["g2_face"], dtype=jnp.float64)
    g3_face = jnp.asarray(record["g3_face"], dtype=jnp.float64)
    f_face = jnp.asarray(record["f_face"], dtype=jnp.float64)
    f_cell = jnp.asarray(record["f_cell"], dtype=jnp.float64)
    phi_b = jnp.asarray(record["phi_b"], dtype=jnp.float64)
    flux_sign = jnp.asarray(record["flux_sign"], dtype=jnp.float64)
    d_face = (
        jnp.zeros_like(rho_face).at[1:].set(g2_face[1:] * g3_face[1:] / rho_face[1:])
    )
    d_mid = 0.5 * (d_face[:-1] + d_face[1:])
    drho = rho_face[1] - rho_face[0]
    eta = jnp.asarray(model.eta.eta0, dtype=jnp.float64) * jnp.exp(
        jnp.asarray(model.eta.contrast, dtype=jnp.float64)
        * jnp.clip(psi_n_cell, 0.0, 1.0)
        ** jnp.asarray(model.eta.shape, dtype=jnp.float64)
    )
    toc_cell = (1.0 / eta) * MU0 * 16.0 * jnp.pi**2 * phi_b**2 * rho_cell / f_cell**2
    toc_face = jnp.concatenate(
        [
            toc_cell[:1],
            0.5 * (toc_cell[:-1] + toc_cell[1:]),
            toc_cell[-1:],
        ]
    )
    times = jnp.asarray(waveforms.time, dtype=jnp.float64)
    requested_current = jnp.asarray(waveforms.plasma_current, dtype=jnp.float64)
    intervals = jnp.diff(times)
    gradients = (
        flux_sign
        * requested_current[1:]
        * 16.0
        * jnp.pi**3
        * MU0
        * phi_b
        / (d_face[-1] * f_face[-1])
    )

    def solve_member(
        multiplier, rho, psi, ion_temperature, electron_temperature, density
    ):
        scaled_toc_face = toc_face / jnp.asarray(multiplier, dtype=jnp.float64)
        psi_history, axis_voltage, boundary_voltage = _diffuse_scan(
            psi,
            d_face,
            d_mid,
            scaled_toc_face,
            drho,
            intervals,
            gradients,
            model.theta,
        )
        final_psi = psi_history[-1]
        initial_current = _native_enclosed_current(geometry, psi)[-1]
        final_current = _native_enclosed_current(geometry, final_psi)[-1]
        boundary_swing = final_psi[-1] - psi[-1]
        axis_swing = final_psi[0] - psi[0]
        state_values = jnp.stack(
            (rho, final_psi, ion_temperature, electron_temperature, density)
        )
        receipt_values = jnp.stack(
            (
                boundary_swing,
                axis_swing,
                boundary_swing - axis_swing,
                jnp.mean(axis_voltage),
                jnp.mean(boundary_voltage),
                requested_current[0],
                requested_current[-1],
                initial_current,
                final_current,
                final_psi[-1],
                final_current,
                ion_temperature[-1],
                electron_temperature[-1],
                density[-1],
            )
        )
        steps = intervals.size
        diagnostics = jnp.full((3,), steps, dtype=jnp.int32)
        return state_values, receipt_values, diagnostics

    mapped = jax.vmap(solve_member, in_axes=(None, 0, 0, 0, 0, 0))
    if jit:
        mapped = jax.jit(mapped)
    return mapped


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


def _make_torax_step(config):
    from torax._src.orchestration.run_simulation import make_step_fn

    return make_step_fn(config)


def _with_resistivity_multiplier(provider, resistivity_multiplier):
    import jax.numpy as jnp
    from torax._src.torax_pydantic.interpolated_param_1d import (
        TimeVaryingScalarUpdate,
    )

    multiplier = jnp.asarray(resistivity_multiplier, dtype=jnp.float64)
    base_multiplier = provider.numerics.resistivity_multiplier
    return provider.update_provider(
        lambda current: (current.numerics.resistivity_multiplier,),
        (
            TimeVaryingScalarUpdate(
                value=jnp.full_like(base_multiplier.value, multiplier)
            ),
        ),
    )


@dataclass(frozen=True)
class _ToraxStepExecution:
    states: tuple[Any, ...]
    error_name: str


def _run_torax_steps(config, step_durations, resistivity_multiplier):
    import jax
    import jax.numpy as jnp
    from torax._src.orchestration.initial_state import (
        get_initial_state_and_post_processed_outputs,
    )

    step_fn = _make_torax_step(config)
    multiplier = jnp.asarray(resistivity_multiplier, dtype=jnp.float64)
    runtime_params = _with_resistivity_multiplier(
        step_fn.runtime_params_provider, multiplier
    )
    initial_state, post_processed = get_initial_state_and_post_processed_outputs(
        step_fn,
        runtime_params_overrides=runtime_params,
    )
    states = [initial_state]
    current_state = initial_state
    error_name = "NO_ERROR"
    tracing = isinstance(multiplier, jax.core.Tracer)
    for duration in step_durations:
        remaining = float(duration)
        if tracing:
            fixed_dt = float(np.min(runtime_params.numerics.fixed_dt.value))
            if runtime_params.numerics.adaptive_dt or fixed_dt < remaining:
                raise ValueError(
                    "TORAX gradients require non-adaptive waveform intervals no "
                    "longer than the configured fixed step"
                )
        while remaining > 1.0e-12:
            previous_time = current_state.t
            current_state, post_processed = step_fn(
                current_state,
                post_processed,
                max_dt=jnp.asarray(remaining, dtype=jnp.float64),
                runtime_params_overrides=runtime_params,
            )
            states.append(current_state)
            if tracing:
                break
            error_name = step_fn.check_for_errors(current_state, post_processed).name
            if error_name != "NO_ERROR":
                break
            elapsed = float(current_state.t - previous_time)
            if elapsed <= 0.0:
                error_name = "NON_ADVANCING_STEP"
                break
            remaining -= elapsed
        if error_name != "NO_ERROR":
            break
    return _ToraxStepExecution(states=tuple(states), error_name=error_name)


def _fixed_step_durations(provider, waveform_times) -> tuple[float, ...]:
    if bool(provider.numerics.adaptive_dt):
        raise ValueError("TORAX ensemble batching requires non-adaptive fixed steps")
    fixed_dt = float(np.min(np.asarray(provider.numerics.fixed_dt.value)))
    if fixed_dt <= 0.0:
        raise ValueError("TORAX fixed step must be positive")
    durations = []
    for interval in np.diff(np.asarray(waveform_times, dtype=np.float64)):
        remaining = float(interval)
        while remaining > 1.0e-12:
            duration = min(remaining, fixed_dt)
            durations.append(duration)
            remaining -= duration
    return tuple(durations)


def _solve_torax_batch(geometry, waveforms, model, state_channels, *, jit: bool):
    import jax
    import jax.numpy as jnp
    from torax._src.orchestration.initial_state import (
        get_initial_state_and_post_processed_outputs,
    )

    member_count = int(np.shape(state_channels[0])[0])
    member_steps = []
    providers = []
    for index in range(member_count):
        member_state = TransportState(
            rho=np.asarray(state_channels[0][index]),
            psi=np.asarray(state_channels[1][index]),
            ion_temperature=np.asarray(state_channels[2][index]),
            electron_temperature=np.asarray(state_channels[3][index]),
            electron_density=np.asarray(state_channels[4][index]),
        )
        member_input = ForwardTransportInput(
            geometry=geometry,
            initial_state=member_state,
            waveforms=waveforms,
            model=model,
        )
        step_fn = _make_torax_step(_prepare_torax_config(member_input))
        member_steps.append(step_fn)
        providers.append(
            _with_resistivity_multiplier(
                step_fn.runtime_params_provider, model.resistivity_multiplier
            )
        )

    step_fn = member_steps[0]
    step_durations = _fixed_step_durations(
        step_fn.runtime_params_provider, waveforms.time
    )
    stacked_providers = jax.tree.map(lambda *values: jnp.stack(values), *providers)
    requested_current = jnp.asarray(waveforms.plasma_current, dtype=jnp.float64)

    def solve_member(provider):
        initial, post_processed = get_initial_state_and_post_processed_outputs(
            step_fn,
            runtime_params_overrides=provider,
        )
        current = initial
        outer_iterations = jnp.zeros((), dtype=jnp.int32)
        inner_iterations = jnp.zeros((), dtype=jnp.int32)
        for duration in step_durations:
            current, post_processed = step_fn(
                current,
                post_processed,
                max_dt=jnp.asarray(duration, dtype=jnp.float64),
                runtime_params_overrides=provider,
            )
            outer_iterations += current.solver_numeric_outputs.outer_solver_iterations
            inner_iterations += current.solver_numeric_outputs.inner_solver_iterations

        initial_profiles = initial.core_profiles
        final_profiles = current.core_profiles
        initial_psi = initial_profiles.psi.cell_plus_boundaries()
        final_psi = final_profiles.psi.cell_plus_boundaries()
        rho = jnp.concatenate(
            (
                jnp.zeros(1, dtype=current.geometry.rho_norm.dtype),
                current.geometry.rho_norm,
                jnp.ones(1, dtype=current.geometry.rho_norm.dtype),
            )
        )
        ion_temperature = final_profiles.T_i.cell_plus_boundaries()
        electron_temperature = final_profiles.T_e.cell_plus_boundaries()
        electron_density = final_profiles.n_e.cell_plus_boundaries()
        initial_current = initial_profiles.Ip_profile_face[-1]
        final_current = final_profiles.Ip_profile_face[-1]
        boundary_swing = final_psi[-1] - initial_psi[-1]
        axis_swing = final_psi[0] - initial_psi[0]
        elapsed = current.t - initial.t
        state_values = jnp.stack(
            (
                rho,
                final_psi,
                ion_temperature,
                electron_temperature,
                electron_density,
            )
        )
        receipt_values = jnp.stack(
            (
                boundary_swing,
                axis_swing,
                boundary_swing - axis_swing,
                axis_swing / elapsed,
                boundary_swing / elapsed,
                requested_current[0],
                requested_current[-1],
                initial_current,
                final_current,
                final_psi[-1],
                final_current,
                ion_temperature[-1],
                electron_temperature[-1],
                electron_density[-1],
            )
        )
        diagnostics = jnp.asarray(
            (len(step_durations), outer_iterations, inner_iterations),
            dtype=jnp.int32,
        )
        return (
            state_values,
            receipt_values,
            diagnostics,
            current,
            post_processed,
        )

    mapped = jax.vmap(solve_member)
    if jit:
        mapped = jax.jit(mapped)
    (
        state_values,
        receipt_values,
        diagnostic_values,
        final_states,
        final_post_processed,
    ) = mapped(stacked_providers)

    if not isinstance(state_values, jax.core.Tracer):
        for index in range(member_count):
            member_state = jax.tree.map(lambda value: value[index], final_states)
            member_post_processed = jax.tree.map(
                lambda value: value[index], final_post_processed
            )
            error_name = step_fn.check_for_errors(
                member_state, member_post_processed
            ).name
            if error_name != "NO_ERROR":
                raise TransportEngineError(
                    f"TORAX failed with simulation status {error_name}"
                )
    return state_values, receipt_values, diagnostic_values


def _solve_torax(inputs: ForwardTransportInput) -> ForwardTransportReceipt:
    execution = _run_torax_steps(
        _prepare_torax_config(inputs),
        np.diff(inputs.waveforms.time),
        inputs.model.resistivity_multiplier,
    )
    if execution.error_name != "NO_ERROR":
        raise TransportEngineError(
            f"TORAX failed with simulation status {execution.error_name}"
        )

    import jax.numpy as jnp

    initial = execution.states[0]
    final = execution.states[-1]
    initial_profiles = initial.core_profiles
    final_profiles = final.core_profiles
    initial_psi = initial_profiles.psi.cell_plus_boundaries()
    final_psi = final_profiles.psi.cell_plus_boundaries()
    rho = jnp.concatenate(
        [
            jnp.zeros(1, dtype=final.geometry.rho_norm.dtype),
            final.geometry.rho_norm,
            jnp.ones(1, dtype=final.geometry.rho_norm.dtype),
        ]
    )
    ion_temperature = final_profiles.T_i.cell_plus_boundaries()
    electron_temperature = final_profiles.T_e.cell_plus_boundaries()
    electron_density = final_profiles.n_e.cell_plus_boundaries()
    initial_current = initial_profiles.Ip_profile_face[-1]
    final_current = final_profiles.Ip_profile_face[-1]
    state = TransportState(
        rho=rho,
        psi=final_psi,
        ion_temperature=ion_temperature,
        electron_temperature=electron_temperature,
        electron_density=electron_density,
    )
    boundary_swing = final_psi[-1] - initial_psi[-1]
    axis_swing = final_psi[0] - initial_psi[0]
    elapsed = final.t - initial.t
    mean_axis_voltage = axis_swing / elapsed
    mean_boundary_voltage = boundary_swing / elapsed
    outer_iterations = sum(
        state.solver_numeric_outputs.outer_solver_iterations
        for state in execution.states[1:]
    )
    inner_iterations = sum(
        state.solver_numeric_outputs.inner_solver_iterations
        for state in execution.states[1:]
    )
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
            achieved_initial=initial_current,
            achieved_final=final_current,
        ),
        boundary=AchievedBoundaryValues(
            psi=state.psi[-1],
            plasma_current=final_current,
            ion_temperature=state.ion_temperature[-1],
            electron_temperature=state.electron_temperature[-1],
            electron_density=state.electron_density[-1],
        ),
        diagnostics=SolverDiagnostics(
            engine_status=execution.error_name,
            steps=len(execution.states) - 1,
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
