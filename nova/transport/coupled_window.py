"""Pure single-sweep primitives for equilibrium--transport exchange.

The two solvers in a coupled window do not normally share a time grid.  A
``Waveform`` therefore owns both its time samples and the normalised radial
grid used by every slice.  The physical coordinate map travels with those
samples: boundary toroidal flux, axis reference, and boundary reference are
interpolated beside the profiles rather than being reconstructed by a
consumer.

Radial profiles are first evaluated on a common *normalised* radial grid in
each bracketing slice and only then interpolated in time.  This ordering is
load-bearing when the physical flux map evolves: interpolating samples at a
fixed physical-flux index would mix different flux surfaces.

``solve_window`` composes the pure side sweeps into a damped waveform fixed
point.  It reports the observed contraction and every exchanged-field
residual, aggregates the interval conservation ledgers, and raises if either
convergence or conservation misses the declared tolerance.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from nova.equilibrium.forward import ForwardEquilibrium, ForwardProfile, SolveRoute
from nova.equilibrium.source import ForwardSource
from nova.transport.forward import (
    FluxConsumptionLedger,
    ForwardTransport,
    ForwardTransportInput,
    ForwardTransportReceipt,
    PlasmaCurrentLedger,
    TransportGeometry,
    TransportModel,
    TransportState,
    TransportWaveforms,
)

__all__ = [
    "EquilibriumSweepReceipt",
    "ExchangeSweepResult",
    "TransportSweepReceipt",
    "Waveform",
    "WaveformSample",
    "WindowConfig",
    "WindowConservationError",
    "WindowConservationReceipt",
    "WindowConvergenceError",
    "WindowConvergenceReceipt",
    "WindowReceipt",
    "equilibrium_sweep",
    "solve_window",
    "transport_sweep",
]


def _readonly(value: object, *, dtype=None) -> np.ndarray:
    """Return an owned, read-only array."""
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _finite(name: str, value: np.ndarray) -> None:
    """Reject a non-finite numeric waveform field."""
    if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class WaveformSample:
    """One immutable waveform evaluation on a normalised radial grid."""

    time: float
    radial_grid: np.ndarray
    phi_boundary: float
    axis_reference: float
    boundary_reference: float
    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        radial_grid = _readonly(self.radial_grid, dtype=np.float64)
        if radial_grid.ndim != 1 or radial_grid.size < 2:
            raise ValueError("a waveform sample needs at least two radial points")
        if not np.all(np.diff(radial_grid) > 0.0):
            raise ValueError("waveform sample radial grid must be strictly increasing")
        if radial_grid[0] < 0.0 or radial_grid[-1] > 1.0:
            raise ValueError("waveform sample radial grid must lie in [0, 1]")
        frozen_values: dict[str, np.ndarray] = {}
        for name, value in self.values.items():
            array = _readonly(value)
            _finite(f"waveform sample channel {name}", array)
            frozen_values[str(name)] = array
        object.__setattr__(self, "radial_grid", radial_grid)
        object.__setattr__(self, "values", MappingProxyType(frozen_values))

    def geometry(self) -> TransportGeometry:
        """Return this sample as an immutable transport-geometry record."""
        record = {
            name: np.array(value, copy=True) for name, value in self.values.items()
        }
        record.update(
            {
                "rho_face": np.array(self.radial_grid, copy=True),
                "rho_cell": 0.5 * (self.radial_grid[:-1] + self.radial_grid[1:]),
                "phi_b": self.phi_boundary,
                "axis_psi": self.axis_reference,
                "boundary_psi": self.boundary_reference,
            }
        )
        return TransportGeometry(record)


@dataclass(frozen=True)
class Waveform:
    """Immutable radial-profile waveform carrying its physical coordinate map.

    ``radial_grid`` has shape ``(time, radius)`` and is always the normalised
    coordinate used for profile interpolation.  A channel with shape
    ``(time, radius)`` is a radial profile.  Scalar channels have shape
    ``(time,)``; other fixed-shape numeric records are interpolated
    element-wise in time.  This permits a complete ``TransportGeometry``
    record to travel through the same type as an equilibrium source profile.
    """

    time: np.ndarray
    radial_grid: np.ndarray
    phi_boundary: np.ndarray
    axis_reference: np.ndarray
    boundary_reference: np.ndarray
    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        time = _readonly(self.time, dtype=np.float64)
        radial_grid = _readonly(self.radial_grid, dtype=np.float64)
        phi_boundary = _readonly(self.phi_boundary, dtype=np.float64)
        axis_reference = _readonly(self.axis_reference, dtype=np.float64)
        boundary_reference = _readonly(self.boundary_reference, dtype=np.float64)
        if time.ndim != 1 or time.size < 2:
            raise ValueError("a waveform needs at least two time samples")
        if not np.all(np.diff(time) > 0.0):
            raise ValueError("waveform time must be strictly increasing")
        if radial_grid.ndim == 1:
            radial_grid = _readonly(
                np.broadcast_to(radial_grid, (time.size, radial_grid.size)),
                dtype=np.float64,
            )
        if radial_grid.ndim != 2 or radial_grid.shape[0] != time.size:
            raise ValueError("radial_grid must have shape (time, radius)")
        if radial_grid.shape[1] < 2 or not np.all(np.diff(radial_grid, axis=1) > 0.0):
            raise ValueError("every waveform radial grid must be strictly increasing")
        if np.any(radial_grid[:, 0] < 0.0) or np.any(radial_grid[:, -1] > 1.0):
            raise ValueError("waveform radial grids must lie in [0, 1]")
        for name, coordinate in (
            ("phi_boundary", phi_boundary),
            ("axis_reference", axis_reference),
            ("boundary_reference", boundary_reference),
        ):
            if coordinate.shape != time.shape:
                raise ValueError(f"{name} must match the waveform time grid")
            _finite(name, coordinate)
        if np.any(phi_boundary == 0.0):
            raise ValueError("phi_boundary must be non-zero at every sample")
        if np.any(boundary_reference == axis_reference):
            raise ValueError("axis and boundary references must remain distinct")

        frozen_values: dict[str, np.ndarray] = {}
        for name, value in self.values.items():
            array = _readonly(value)
            if array.ndim == 0 or array.shape[0] != time.size:
                raise ValueError(
                    f"waveform channel {name} must carry time as its leading axis"
                )
            _finite(f"waveform channel {name}", array)
            frozen_values[str(name)] = array
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "radial_grid", radial_grid)
        object.__setattr__(self, "phi_boundary", phi_boundary)
        object.__setattr__(self, "axis_reference", axis_reference)
        object.__setattr__(self, "boundary_reference", boundary_reference)
        object.__setattr__(self, "values", MappingProxyType(frozen_values))

    @classmethod
    def from_geometries(
        cls,
        time: Sequence[float],
        geometries: Sequence[TransportGeometry],
    ) -> Waveform:
        """Stack compatible geometry records into one coordinate-aware waveform."""
        time_array = np.asarray(time, dtype=np.float64)
        if len(geometries) != time_array.size:
            raise ValueError("one transport geometry is required per time sample")
        if not geometries:
            raise ValueError("at least one transport geometry is required")
        keys = tuple(geometries[0].record)
        if any(set(geometry.record) != set(keys) for geometry in geometries[1:]):
            raise ValueError("all geometry samples must carry the same record fields")
        values: dict[str, np.ndarray] = {}
        coordinate_fields = {"rho_face", "phi_b", "axis_psi", "boundary_psi"}
        for key in keys:
            if key in coordinate_fields:
                continue
            try:
                values[key] = np.stack(
                    [np.asarray(geometry.record[key]) for geometry in geometries]
                )
            except ValueError as error:
                raise ValueError(
                    f"geometry field {key} must have one fixed shape across "
                    "the waveform"
                ) from error
        return cls(
            time=time_array,
            radial_grid=np.stack(
                [np.asarray(geometry.record["rho_face"]) for geometry in geometries]
            ),
            phi_boundary=np.asarray(
                [geometry.record["phi_b"] for geometry in geometries]
            ),
            axis_reference=np.asarray(
                [geometry.record["axis_psi"] for geometry in geometries]
            ),
            boundary_reference=np.asarray(
                [geometry.record["boundary_psi"] for geometry in geometries]
            ),
            values=values,
        )

    def _bracket(self, time: float) -> tuple[int, int, float]:
        """Return the bracketing indices and right-sample weight."""
        query = float(time)
        if query < self.time[0] or query > self.time[-1]:
            raise ValueError(
                f"waveform query {query} lies outside [{self.time[0]}, {self.time[-1]}]"
            )
        right = int(np.searchsorted(self.time, query, side="right"))
        if right == 0:
            return 0, 0, 0.0
        if right == self.time.size:
            return self.time.size - 1, self.time.size - 1, 0.0
        left = right - 1
        weight = (query - self.time[left]) / (self.time[right] - self.time[left])
        return left, right, float(weight)

    def sample(
        self, time: float, *, radial_grid: np.ndarray | None = None
    ) -> WaveformSample:
        """Interpolate one sample without mixing evolving radial coordinates."""
        left, right, weight = self._bracket(time)
        if radial_grid is None:
            target_grid = (1.0 - weight) * self.radial_grid[
                left
            ] + weight * self.radial_grid[right]
        else:
            target_grid = np.asarray(radial_grid, dtype=np.float64)
        values: dict[str, np.ndarray] = {}
        radial_size = self.radial_grid.shape[1]
        for name, channel in self.values.items():
            if name == "rho_cell":
                values[name] = 0.5 * (target_grid[:-1] + target_grid[1:])
                continue
            left_value = channel[left]
            right_value = channel[right]
            if channel.ndim == 2 and channel.shape[1] == radial_size:
                left_value = np.interp(target_grid, self.radial_grid[left], left_value)
                right_value = np.interp(
                    target_grid, self.radial_grid[right], right_value
                )
            if np.issubdtype(channel.dtype, np.bool_):
                if not np.array_equal(left_value, right_value):
                    raise ValueError(
                        f"boolean waveform channel {name} changes within an interval"
                    )
                values[name] = left_value
            else:
                values[name] = (1.0 - weight) * left_value + weight * right_value
        return WaveformSample(
            time=float(time),
            radial_grid=target_grid,
            phi_boundary=(1.0 - weight) * self.phi_boundary[left]
            + weight * self.phi_boundary[right],
            axis_reference=(1.0 - weight) * self.axis_reference[left]
            + weight * self.axis_reference[right],
            boundary_reference=(1.0 - weight) * self.boundary_reference[left]
            + weight * self.boundary_reference[right],
            values=values,
        )


@dataclass(frozen=True)
class TransportSweepReceipt:
    """One transport pass through a geometry waveform."""

    time: np.ndarray
    geometry_time: np.ndarray
    receipts: tuple[ForwardTransportReceipt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _readonly(self.time, dtype=np.float64))
        object.__setattr__(
            self, "geometry_time", _readonly(self.geometry_time, dtype=np.float64)
        )

    @property
    def state(self) -> TransportState:
        """Return the state at the end of the sweep."""
        return self.receipts[-1].state


@dataclass(frozen=True)
class EquilibriumSweepReceipt:
    """Equilibrium results and their full conservation receipts over one pass."""

    time: np.ndarray
    source_samples: tuple[WaveformSample, ...]
    equilibria: tuple[ForwardEquilibrium, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _readonly(self.time, dtype=np.float64))

    @property
    def conservation(self) -> tuple[Any, ...]:
        """Return the conservation ledger emitted at every coarse sample."""
        return tuple(equilibrium.conservation for equilibrium in self.equilibria)


@dataclass(frozen=True)
class WindowConfig:
    """The four declared controls of one coupled exchange window.

    ``length`` fixes the physical window.  The equilibrium and transport time
    arrays jointly form the per-side sample-grid control; both span the whole
    window but may have different interior samples.  ``iteration_cap`` and
    ``tolerance`` control the fixed-point solve without changing either side's
    native temporal resolution.
    """

    length: float
    equilibrium_grid: np.ndarray
    transport_grid: np.ndarray
    iteration_cap: int
    tolerance: float

    def __post_init__(self) -> None:
        equilibrium_grid = _readonly(self.equilibrium_grid, dtype=np.float64)
        transport_grid = _readonly(self.transport_grid, dtype=np.float64)
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("window length must be finite and positive")
        if self.iteration_cap < 1:
            raise ValueError("window iteration cap must be at least one")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("window convergence tolerance must be finite and positive")
        for name, grid in (
            ("equilibrium", equilibrium_grid),
            ("transport", transport_grid),
        ):
            if grid.ndim != 1 or grid.size < 2:
                raise ValueError(f"{name} sample grid needs at least two times")
            if not np.all(np.diff(grid) > 0.0):
                raise ValueError(f"{name} sample grid must be strictly increasing")
            scale = max(abs(self.length), 1.0)
            if not np.isclose(grid[0], 0.0, rtol=0.0, atol=1.0e-14 * scale):
                raise ValueError(f"{name} sample grid must start at zero")
            if not np.isclose(grid[-1], self.length, rtol=0.0, atol=1.0e-14 * scale):
                raise ValueError(f"{name} sample grid must end at the window length")
        object.__setattr__(self, "equilibrium_grid", equilibrium_grid)
        object.__setattr__(self, "transport_grid", transport_grid)


@dataclass(frozen=True)
class ExchangeSweepResult:
    """A waveform produced by one side together with that side's receipt."""

    waveform: Waveform
    receipt: Any


@dataclass(frozen=True)
class WindowConvergenceReceipt:
    """Measured behaviour of a converged or exhausted waveform iteration."""

    iterations_used: int
    contraction_estimate: float | None
    exit_residual: Mapping[str, float]
    damping_applied: float
    residual_trace: tuple[Mapping[str, float], ...]

    def __post_init__(self) -> None:
        exit_residual = MappingProxyType(
            {str(name): float(value) for name, value in self.exit_residual.items()}
        )
        trace = tuple(
            MappingProxyType({str(name): float(value) for name, value in row.items()})
            for row in self.residual_trace
        )
        object.__setattr__(self, "exit_residual", exit_residual)
        object.__setattr__(self, "residual_trace", trace)

    @property
    def maximum_residual(self) -> float:
        """Return the largest final exchanged-field residual."""
        return max(self.exit_residual.values(), default=0.0)


@dataclass(frozen=True)
class WindowConservationReceipt:
    """Aggregated flux and current closure across a transport window."""

    flux_consumption: FluxConsumptionLedger
    plasma_current: PlasmaCurrentLedger
    flux_closure_error: float
    flux_closure_residual: float
    current_continuity_error: float
    current_continuity_residual: float


@dataclass(frozen=True)
class WindowReceipt:
    """Converged exchanged waveforms with side and window-level receipts."""

    geometry_waveform: Waveform
    source_waveform: Waveform
    equilibrium_receipt: Any
    transport_receipt: TransportSweepReceipt
    convergence: WindowConvergenceReceipt
    conservation: WindowConservationReceipt


class WindowConvergenceError(RuntimeError):
    """The waveform iteration exhausted its cap before reaching tolerance."""

    def __init__(
        self,
        convergence: WindowConvergenceReceipt,
        geometry_waveform: Waveform,
        source_waveform: Waveform,
        equilibrium_receipt: Any,
        transport_receipt: TransportSweepReceipt,
    ) -> None:
        super().__init__(
            "window exchange did not converge: "
            f"residual {convergence.maximum_residual:.6g} after "
            f"{convergence.iterations_used} iterations"
        )
        self.convergence = convergence
        self.geometry_waveform = geometry_waveform
        self.source_waveform = source_waveform
        self.equilibrium_receipt = equilibrium_receipt
        self.transport_receipt = transport_receipt


class WindowConservationError(RuntimeError):
    """A converged exchange failed its flux or current closure."""

    def __init__(self, conservation: WindowConservationReceipt) -> None:
        super().__init__(
            "window conservation did not close: "
            f"flux residual {conservation.flux_closure_residual:.6g}, "
            "current residual "
            f"{conservation.current_continuity_residual:.6g}"
        )
        self.conservation = conservation


def _require_waveform_grid(waveform: Waveform, expected: np.ndarray, name: str) -> None:
    """Require one exchanged waveform to use its declared side time grid."""
    if waveform.time.shape != expected.shape or not np.array_equal(
        waveform.time, expected
    ):
        raise ValueError(f"{name} waveform must use its declared sample grid")


def _relative_residual(previous, candidate) -> float:
    """Return a scale-normalised sup residual between two exchanged fields."""
    previous_array = np.asarray(previous)
    candidate_array = np.asarray(candidate)
    if previous_array.shape != candidate_array.shape:
        raise ValueError("an exchanged field changed shape within a window")
    if np.issubdtype(previous_array.dtype, np.bool_) or np.issubdtype(
        candidate_array.dtype, np.bool_
    ):
        return 0.0 if np.array_equal(previous_array, candidate_array) else 1.0
    scale = max(
        float(np.max(np.abs(previous_array), initial=0.0)),
        float(np.max(np.abs(candidate_array), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    return float(np.max(np.abs(candidate_array - previous_array), initial=0.0)) / scale


def _waveform_residual(
    name: str, previous: Waveform, candidate: Waveform
) -> dict[str, float]:
    """Return one residual for every coordinate-map and profile field."""
    if set(previous.values) != set(candidate.values):
        raise ValueError(f"{name} waveform fields changed within the window")
    fields = {
        "radial_grid": (previous.radial_grid, candidate.radial_grid),
        "phi_boundary": (previous.phi_boundary, candidate.phi_boundary),
        "axis_reference": (previous.axis_reference, candidate.axis_reference),
        "boundary_reference": (
            previous.boundary_reference,
            candidate.boundary_reference,
        ),
        **{
            field: (previous.values[field], candidate.values[field])
            for field in sorted(previous.values)
        },
    }
    return {
        f"{name}.{field}": _relative_residual(left, right)
        for field, (left, right) in fields.items()
    }


def _blend_field(previous, candidate, damping: float):
    """Relax one numeric field while preserving stable boolean metadata."""
    previous_array = np.asarray(previous)
    candidate_array = np.asarray(candidate)
    if previous_array.shape != candidate_array.shape:
        raise ValueError("an exchanged field changed shape within a window")
    if np.issubdtype(previous_array.dtype, np.bool_) or np.issubdtype(
        candidate_array.dtype, np.bool_
    ):
        if not np.array_equal(previous_array, candidate_array):
            raise ValueError("boolean exchange metadata changed within a window")
        return previous_array
    return previous_array + damping * (candidate_array - previous_array)


def _blend_waveform(
    previous: Waveform, candidate: Waveform, damping: float
) -> Waveform:
    """Return an immutable relaxed waveform on one unchanged time grid."""
    if not np.array_equal(previous.time, candidate.time):
        raise ValueError("an exchanged waveform changed its time grid")
    if set(previous.values) != set(candidate.values):
        raise ValueError("an exchanged waveform changed its fields")
    return Waveform(
        time=previous.time,
        radial_grid=_blend_field(previous.radial_grid, candidate.radial_grid, damping),
        phi_boundary=_blend_field(
            previous.phi_boundary, candidate.phi_boundary, damping
        ),
        axis_reference=_blend_field(
            previous.axis_reference, candidate.axis_reference, damping
        ),
        boundary_reference=_blend_field(
            previous.boundary_reference, candidate.boundary_reference, damping
        ),
        values={
            name: _blend_field(previous.values[name], candidate.values[name], damping)
            for name in previous.values
        },
    )


def _contraction_estimate(trace: Sequence[Mapping[str, float]]) -> float | None:
    """Estimate the final observed sup-residual contraction."""
    if len(trace) < 2:
        return None
    previous = max(trace[-2].values(), default=0.0)
    current = max(trace[-1].values(), default=0.0)
    if previous == 0.0:
        return 0.0 if current == 0.0 else None
    return current / previous


def _window_conservation(
    transport: TransportSweepReceipt,
) -> WindowConservationReceipt:
    """Aggregate interval ledgers and quantify their closure."""
    if not transport.receipts:
        raise ValueError("a window transport receipt needs at least one interval")
    durations = np.diff(transport.time)
    if durations.size != len(transport.receipts):
        raise ValueError("transport interval receipts must match the transport grid")
    elapsed = float(np.sum(durations))
    ledgers = [receipt.flux_consumption for receipt in transport.receipts]
    boundary = sum(float(ledger.boundary) for ledger in ledgers)
    resistive = sum(float(ledger.resistive) for ledger in ledgers)
    internal = sum(float(ledger.internal) for ledger in ledgers)
    flux_error = abs(boundary - resistive - internal)
    flux_scale = max(abs(boundary), abs(resistive), abs(internal), 1.0e-30)
    flux = FluxConsumptionLedger(
        boundary=boundary,
        resistive=resistive,
        internal=internal,
        mean_axis_voltage=sum(
            float(ledger.mean_axis_voltage) * duration
            for ledger, duration in zip(ledgers, durations, strict=True)
        )
        / elapsed,
        mean_boundary_voltage=sum(
            float(ledger.mean_boundary_voltage) * duration
            for ledger, duration in zip(ledgers, durations, strict=True)
        )
        / elapsed,
    )

    currents = [receipt.plasma_current for receipt in transport.receipts]
    boundary_errors = [
        abs(float(currents[0].requested_initial - currents[0].achieved_initial)),
        abs(float(currents[-1].requested_final - currents[-1].achieved_final)),
    ]
    for left, right in zip(currents[:-1], currents[1:], strict=True):
        boundary_errors.extend(
            [
                abs(float(left.achieved_final - right.achieved_initial)),
                abs(float(left.requested_final - right.requested_initial)),
            ]
        )
    current_error = max(boundary_errors, default=0.0)
    current_scale = max(
        *(
            abs(float(value))
            for ledger in currents
            for value in (
                ledger.requested_initial,
                ledger.requested_final,
                ledger.achieved_initial,
                ledger.achieved_final,
            )
        ),
        1.0,
    )
    current = PlasmaCurrentLedger(
        requested_initial=float(currents[0].requested_initial),
        requested_final=float(currents[-1].requested_final),
        achieved_initial=float(currents[0].achieved_initial),
        achieved_final=float(currents[-1].achieved_final),
    )
    return WindowConservationReceipt(
        flux_consumption=flux,
        plasma_current=current,
        flux_closure_error=flux_error,
        flux_closure_residual=flux_error / flux_scale,
        current_continuity_error=current_error,
        current_continuity_residual=current_error / current_scale,
    )


def solve_window(
    initial_geometry: Waveform,
    initial_source: Waveform,
    config: WindowConfig,
    equilibrium_update: Callable[[Waveform, np.ndarray], ExchangeSweepResult],
    transport_update: Callable[[Waveform, np.ndarray], ExchangeSweepResult],
    *,
    damping: float = 1.0,
) -> WindowReceipt:
    """Converge one equilibrium--transport waveform exchange or raise.

    The transport side first consumes the current geometry trajectory and
    emits a source trajectory.  The equilibrium side consumes that source and
    emits the next geometry trajectory.  Residuals are measured on every
    exchanged profile and coordinate-map field before relaxation.  Exhausting
    the cap raises :class:`WindowConvergenceError`; the exception retains the
    final candidate and convergence receipt for diagnosis, but no degraded
    :class:`WindowReceipt` is returned.
    """
    if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("window damping must lie in (0, 1]")
    _require_waveform_grid(initial_geometry, config.equilibrium_grid, "geometry")
    _require_waveform_grid(initial_source, config.transport_grid, "source")
    geometry = initial_geometry
    source = initial_source
    residual_trace: list[Mapping[str, float]] = []

    for iteration in range(1, config.iteration_cap + 1):
        transported = transport_update(geometry, config.transport_grid)
        if not isinstance(transported, ExchangeSweepResult):
            raise TypeError("transport update must return ExchangeSweepResult")
        if not isinstance(transported.receipt, TransportSweepReceipt):
            raise TypeError("transport update must return a TransportSweepReceipt")
        source_candidate = transported.waveform
        _require_waveform_grid(source_candidate, config.transport_grid, "source")

        equilibrated = equilibrium_update(source_candidate, config.equilibrium_grid)
        if not isinstance(equilibrated, ExchangeSweepResult):
            raise TypeError("equilibrium update must return ExchangeSweepResult")
        geometry_candidate = equilibrated.waveform
        _require_waveform_grid(geometry_candidate, config.equilibrium_grid, "geometry")

        residual = {
            **_waveform_residual("geometry", geometry, geometry_candidate),
            **_waveform_residual("source", source, source_candidate),
        }
        residual_trace.append(residual)
        convergence = WindowConvergenceReceipt(
            iterations_used=iteration,
            contraction_estimate=_contraction_estimate(residual_trace),
            exit_residual=residual,
            damping_applied=float(damping),
            residual_trace=tuple(residual_trace),
        )
        if convergence.maximum_residual <= config.tolerance:
            conservation = _window_conservation(transported.receipt)
            if (
                conservation.flux_closure_residual > config.tolerance
                or conservation.current_continuity_residual > config.tolerance
            ):
                raise WindowConservationError(conservation)
            return WindowReceipt(
                geometry_waveform=geometry_candidate,
                source_waveform=source_candidate,
                equilibrium_receipt=equilibrated.receipt,
                transport_receipt=transported.receipt,
                convergence=convergence,
                conservation=conservation,
            )
        if iteration == config.iteration_cap:
            raise WindowConvergenceError(
                convergence,
                geometry_candidate,
                source_candidate,
                equilibrated.receipt,
                transported.receipt,
            )
        geometry = _blend_waveform(geometry, geometry_candidate, damping)
        source = _blend_waveform(source, source_candidate, damping)

    raise AssertionError("unreachable window iteration state")


def transport_sweep(
    geometry_waveform: Waveform,
    initial_state: TransportState,
    time: Sequence[float],
    plasma_current: Sequence[float],
    model: TransportModel,
    *,
    solve: Callable[[ForwardTransportInput], ForwardTransportReceipt] | None = None,
) -> TransportSweepReceipt:
    """Advance transport once against midpoint-interpolated window geometry.

    Each transport interval is an ordinary :class:`ForwardTransport` facade
    solve.  Geometry is sampled at the interval midpoint; this is the public
    staggered/single-sweep primitive on which the converged window iteration
    can build without giving this function any convergence state of its own.
    """
    time_array = np.asarray(time, dtype=np.float64)
    current_array = np.asarray(plasma_current, dtype=np.float64)
    if time_array.ndim != 1 or time_array.size < 2:
        raise ValueError("a transport sweep needs at least two time samples")
    if current_array.shape != time_array.shape:
        raise ValueError("plasma_current must match the transport time grid")
    if not np.all(np.diff(time_array) > 0.0):
        raise ValueError("transport sweep time must be strictly increasing")
    if (
        time_array[0] < geometry_waveform.time[0]
        or time_array[-1] > geometry_waveform.time[-1]
    ):
        raise ValueError("transport times must lie inside the geometry waveform")

    solve_interval = ForwardTransport().solve if solve is None else solve
    state = initial_state
    receipts: list[ForwardTransportReceipt] = []
    geometry_times = 0.5 * (time_array[:-1] + time_array[1:])
    for index, geometry_time in enumerate(geometry_times):
        geometry = geometry_waveform.sample(float(geometry_time)).geometry()
        receipt = solve_interval(
            ForwardTransportInput(
                geometry=geometry,
                initial_state=state,
                waveforms=TransportWaveforms(
                    time=time_array[index : index + 2],
                    plasma_current=current_array[index : index + 2],
                ),
                model=model,
            )
        )
        receipts.append(receipt)
        state = receipt.state
    return TransportSweepReceipt(
        time=time_array,
        geometry_time=geometry_times,
        receipts=tuple(receipts),
    )


def equilibrium_sweep(
    profile: ForwardProfile,
    initial_flux,
    source_waveform: Waveform,
    time: Sequence[float],
    source_from_sample: Callable[[WaveformSample], ForwardSource],
    *,
    route: SolveRoute = "newton_krylov",
    current=None,
    solve_options: Mapping[str, Any] | None = None,
) -> EquilibriumSweepReceipt:
    """Solve equilibrium at coarse times against an interpolated source waveform.

    A fresh profile/operator pair is created for every source sample.  The
    supplied profile, source waveform, and initial flux remain untouched;
    only the returned flux is threaded forward as the next solve's seed.
    """
    time_array = np.asarray(time, dtype=np.float64)
    if time_array.ndim != 1 or time_array.size == 0:
        raise ValueError("an equilibrium sweep needs at least one sample time")
    if not np.all(np.diff(time_array) > 0.0):
        raise ValueError("equilibrium sweep time must be strictly increasing")
    options = dict(solve_options or {})
    seed = initial_flux
    samples: list[WaveformSample] = []
    equilibria: list[ForwardEquilibrium] = []
    for sample_time in time_array:
        sample = source_waveform.sample(float(sample_time))
        source = source_from_sample(sample)
        operator = dataclasses.replace(profile.operator, source=source)
        sampled_profile = dataclasses.replace(profile, operator=operator)
        equilibrium = sampled_profile.solve(
            seed,
            route=route,
            current=current,
            **options,
        )
        samples.append(sample)
        equilibria.append(equilibrium)
        seed = equilibrium.flux
    return EquilibriumSweepReceipt(
        time=time_array,
        source_samples=tuple(samples),
        equilibria=tuple(equilibria),
    )
