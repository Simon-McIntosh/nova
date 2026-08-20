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
    ForwardTransport,
    ForwardTransportInput,
    ForwardTransportReceipt,
    TransportGeometry,
    TransportModel,
    TransportState,
    TransportWaveforms,
)

__all__ = [
    "EquilibriumSweepReceipt",
    "TransportSweepReceipt",
    "Waveform",
    "WaveformSample",
    "equilibrium_sweep",
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
