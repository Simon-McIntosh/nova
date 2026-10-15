"""Equilibrium current prescriptions paired with independently clocked coil currents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from nova.imas.profiles import PrescribedCurrent, prescribe_current_density
from nova.media.sources.frame import EquilibriumFrame


@dataclass(frozen=True)
class CoilCurrentSeries:
    """Named coil currents in amperes on their own time base."""

    time: np.ndarray
    names: tuple[str, ...]
    current: np.ndarray

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        current = np.asarray(self.current, dtype=float)
        if (
            time.ndim != 1
            or time.size == 0
            or not np.all(np.isfinite(time))
            or not np.all(np.diff(time) > 0)
            or current.shape != (time.size, len(self.names))
            or not np.all(np.isfinite(current))
        ):
            raise ValueError(
                "coil currents need finite rows on an increasing time base"
            )

    def at(self, time: float) -> np.ndarray:
        """Interpolate coil currents without extrapolating beyond their clock."""
        if time < self.time[0] or time > self.time[-1]:
            raise ValueError("coil current clock does not cover equilibrium time")
        return np.array(
            [
                np.interp(time, self.time, self.current[:, index])
                for index in range(len(self.names))
            ]
        )


@dataclass(frozen=True)
class PlasmaSlice:
    """One equilibrium map and its flux functions in the stored convention."""

    time: float
    radius: np.ndarray
    height: np.ndarray
    psi: np.ndarray
    psi_samples: np.ndarray
    p_prime: np.ndarray
    ff_prime: np.ndarray
    psi_axis: float
    psi_boundary: float
    plasma_current: float
    outline: np.ndarray
    axis: tuple[float, float]
    flux_derivative_scale: float
    x_points: tuple[tuple[float, float], ...] = ()
    boundary_outline: np.ndarray | None = None

    def prescribe(self) -> PrescribedCurrent:
        """Prescribe current after checking the declared flux orientation."""
        span = self.psi_boundary - self.psi_axis
        if (
            not np.isfinite(span)
            or not np.isfinite(self.plasma_current)
            or self.plasma_current == 0
            or span * self.plasma_current >= 0
        ):
            raise ValueError("plasma current sign and psi ordering disagree with COCOS")
        return prescribe_current_density(
            self.radius,
            self.height,
            self.psi,
            self.psi_samples,
            self.p_prime,
            self.ff_prime,
            self.psi_axis,
            self.psi_boundary,
            self.plasma_current,
            self.outline,
            self.axis,
            self.flux_derivative_scale,
            x_points=self.x_points,
            boundary_outline=self.boundary_outline,
        )


@dataclass(frozen=True)
class PlasmaSource:
    """Equilibrium slices and coil currents from the same discharge."""

    slices: tuple[PlasmaSlice, ...]
    coils: CoilCurrentSeries
    provenance: str

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("plasma source needs an equilibrium slice")
        for slice_ in self.slices:
            self.coils.at(slice_.time)

    def prescribe(self, index: int) -> PrescribedCurrent:
        return self.slices[index].prescribe()

    def coil_currents(self, index: int) -> np.ndarray:
        return self.coils.at(self.slices[index].time)

    @classmethod
    def from_mast(
        cls,
        frames: Sequence[EquilibriumFrame],
        coils: CoilCurrentSeries,
        *,
        provenance: str,
    ) -> PlasmaSource:
        """Use the existing MAST EFIT reader's COCOS 3 to 17 flux conversion."""
        slices = []
        for frame in frames:
            finite_x = np.asarray(frame.x_points, float).reshape(-1, 2)
            finite_x = finite_x[np.all(np.isfinite(finite_x), axis=1)]
            slices.append(
                PlasmaSlice(
                    time=frame.time,
                    radius=frame.radius,
                    height=frame.height,
                    psi=frame.flux.T,
                    psi_samples=frame.flux_axis
                    + frame.psi_norm * (frame.flux_boundary - frame.flux_axis),
                    p_prime=frame.p_prime,
                    ff_prime=frame.ff_prime,
                    psi_axis=frame.flux_axis,
                    psi_boundary=frame.flux_boundary,
                    plasma_current=frame.plasma_current,
                    outline=frame.boundary,
                    axis=tuple(frame.magnetic_axis),
                    flux_derivative_scale=1.0,
                    x_points=tuple(map(tuple, finite_x)),
                    boundary_outline=frame.boundary,
                )
            )
        return cls(tuple(slices), coils, provenance)

    @classmethod
    def from_imas(
        cls,
        equilibrium: Any,
        active: Any,
        *,
        provenance: str,
        indices: Sequence[int] | None = None,
    ) -> PlasmaSource:
        """Use native IMAS equilibrium slices and the active-coil time base."""
        if int(equilibrium.ids_properties.homogeneous_time) != 1:
            raise ValueError("equilibrium needs its IDS time base")
        if int(active.ids_properties.homogeneous_time) != 1:
            raise ValueError("active coils need their IDS time base")
        coil_time = np.asarray(active.time, float)
        coils = CoilCurrentSeries(
            coil_time,
            tuple(str(coil.name).strip() for coil in active.coil),
            np.column_stack(
                [np.asarray(coil.current.data, float) for coil in active.coil]
            ),
        )
        slices = []
        selected = range(len(equilibrium.time_slice)) if indices is None else indices
        for index in selected:
            slice_ = equilibrium.time_slice[index]
            global_ = slice_.global_quantities
            profile = slice_.profiles_1d
            map_ = slice_.profiles_2d[0]
            outline = slice_.boundary_separatrix.outline
            boundary = slice_.boundary.outline
            slices.append(
                PlasmaSlice(
                    time=float(slice_.time),
                    radius=np.asarray(map_.grid.dim1),
                    height=np.asarray(map_.grid.dim2),
                    psi=np.asarray(map_.psi),
                    psi_samples=np.asarray(profile.psi),
                    p_prime=np.asarray(profile.dpressure_dpsi),
                    ff_prime=np.asarray(profile.f_df_dpsi),
                    psi_axis=float(global_.psi_axis),
                    psi_boundary=float(global_.psi_boundary),
                    plasma_current=float(global_.ip),
                    outline=np.c_[np.asarray(outline.r), np.asarray(outline.z)],
                    axis=(
                        float(global_.magnetic_axis.r),
                        float(global_.magnetic_axis.z),
                    ),
                    flux_derivative_scale=2 * np.pi,
                    x_points=tuple(
                        (float(point.r), float(point.z))
                        for point in slice_.boundary_separatrix.x_point
                    ),
                    boundary_outline=np.c_[
                        np.asarray(boundary.r), np.asarray(boundary.z)
                    ],
                )
            )
        return cls(tuple(slices), coils, provenance)
