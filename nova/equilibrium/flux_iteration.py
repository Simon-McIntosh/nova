"""Least-squares profile fit inside the equilibrium iteration.

The nested inverse class converges one complete free-boundary forward solve for
every coefficient proposal it compares, so a proposal the data did not choose
still has to find its own axis and boundary before a row is compared. The class
here holds the flux map within a pass: it evaluates the measured rows that map
drives, solves one weighted least-squares problem for the profile coefficients
and the fitted conductor currents, lays down the plasma current that fit
implies, and applies one Grad-Shafranov read-write cycle to update the flux map
from it. A wrong proposal is corrected by the next fit instead of being solved
to the end, and passes repeat until the flux map stops changing.

The current distribution within a pass is the one the held flux map books, so
the measured rows are linear in the fitted coefficients and their response
needs no forward solve. The caller supplies one reader of a per-cell current
vector into the measured row order; one response column is the difference of
that reader two coefficient vectors apart at the same held map.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.measurement import Magnetics, SliceMeasurement, whitened_solve
from nova.equilibrium.source import ForwardSource, PolynomialFluxFunction
from nova.jax.config import configure_dtypes

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from nova.equilibrium.forward import ForwardProfile

__all__ = [
    "FluxIterationFit",
    "FluxIterationReceipt",
    "IterationPass",
    "IterationTerminalState",
]


class IterationTerminalState(NamedTuple):
    """Terminal flux map, its booked cell currents and its topology read."""

    flux: np.ndarray
    cell_current: np.ndarray
    topology: object


@dataclass(frozen=True)
class IterationPass:
    """One pass of the fit-inside-the-iteration loop.

    ``misfit`` is the largest whitened row residual the pass's held map drives,
    ``flux_update`` the largest flux change of the update that pass applies
    divided by the axis-to-boundary span, and ``coefficients`` the coefficients
    its least-squares solve laid down.
    """

    index: int
    misfit: float
    flux_update: float
    coefficients: np.ndarray


@dataclass(frozen=True)
class FluxIterationReceipt:
    """One terminal reconstruction from the fit inside the iteration.

    ``passes`` carries every pass in order, so the pass at which the flux map
    stopped improving and the size of the last flux update beside the terminal
    misfit are read from it. ``flux_evaluations`` counts the Grad-Shafranov
    read-write cycles applied; no forward solve is converged for a proposed
    coefficient set, so it is the whole flux cost of the reconstruction.
    """

    coefficients: np.ndarray
    coefficient_names: tuple[str, ...]
    source_profile: ForwardSource
    terminal_state: IterationTerminalState
    passes: tuple[IterationPass, ...]
    converged: bool
    terminal_misfit: float
    last_flux_update: float
    stopped_improving_pass: int | None
    axis_lost_pass: int | None
    whitened_residuals: np.ndarray
    singular_values: np.ndarray
    covariance: np.ndarray
    undetermined_coefficients: tuple[str, ...]
    null_directions: np.ndarray
    conductor_currents: np.ndarray
    relaxation: float
    flux_evaluations: int
    sensor_evaluations: int
    wall_seconds: float
    device: str


class FluxIterationFit:
    """Fit profile coefficients against the rows a held flux map drives.

    ``profile`` owns the immutable problem geometry: the lattice, the wall and
    the source whose polynomial coefficients the fit varies. ``sensor_rows``
    reads one per-cell plasma current vector [A] into the measured row order
    [Wb or T], with no conductor contribution, so the same reader supplies both
    a pass's residual and its response columns. Fitted conductor currents enter
    through ``conductor_response`` [Wb/A or T/A], whose columns match the
    request field named by ``conductor_current_field``; the measurement's
    ``vacuum`` then carries only fixed exterior contributions and the fitted
    currents are laid down through the same field on the flux update.

    ``coefficient_basis`` optionally maps the fitted amplitudes onto the raw
    polynomial coefficient vector, exactly as :class:`MagneticsFit` does.
    ``conductor_names`` name the extra rows.
    """

    def __init__(
        self,
        profile: ForwardProfile,
        magnetics: Magnetics,
        sensor_rows: Callable[[np.ndarray], np.ndarray],
        *,
        coefficient_basis: np.ndarray | None = None,
        conductor_response: np.ndarray | None = None,
        conductor_current_field: str = "current",
        conductor_names: tuple[str, ...] | None = None,
    ) -> None:
        configure_dtypes()
        assert jax.config.jax_enable_x64 is True
        core = profile.source.core
        if not all(
            isinstance(function, PolynomialFluxFunction)
            for function in (core.p_prime, core.ff_prime)
        ):
            raise TypeError("fitted flux functions must be PolynomialFluxFunction")
        if not isinstance(magnetics, Magnetics):
            raise TypeError("the fit requires magnetics row geometry")
        self.profile = profile
        self.magnetics = magnetics
        self.sensor_rows = sensor_rows
        self._pressure_count = core.p_prime.coefficients.size
        self._names = tuple(
            f"{family}_{order}"
            for family, function in (
                ("pressure", core.p_prime),
                ("diamagnetic", core.ff_prime),
            )
            for order in range(function.coefficients.size)
        )
        raw = np.r_[core.p_prime.coefficients, core.ff_prime.coefficients]
        basis = (
            np.eye(raw.size)
            if coefficient_basis is None
            else np.asarray(coefficient_basis, dtype=np.float64).copy()
        )
        if (
            basis.ndim != 2
            or basis.shape[0] != raw.size
            or basis.shape[1] < 1
            or not np.all(np.isfinite(basis))
            or np.linalg.matrix_rank(basis) != basis.shape[1]
        ):
            raise ValueError("coefficient basis must have finite independent columns")
        initial = np.linalg.lstsq(basis, raw, rcond=None)[0]
        if not np.allclose(basis @ initial, raw, rtol=1e-12, atol=1e-12):
            raise ValueError("the initial source must belong to the coefficient basis")
        basis.setflags(write=False)
        self._basis = basis
        self._initial = initial
        if coefficient_basis is not None:
            self._names = tuple(f"basis_{index}" for index in range(basis.shape[1]))
        self._profile_count = initial.size
        self._conductor_field = conductor_current_field
        self._conductor_response = np.empty((magnetics.number, 0))
        if conductor_response is not None:
            if conductor_current_field not in {"current", "prescribed_current"}:
                raise ValueError("fitted currents require a conductor request field")
            response = np.asarray(conductor_response, dtype=np.float64).copy()
            if (
                response.ndim != 2
                or response.shape[0] != magnetics.number
                or response.shape[1] == 0
                or not np.all(np.isfinite(response))
            ):
                raise ValueError("conductor response must align with magnetics rows")
            names = (
                tuple(f"conductor_{index}" for index in range(response.shape[1]))
                if conductor_names is None
                else tuple(conductor_names)
            )
            if len(names) != response.shape[1] or len(set(names)) != len(names):
                raise ValueError("conductor names must be unique and column-aligned")
            response.setflags(write=False)
            self._conductor_response = response
            self._initial = np.r_[initial, np.zeros(response.shape[1])]
            self._names += names
        self._conductor_count = self._conductor_response.shape[1]

    def _source(self, coefficients) -> ForwardSource:
        coefficients = self._basis @ np.asarray(coefficients)[: self._profile_count]
        source = self.profile.source
        core = source.core
        split = self._pressure_count
        return replace(
            source,
            core=replace(
                core,
                p_prime=replace(
                    core.p_prime, coefficients=jnp.asarray(coefficients[:split])
                ),
                ff_prime=replace(
                    core.ff_prime, coefficients=jnp.asarray(coefficients[split:])
                ),
            ),
        )

    def _profile(self, coefficients) -> ForwardProfile:
        return self.profile._with_source(self._source(coefficients))

    def _cell_current(self, profile, flux) -> np.ndarray:
        return np.asarray(
            profile.operator.cell_current(jnp.asarray(flux), None, None),
            dtype=np.float64,
        )

    def _rows(self, cells: np.ndarray, currents: np.ndarray) -> np.ndarray:
        values = np.empty(self.magnetics.number + 1 + self._conductor_count)
        values[: self.magnetics.number] = np.asarray(
            self.sensor_rows(cells), dtype=np.float64
        )
        if self._conductor_count:
            values[: self.magnetics.number] += self._conductor_response @ currents
            values[self.magnetics.number + 1 :] = currents
        values[self.magnetics.number] = float(np.sum(cells))
        if not np.all(np.isfinite(values)):
            raise ValueError("the held-map observation contains non-finite values")
        return values

    def fit(
        self,
        measurement: SliceMeasurement,
        initial_flux,
        *,
        plasma_current_scale: float,
        max_passes: int = 64,
        flux_tolerance: float = 1.0e-10,
        relaxation: float = 1.0,
        rcond: float = 1.0e-8,
    ) -> FluxIterationReceipt:
        """Return one reconstruction from the held-map least-squares passes.

        ``initial_flux`` is the starting map on the profile's node set — the
        stored state of a stored case, or the starting field of a known state's
        study. ``relaxation`` is the fraction of each flux update laid down; the
        default lays down the whole update. Convergence is the flux update
        falling to ``flux_tolerance`` of the axis-to-boundary span.
        """
        started = perf_counter()
        if (
            not np.isfinite(plasma_current_scale)
            or plasma_current_scale <= 0.0
            or max_passes < 1
            or flux_tolerance <= 0.0
            or not 0.0 < relaxation <= 1.0
            or not 0.0 <= rcond < 1.0
        ):
            raise ValueError("fit budgets and scales must be finite and positive")
        size = self.magnetics.number
        measured = np.asarray(measurement.measured, dtype=np.float64)
        vacuum = np.asarray(measurement.vacuum, dtype=np.float64)
        mask = np.asarray(measurement.mask, dtype=bool)
        scale = np.asarray(measurement.scale, dtype=np.float64)
        if any(value.shape != (size,) for value in (measured, vacuum, mask, scale)):
            raise ValueError("measurement rows must align with magnetics geometry")
        if (
            not np.all(np.isfinite(scale[mask]))
            or np.any(scale[mask] <= 0.0)
            or not np.isfinite(measurement.plasma_current)
        ):
            raise ValueError("active measurement rows and scales must be finite")
        target = np.r_[
            np.where(mask, measurement.signature, 0.0), measurement.plasma_current
        ]
        weight = np.r_[measurement.weight, 1.0 / plasma_current_scale]
        if self._conductor_count:
            currents = np.asarray(measurement.conductor_current, dtype=np.float64)
            uncertainties = np.asarray(measurement.conductor_scale, dtype=np.float64)
            if (
                currents.shape != (self._conductor_count,)
                or uncertainties.shape != (self._conductor_count,)
                or not np.all(np.isfinite(currents))
                or not np.all(np.isfinite(uncertainties))
                or np.any(uncertainties <= 0.0)
            ):
                raise ValueError(
                    "fitted currents require aligned positive uncertainties"
                )
            target = np.r_[target, currents]
            weight = np.r_[weight, 1.0 / uncertainties]
        elif (
            measurement.conductor_current is not None
            or measurement.conductor_scale is not None
        ):
            raise ValueError("current measurements require a conductor response")

        flux = np.asarray(initial_flux, dtype=np.float64)
        coefficients = np.array(self._initial, dtype=np.float64)
        if self._conductor_count:
            coefficients[self._profile_count :] = np.asarray(
                measurement.conductor_current, dtype=np.float64
            )
        initial_profile = self._profile(coefficients)
        operator = initial_profile.operator
        try:
            _unused, topology = operator.read(jnp.asarray(flux))
            span = abs(float(topology.flux_span))
        except Exception:  # an unqualified starting map still has a flux scale
            span = abs(float(np.max(flux) - np.min(flux)))
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError("the starting flux map has no finite scale")

        passes: list[IterationPass] = []
        flux_evaluations = 0
        sensor_evaluations = 0
        axis_lost_pass: int | None = None
        converged = False
        residual = np.full(target.size, np.nan)
        spectrum = np.zeros(0)
        null = np.zeros((0, coefficients.size))
        covariance = np.zeros((coefficients.size, coefficients.size))
        misfit = np.inf
        best_update = np.inf
        stopped_improving: int | None = None
        values = None
        for index in range(max_passes):
            active = self._profile(coefficients)
            operator = active.operator
            columns = []
            try:
                cells = self._cell_current(active, flux)
                sensor_evaluations += 1
                values = self._rows(cells, coefficients[self._profile_count :])
                for column in range(self._profile_count):
                    direction = np.zeros_like(coefficients)
                    direction[column] = 1.0
                    stepped = self._cell_current(
                        self._profile(coefficients + direction), flux
                    )
                    sensor_evaluations += 1
                    difference = stepped - cells
                    columns.append(
                        np.r_[
                            np.asarray(self.sensor_rows(difference), dtype=np.float64),
                            float(np.sum(difference)),
                        ]
                    )
            except ValueError:
                if index == 0:
                    raise
                axis_lost_pass = index
                break
            residual = (values - target) * weight
            misfit = float(np.max(np.abs(residual)))
            response = np.column_stack(columns)
            if self._conductor_count:
                response = np.vstack(
                    (response, np.zeros((self._conductor_count, response.shape[1])))
                )
                conductor = np.zeros(
                    (size + 1 + self._conductor_count, self._conductor_count)
                )
                conductor[:size] = self._conductor_response
                conductor[size + 1 :] = np.eye(self._conductor_count)
                response = np.column_stack((response, conductor))
            correction, covariance, spectrum, null = whitened_solve(
                response,
                target - values,
                weight,
                0.0,
                diagnostics=True,
                rcond=rcond,
            )
            coefficients = coefficients + correction
            try:
                updated = np.asarray(
                    operator(
                        jnp.asarray(flux),
                        jnp.asarray(coefficients[self._profile_count :]),
                        None,
                        None,
                    ),
                    dtype=np.float64,
                )
                flux_evaluations += 1
            except Exception:
                axis_lost_pass = index
                break
            delta = updated - flux
            update = float(np.max(np.abs(delta)) / span)
            flux = flux + relaxation * delta
            passes.append(
                IterationPass(
                    index=index,
                    misfit=misfit,
                    flux_update=update,
                    coefficients=coefficients.copy(),
                )
            )
            if update <= flux_tolerance:
                converged = True
                break
            if update >= best_update and stopped_improving is None:
                stopped_improving = index
            best_update = min(best_update, update)

        active = self._profile(coefficients)
        try:
            _unused, terminal_topology = active.operator.read(jnp.asarray(flux))
        except Exception:
            terminal_topology = None
        try:
            terminal_cells = self._cell_current(active, flux)
            sensor_evaluations += 1
            terminal_values = self._rows(
                terminal_cells, coefficients[self._profile_count :]
            )
            residual = (terminal_values - target) * weight
            misfit = float(np.max(np.abs(residual)))
        except ValueError:
            terminal_cells = np.zeros(0)
            if values is not None:
                residual = (values - target) * weight
                misfit = float(np.max(np.abs(residual)))
        last_update = passes[-1].flux_update if passes else np.inf
        undetermined = (
            np.any(np.abs(null) > rcond, axis=0)
            if null.size
            else np.zeros(coefficients.size, dtype=bool)
        )
        return FluxIterationReceipt(
            coefficients=coefficients.copy(),
            coefficient_names=self._names,
            source_profile=self._source(coefficients),
            terminal_state=IterationTerminalState(
                flux=np.asarray(flux, dtype=np.float64),
                cell_current=np.asarray(terminal_cells, dtype=np.float64),
                topology=terminal_topology,
            ),
            passes=tuple(passes),
            converged=converged,
            terminal_misfit=misfit,
            last_flux_update=float(last_update),
            stopped_improving_pass=stopped_improving,
            axis_lost_pass=axis_lost_pass,
            whitened_residuals=residual.copy(),
            singular_values=np.asarray(spectrum, dtype=np.float64),
            covariance=np.asarray(covariance, dtype=np.float64),
            undetermined_coefficients=tuple(
                name
                for name, unknown in zip(self._names, undetermined, strict=True)
                if unknown
            ),
            null_directions=np.asarray(null, dtype=np.float64),
            conductor_currents=np.asarray(
                coefficients[self._profile_count :], dtype=np.float64
            ),
            relaxation=relaxation,
            flux_evaluations=flux_evaluations,
            sensor_evaluations=sensor_evaluations,
            wall_seconds=perf_counter() - started,
            device=str(jax.devices()[0]),
        )
