"""Least-squares profile fit inside the equilibrium iteration.

The nested inverse class converges one complete free-boundary forward solve for
every coefficient proposal it compares, so a proposal the data did not choose
still has to find its own axis and boundary before a row is compared. The class
here holds the flux map within a pass: it evaluates the measured rows that map
drives, solves one weighted least-squares problem for the profile coefficients
and the fitted conductor currents, and applies one Grad-Shafranov read-write
cycle at those coefficients and currents to update the flux map. A forward
solve is never converged for a proposal the data did not choose; a wrong
proposal is corrected by the next fit instead, and passes repeat until the flux
map stops changing.

The current distribution within a pass is the one the held flux map books, so
the measured rows are linear in the fitted coefficients and their response
needs no forward solve. The caller supplies one reader of a per-cell current
vector into the measured row order; one response column is the difference of
that reader two coefficient vectors apart at the same held map.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.magnetics_fit import CoefficientBasis
from nova.equilibrium.measurement import (
    Magnetics,
    SliceMeasurement,
    whitened_solve,
)
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


class RowFit(NamedTuple):
    """The coefficients the rows at the held map name, and the solve's readings."""

    coefficients: np.ndarray
    values: np.ndarray
    covariance: np.ndarray
    spectrum: np.ndarray
    null: np.ndarray


@dataclass(frozen=True)
class IterationPass:
    """One pass of the fit-inside-the-iteration loop.

    ``misfit`` is the largest whitened row residual the pass's held map drives,
    ``flux_update`` the largest flux change of the update that pass applies
    divided by the axis-to-boundary span, and ``coefficients`` the coefficients
    its least-squares solve laid down. ``fraction`` is how much of the
    one-cycle update at those coefficients the pass laid down, so a pass whose
    full cycle would leave the branch the rows were measured on shows the
    smaller step it took instead, and ``trials`` counts the cycle evaluations
    the pass paid beyond the one its step was built from.
    """

    index: int
    misfit: float
    flux_update: float
    coefficients: np.ndarray
    fraction: float = 1.0
    trials: int = 0


@dataclass(frozen=True)
class FluxIterationReceipt:
    """One terminal reconstruction from the fit inside the iteration.

    ``passes`` carries every pass in order, so the pass at which the flux map
    stopped improving and the size of the last flux update beside the terminal
    misfit are read from it. ``flux_evaluations`` counts the Grad-Shafranov
    read-write cycles evaluated — one for each pass's step and one for each
    fraction it backs off from; no forward solve is converged for a proposed
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
    axis_loss_message: str | None
    rejected_pass: int | None
    rejected_message: str | None
    stalled_pass: int | None
    stalled_message: str | None
    topology_error: str | None
    first_pass_values: np.ndarray
    first_pass_residual: np.ndarray
    whitened_residuals: np.ndarray
    singular_values: np.ndarray
    covariance: np.ndarray
    undetermined_coefficients: tuple[str, ...]
    null_directions: np.ndarray
    conductor_currents: np.ndarray
    relaxation: float
    max_coefficient_step: float
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
    measurement's ``conductor_current`` vector; the measurement's ``vacuum``
    then carries only fixed exterior contributions and the fitted currents are
    laid down as the operator's ``current`` on the flux update.

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
        conductor_names: tuple[str, ...] | None = None,
        prescribed_current: np.ndarray | None = None,
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
        space = CoefficientBasis.of(core, coefficient_basis)
        self._space = space
        self._names = space.names
        self._profile_count = space.initial.size
        self._initial = space.initial
        self._conductor_response = np.empty((magnetics.number, 0))
        if conductor_response is not None:
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
            self._initial = np.r_[space.initial, np.zeros(response.shape[1])]
            self._names += names
        self._conductor_count = self._conductor_response.shape[1]
        prescribed = (
            None
            if prescribed_current is None
            else np.array(prescribed_current, dtype=np.float64, copy=True)
        )
        if prescribed is not None:
            prescribed.setflags(write=False)
        self._prescribed_current = prescribed

    def _source(self, coefficients: np.ndarray) -> ForwardSource:
        """Return the source these amplitudes name, on the fit's own geometry."""
        return self._space.source(self.profile.source, coefficients)

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

    def _whitened_misfit(self, coefficients, flux, target, weight) -> float:
        """Return the whitened rows a map and a coefficient set produce.

        Costs one cell-current evaluation and one row read: no flux cycle, so a
        backtracking step can ask what a trial map's rows look like before
        laying the map down.
        """
        active = self._profile(coefficients)
        cells = self._cell_current(active, flux)
        values = self._rows(cells, coefficients[self._profile_count :])
        return float(np.max(np.abs((values - target) * weight)))

    def _cycle(self, coefficients, flux) -> np.ndarray:
        """Return the one-cycle Grad-Shafranov read-write map at coefficients.

        The conductor currents the fit laid down are the operator's current, and
        a prescribed field keeps its own argument; an empty current vector is
        not the same request as no current at all.
        """
        profile = self._profile(coefficients)
        conductor = (
            jnp.asarray(coefficients[self._profile_count :])
            if self._conductor_count
            else None
        )
        if self._prescribed_current is None:
            updated = profile.operator(jnp.asarray(flux), conductor)
        else:
            updated = profile.operator(
                jnp.asarray(flux),
                conductor,
                None,
                None,
                jnp.asarray(self._prescribed_current),
            )
        return np.asarray(updated, dtype=np.float64)

    def _cycle_residual(self, coefficients, flux, span) -> float:
        """Return one cycle's own residual, divided by the span.

        The map a coefficient set asks for is the fixed point of that cycle, so
        this is how far the held map sits from the map those coefficients
        describe: the quantity a damped step has to lower.
        """
        return float(np.max(np.abs(self._cycle(coefficients, flux) - flux)) / span)

    def _mixed_step(
        self,
        state,
        change,
        history,
        *,
        depth: int = 3,
        relaxation: float = 0.5,
    ) -> np.ndarray:
        """Return the next map of a fixed-point walk from the steps it has taken.

        A plain damped step is what the walk takes until it has two steps to
        compare; after that the mixing coefficients are the least-squares
        combination of the recent steps whose images cancel the residual, which
        is Anderson mixing. The step is capped at twice the damped step and
        falls back to that step whenever the Gram solve is singular or the
        mixture is not finite, so an ill-conditioned history cannot throw the
        walk away.
        """
        relaxed = state + relaxation * change
        if len(history) < 2:
            return relaxed
        difference_state = np.column_stack(
            [history[i + 1][0] - history[i][0] for i in range(len(history) - 1)]
        )
        difference_map = np.column_stack(
            [history[i + 1][1] - history[i][1] for i in range(len(history) - 1)]
        )
        gram = difference_map.T @ difference_map
        gram = gram + 1.0e-10 * (np.trace(gram) / depth + 1.0e-30) * np.eye(
            gram.shape[0]
        )
        try:
            mixing = np.linalg.solve(gram, difference_map.T @ change)
        except np.linalg.LinAlgError:
            return relaxed
        mixed = (
            state
            + relaxation * change
            - (difference_state + relaxation * difference_map) @ mixing
        )
        if not np.all(np.isfinite(mixed)):
            return relaxed
        cap = 2.0 * float(np.max(np.abs(relaxed - state)))
        moved = float(np.max(np.abs(mixed - state)))
        if moved > cap:
            mixed = state + (mixed - state) * (cap / moved)
        return mixed

    def _fit_rows(
        self,
        coefficients,
        flux,
        target,
        weight,
        rcond: float,
        max_coefficient_step: float,
    ) -> RowFit:
        """Return the coefficients the rows at the held map name.

        The measured rows are linear in the coefficients at a fixed map, so one
        weighted least-squares step solves them exactly; the returned
        coefficients are that step added to the ones the map was read at. The
        step is bounded by ``max_coefficient_step`` times the largest amplitude
        the pass holds, because a column the rows barely resolve answers a
        residual of its own size with an amplitude the data never saw.
        """
        size = self.magnetics.number
        cells = self._cell_current(self._profile(coefficients), flux)
        values = self._rows(cells, coefficients[self._profile_count :])
        columns = []
        for column in range(self._profile_count):
            direction = np.zeros_like(coefficients)
            direction[column] = 1.0
            stepped = self._cell_current(self._profile(coefficients + direction), flux)
            difference = stepped - cells
            columns.append(
                np.r_[
                    np.asarray(self.sensor_rows(difference), dtype=np.float64),
                    float(np.sum(difference)),
                ]
            )
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
        largest = float(np.max(np.abs(coefficients))) if coefficients.size else 0.0
        radius = max_coefficient_step * max(largest, 0.0)
        reached = float(np.max(np.abs(correction))) if correction.size else 0.0
        if radius > 0.0 and reached > radius:
            correction = correction * (radius / reached)
        return RowFit(
            coefficients=coefficients + correction,
            values=np.asarray(values, dtype=np.float64),
            covariance=covariance,
            spectrum=spectrum,
            null=null,
        )

    def _coupled_merit(
        self, coefficients, flux, target, weight, span
    ) -> tuple[float, float]:
        """Return the flux-equation residual and the whitened row misfit.

        The two equations a pass must satisfy are that the held map is the fixed
        point of the operator its coefficients describe and that the rows that
        map drives match the measurement. The first is the operator's own
        residual divided by the span, the quantity the free-boundary solve
        measures; the second is the whitened misfit a map and a coefficient set
        produce. A step criterion read from both descends the coupled system,
        where one read from the flux residual alone descends the fixed point of
        whichever coefficients happen to be held and can walk away from the rows.
        """
        active = self._profile(coefficients)
        conductor = (
            jnp.asarray(coefficients[self._profile_count :])
            if self._conductor_count
            else None
        )
        if self._prescribed_current is None:
            mapped = active.operator(jnp.asarray(flux), conductor)
        else:
            mapped = active.operator(
                jnp.asarray(flux),
                conductor,
                None,
                None,
                jnp.asarray(self._prescribed_current),
            )
        flux_residual = float(np.max(np.abs(np.asarray(mapped) - flux)) / span)
        row_misfit = self._whitened_misfit(coefficients, flux, target, weight)
        return flux_residual, row_misfit

    def fit(
        self,
        measurement: SliceMeasurement,
        initial_flux,
        *,
        plasma_current_scale: float,
        max_passes: int = 96,
        flux_tolerance: float = 1.0e-10,
        relaxation: float = 1.0,
        rcond: float = 1.0e-8,
        max_update: float = 1.0,
        max_coefficient_step: float = 1.0,
        max_backtracks: int = 3,
        step_rule: str = "coupled",
        on_pass: Callable[[int, float, float], None] | None = None,
    ) -> FluxIterationReceipt:
        """Return one reconstruction from the held-map least-squares passes.

        ``initial_flux`` is the starting map on the profile's node set — the
        stored state of a stored case, or the starting field of a known state's
        study. A pass lays down a fraction of the one-cycle flux update at the
        coefficients it fitted, halving that fraction ``max_backtracks`` times
        and laying down the one that leaves the held map closest to the map
        those coefficients describe: a whole cycle is the fixed-point iteration
        itself, and from a starting field more than a span away it leaves the
        branch the rows were measured on instead of approaching the answer.
        ``relaxation`` caps the fraction a pass may lay down. Convergence is
        the applied flux update falling to ``flux_tolerance`` of the
        axis-to-boundary span.

        ``step_rule`` selects the fraction. ``cycle`` is the flux residual
        alone and is right only while the fitted coefficients are close.
        ``contraction`` descends the coupled merit of the flux residual and the
        whitened rows. ``picard`` lays a fixed relaxed step and is the rule the
        free-boundary solve uses. ``coupled`` fits the coefficients the held
        map's rows name and then lays one whole cycle of them down, mixing the
        walk over the steps it has taken (Anderson) instead of damping it. It follows
        the cycle map that the rows and the flux equation agree on: a plain
        cycle is not a contraction away from its own fixed point, so
        damping it only slows the walk to a stall, and re-solving the
        coefficients against a map more than a span from its answer lays the
        row misfit that map caused to the coefficients and moves a profile the
        data never asked for.

        The coupled walk contracts by about three quarters a pass on the known
        state that starts farthest from its answer, so ``max_passes`` has to
        cover the decades between the start's residual and ``flux_tolerance``.
        """
        started = perf_counter()
        if (
            not np.isfinite(plasma_current_scale)
            or plasma_current_scale <= 0.0
            or max_passes < 1
            or flux_tolerance <= 0.0
            or not 0.0 < relaxation <= 1.0
            or not np.isfinite(max_update)
            or max_update <= 0.0
            or not np.isfinite(max_coefficient_step)
            or max_coefficient_step <= 0.0
            or max_backtracks < 0
            or not 0.0 <= rcond < 1.0
            or step_rule not in ("contraction", "coupled", "cycle", "picard")
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
        axis_loss_message: str | None = None
        rejected_pass: int | None = None
        rejected_message: str | None = None
        stalled_pass: int | None = None
        stalled_message: str | None = None
        topology_error: str | None = None
        first_pass_values: np.ndarray | None = None
        first_pass_residual: np.ndarray | None = None
        converged = False
        residual = np.full(target.size, np.nan)
        spectrum = np.zeros(0)
        null = np.zeros((0, coefficients.size))
        covariance = np.zeros((coefficients.size, coefficients.size))
        misfit = np.inf
        best_update = np.inf
        history: list[tuple[np.ndarray, np.ndarray]] = []
        stopped_improving: int | None = None
        values = None
        for index in range(max_passes):
            try:
                fitted = self._fit_rows(
                    coefficients, flux, target, weight, rcond, max_coefficient_step
                )
            except ValueError as error:
                if index == 0:
                    raise
                axis_lost_pass = index
                axis_loss_message = f"{type(error).__name__}: {error}"
                break
            sensor_evaluations += 1 + self._profile_count
            values = fitted.values
            coefficients = fitted.coefficients
            covariance = fitted.covariance
            spectrum = fitted.spectrum
            null = fitted.null
            residual = (values - target) * weight
            misfit = float(np.max(np.abs(residual)))
            if index == 0:
                first_pass_values = np.array(values, dtype=np.float64)
                first_pass_residual = np.array(residual, dtype=np.float64)
            if step_rule == "coupled":
                # a pass fits the coefficients the held map's rows name and then
                # takes one cycle of them, so the map follows its own
                # coefficients instead of the coefficients chasing a map that
                # never settles. That cycle map is not a contraction away from
                # its own fixed point, so the walk is mixed over the steps it
                # has already taken rather than damped, and what the pass
                # reports is the residual it found: how far the held map sat
                # from the map its own coefficients describe.
                try:
                    mapped = self._cycle(coefficients, flux)
                except ValueError as error:
                    axis_lost_pass = index
                    axis_loss_message = f"{type(error).__name__}: {error}"
                    break
                flux_evaluations += 1
                change = mapped - flux
                update = float(np.max(np.abs(change)) / span)
                history.append((flux, change))
                if len(history) > max_backtracks + 2:
                    history.pop(0)
                # the mixture extrapolates past the step it is given, and that
                # extrapolation is capped at twice the step, so the step itself
                # is taken at no more than half a cycle: a whole cycle plus a
                # doubling of it is two spans of travel, a branch change the
                # pass would have to reject. A measured walk at a whole cycle
                # swings between 0.13 and 0.90 of the span and is rejected; at
                # half it contracts every pass.
                base = min(relaxation, 0.5)
                candidate = self._mixed_step(
                    flux,
                    change,
                    history,
                    depth=max_backtracks,
                    relaxation=base,
                )
                travel = float(np.max(np.abs(candidate - flux)) / span)
                if passes and travel > max_update:
                    rejected_pass = index
                    rejected_message = (
                        f"the map update {travel:.3g} of the span exceeded "
                        f"{max_update:g}"
                    )
                    break
                flux = candidate
                passes.append(
                    IterationPass(
                        index=index,
                        misfit=misfit,
                        flux_update=update,
                        coefficients=coefficients.copy(),
                        fraction=1.0,
                        trials=len(history),
                    )
                )
                if on_pass is not None:
                    on_pass(index, misfit, update)
                if update <= flux_tolerance:
                    converged = True
                    break
                if update >= best_update and stopped_improving is None:
                    stopped_improving = index
                best_update = min(best_update, update)
                continue
            # the map this pass lays down belongs to the coefficients just
            # fitted, and one whole cycle of it is the fixed-point iteration
            # itself: from a starting field more than a span from the answer it
            # leaves the branch the rows were measured on, so the pass lays
            # down the largest fraction of that cycle which brings the held map
            # nearer the map its own coefficients describe
            try:
                updated = self._cycle(coefficients, flux)
                flux_evaluations += 1
            except ValueError as error:
                axis_lost_pass = index
                axis_loss_message = f"{type(error).__name__}: {error}"
                break
            delta = updated - flux
            residual_at_hold = float(np.max(np.abs(delta)) / span)
            fraction = relaxation
            if step_rule not in ("contraction", "coupled", "cycle", "picard"):
                raise ValueError(f"unknown step rule {step_rule!r}")
            # the pass lays down a fraction of the one-cycle update. Which
            # fraction is the whole question: a whole cycle is the fixed point
            # iteration itself, and from a starting field more than a span from
            # its answer it leaves the branch the rows were measured on. The
            # ``cycle`` rule descends the cycle's own residual, which is right
            # only while the coefficients are close; the ``contraction`` rule
            # descends the coupled merit of the flux equation and the rows, so a
            # step that fits the operator at coefficients the rows did not
            # choose is not preferred over one that fits the data.
            accepted = residual_at_hold <= flux_tolerance
            trials = 0
            best_fraction = 1.0 if accepted else 0.0
            best_residual = residual_at_hold
            if step_rule == "contraction" and not accepted:
                # the trial rungs are scored on the coupled merit, so the map
                # they are compared against must be scored the same way: a
                # flux-only reference stalls the pass on rungs whose flux
                # residual falls but whose row misfit, which the reference
                # omits, has not yet fallen with it
                best_residual = max(
                    residual_at_hold,
                    self._whitened_misfit(coefficients, flux, target, weight),
                )
                sensor_evaluations += 1
            fixed = step_rule == "picard"
            if fixed:
                # a fixed relaxed step, the rule the free-boundary solve uses
                best_fraction = fraction
            while not fixed and not accepted and trials <= max_backtracks:
                trial_flux = flux + fraction * delta
                try:
                    if step_rule == "cycle":
                        score = self._cycle_residual(coefficients, trial_flux, span)
                        flux_evaluations += 1
                    else:
                        flux_residual, row_misfit = self._coupled_merit(
                            coefficients, trial_flux, target, weight, span
                        )
                        flux_evaluations += 1
                        sensor_evaluations += 1
                        score = max(flux_residual, row_misfit)
                except ValueError:  # the trial map carries no usable axis
                    score = np.inf
                trials += 1
                if score < best_residual:
                    best_fraction = fraction
                    best_residual = score
                fraction *= 0.5
            if not best_fraction:
                stalled_pass = index
                stalled_message = (
                    f"no fraction down to {fraction * 2.0:.3g} of the cycle left "
                    f"the held map nearer the one its coefficients describe than "
                    f"{residual_at_hold:.3g} of the span"
                )
                break
            fraction = best_fraction
            step = fraction * delta
            # the pass reports the map it applies, so a damped pass's row is the
            # step it laid down and not the undamped one it was scaled from
            update = float(np.max(np.abs(step)) / span)
            if passes and update > max_update:
                # a solve that moves the map by more than the span has left the
                # branch the fit was measured on; the previous map stays
                rejected_pass = index
                rejected_message = (
                    f"the map update {update:.3g} of the span exceeded {max_update:g}"
                )
                break
            flux = flux + step
            passes.append(
                IterationPass(
                    index=index,
                    misfit=misfit,
                    flux_update=update,
                    coefficients=coefficients.copy(),
                    fraction=fraction,
                    trials=trials,
                )
            )
            if on_pass is not None:
                on_pass(index, misfit, update)
            if update <= flux_tolerance:
                converged = True
                break
            if update >= best_update and stopped_improving is None:
                stopped_improving = index
            best_update = min(best_update, update)

        active = self._profile(coefficients)
        try:
            _unused, terminal_topology = active.operator.read(jnp.asarray(flux))
        except Exception as error:  # reported, since a map still has a state
            terminal_topology = None
            topology_error = f"{type(error).__name__}: {error}"
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
            axis_loss_message=axis_loss_message,
            rejected_pass=rejected_pass,
            rejected_message=rejected_message,
            stalled_pass=stalled_pass,
            stalled_message=stalled_message,
            topology_error=topology_error,
            first_pass_values=(
                first_pass_values if first_pass_values is not None else np.zeros(0)
            ),
            first_pass_residual=(
                first_pass_residual if first_pass_residual is not None else np.zeros(0)
            ),
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
            max_coefficient_step=max_coefficient_step,
            flux_evaluations=flux_evaluations,
            sensor_evaluations=sensor_evaluations,
            wall_seconds=perf_counter() - started,
            device=str(jax.devices()[0]),
        )
