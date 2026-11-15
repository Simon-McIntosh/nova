"""Weighted profile-coefficient inversion around the public forward solve.

Geometry belongs to the supplied forward problem. A
source adapter binds a terminal receipt's currents to that same magnetic
geometry; the observation kernels own the sensor evaluation. Neither the
measurement nor the least-squares state enters a forward request.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.measurement import Magnetics, SliceMeasurement, whitened_solve
from nova.equilibrium.observation_kernels import (
    virtual_flux_loops,
    virtual_poloidal_probes,
)
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    ForwardSolveReceipt,
    ForwardSolveRequest,
)
from nova.equilibrium.source import ForwardSource, PolynomialFluxFunction
from nova.jax.config import configure_dtypes

if TYPE_CHECKING:
    from nova.biot.biot import Biot
    from nova.equilibrium.forward import ForwardProfile

__all__ = ["MagneticsFit", "MagneticsFitReceipt"]


@dataclass(frozen=True)
class MagneticsFitReceipt:
    """One terminal fit, with row residuals and data identifiability.

    Coefficients use the declared fitting basis. The default concatenates
    ascending-power pressure and diamagnetic vectors; ``source_profile`` always
    carries the recovered physical functions for an independent forward solve.
    Residuals keep the magnetics row order, then plasma current and any measured
    conductor currents. Coefficients concatenate profiles and fitted currents;
    excluded rows carry zero. Singular values describe the whitened,
    column-normalised response before regularisation. ``null_directions`` are
    unit vectors in the coefficient frame, and ``undetermined_coefficients``
    names every coefficient with a component in their span.
    """

    coefficients: np.ndarray
    coefficient_names: tuple[str, ...]
    source_profile: ForwardSource
    forward: ForwardSolveReceipt
    iterations: int
    singular_values: np.ndarray
    undetermined_coefficients: tuple[str, ...]
    null_directions: np.ndarray
    whitened_residuals: np.ndarray
    covariance: np.ndarray
    converged: bool
    forward_evaluations: int
    conductor_currents: np.ndarray
    wall_seconds: float
    device: str


class MagneticsFit:
    """Fit polynomial source coefficients by damped Gauss--Newton corrections.

    ``magnetic_source`` maps a qualified forward receipt onto a Biot source
    carrying its terminal plasma currents. Known conductor signals are supplied
    in ``SliceMeasurement.vacuum`` and subtracted at the declared sensor geometry. Its
    geometry can be built once and reused; it receives no measurements and
    must not infer currents from sampled flux. The existing virtual diagnostic
    kernels then read it at ``magnetics`` coordinates. All forward evaluations,
    including finite-difference response columns and line-search trials, use
    the supplied request's public policy and fixed-shape source representation.
    ``coefficient_basis`` optionally maps fitted coefficients to the concatenated
    polynomial coefficients. It can declare fixed profile shapes with fitted
    amplitudes, without introducing an unobserved polynomial direction.

    ``conductor_response`` adds current unknowns and their diagnostic responses
    [Wb/A or T/A]. Its columns match the complete vector in the request field
    named by ``conductor_current_field``. The measurement's ``vacuum`` excludes
    those fitted currents and retains only fixed exterior contributions.
    Measured conductor currents and their uncertainties supply extra fit rows;
    every trial carries the currents through the existing forward request.
    """

    def __init__(
        self,
        profile: ForwardProfile,
        request: ForwardSolveRequest,
        magnetics: Magnetics,
        magnetic_source: Callable[[ForwardSolveReceipt], Biot],
        *,
        coefficient_basis: np.ndarray | None = None,
        conductor_response: np.ndarray | None = None,
        conductor_current_field: Literal["current", "prescribed_current"] = "current",
        conductor_names: tuple[str, ...] | None = None,
    ) -> None:
        configure_dtypes()
        assert jax.config.jax_enable_x64 is True
        if not isinstance(request, ForwardSolveRequest):
            raise TypeError("a fit requires a ForwardSolveRequest")
        core = request.source_profile.core
        if not all(
            isinstance(function, PolynomialFluxFunction)
            for function in (core.p_prime, core.ff_prime)
        ):
            raise TypeError("fitted flux functions must be PolynomialFluxFunction")
        if request.target_current is not None:
            raise ValueError("the plasma-current measurement must remain a fit row")
        self.profile = profile
        self.request = request
        self.magnetics = magnetics
        self.magnetic_source = magnetic_source
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
            currents = np.asarray(
                getattr(request, conductor_current_field), dtype=np.float64
            )
            response = np.asarray(conductor_response, dtype=np.float64).copy()
            if (
                currents.ndim != 1
                or currents.size == 0
                or not np.all(np.isfinite(currents))
                or response.shape != (magnetics.number, currents.size)
                or not np.all(np.isfinite(response))
            ):
                raise ValueError("conductor response must align with request currents")
            names = (
                tuple(f"conductor_{index}" for index in range(currents.size))
                if conductor_names is None
                else tuple(conductor_names)
            )
            if len(names) != currents.size or len(set(names)) != len(names):
                raise ValueError("conductor names must be unique and column-aligned")
            response.setflags(write=False)
            self._conductor_response = response
            self._initial = np.r_[initial, currents]
            self._names += names

    def _source(self, coefficients):
        coefficients = self._basis @ coefficients[: self._profile_count]
        source = self.request.source_profile
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

    def _evaluate(self, coefficients, seed):
        request = replace(self.request, source_profile=self._source(coefficients))
        currents = coefficients[self._profile_count :]
        if currents.size:
            request = replace(request, **{self._conductor_field: jnp.asarray(currents)})
        if seed is not None:
            request = replace(request, seed_policy=ExplicitSolveSeed(seed))
        receipt = self.profile.solve(request)
        if not isinstance(receipt, ForwardSolveReceipt):
            raise TypeError(
                "the public forward solve must return a ForwardSolveReceipt"
            )
        if not bool(receipt.qualified):
            error = RuntimeError("the forward evaluation did not qualify")
            error.receipt = receipt
            raise error
        biot = self.magnetic_source(receipt)
        coordinates = np.column_stack((self.magnetics.r, self.magnetics.z))
        loops = np.asarray(self.magnetics.flux_loop, dtype=bool)
        values = np.empty(self.magnetics.number + 1, dtype=np.float64)
        if np.any(loops):
            values[:-1][loops] = (
                virtual_flux_loops(biot, coordinates[loops]).values
                * self.magnetics.loop_fraction[loops]
            )
        if np.any(~loops):
            angle = np.deg2rad(np.asarray(self.magnetics.angle)[~loops])
            orientation = np.column_stack((np.cos(angle), np.sin(angle)))
            values[:-1][~loops] = virtual_poloidal_probes(
                biot, coordinates[~loops], orientation
            ).values
        values[-1] = float(np.sum(receipt.terminal_state.cell_current))
        if currents.size:
            values[:-1] += self._conductor_response @ currents
            values = np.r_[values, currents]
        if not np.all(np.isfinite(values)):
            raise ValueError("the forward observation contains non-finite values")
        return receipt, values

    def fit(
        self,
        measurement: SliceMeasurement,
        *,
        plasma_current_scale: float,
        max_iterations: int = 16,
        tolerance: float = 1.0e-8,
        difference_step: float = 1.0e-4,
        ridge: float = 0.0,
        penalty: np.ndarray | None = None,
        rcond: float = 1.0e-8,
    ) -> MagneticsFitReceipt:
        """Return the terminal fit without changing the forward problem.

        The explicit current scale supplies the whitening uncertainty of the
        Rogowski row. ``ridge`` is a dimensionless penalty on a coefficient
        correction in the column-normalised response frame, with an optional
        per-column ``penalty``. A zero ridge declares unregularised least
        squares; unobserved directions remain reported even with a ridge.
        Convergence means either an exact data fit or a stationary least-squares
        objective: the column-normalised data gradient is below ``tolerance``.
        Nonzero sensor residuals therefore remain visible on a converged fit.
        """
        started = perf_counter()
        if (
            not np.isfinite(plasma_current_scale)
            or plasma_current_scale <= 0.0
            or tolerance <= 0.0
            or difference_step <= 0.0
            or ridge < 0.0
            or not 0.0 <= rcond < 1.0
            or max_iterations < 0
        ):
            raise ValueError("fit scales and budgets must be finite and positive")
        size = self.magnetics.number
        measured = np.asarray(measurement.measured, dtype=np.float64)
        mask = np.asarray(measurement.mask, dtype=bool)
        scale = np.asarray(measurement.scale, dtype=np.float64)
        vacuum = np.asarray(measurement.vacuum, dtype=np.float64)
        if any(value.shape != (size,) for value in (measured, vacuum, mask, scale)):
            raise ValueError("measurement rows must align with magnetics geometry")
        if (
            not np.all(np.isfinite(measured[mask]))
            or not np.all(np.isfinite(vacuum[mask]))
            or not np.all(np.isfinite(scale[mask]))
            or np.any(scale[mask] <= 0.0)
            or not np.isfinite(measurement.plasma_current)
        ):
            raise ValueError("active measurement rows and scales must be finite")
        target = np.r_[
            np.where(mask, measurement.signature, 0.0), measurement.plasma_current
        ]
        weight = np.r_[measurement.weight, 1.0 / plasma_current_scale]
        conductor_count = self._initial.size - self._profile_count
        if conductor_count:
            currents = np.asarray(measurement.conductor_current, dtype=np.float64)
            scales = np.asarray(measurement.conductor_scale, dtype=np.float64)
            if (
                currents.shape != (conductor_count,)
                or scales.shape != (conductor_count,)
                or not np.all(np.isfinite(currents))
                or not np.all(np.isfinite(scales))
                or np.any(scales <= 0.0)
            ):
                raise ValueError(
                    "fitted currents require aligned positive uncertainties"
                )
            target = np.r_[target, currents]
            weight = np.r_[weight, 1.0 / scales]
        elif (
            measurement.conductor_current is not None
            or measurement.conductor_scale is not None
        ):
            raise ValueError("current measurements require a conductor response")
        coefficients = self._initial.copy()
        forward, values = self._evaluate(coefficients, None)
        evaluations = 1
        iterations = 0
        while True:
            residual = (values - target) * weight
            columns = []
            seed = forward.terminal_state.flux
            for index in range(coefficients.size):
                step = difference_step * max(abs(coefficients[index]), 1.0)
                direction = np.zeros_like(coefficients)
                direction[index] = step
                _, upper = self._evaluate(coefficients + direction, seed)
                _, lower = self._evaluate(coefficients - direction, seed)
                evaluations += 2
                columns.append((upper - lower) / (2.0 * step))
            response = np.column_stack(columns)
            correction, covariance, spectrum, null = whitened_solve(
                response,
                target - values,
                weight,
                ridge,
                penalty=penalty,
                diagnostics=True,
                rcond=rcond,
            )
            weighted = response * weight[:, None]
            column_scale = np.linalg.norm(weighted, axis=0)
            gradient = (weighted.T @ residual) / np.where(
                column_scale > 0, column_scale, 1.0
            )
            converged = bool(
                np.max(np.abs(residual)) <= tolerance
                or np.max(np.abs(gradient)) <= tolerance
            )
            if converged or iterations == max_iterations:
                break
            merit = np.linalg.norm(residual)
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                candidate = coefficients + fraction * correction
                trial, prediction = self._evaluate(candidate, seed)
                evaluations += 1
                if np.linalg.norm((prediction - target) * weight) < merit:
                    coefficients, forward, values = candidate, trial, prediction
                    iterations += 1
                    accepted = True
                    break
            if not accepted:
                break
        undetermined = np.any(np.abs(null) > rcond, axis=0)
        return MagneticsFitReceipt(
            coefficients=coefficients,
            coefficient_names=self._names,
            source_profile=self._source(coefficients),
            forward=forward,
            iterations=iterations,
            singular_values=spectrum,
            undetermined_coefficients=tuple(
                name
                for name, unknown in zip(self._names, undetermined, strict=True)
                if unknown
            ),
            null_directions=null,
            whitened_residuals=residual,
            covariance=covariance,
            converged=converged,
            forward_evaluations=evaluations,
            conductor_currents=coefficients[self._profile_count :].copy(),
            wall_seconds=perf_counter() - started,
            device=str(jax.devices()[0]),
        )
