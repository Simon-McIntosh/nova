"""Weighted profile-coefficient inversion around the public forward solve.

Geometry and conductor currents belong to the supplied forward problem. A
source adapter binds a terminal receipt's currents to that same magnetic
geometry; the observation kernels own the sensor evaluation. Neither the
measurement nor the least-squares state enters a forward request.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

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
from nova.equilibrium.source import PolynomialFluxFunction
from nova.jax.config import configure_dtypes

if TYPE_CHECKING:
    from nova.biot.biot import Biot
    from nova.equilibrium.forward import ForwardProfile

__all__ = ["MagneticsFit", "MagneticsFitReceipt"]


@dataclass(frozen=True)
class MagneticsFitReceipt:
    """One terminal fit, with row residuals and data identifiability.

    Coefficients concatenate ascending-power pressure and diamagnetic vectors.
    Residuals keep the magnetics row order, with the plasma-current row last;
    excluded rows carry zero. Singular values describe the whitened,
    column-normalised response before regularisation. ``null_directions`` are
    unit vectors in the coefficient frame, and ``undetermined_coefficients``
    names every coefficient with a component in their span.
    """

    coefficients: np.ndarray
    coefficient_names: tuple[str, ...]
    forward: ForwardSolveReceipt
    iterations: int
    singular_values: np.ndarray
    undetermined_coefficients: tuple[str, ...]
    null_directions: np.ndarray
    whitened_residuals: np.ndarray
    covariance: np.ndarray
    converged: bool
    forward_evaluations: int


class MagneticsFit:
    """Fit polynomial source coefficients by damped Gauss--Newton steps.

    ``magnetic_source`` maps a qualified forward receipt onto a Biot source
    carrying its terminal plasma currents and the prescribed conductors. Its
    geometry can be built once and reused; it receives no measurements and
    must not infer currents from sampled flux. The existing virtual diagnostic
    kernels then read it at ``magnetics`` coordinates. All forward evaluations,
    including finite-difference response columns and line-search trials, use
    the supplied request's public policy and fixed-shape source representation.
    """

    def __init__(
        self,
        profile: ForwardProfile,
        request: ForwardSolveRequest,
        magnetics: Magnetics,
        magnetic_source: Callable[[ForwardSolveReceipt], Biot],
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

    def _source(self, coefficients):
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
            values[:-1][loops] = virtual_flux_loops(biot, coordinates[loops]).values
        if np.any(~loops):
            angle = np.deg2rad(np.asarray(self.magnetics.angle)[~loops])
            orientation = np.column_stack((np.cos(angle), np.sin(angle)))
            values[:-1][~loops] = virtual_poloidal_probes(
                biot, coordinates[~loops], orientation
            ).values
        values[-1] = float(np.sum(receipt.terminal_state.cell_current))
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
        """
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
        if any(value.shape != (size,) for value in (measured, mask, scale)):
            raise ValueError("measurement rows must align with magnetics geometry")
        if (
            not np.all(np.isfinite(measured[mask]))
            or not np.all(np.isfinite(scale[mask]))
            or np.any(scale[mask] <= 0.0)
            or not np.isfinite(measurement.plasma_current)
        ):
            raise ValueError("active measurement rows and scales must be finite")
        target = np.r_[np.where(mask, measured, 0.0), measurement.plasma_current]
        weight = np.r_[measurement.weight, 1.0 / plasma_current_scale]
        core = self.request.source_profile.core
        coefficients = np.r_[core.p_prime.coefficients, core.ff_prime.coefficients]
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
            converged = bool(np.max(np.abs(residual)) <= tolerance)
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
        )
