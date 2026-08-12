"""Noise-weighted geometry inversion with identifiable-mode regularisation.

Geometry inference is a small nonlinear least-squares problem with an unusual
but important metric.  The parameters are generators of vertex displacement,
so ``p.T @ gram @ p`` is the pack's mean-square displacement.  Solving in the
orthonormal coordinates of that metric makes a singular value directly mean
signal-to-noise per metre of physical pack displacement.

The routines here use Gauss--Newton steps with Levenberg--Marquardt damping.
At every iteration they project nuisance columns out of the whitened sensor
space, retain only directions whose one-noise displacement is below a stated
limit, and leave all other directions unchanged.  The same decomposition gives
confidence bounds without pretending that frozen directions were measured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


Array = np.ndarray
ModelJacobian = Callable[[Array], tuple[Array, Array]]


@dataclass(frozen=True)
class GeometryFit:
    """Result of an identifiable-mode geometry fit.

    ``parameters`` and ``standard_error`` use the caller's parameter basis.
    ``mode_amplitudes`` and ``mode_standard_error`` use orthonormal physical
    displacement coordinates, in metres.  Frozen modes have infinite standard
    error and zero fitted increment.
    """

    parameters: Array
    standard_error: Array
    modes: Array
    mode_amplitudes: Array
    mode_standard_error: Array
    singular_values: Array
    resolved: Array
    predicted: Array
    residual: Array
    whitened_chi_squared: float
    iterations: int
    converged: bool

    @property
    def resolved_count(self) -> int:
        """Return the number of geometry directions admitted to the fit."""
        return int(self.resolved.sum())


@dataclass(frozen=True)
class RecoveryLadder:
    """Banked recoveries of known perturbations at stated sensor noise."""

    amplitudes: Array
    truth: Array
    recovered: Array
    resolved_modes: Array
    mode_standard_error: Array

    @property
    def error(self) -> Array:
        """Return recovered minus injected parameters."""
        return self.recovered - self.truth

    @property
    def bias(self) -> Array:
        """Return the mean parameter error at each injected amplitude."""
        return self.error.mean(axis=1)


def _as_vector(value, name: str) -> Array:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def _metric_factors(gram: Array) -> tuple[Array, Array]:
    """Return the square root and inverse root of a positive metric."""
    gram = np.asarray(gram, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram must be square, got {gram.shape}")
    weight, direction = np.linalg.eigh(gram)
    floor = np.finfo(float).eps * max(gram.shape) * max(float(weight.max()), 1.0)
    if np.any(weight <= floor):
        raise ValueError("gram must be positive definite on the fitted parameters")
    root = (direction * np.sqrt(weight)) @ direction.T
    inverse_root = (direction / np.sqrt(weight)) @ direction.T
    return root, inverse_root


def nuisance_basis(span: Array | None, noise: Array) -> Array:
    """Return an orthonormal basis for nuisance columns in whitened space."""
    noise = _as_vector(noise, "noise")
    if span is None:
        return np.empty((noise.size, 0))
    span = np.asarray(span, dtype=float)
    if span.ndim != 2 or span.shape[0] != noise.size:
        raise ValueError(f"nuisance span must be ({noise.size}, n), got {span.shape}")
    left, values, _ = np.linalg.svd(span / noise[:, None], full_matrices=False)
    if values.size == 0:
        return np.empty((noise.size, 0))
    keep = values > np.finfo(float).eps * max(span.shape) * values[0]
    return left[:, keep]


def project_nuisance(values: Array, basis: Array) -> Array:
    """Remove the nuisance basis from one vector or a matrix of sensor rows."""
    values = np.asarray(values, dtype=float)
    return values - basis @ (basis.T @ values)


def _decompose(
    jacobian: Array,
    noise: Array,
    inverse_root: Array,
    nuisance: Array,
    resolution_limit: float,
) -> tuple[Array, Array, Array, Array]:
    """Return the metric-aware singular system and its resolved mask."""
    whitened = project_nuisance(jacobian / noise[:, None], nuisance)
    left, values, right = np.linalg.svd(whitened @ inverse_root, full_matrices=False)
    resolved = values >= 1.0 / resolution_limit
    return left, values, right, resolved


def solve_geometry(
    model_jacobian: ModelJacobian,
    target,
    noise,
    initial,
    gram,
    *,
    nuisance_span=None,
    resolution_limit: float = 0.01,
    damping: float = 1e-3,
    max_iterations: int = 30,
    step_tolerance: float = 1e-9,
    objective_tolerance: float = 1e-9,
) -> GeometryFit:
    """Fit geometry with damped Gauss--Newton steps on resolvable modes.

    ``model_jacobian(parameters)`` returns the predicted sensor vector and its
    exact Jacobian.  ``noise`` is one standard deviation per sensor channel,
    ``gram`` maps the parameter basis to mean-square physical displacement, and
    ``nuisance_span`` contains sensor patterns that may be fitted independently
    of geometry.  Modes needing more than ``resolution_limit`` metres for unit
    signal-to-noise are frozen.
    """
    target = _as_vector(target, "target")
    noise = _as_vector(noise, "noise")
    parameters = _as_vector(initial, "initial").copy()
    if target.shape != noise.shape:
        raise ValueError("target and noise must have the same shape")
    if np.any(~np.isfinite(noise)) or np.any(noise <= 0.0):
        raise ValueError("noise must be finite and strictly positive")
    if not np.isfinite(resolution_limit) or resolution_limit <= 0.0:
        raise ValueError("resolution_limit must be finite and positive")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")

    root, inverse_root = _metric_factors(gram)
    if root.shape[0] != parameters.size:
        raise ValueError("gram and initial parameter dimensions disagree")
    nuisance = nuisance_basis(nuisance_span, noise)
    current_damping = max(float(damping), np.finfo(float).eps)
    converged = False
    iterations = 0

    def evaluate(taken: Array):
        predicted, jacobian = model_jacobian(taken)
        predicted = _as_vector(predicted, "predicted")
        jacobian = np.asarray(jacobian, dtype=float)
        if predicted.shape != target.shape:
            raise ValueError("model prediction and target dimensions disagree")
        if jacobian.shape != (target.size, parameters.size):
            raise ValueError(
                "jacobian must have shape "
                f"{(target.size, parameters.size)}, got {jacobian.shape}"
            )
        residual = target - predicted
        whitened = project_nuisance(residual / noise, nuisance)
        return predicted, jacobian, residual, whitened

    predicted, jacobian, residual, whitened_residual = evaluate(parameters)
    objective = float(whitened_residual @ whitened_residual)
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        left, values, right, resolved = _decompose(
            jacobian, noise, inverse_root, nuisance, resolution_limit
        )
        if not np.any(resolved):
            converged = True
            break
        gain = values[resolved] / (values[resolved] ** 2 + current_damping)
        physical_step = right[resolved].T @ (
            gain * (left[:, resolved].T @ whitened_residual)
        )
        step = inverse_root @ physical_step
        trial = parameters + step
        trial_state = evaluate(trial)
        trial_objective = float(trial_state[3] @ trial_state[3])
        if trial_objective <= objective:
            improvement = objective - trial_objective
            parameters = trial
            predicted, jacobian, residual, whitened_residual = trial_state
            objective = trial_objective
            current_damping = max(current_damping / 3.0, np.finfo(float).eps)
            physical_norm = float(np.linalg.norm(root @ step))
            if physical_norm <= step_tolerance or improvement <= objective_tolerance:
                converged = True
                break
        else:
            current_damping *= 10.0

    left, values, right, resolved = _decompose(
        jacobian, noise, inverse_root, nuisance, resolution_limit
    )
    del left
    modes = inverse_root @ right.T
    physical_parameters = root @ parameters
    mode_amplitudes = right @ physical_parameters
    mode_error = np.full(values.shape, np.inf)
    mode_error[resolved] = 1.0 / values[resolved]
    parameter_covariance = (
        modes[:, resolved] @ np.diag(mode_error[resolved] ** 2) @ modes[:, resolved].T
    )
    parameter_error = np.sqrt(np.maximum(np.diag(parameter_covariance), 0.0))
    return GeometryFit(
        parameters=parameters,
        standard_error=parameter_error,
        modes=modes.T,
        mode_amplitudes=mode_amplitudes,
        mode_standard_error=mode_error,
        singular_values=values,
        resolved=resolved,
        predicted=predicted,
        residual=residual,
        whitened_chi_squared=objective,
        iterations=iterations,
        converged=converged,
    )


def solve_linear_geometry(
    jacobian,
    target,
    noise,
    initial,
    gram,
    **kwargs,
) -> GeometryFit:
    """Fit a banked local geometry model with the same nonlinear solver."""
    jacobian = np.asarray(jacobian, dtype=float)

    def model(parameters):
        return jacobian @ parameters, jacobian

    return solve_geometry(model, target, noise, initial, gram, **kwargs)


def synthetic_recovery_ladder(
    jacobian,
    noise,
    gram,
    amplitudes,
    *,
    nuisance_span=None,
    resolution_limit: float = 0.01,
    samples: int = 128,
    seed: int = 0,
) -> RecoveryLadder:
    """Recover random known deformations from exact local-model sensor data.

    Each rung has fixed root-mean-square pack displacement ``amplitudes`` and a
    random direction inside the resolvable geometry subspace.  The exact banked
    local model generates the noiseless target, then independent normal sensor
    noise is added at the supplied per-channel floors.  This separates recovery
    bias from the array's predicted one-noise resolution.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    noise = _as_vector(noise, "noise")
    gram = np.asarray(gram, dtype=float)
    amplitudes = _as_vector(amplitudes, "amplitudes")
    if np.any(amplitudes <= 0.0):
        raise ValueError("amplitudes must be strictly positive")
    if samples < 1:
        raise ValueError("samples must be positive")
    parameter_count = jacobian.shape[1]
    reference = solve_linear_geometry(
        jacobian,
        np.zeros(jacobian.shape[0]),
        noise,
        np.zeros(parameter_count),
        gram,
        nuisance_span=nuisance_span,
        resolution_limit=resolution_limit,
    )
    resolved_modes = reference.modes[reference.resolved]
    if resolved_modes.size == 0:
        raise ValueError("the requested resolution limit admits no geometry modes")

    generator = np.random.default_rng(seed)
    truth = np.empty((amplitudes.size, samples, parameter_count))
    recovered = np.empty_like(truth)
    for rung, amplitude in enumerate(amplitudes):
        directions = generator.normal(size=(samples, reference.resolved_count))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        coefficients = amplitude * directions
        truth[rung] = coefficients @ resolved_modes
        for sample in range(samples):
            target = jacobian @ truth[rung, sample]
            target += generator.normal(size=noise.size) * noise
            recovered[rung, sample] = solve_linear_geometry(
                jacobian,
                target,
                noise,
                np.zeros(parameter_count),
                gram,
                nuisance_span=nuisance_span,
                resolution_limit=resolution_limit,
            ).parameters
    return RecoveryLadder(
        amplitudes=amplitudes,
        truth=truth,
        recovered=recovered,
        resolved_modes=resolved_modes,
        mode_standard_error=reference.mode_standard_error[reference.resolved],
    )


__all__ = [
    "GeometryFit",
    "RecoveryLadder",
    "nuisance_basis",
    "project_nuisance",
    "solve_geometry",
    "solve_linear_geometry",
    "synthetic_recovery_ladder",
]
