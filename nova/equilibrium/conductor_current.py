"""Resolve conductor-current authority around an unchanged equilibrium solve.

Unknown currents deliberately remain non-finite in the resolved template.  The
only way to obtain a current vector suitable for a forward solve is to apply an
explicit candidate carrying every unknown parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import scipy.optimize


class CurrentTier(StrEnum):
    """State how one conductor current is available at inference."""

    KNOWN = "known"
    KNOWABLE = "knowable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StaticCurrentRelation:
    """A fit-once linear relation from one known conductor current."""

    source: str
    scale: float
    relative_residual: float
    provenance: str
    transfer_caveat: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale):
            raise ValueError("relation scale must be finite")
        if not np.isfinite(self.relative_residual) or self.relative_residual < 0.0:
            raise ValueError("relation residual must be finite and nonnegative")


@dataclass(frozen=True)
class UnknownCurrentPrior:
    """A proper prior and admissible interval for one free current."""

    mean_a: float
    standard_deviation_a: float
    lower_a: float
    upper_a: float
    provenance: str

    def __post_init__(self) -> None:
        values = (
            self.mean_a,
            self.standard_deviation_a,
            self.lower_a,
            self.upper_a,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("unknown-current prior values must be finite")
        if self.standard_deviation_a <= 0.0:
            raise ValueError("unknown-current prior width must be positive")
        if not self.lower_a < self.mean_a < self.upper_a:
            raise ValueError(
                "unknown-current prior mean must lie strictly inside bounds"
            )


@dataclass(frozen=True)
class ConductorCurrentDeclaration:
    """Declare the authority for one conductor in a response matrix."""

    name: str
    tier: CurrentTier
    provenance: str
    relation: StaticCurrentRelation | None = None
    prior: UnknownCurrentPrior | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("conductor name must be nonempty")
        if self.tier is CurrentTier.KNOWABLE and self.relation is None:
            raise ValueError(f"knowable conductor {self.name} requires a relation")
        if self.tier is CurrentTier.UNKNOWN and self.prior is None:
            raise ValueError(f"unknown conductor {self.name} requires a prior")
        if self.tier is not CurrentTier.KNOWABLE and self.relation is not None:
            raise ValueError(
                f"only knowable conductor {self.name} may carry a relation"
            )
        if self.tier is not CurrentTier.UNKNOWN and self.prior is not None:
            raise ValueError(f"only unknown conductor {self.name} may carry a prior")


@dataclass(frozen=True)
class ConductorCurrentRow:
    """Receipt entry for one current in response-column order."""

    name: str
    tier: str
    disposition: str
    value_a: float
    prior_mean_a: float | None
    prior_standard_deviation_a: float | None
    posterior_standard_deviation_a: float
    provenance: str
    transfer_caveat: str | None
    posterior_status: str

    def as_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible representation."""

        return {
            "name": self.name,
            "tier": self.tier,
            "disposition": self.disposition,
            "value_a": float(self.value_a),
            "prior_mean_a": self.prior_mean_a,
            "prior_standard_deviation_a": self.prior_standard_deviation_a,
            "posterior_standard_deviation_a": float(
                self.posterior_standard_deviation_a
            ),
            "provenance": self.provenance,
            "transfer_caveat": self.transfer_caveat,
            "posterior_status": self.posterior_status,
        }


@dataclass(frozen=True)
class CurrentResolution:
    """Ordered prescribed entries and explicit free-current parameterization."""

    names: tuple[str, ...]
    declarations: tuple[ConductorCurrentDeclaration, ...]
    template_a: np.ndarray
    prescribed_standard_deviation_a: np.ndarray
    unknown_indices: np.ndarray
    prior_mean_a: np.ndarray
    prior_standard_deviation_a: np.ndarray
    bounds_a: tuple[tuple[float, float], ...]

    @property
    def unknown_names(self) -> tuple[str, ...]:
        """Return free-parameter names in optimization order."""

        return tuple(self.names[index] for index in self.unknown_indices)

    def current(self, unknown_a: Sequence[float]) -> np.ndarray:
        """Apply every free parameter and return a complete finite vector."""

        candidate = np.asarray(unknown_a, dtype=float)
        if candidate.shape != self.prior_mean_a.shape:
            raise ValueError(
                f"expected {len(self.prior_mean_a)} unknown currents, got "
                f"shape {candidate.shape}"
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError("unknown-current candidate must be finite")
        for value, (lower, upper), name in zip(
            candidate, self.bounds_a, self.unknown_names, strict=True
        ):
            if not lower <= value <= upper:
                raise ValueError(f"candidate for {name} lies outside its prior bounds")
        result = self.template_a.copy()
        result[self.unknown_indices] = candidate
        if not np.all(np.isfinite(result)):
            raise RuntimeError("resolved current vector remains incomplete")
        return result


def resolve_conductor_currents(
    names: Sequence[str],
    declarations: Sequence[ConductorCurrentDeclaration],
    known_current_a: Mapping[str, float],
) -> CurrentResolution:
    """Resolve known and fit-once values while preserving unknown holes."""

    ordered_names = tuple(str(name) for name in names)
    ordered_declarations = tuple(declarations)
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("response-column conductor names must be unique")
    by_name = {item.name: item for item in ordered_declarations}
    if len(by_name) != len(ordered_declarations):
        raise ValueError("conductor declarations must be unique")
    if set(by_name) != set(ordered_names):
        missing = sorted(set(ordered_names) - set(by_name))
        extra = sorted(set(by_name) - set(ordered_names))
        raise ValueError(
            "declarations do not match response order; "
            f"missing={missing}, extra={extra}"
        )

    template = np.full(len(ordered_names), np.nan, dtype=float)
    uncertainty = np.full(len(ordered_names), np.nan, dtype=float)
    unknown_indices = []
    prior_mean = []
    prior_standard_deviation = []
    bounds = []
    ordered = []
    for index, name in enumerate(ordered_names):
        declaration = by_name[name]
        ordered.append(declaration)
        if declaration.tier is CurrentTier.KNOWN:
            if name not in known_current_a:
                raise ValueError(f"known current {name} was not supplied")
            value = float(known_current_a[name])
            if not np.isfinite(value):
                raise ValueError(f"known current {name} must be finite")
            template[index] = value
            uncertainty[index] = 0.0
        elif declaration.tier is CurrentTier.KNOWABLE:
            relation = declaration.relation
            assert relation is not None
            if relation.source not in known_current_a:
                raise ValueError(
                    f"relation source {relation.source} for {name} was not supplied"
                )
            source = float(known_current_a[relation.source])
            if not np.isfinite(source):
                raise ValueError(f"relation source {relation.source} must be finite")
            template[index] = relation.scale * source
            uncertainty[index] = abs(template[index]) * relation.relative_residual
        else:
            prior = declaration.prior
            assert prior is not None
            unknown_indices.append(index)
            prior_mean.append(prior.mean_a)
            prior_standard_deviation.append(prior.standard_deviation_a)
            bounds.append((prior.lower_a, prior.upper_a))

    return CurrentResolution(
        names=ordered_names,
        declarations=tuple(ordered),
        template_a=template,
        prescribed_standard_deviation_a=uncertainty,
        unknown_indices=np.asarray(unknown_indices, dtype=int),
        prior_mean_a=np.asarray(prior_mean, dtype=float),
        prior_standard_deviation_a=np.asarray(prior_standard_deviation, dtype=float),
        bounds_a=tuple(bounds),
    )


@dataclass(frozen=True)
class LikelihoodValue:
    """Residual and covariance returned by an inference-time observation."""

    residual: np.ndarray
    covariance: np.ndarray


class LikelihoodEvaluator(Protocol):
    """Evaluate one inference-available observation on an inner solution."""

    def __call__(self, equilibrium: Any, current_a: np.ndarray) -> LikelihoodValue: ...


@dataclass(frozen=True)
class InferenceLikelihood:
    """Name and provenance an observation operator admitted at inference."""

    name: str
    evaluator: LikelihoodEvaluator
    provenance: str
    available_at_inference: bool = True
    uses_label_artifact: bool = False

    def __post_init__(self) -> None:
        if not self.available_at_inference or self.uses_label_artifact:
            raise ValueError(
                "outer likelihood must be inference-available and label-free"
            )


@dataclass(frozen=True)
class ConductorCurrentRun:
    """Outer-loop result, final inner equilibrium, and complete run receipt."""

    current_a: np.ndarray
    equilibrium: Any
    unknown_posterior_mean_a: np.ndarray
    unknown_posterior_covariance_a2: np.ndarray
    objective: float
    evaluations: int
    receipt: dict[str, Any]


class ConductorCurrentInfeasible(RuntimeError):
    """Report an inner failure without fabricating a current candidate."""


def _likelihood_value(value: LikelihoodValue) -> tuple[np.ndarray, np.ndarray]:
    residual = np.atleast_1d(np.asarray(value.residual, dtype=float))
    covariance = np.atleast_2d(np.asarray(value.covariance, dtype=float))
    if residual.ndim != 1 or covariance.shape != (len(residual), len(residual)):
        raise ValueError("likelihood residual and covariance shapes do not agree")
    if not np.all(np.isfinite(residual + covariance.sum(axis=0))):
        raise ValueError("likelihood values must be finite")
    try:
        precision = np.linalg.inv(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("likelihood covariance must be nonsingular") from error
    if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
        raise ValueError("likelihood covariance must be positive definite")
    return residual, precision


def _receipt(
    resolution: CurrentResolution,
    posterior_mean: np.ndarray,
    posterior_covariance: np.ndarray,
    *,
    status: str,
    likelihood: InferenceLikelihood | None,
    likelihood_rank: int,
    evaluations: int,
    optimizer_success: bool,
) -> dict[str, Any]:
    full = resolution.current(posterior_mean)
    unknown_position = {
        int(index): position
        for position, index in enumerate(resolution.unknown_indices)
    }
    rows = []
    for index, declaration in enumerate(resolution.declarations):
        if declaration.tier is CurrentTier.UNKNOWN:
            position = unknown_position[index]
            prior = declaration.prior
            assert prior is not None
            row = ConductorCurrentRow(
                name=declaration.name,
                tier=declaration.tier.value,
                disposition="solved",
                value_a=float(full[index]),
                prior_mean_a=prior.mean_a,
                prior_standard_deviation_a=prior.standard_deviation_a,
                posterior_standard_deviation_a=float(
                    np.sqrt(posterior_covariance[position, position])
                ),
                provenance=prior.provenance,
                transfer_caveat=None,
                posterior_status=status,
            )
        else:
            relation = declaration.relation
            row = ConductorCurrentRow(
                name=declaration.name,
                tier=declaration.tier.value,
                disposition=(
                    "predicted"
                    if declaration.tier is CurrentTier.KNOWABLE
                    else "prescribed"
                ),
                value_a=float(full[index]),
                prior_mean_a=None,
                prior_standard_deviation_a=None,
                posterior_standard_deviation_a=float(
                    resolution.prescribed_standard_deviation_a[index]
                ),
                provenance=(
                    relation.provenance if relation else declaration.provenance
                ),
                transfer_caveat=(relation.transfer_caveat if relation else None),
                posterior_status="prescribed",
            )
        rows.append(row.as_dict())
    return {
        "posterior_status": status,
        "likelihood": None if likelihood is None else likelihood.name,
        "likelihood_provenance": (
            None if likelihood is None else likelihood.provenance
        ),
        "likelihood_rank": int(likelihood_rank),
        "unknown_parameter_count": len(resolution.unknown_indices),
        "optimizer_success": bool(optimizer_success),
        "inner_solve_evaluations": int(evaluations),
        "response_order": list(resolution.names),
        "conductors": rows,
    }


def solve_conductor_currents(
    profile: Any,
    initial_flux: Any,
    resolution: CurrentResolution,
    *,
    likelihood: InferenceLikelihood | None = None,
    solve_options: Mapping[str, Any] | None = None,
    relative_step: float = 1.0e-4,
) -> ConductorCurrentRun:
    """Run a constrained MAP outer loop through ``profile.solve(current=...)``.

    With no likelihood, the prior mean is evaluated once and reported as
    prior-dominated.  A supplied likelihood must inform every free direction
    before the result is described as recovered.
    """

    options = dict(solve_options or {})
    cache: dict[bytes, tuple[Any, np.ndarray | None, np.ndarray | None]] = {}

    def evaluate(unknown: np.ndarray):
        candidate = resolution.current(unknown)
        key = np.ascontiguousarray(candidate).tobytes()
        if key in cache:
            return cache[key]
        try:
            equilibrium = profile.solve(initial_flux, current=candidate, **options)
        except Exception as error:
            names = ", ".join(resolution.unknown_names)
            raise ConductorCurrentInfeasible(
                f"inner equilibrium failed for explicit candidate ({names})"
            ) from error
        if likelihood is None:
            result = (equilibrium, None, None)
        else:
            residual, precision = _likelihood_value(
                likelihood.evaluator(equilibrium, candidate)
            )
            result = (equilibrium, residual, precision)
        cache[key] = result
        return result

    mean = resolution.prior_mean_a
    prior_precision = np.diag(1.0 / resolution.prior_standard_deviation_a**2)

    def objective(unknown: np.ndarray) -> float:
        displacement = unknown - mean
        value = 0.5 * float(displacement @ prior_precision @ displacement)
        _equilibrium, residual, precision = evaluate(unknown)
        if residual is not None and precision is not None:
            value += 0.5 * float(residual @ precision @ residual)
        return value

    if likelihood is None:
        posterior_mean = mean.copy()
        objective_value = objective(posterior_mean)
        posterior_covariance = np.diag(resolution.prior_standard_deviation_a**2)
        rank = 0
        optimizer_success = True
        status = "prior-dominated"
    else:
        optimized = scipy.optimize.minimize(
            objective,
            mean,
            method="L-BFGS-B",
            bounds=resolution.bounds_a,
        )
        posterior_mean = np.asarray(optimized.x, dtype=float)
        objective_value = float(objective(posterior_mean))
        _equilibrium, centre_residual, centre_precision = evaluate(posterior_mean)
        assert centre_residual is not None and centre_precision is not None
        jacobian = np.empty((len(centre_residual), len(posterior_mean)), dtype=float)
        for column, (value, width, bounds) in enumerate(
            zip(
                posterior_mean,
                resolution.prior_standard_deviation_a,
                resolution.bounds_a,
                strict=True,
            )
        ):
            step = max(abs(value), width, 1.0) * relative_step
            lower, upper = bounds
            plus = posterior_mean.copy()
            minus = posterior_mean.copy()
            plus[column] = min(value + step, upper)
            minus[column] = max(value - step, lower)
            denominator = plus[column] - minus[column]
            if denominator <= 0.0:
                raise ValueError("unknown-current bounds leave no sensitivity step")
            _plus_equilibrium, plus_residual, _plus_precision = evaluate(plus)
            _minus_equilibrium, minus_residual, _minus_precision = evaluate(minus)
            assert plus_residual is not None and minus_residual is not None
            jacobian[:, column] = (plus_residual - minus_residual) / denominator
        likelihood_information = jacobian.T @ centre_precision @ jacobian
        rank = int(np.linalg.matrix_rank(likelihood_information, tol=1.0e-12))
        posterior_covariance = np.linalg.inv(prior_precision + likelihood_information)
        optimizer_success = bool(optimized.success)
        status = (
            "recovered"
            if optimizer_success and rank == len(resolution.unknown_indices)
            else "prior-dominated"
        )

    final_equilibrium, _residual, _precision = evaluate(posterior_mean)
    receipt = _receipt(
        resolution,
        posterior_mean,
        posterior_covariance,
        status=status,
        likelihood=likelihood,
        likelihood_rank=rank,
        evaluations=len(cache),
        optimizer_success=optimizer_success,
    )
    return ConductorCurrentRun(
        current_a=resolution.current(posterior_mean),
        equilibrium=final_equilibrium,
        unknown_posterior_mean_a=posterior_mean,
        unknown_posterior_covariance_a2=posterior_covariance,
        objective=objective_value,
        evaluations=len(cache),
        receipt=receipt,
    )
