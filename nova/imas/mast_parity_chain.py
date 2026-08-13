"""Assemble corrected MAST measurements into scored equilibrium slices.

The public :func:`run_parity_chain` entry point owns the reconstruction order:
corrected waveforms, a current-moment seed, the fixed-shape accelerated profile
map, and topology labels.  Reference equilibria are deliberately absent from
this module.  Comparisons against an independent reconstruction belong to the
later adjudication layer; the raw-magnetics score here is measured directly
against the corrected machine signals.

Machine geometry is immutable but campaign-specific, so callers provide the
already-constructed moment/profile solvers and a topology labeler.  The
numerical profile solve itself is not injectable: every slice goes through the
same ``vmap`` of the exact-tangent Newton--Krylov route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from nova.equilibrium.fixed_point import newton_krylov
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.measurement import SliceMeasurement
from nova.imas.mast_solve_inputs import (
    CorrectedSolveInputs,
    read_corrected_solve_inputs,
)
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.imas.parity_tolerances import (
    MagneticsBudgetClass,
    ScorecardField,
    scorecard_verdicts,
)


@dataclass(frozen=True)
class AcceleratorSettings:
    """Fixed evaluation budget for the batched profile solve."""

    newton_steps: int = 2
    gmres_iterations: int = 4
    warmup: int = 8
    relaxation: float = 0.6
    step_cap: float = 10.0

    def __post_init__(self) -> None:
        """Reject budgets that cannot produce a measured residual."""

        if self.newton_steps < 0 or self.gmres_iterations < 1 or self.warmup < 0:
            raise ValueError("accelerator iteration counts must be non-negative")
        if self.evaluation_count < 1:
            raise ValueError("accelerator budget must contain an evaluated map")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("accelerator relaxation must be in (0, 1]")
        if self.step_cap <= 0.0:
            raise ValueError("accelerator step_cap must be positive")

    @property
    def evaluation_count(self) -> int:
        """Return map evaluations represented by one residual trace."""

        return self.warmup + self.newton_steps * (2 + self.gmres_iterations)


@dataclass(frozen=True)
class TopologyLabels:
    """Fixed-shape topology observables for a batch of reconstructed slices."""

    magnetic_axis_m: np.ndarray
    x_point_m: np.ndarray
    lcfs_m: np.ndarray
    diverted: np.ndarray
    core_mask: np.ndarray
    common_scrape_off_mask: np.ndarray
    private_flux_mask: np.ndarray
    excluded_material_mask: np.ndarray

    def validate(self, slice_count: int) -> None:
        """Reject a label batch whose leading slice axes disagree."""

        arrays = {
            "magnetic_axis_m": self.magnetic_axis_m,
            "x_point_m": self.x_point_m,
            "lcfs_m": self.lcfs_m,
            "diverted": self.diverted,
            "core_mask": self.core_mask,
            "common_scrape_off_mask": self.common_scrape_off_mask,
            "private_flux_mask": self.private_flux_mask,
            "excluded_material_mask": self.excluded_material_mask,
        }
        mismatched = {
            name: np.asarray(value).shape
            for name, value in arrays.items()
            if np.asarray(value).ndim == 0 or np.asarray(value).shape[0] != slice_count
        }
        if mismatched:
            raise ValueError(
                f"topology labels do not carry {slice_count} slices: {mismatched}"
            )
        if np.asarray(self.magnetic_axis_m).shape != (slice_count, 2):
            raise ValueError("magnetic_axis_m shape must be (slice_count, 2)")
        if np.asarray(self.x_point_m).shape != (slice_count, 2):
            raise ValueError("x_point_m shape must be (slice_count, 2)")
        if np.asarray(self.lcfs_m).ndim != 3 or np.asarray(self.lcfs_m).shape[2] != 2:
            raise ValueError("lcfs_m shape must be (slice_count, point_count, 2)")


class TopologyLabeler(Protocol):
    """Machine adapter turning solved flux maps into fixed-shape labels."""

    def __call__(self, flux: jax.Array) -> TopologyLabels: ...


class TemporalScorer(Protocol):
    """Shot-level adapter for a transport flux-ledger consistency metric."""

    def __call__(
        self, inputs: CorrectedSolveInputs, flux: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class AcceleratedProfileSolve:
    """Batched fixed-point states and their solve-health traces."""

    flux: np.ndarray
    residual: np.ndarray
    trace: np.ndarray
    elapsed_s: float

    @property
    def slice_count(self) -> int:
        """Return the number of reconstructed slices."""

        return int(self.flux.shape[0])


@dataclass(frozen=True)
class GeometryScores:
    """Per-slice geometry observables emitted without a fitted reference."""

    magnetic_axis_m: np.ndarray
    lcfs_m: np.ndarray
    x_point_m: np.ndarray
    diverted: np.ndarray
    seed_to_solved_lcfs_distance_m: np.ndarray


@dataclass(frozen=True)
class PhysicsScores:
    """Per-slice force-balance and corrected-machine residual metrics."""

    profile_residual: np.ndarray
    whitened_raw_magnetics_residual: np.ndarray
    fixed_point_defect: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )


@dataclass(frozen=True)
class SolveHealthScores:
    """Per-slice convergence, confinement, work and throughput metrics."""

    convergence_fraction: np.ndarray
    confinement_fraction: np.ndarray
    iteration_count: np.ndarray
    throughput_slices_per_second: np.ndarray


@dataclass(frozen=True)
class TemporalScores:
    """Per-slice transport consistency supplied by the diffusion adapter."""

    current_diffusion_flux_ledger_consistency: np.ndarray


@dataclass(frozen=True)
class SliceScorecard:
    """The four metric groups carried beside every reconstructed slice."""

    shot: int
    time_s: np.ndarray
    geometry: GeometryScores
    physics: PhysicsScores
    solve_health: SolveHealthScores
    temporal: TemporalScores
    registered_metrics: Mapping[str, float]
    magnetics_budget: MagneticsBudgetClass = MagneticsBudgetClass.SAME_SOURCE
    verdicts: Mapping[str, bool] = field(init=False)

    def __post_init__(self) -> None:
        """Validate and adjudicate the complete registered metric schema."""

        metrics = {
            str(name): float(value) for name, value in self.registered_metrics.items()
        }
        verdicts = scorecard_verdicts(metrics, self.magnetics_budget)
        object.__setattr__(self, "registered_metrics", MappingProxyType(metrics))
        object.__setattr__(self, "verdicts", verdicts)

    @property
    def slice_count(self) -> int:
        """Return the scorecard row count."""

        return int(self.time_s.size)

    def as_dict(self) -> dict[str, Any]:
        """Return named groups for artifact serialization."""

        return {
            "shot": self.shot,
            "time_s": self.time_s,
            "geometry": asdict(self.geometry),
            "physics": asdict(self.physics),
            "solve_health": asdict(self.solve_health),
            "temporal": asdict(self.temporal),
            "registered_metrics": dict(self.registered_metrics),
            "verdicts": dict(self.verdicts),
        }


@dataclass(frozen=True)
class ParityChainResult:
    """Corrected inputs, both reconstruction stages, labels and scorecard."""

    inputs: CorrectedSolveInputs
    moment_seeds: tuple[Any, ...]
    solve: AcceleratedProfileSolve
    topology: TopologyLabels
    scorecard: SliceScorecard


def _sensor_scales(signals: np.ndarray, supplied: np.ndarray | None) -> np.ndarray:
    """Return positive per-sensor whitening scales over the complete shot."""

    sensor_count = signals.shape[1]
    if supplied is not None:
        scale = np.asarray(supplied, dtype=float)
        if scale.shape != (sensor_count,):
            raise ValueError(f"sensor_scale shape must be ({sensor_count},)")
    else:
        scale = np.nanstd(signals, axis=0)
        amplitude = np.nanmax(np.abs(signals), axis=0)
        scale = np.where(scale > 0.0, scale, amplitude)
    finite = scale[np.isfinite(scale) & (scale > 0.0)]
    floor = max(float(np.min(finite)) * 1.0e-9, 1.0e-30) if finite.size else 1.0
    return np.where(np.isfinite(scale) & (scale > floor), scale, floor)


def _pack_source_currents(
    profile_solver: Any, inputs: CorrectedSolveInputs
) -> np.ndarray:
    """Reorder corrected conductor currents into the solver's fixed source order."""

    columns = {name: index for index, name in enumerate(inputs.coil_channels)}
    missing = set(profile_solver.source_names).difference(columns)
    if missing:
        raise ValueError(
            "corrected inputs are missing profile sources " + ", ".join(sorted(missing))
        )
    return np.stack(
        [
            inputs.coil_currents_a[:, columns[name]]
            for name in profile_solver.source_names
        ],
        axis=1,
    )


def _moment_seeds(
    moment_solver: Any,
    profile_solver: Any,
    inputs: CorrectedSolveInputs,
    source_current: np.ndarray,
    scale: np.ndarray,
) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Run the host moment seed and return its aligned solve ingredients."""

    vacuum_sensor = source_current @ np.asarray(profile_solver.source_to_sensor).T
    vacuum_flux = source_current @ np.asarray(profile_solver.source_to_grid).T
    mask = np.isfinite(inputs.sensor_signals)
    seeds = []
    for index in range(inputs.slice_count):
        measurement = SliceMeasurement(
            measured=inputs.sensor_signals[index],
            vacuum=vacuum_sensor[index],
            mask=mask[index],
            scale=scale,
            plasma_current=float(inputs.plasma_current_a[index]),
            vacuum_flux=vacuum_flux[index],
        )
        seeds.append(moment_solver.solve(measurement))
    initial_flux = np.stack([np.asarray(seed.flux).reshape(-1) for seed in seeds])
    expected = (inputs.slice_count, np.asarray(profile_solver.source_to_grid).shape[0])
    if initial_flux.shape != expected:
        raise ValueError(
            f"moment seed flux shape {initial_flux.shape} must be {expected}"
        )
    return tuple(seeds), initial_flux, mask, vacuum_sensor


def _accelerated_profile_solve(
    profile_solver: Any,
    source_current: np.ndarray,
    inputs: CorrectedSolveInputs,
    scale: np.ndarray,
    mask: np.ndarray,
    initial_flux: np.ndarray,
    settings: AcceleratorSettings,
) -> AcceleratedProfileSolve:
    """Run the fixed-shape profile map through one leading-axis ``vmap``."""

    source = jnp.asarray(source_current, dtype=profile_solver.source_to_grid.dtype)
    plasma = jnp.asarray(
        inputs.plasma_current_a, dtype=profile_solver.source_to_grid.dtype
    )
    measured = jnp.asarray(
        inputs.sensor_signals, dtype=profile_solver.source_to_grid.dtype
    )
    scales = jnp.broadcast_to(
        jnp.asarray(scale, dtype=profile_solver.source_to_grid.dtype), measured.shape
    )
    masks = jnp.asarray(mask, dtype=bool)
    initial = jnp.asarray(initial_flux, dtype=profile_solver.source_to_grid.dtype)

    def solve_slice(source_row, plasma_value, measured_row, scale_row, mask_row, seed):
        map_fn = profile_solver.least_squares_map(
            source_row, plasma_value, measured_row, scale_row, mask_row
        )
        return newton_krylov(
            map_fn,
            seed,
            newton_steps=settings.newton_steps,
            gmres_iterations=settings.gmres_iterations,
            warmup=settings.warmup,
            relaxation=settings.relaxation,
            step_cap=settings.step_cap,
            precision=profile_solver.precision,
        )

    started = perf_counter()
    result = jax.vmap(solve_slice)(source, plasma, measured, scales, masks, initial)
    jax.block_until_ready(result.state)
    elapsed = perf_counter() - started
    return AcceleratedProfileSolve(
        flux=np.asarray(result.state),
        residual=np.asarray(result.residual),
        trace=np.asarray(result.trace),
        elapsed_s=elapsed,
    )


def _raw_magnetics_residuals(
    profile_solver: Any,
    source_current: np.ndarray,
    inputs: CorrectedSolveInputs,
    scale: np.ndarray,
    mask: np.ndarray,
    flux: np.ndarray,
) -> np.ndarray:
    """Score each solved profile directly against corrected machine signals."""

    residuals = []
    for index, state in enumerate(flux):
        source = jnp.asarray(
            source_current[index], dtype=profile_solver.source_to_grid.dtype
        )
        plasma = jnp.asarray(
            inputs.plasma_current_a[index], dtype=profile_solver.source_to_grid.dtype
        )
        measured = jnp.asarray(
            inputs.sensor_signals[index], dtype=profile_solver.source_to_grid.dtype
        )
        scales = jnp.asarray(scale, dtype=profile_solver.source_to_grid.dtype)
        keep = jnp.asarray(mask[index], dtype=bool)
        basis, _labels = profile_solver._profile_basis(jnp.asarray(state))
        coefficients = profile_solver._least_squares_coefficients(
            basis, source, plasma, measured, scales, keep
        )
        predicted = profile_solver.source_to_sensor @ source + (
            profile_solver.plasma_to_sensor @ (basis @ coefficients)
        )
        whitened = jnp.where(keep, (predicted - measured) / scales, 0.0)
        count = jnp.maximum(jnp.sum(keep), 1)
        residuals.append(jnp.sqrt(jnp.sum(whitened**2) / count))
    return np.asarray(jnp.stack(residuals))


def _profile_reproduction_residuals(
    profile_solver: Any,
    source_current: np.ndarray,
    inputs: CorrectedSolveInputs,
    scale: np.ndarray,
    mask: np.ndarray,
    flux: np.ndarray,
) -> np.ndarray:
    """Compare Nova profile current with an independent grid-operator read.

    Each row is ``RMS(j_phi_nova - j_phi_grid) / max(abs(j_phi_grid))``.
    ``j_phi_nova`` comes from the fitted profile basis, while ``j_phi_grid``
    is read independently by applying the structured-grid Grad--Shafranov
    operator to the plasma-generated part of the solved flux.  Only interior
    in-limiter cells with a complete central-difference stencil are scored.
    """

    grid_r = np.asarray(profile_solver.grid_r, dtype=float)
    grid_z = np.asarray(profile_solver.grid_z, dtype=float)
    grid_shape = (grid_z.size, grid_r.size)
    if grid_r.ndim != 1 or grid_z.ndim != 1 or min(grid_shape) < 3:
        return np.full(inputs.slice_count, np.nan)
    radial_step = np.diff(grid_r)
    vertical_step = np.diff(grid_z)
    if not (
        np.all(radial_step > 0.0)
        and np.all(vertical_step > 0.0)
        and np.allclose(radial_step, radial_step[0], rtol=1.0e-12)
        and np.allclose(vertical_step, vertical_step[0], rtol=1.0e-12)
    ):
        raise ValueError("profile reproduction requires a uniform structured grid")

    cell_area = np.asarray(profile_solver.cell_area, dtype=float)
    inside = np.asarray(profile_solver.inside_limiter, dtype=bool)
    if cell_area.shape != (grid_r.size * grid_z.size,):
        raise ValueError("cell_area does not match the profile grid")
    if inside.shape != grid_shape:
        raise ValueError("inside_limiter does not match the profile grid")

    interior = np.zeros(grid_shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    keep = interior & inside
    radius = grid_r[np.newaxis, :]
    residuals: list[float] = []
    for index, state in enumerate(np.asarray(flux)):
        source = jnp.asarray(
            source_current[index], dtype=profile_solver.source_to_grid.dtype
        )
        plasma = jnp.asarray(
            inputs.plasma_current_a[index], dtype=profile_solver.source_to_grid.dtype
        )
        measured = jnp.asarray(
            inputs.sensor_signals[index], dtype=profile_solver.source_to_grid.dtype
        )
        scales = jnp.asarray(scale, dtype=profile_solver.source_to_grid.dtype)
        measured_mask = jnp.asarray(mask[index], dtype=bool)
        state_array = jnp.asarray(state, dtype=profile_solver.source_to_grid.dtype)
        basis, _labels = profile_solver._profile_basis(state_array)
        coefficients = profile_solver._least_squares_coefficients(
            basis, source, plasma, measured, scales, measured_mask
        )
        nova_current_density = np.asarray(basis @ coefficients) / cell_area

        vacuum_flux = np.asarray(profile_solver.source_to_grid @ source)
        plasma_flux = (np.asarray(state) - vacuum_flux).reshape(grid_shape)
        radial_first = (plasma_flux[:, 2:] - plasma_flux[:, :-2]) / (
            2.0 * radial_step[0]
        )
        radial_second = (
            plasma_flux[:, 2:] - 2.0 * plasma_flux[:, 1:-1] + plasma_flux[:, :-2]
        ) / radial_step[0] ** 2
        vertical_second = (
            plasma_flux[2:, :] - 2.0 * plasma_flux[1:-1, :] + plasma_flux[:-2, :]
        ) / vertical_step[0] ** 2
        delta_star = (
            radial_second[1:-1]
            - radial_first[1:-1] / radius[:, 1:-1]
            + vertical_second[:, 1:-1]
        )
        grid_current_density = np.full(grid_shape, np.nan)
        grid_current_density[1:-1, 1:-1] = -delta_star / (
            TOTAL_FLUX_FACTOR * mu_0 * radius[:, 1:-1]
        )
        finite = (
            keep
            & np.isfinite(grid_current_density)
            & np.isfinite(nova_current_density.reshape(grid_shape))
        )
        reference = grid_current_density[finite]
        if not reference.size:
            residuals.append(float("nan"))
            continue
        reference_scale = float(np.max(np.abs(reference)))
        if not np.isfinite(reference_scale) or reference_scale <= 0.0:
            residuals.append(float("nan"))
            continue
        difference = (
            nova_current_density.reshape(grid_shape)[finite] - reference
        ) / reference_scale
        residuals.append(float(np.sqrt(np.mean(np.square(difference)))))
    return np.asarray(residuals)


def _lcfs_distance(seed: Any, solved_lcfs: np.ndarray) -> float:
    """Return symmetric nearest-point distance between seed and solved boundaries."""

    seed_ring = getattr(seed, "ring", None)
    if seed_ring is None:
        return float("nan")
    first = np.asarray(seed_ring, dtype=float)
    second = np.asarray(solved_lcfs, dtype=float)
    first = first[np.all(np.isfinite(first), axis=1)]
    second = second[np.all(np.isfinite(second), axis=1)]
    if not first.size or not second.size:
        return float("nan")
    separation = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    forward = np.mean(np.min(separation, axis=0))
    reverse = np.mean(np.min(separation, axis=1))
    return float(0.5 * (forward + reverse))


def _convergence_fraction(trace: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Return the fractional residual reduction from first to final read."""

    trace_array = np.asarray(trace, dtype=float)
    final_array = np.asarray(final, dtype=float)
    if trace_array.ndim != 2 or trace_array.shape[0] != final_array.shape[0]:
        raise ValueError("residual trace rows must match the final residual row count")
    finite = np.isfinite(trace_array)
    has_finite = np.any(finite, axis=1)
    first_index = np.argmax(finite, axis=1)
    first = np.full(final_array.shape, np.nan, dtype=float)
    rows = np.flatnonzero(has_finite)
    first[rows] = trace_array[rows, first_index[rows]]
    ratio = np.divide(
        final_array,
        first,
        out=np.ones_like(final_array, dtype=float),
        where=has_finite & np.isfinite(final_array) & (first > 0.0),
    )
    return np.clip(1.0 - ratio, 0.0, 1.0)


def _registered_scorecard_metrics(
    solve: AcceleratedProfileSolve,
    profile_reproduction: np.ndarray,
    raw_residual: np.ndarray,
    topology: TopologyLabels,
    temporal: np.ndarray,
    accelerator: AcceleratorSettings,
) -> Mapping[str, float]:
    """Reduce emitted observations to the registered shot-level metric schema.

    Geometry comparisons require the independent reference reconstruction, which
    is intentionally absent from this chain.  Those fields remain explicit NaNs
    and therefore receive failing verdicts until the referee layer supplies the
    comparisons; they are never replaced by seed-relative proxy scores.
    """

    reference_dependent = float("nan")
    core = np.asarray(topology.core_mask, dtype=bool).reshape(solve.slice_count, -1)
    confined = np.any(core, axis=1)
    converged = np.isfinite(solve.residual) & (solve.residual < 1.0e-8)
    throughput = solve.slice_count / max(solve.elapsed_s, np.finfo(float).tiny)
    return MappingProxyType(
        {
            ScorecardField.MAGNETIC_AXIS_DISTANCE_M.value: reference_dependent,
            ScorecardField.LCFS_DISTANCE_M.value: reference_dependent,
            ScorecardField.X_POINT_DISTANCE_M.value: reference_dependent,
            ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION.value: (
                reference_dependent
            ),
            ScorecardField.PROFILE_RESIDUAL_RMS.value: float(
                np.median(profile_reproduction)
            ),
            ScorecardField.FIXED_POINT_DEFECT.value: float(
                np.max(np.abs(solve.residual))
            ),
            ScorecardField.MAGNETICS_RESIDUAL_WHITENED_RMS.value: float(
                np.sqrt(np.mean(np.square(raw_residual)))
            ),
            ScorecardField.CONVERGED_FRACTION.value: float(np.mean(converged)),
            ScorecardField.CONFINED_FRACTION.value: float(np.mean(confined)),
            ScorecardField.ITERATION_COUNT.value: float(accelerator.evaluation_count),
            ScorecardField.THROUGHPUT_SLICES_PER_CORE_S.value: float(throughput),
            ScorecardField.CURRENT_DIFFUSION_FLUX_LEDGER_RMS_FRACTION.value: float(
                np.sqrt(np.mean(np.square(temporal)))
            ),
        }
    )


def run_parity_chain(
    shot: int,
    *,
    moment_solver: Any,
    profile_solver: Any,
    topology_labeler: TopologyLabeler,
    temporal_scorer: TemporalScorer | None = None,
    sensor_scale: np.ndarray | None = None,
    accelerator: AcceleratorSettings = AcceleratorSettings(),
    magnetics_budget: MagneticsBudgetClass = MagneticsBudgetClass.SAME_SOURCE,
    store: Path | str = SHOT_STORE,
) -> ParityChainResult:
    """Reconstruct and score every corrected slice of one MAST shot.

    ``topology_labeler`` owns the machine-specific fixed-shape null and LCFS
    extraction.  ``temporal_scorer`` owns the separate diffusion ledger; when
    absent, its score is explicitly unavailable (NaN), never silently replaced
    by a proxy.  The default reader is the promoted corrected-waveform door.
    """

    inputs = read_corrected_solve_inputs(int(shot), store=store)
    if inputs.slice_count == 0:
        raise ValueError(f"shot {shot} carries no corrected solve slices")
    source_current = _pack_source_currents(profile_solver, inputs)
    scale = _sensor_scales(inputs.sensor_signals, sensor_scale)
    seeds, initial_flux, mask, _vacuum_sensor = _moment_seeds(
        moment_solver, profile_solver, inputs, source_current, scale
    )
    solve = _accelerated_profile_solve(
        profile_solver,
        source_current,
        inputs,
        scale,
        mask,
        initial_flux,
        accelerator,
    )
    topology = topology_labeler(jnp.asarray(solve.flux))
    topology.validate(inputs.slice_count)

    raw_residual = _raw_magnetics_residuals(
        profile_solver, source_current, inputs, scale, mask, solve.flux
    )
    profile_reproduction = _profile_reproduction_residuals(
        profile_solver, source_current, inputs, scale, mask, solve.flux
    )
    if temporal_scorer is None:
        temporal = np.full(inputs.slice_count, np.nan)
    else:
        temporal = np.asarray(temporal_scorer(inputs, solve.flux), dtype=float)
        if temporal.shape != (inputs.slice_count,):
            raise ValueError(
                "temporal scorer shape "
                f"{temporal.shape} must be ({inputs.slice_count},)"
            )

    core = np.asarray(topology.core_mask, dtype=bool)
    confinement = core.reshape(inputs.slice_count, -1).mean(axis=1)
    throughput = inputs.slice_count / max(solve.elapsed_s, np.finfo(float).tiny)
    scorecard = SliceScorecard(
        shot=int(shot),
        time_s=np.asarray(inputs.time_s),
        geometry=GeometryScores(
            magnetic_axis_m=np.asarray(topology.magnetic_axis_m),
            lcfs_m=np.asarray(topology.lcfs_m),
            x_point_m=np.asarray(topology.x_point_m),
            diverted=np.asarray(topology.diverted, dtype=bool),
            seed_to_solved_lcfs_distance_m=np.asarray(
                [
                    _lcfs_distance(seed, topology.lcfs_m[index])
                    for index, seed in enumerate(seeds)
                ]
            ),
        ),
        physics=PhysicsScores(
            profile_residual=profile_reproduction,
            whitened_raw_magnetics_residual=raw_residual,
            fixed_point_defect=np.asarray(solve.residual),
        ),
        solve_health=SolveHealthScores(
            convergence_fraction=_convergence_fraction(solve.trace, solve.residual),
            confinement_fraction=confinement,
            iteration_count=np.full(inputs.slice_count, accelerator.evaluation_count),
            throughput_slices_per_second=np.full(inputs.slice_count, throughput),
        ),
        temporal=TemporalScores(current_diffusion_flux_ledger_consistency=temporal),
        registered_metrics=_registered_scorecard_metrics(
            solve,
            profile_reproduction,
            raw_residual,
            topology,
            temporal,
            accelerator,
        ),
        magnetics_budget=magnetics_budget,
    )
    return ParityChainResult(inputs, seeds, solve, topology, scorecard)


__all__ = [
    "AcceleratedProfileSolve",
    "AcceleratorSettings",
    "GeometryScores",
    "ParityChainResult",
    "PhysicsScores",
    "SliceScorecard",
    "SolveHealthScores",
    "TemporalScores",
    "TopologyLabels",
    "run_parity_chain",
]
