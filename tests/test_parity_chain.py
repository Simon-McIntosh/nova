"""End-to-end contract for the corrected reconstruction-chain entry point."""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
import pytest

import nova.imas.mast_parity_chain as chain_module
from nova.imas.mast_parity_chain import (
    AcceleratorSettings,
    TopologyLabels,
    run_parity_chain,
)
from nova.imas.mast_solve_inputs import CorrectedSolveInputs
from nova.imas.parity_tolerances import SCORECARD_FIELDS
from nova.jax.config import Precision


PILOT_SHOT = 21978
SLICE_COUNT = 3
GRID_SIZE = 4


@dataclass(frozen=True)
class Seed:
    """Minimal physical seed returned by the moment-stage test double."""

    flux: np.ndarray
    ring: np.ndarray


class MomentSolver:
    """Record corrected measurements and provide one finite flux seed each."""

    def __init__(self):
        self.measurements = []

    def solve(self, measurement):
        self.measurements.append(measurement)
        level = float(measurement.plasma_current) / 1.0e5
        angle = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        return Seed(
            flux=np.full(GRID_SIZE, level),
            ring=np.column_stack([1.0 + 0.3 * np.cos(angle), 0.4 * np.sin(angle)]),
        )


class ProfileSolver:
    """Small differentiable contraction with the production profile interface."""

    source_names = ("upper", "lower")
    precision = Precision.DOUBLE

    def __init__(self):
        self.grid_r = jnp.asarray([1.0, 2.0])
        self.grid_z = jnp.asarray([-1.0, 1.0])
        self.source_to_grid = jnp.asarray(
            [[1.0e-5, 0.0], [0.0, 1.0e-5], [1.0e-5, 0.0], [0.0, 1.0e-5]]
        )
        self.source_to_sensor = jnp.asarray([[1.0e-6, 0.0], [0.0, 1.0e-6]])
        self.plasma_to_sensor = jnp.asarray(
            [[2.0e-6, 0.0, 0.0, 0.0], [0.0, 2.0e-6, 0.0, 0.0]]
        )

    def least_squares_map(self, source, plasma, measured, scale, mask):
        del measured, scale, mask
        target = self.source_to_grid @ source + jnp.full(GRID_SIZE, plasma / 4.0e5)
        return lambda flux: 0.25 * flux + 0.75 * target

    def _profile_basis(self, flux):
        del flux
        return jnp.eye(GRID_SIZE), {}

    def _least_squares_coefficients(self, basis, source, plasma, measured, scale, mask):
        del basis, plasma, scale, mask
        target = measured - self.source_to_sensor @ source
        return jnp.asarray([target[0] / 2.0e-6, target[1] / 2.0e-6, 0.0, 0.0])


def corrected_inputs(shot, *, store):
    """Return three corrected pilot slices from the sole reader seam."""

    assert shot == PILOT_SHOT
    assert str(store) == "/pilot"
    time = np.array([0.10, 0.11, 0.12])
    currents = np.array([[10.0, -8.0], [11.0, -7.0], [12.0, -6.0]])
    plasma = np.array([1.0e5, 1.1e5, 1.2e5])
    sensors = np.column_stack(
        [1.0e-6 * currents[:, 0] + 0.5 * plasma, 1.0e-6 * currents[:, 1] + 0.5 * plasma]
    )
    return CorrectedSolveInputs(
        shot=shot,
        time_s=time,
        coil_channels=("upper", "lower"),
        coil_currents_a=currents,
        sensor_channels=("probe", "loop"),
        sensor_signals=sensors,
        sensor_units=("T", "Wb"),
        plasma_current_a=plasma,
        corrections=(),
    )


def label_topology(flux):
    """Emit fixed-size axis, null, boundary and four-domain labels."""

    count = flux.shape[0]
    angle = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    lcfs = np.stack(
        [
            np.column_stack(
                [1.0 + (0.30 + 0.01 * index) * np.cos(angle), 0.4 * np.sin(angle)]
            )
            for index in range(count)
        ]
    )
    core = np.array([[True, True, False, False]] * count)
    return TopologyLabels(
        magnetic_axis_m=np.column_stack([np.ones(count), np.zeros(count)]),
        x_point_m=np.column_stack([np.ones(count), -0.4 * np.ones(count)]),
        lcfs_m=lcfs,
        diverted=np.array([True, False, True]),
        core_mask=core,
        common_scrape_off_mask=~core,
        private_flux_mask=np.zeros_like(core),
        excluded_material_mask=np.zeros_like(core),
    )


def flux_ledger(inputs, flux):
    """Return one finite transport consistency score per pilot slice."""

    assert inputs.slice_count == flux.shape[0]
    return np.array([0.01, 0.02, 0.03])


def test_pilot_shot_runs_the_scored_chain_end_to_end(monkeypatch):
    """One entry point returns all four score groups beside a batched solve."""

    monkeypatch.setattr(chain_module, "read_corrected_solve_inputs", corrected_inputs)
    registry_calls = []
    registry_verdicts = chain_module.scorecard_verdicts

    def record_registry_call(metrics, magnetics_budget):
        registry_calls.append((set(metrics), magnetics_budget))
        return registry_verdicts(metrics, magnetics_budget)

    monkeypatch.setattr(chain_module, "scorecard_verdicts", record_registry_call)
    reproduction = np.array([0.011, 0.022, 0.033])
    monkeypatch.setattr(
        chain_module,
        "_profile_reproduction_residuals",
        lambda *_args: reproduction,
    )
    moment = MomentSolver()
    settings = AcceleratorSettings(
        newton_steps=1,
        gmres_iterations=GRID_SIZE,
        warmup=2,
        relaxation=0.5,
    )
    result = run_parity_chain(
        PILOT_SHOT,
        moment_solver=moment,
        profile_solver=ProfileSolver(),
        topology_labeler=label_topology,
        temporal_scorer=flux_ledger,
        accelerator=settings,
        store="/pilot",
    )

    assert result.inputs.slice_count == SLICE_COUNT
    assert result.solve.slice_count == SLICE_COUNT
    assert result.scorecard.slice_count == SLICE_COUNT
    assert len(result.moment_seeds) == SLICE_COUNT
    assert len(moment.measurements) == SLICE_COUNT
    np.testing.assert_allclose(
        [row.measured for row in moment.measurements], result.inputs.sensor_signals
    )

    groups = result.scorecard.as_dict()
    assert {"geometry", "physics", "solve_health", "temporal"} <= groups.keys()
    assert set(groups["registered_metrics"]) == SCORECARD_FIELDS
    assert set(groups["verdicts"]) == SCORECARD_FIELDS
    assert registry_calls == [(SCORECARD_FIELDS, result.scorecard.magnetics_budget)]
    assert groups["geometry"]["magnetic_axis_m"].shape == (SLICE_COUNT, 2)
    assert groups["geometry"]["lcfs_m"].shape == (SLICE_COUNT, 8, 2)
    assert groups["geometry"]["x_point_m"].shape == (SLICE_COUNT, 2)
    assert groups["geometry"]["diverted"].shape == (SLICE_COUNT,)
    assert groups["physics"]["profile_residual"].shape == (SLICE_COUNT,)
    assert groups["physics"]["fixed_point_defect"].shape == (SLICE_COUNT,)
    assert groups["physics"]["whitened_raw_magnetics_residual"].shape == (SLICE_COUNT,)
    assert groups["solve_health"]["convergence_fraction"].shape == (SLICE_COUNT,)
    assert groups["solve_health"]["confinement_fraction"].shape == (SLICE_COUNT,)
    assert (
        groups["solve_health"]["iteration_count"].tolist()
        == [settings.evaluation_count] * SLICE_COUNT
    )
    assert np.all(groups["solve_health"]["throughput_slices_per_second"] > 0.0)
    np.testing.assert_allclose(
        groups["temporal"]["current_diffusion_flux_ledger_consistency"],
        [0.01, 0.02, 0.03],
    )
    assert result.solve.trace.shape == (SLICE_COUNT, settings.evaluation_count)
    assert np.all(np.isfinite(result.solve.flux))
    assert np.all(result.solve.residual < 1.0e-8)
    np.testing.assert_allclose(groups["physics"]["profile_residual"], reproduction)
    np.testing.assert_allclose(
        groups["physics"]["fixed_point_defect"], result.solve.residual
    )
    assert result.scorecard.registered_metrics["profile_residual_rms"] == pytest.approx(
        np.median(reproduction)
    )
    assert result.scorecard.registered_metrics["fixed_point_defect"] == pytest.approx(
        np.max(np.abs(result.solve.residual))
    )
    assert result.scorecard.registered_metrics["profile_residual_rms"] != pytest.approx(
        result.scorecard.registered_metrics["fixed_point_defect"]
    )


def test_nonfinite_trace_row_reports_zero_convergence():
    trace = np.array([[np.nan, np.nan], [4.0, 2.0]])
    final = np.array([np.nan, 1.0])

    fraction = chain_module._convergence_fraction(trace, final)

    np.testing.assert_allclose(fraction, [0.0, 0.75])


def test_profile_reproduction_is_current_density_rms_against_grid_read():
    grid_r = np.linspace(1.0, 2.0, 5)
    grid_z = np.linspace(-1.0, 1.0, 5)
    radius, _height = np.meshgrid(grid_r, grid_z)
    current_density = 7.0
    plasma_flux = (
        -chain_module.TOTAL_FLUX_FACTOR
        * chain_module.mu_0
        * current_density
        * radius**3
        / 3.0
    )

    class CurrentReadSolver:
        def __init__(self):
            self.grid_r = jnp.asarray(grid_r)
            self.grid_z = jnp.asarray(grid_z)
            self.cell_area = jnp.ones(grid_r.size * grid_z.size)
            self.inside_limiter = jnp.ones((grid_z.size, grid_r.size), dtype=bool)
            self.source_to_grid = jnp.zeros((grid_r.size * grid_z.size, 1))

        def _profile_basis(self, _flux):
            return jnp.zeros((grid_r.size * grid_z.size, 1)), {}

        def _least_squares_coefficients(self, *_args):
            return jnp.zeros(1)

    inputs = replace(
        corrected_inputs(PILOT_SHOT, store="/pilot"),
        time_s=np.array([0.1]),
        coil_currents_a=np.zeros((1, 2)),
        sensor_signals=np.zeros((1, 2)),
        plasma_current_a=np.array([1.0]),
    )
    radial_step = grid_r[1] - grid_r[0]
    grid_current = current_density * (1.0 - radial_step**2 / (3.0 * grid_r[1:-1] ** 2))
    expected = np.sqrt(np.mean(np.square(grid_current))) / np.max(grid_current)

    observed = chain_module._profile_reproduction_residuals(
        CurrentReadSolver(),
        np.zeros((1, 1)),
        inputs,
        np.ones(2),
        np.ones((1, 2), dtype=bool),
        plasma_flux.reshape(1, -1),
    )

    np.testing.assert_allclose(observed, [expected], rtol=1.0e-12)


def test_scorecard_refuses_an_unregistered_metric(monkeypatch):
    """An unknown metric cannot reach an accepted scorecard or verdict map."""

    monkeypatch.setattr(chain_module, "read_corrected_solve_inputs", corrected_inputs)
    result = run_parity_chain(
        PILOT_SHOT,
        moment_solver=MomentSolver(),
        profile_solver=ProfileSolver(),
        topology_labeler=label_topology,
        temporal_scorer=flux_ledger,
        accelerator=AcceleratorSettings(
            newton_steps=1,
            gmres_iterations=GRID_SIZE,
            warmup=2,
            relaxation=0.5,
        ),
        store="/pilot",
    )
    metrics = dict(result.scorecard.registered_metrics)
    metrics["unregistered_metric"] = 0.0

    with pytest.raises(ValueError, match="unregistered fields"):
        replace(result.scorecard, registered_metrics=metrics)
