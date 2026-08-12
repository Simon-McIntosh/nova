"""End-to-end contract for the corrected reconstruction-chain entry point."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import nova.imas.mast_parity_chain as chain_module
from nova.imas.mast_parity_chain import (
    AcceleratorSettings,
    TopologyLabels,
    run_parity_chain,
)
from nova.imas.mast_solve_inputs import CorrectedSolveInputs
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
    assert groups["geometry"]["magnetic_axis_m"].shape == (SLICE_COUNT, 2)
    assert groups["geometry"]["lcfs_m"].shape == (SLICE_COUNT, 8, 2)
    assert groups["geometry"]["x_point_m"].shape == (SLICE_COUNT, 2)
    assert groups["geometry"]["diverted"].shape == (SLICE_COUNT,)
    assert groups["physics"]["profile_residual"].shape == (SLICE_COUNT,)
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
