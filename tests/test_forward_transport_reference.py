"""Published TORAX ramp-up case through the public transport facade."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import time

import jax
import numpy as np

import nova.transport.forward as forward_module
from nova.transport import (
    ForwardTransport,
    ForwardTransportInput,
    TransportGeometry,
    TransportModel,
    TransportRung,
    TransportState,
    TransportWaveforms,
)

jax.config.update("jax_platforms", "cpu")


REFERENCE_INTERVAL_SECONDS = 2.0
TRAJECTORY_INTERVAL_SECONDS = 20.0
MOVING_GEOMETRY_FINAL_FIELD_RATIO = 0.98
MINIMUM_TRAJECTORY_STEPS = 10
PROFILE_RELATIVE_TOLERANCE = 1.0e-10
PROFILE_ABSOLUTE_TOLERANCE = 1.0e-12
DENSITY_ABSOLUTE_TOLERANCE = 1.0e8


def _published_config(final_time=REFERENCE_INTERVAL_SECONDS):
    from torax.examples.iterhybrid_rampup import CONFIG

    config = copy.deepcopy(CONFIG)
    config["numerics"]["t_final"] = final_time
    config["numerics"]["exact_t_final"] = True
    return config


def _moving_geometry_config():
    config = _published_config(TRAJECTORY_INTERVAL_SECONDS)
    geometry = config["geometry"]
    initial_field = geometry.pop("B_0")
    geometry["calcphibdot"] = True
    geometry["geometry_configs"] = {
        0.0: {"B_0": initial_field},
        TRAJECTORY_INTERVAL_SECONDS: {
            "B_0": initial_field * MOVING_GEOMETRY_FINAL_FIELD_RATIO
        },
    }
    return config


def _initial_state(config):
    from torax._src.orchestration.initial_state import (
        get_initial_state_and_post_processed_outputs,
    )

    step_fn = forward_module._make_torax_step(config)
    initial, _post_processed = get_initial_state_and_post_processed_outputs(step_fn)
    profiles = initial.core_profiles
    rho = np.concatenate(([0.0], np.asarray(initial.geometry.rho_norm), [1.0]))
    return TransportState(
        rho=rho,
        psi=np.asarray(profiles.psi.cell_plus_boundaries()),
        ion_temperature=np.asarray(profiles.T_i.cell_plus_boundaries()),
        electron_temperature=np.asarray(profiles.T_e.cell_plus_boundaries()),
        electron_density=np.asarray(profiles.n_e.cell_plus_boundaries()),
    )


def _published_request(
    config_data,
    config,
    interval_seconds=REFERENCE_INTERVAL_SECONDS,
):
    initial_state = _initial_state(config)
    initial_current = 3.0e6
    final_current = initial_current + (10.5e6 - initial_current) * (
        interval_seconds / 80.0
    )
    return ForwardTransportInput(
        geometry=TransportGeometry({"valid": True}),
        initial_state=initial_state,
        waveforms=TransportWaveforms(
            time=np.array([0.0, interval_seconds]),
            plasma_current=np.array([initial_current, final_current]),
        ),
        model=TransportModel(
            TransportRung.TORAX_MULTI_CHANNEL,
            torax_config=config_data,
        ),
    )


def _numeric_reference_candidates():
    specification = importlib.util.find_spec("torax")
    assert specification is not None
    assert specification.submodule_search_locations is not None
    package_root = Path(next(iter(specification.submodule_search_locations)))
    numeric_suffixes = {".csv", ".json", ".nc", ".npy", ".npz", ".zarr"}
    return tuple(
        path
        for path in package_root.rglob("*")
        if path.suffix.lower() in numeric_suffixes
        and ("iterhybrid" in path.name.lower() or "rampup" in path.name.lower())
    )


def test_published_iterhybrid_rampup_matches_torax_entry_point(monkeypatch):
    """The published physics and CHEASE provider survive the typed facade."""
    from torax._src.orchestration.run_simulation import run_simulation
    from torax._src.torax_pydantic.model_config import ToraxConfig

    assert not _numeric_reference_candidates()
    config_data = _published_config()
    direct_config = ToraxConfig.from_dict(copy.deepcopy(config_data))
    request = _published_request(config_data, direct_config)
    config_snapshot = copy.deepcopy(config_data)

    def reject_nova_geometry(_record):
        raise AssertionError("the published case must retain its CHEASE provider")

    monkeypatch.setattr(
        forward_module,
        "torax_geometry_from_fsa",
        reject_nova_geometry,
    )
    facade_start = time.perf_counter()
    receipt = ForwardTransport().solve(request)
    facade_wall_seconds = time.perf_counter() - facade_start

    direct_output, direct_history = run_simulation(
        direct_config,
        progress_bar=False,
    )
    assert direct_history.sim_error.name == "NO_ERROR"
    direct_profiles = direct_output.children["profiles"].dataset
    direct_steps = direct_profiles.sizes["time"] - 1

    assert config_data == config_snapshot
    assert receipt.diagnostics.engine_status == "NO_ERROR"
    assert receipt.diagnostics.steps == direct_steps == 1
    assert receipt.provenance.rung is TransportRung.TORAX_MULTI_CHANNEL
    np.testing.assert_allclose(
        receipt.state.ion_temperature,
        np.asarray(direct_profiles["T_i"][-1]),
        rtol=PROFILE_RELATIVE_TOLERANCE,
        atol=PROFILE_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_allclose(
        receipt.state.electron_temperature,
        np.asarray(direct_profiles["T_e"][-1]),
        rtol=PROFILE_RELATIVE_TOLERANCE,
        atol=PROFILE_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_allclose(
        receipt.state.electron_density,
        np.asarray(direct_profiles["n_e"][-1]),
        rtol=PROFILE_RELATIVE_TOLERANCE,
        atol=DENSITY_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_allclose(
        receipt.state.psi,
        np.asarray(direct_profiles["psi"][-1]),
        rtol=PROFILE_RELATIVE_TOLERANCE,
        atol=PROFILE_ABSOLUTE_TOLERANCE,
    )
    print(
        "published-rampup-metrics "
        f"facade_wall_seconds={facade_wall_seconds:.6f} "
        f"facade_steps={receipt.diagnostics.steps} direct_steps={direct_steps}"
    )


def test_published_rampup_trajectory_exercises_moving_grid(monkeypatch):
    """Every saved facade state matches TORAX while the flux grid moves."""
    from torax._src.orchestration.run_simulation import run_simulation
    from torax._src.torax_pydantic.model_config import ToraxConfig

    config_data = _moving_geometry_config()
    direct_config = ToraxConfig.from_dict(copy.deepcopy(config_data))
    request = _published_request(
        config_data,
        direct_config,
        interval_seconds=TRAJECTORY_INTERVAL_SECONDS,
    )
    config_snapshot = copy.deepcopy(config_data)
    facade_executions = []
    run_steps = forward_module._run_torax_steps

    def capture_execution(*args, **kwargs):
        execution = run_steps(*args, **kwargs)
        facade_executions.append(execution)
        return execution

    def reject_nova_geometry(_record):
        raise AssertionError("the published case must retain its CHEASE provider")

    monkeypatch.setattr(forward_module, "_run_torax_steps", capture_execution)
    monkeypatch.setattr(
        forward_module,
        "torax_geometry_from_fsa",
        reject_nova_geometry,
    )
    facade_start = time.perf_counter()
    receipt = ForwardTransport().solve(request)
    facade_wall_seconds = time.perf_counter() - facade_start

    direct_start = time.perf_counter()
    direct_output, direct_history = run_simulation(
        direct_config,
        progress_bar=False,
    )
    direct_wall_seconds = time.perf_counter() - direct_start
    assert direct_history.sim_error.name == "NO_ERROR"
    direct_profiles = direct_output.children["profiles"].dataset
    direct_steps = direct_profiles.sizes["time"] - 1

    assert config_data == config_snapshot
    assert len(facade_executions) == 1
    facade_states = facade_executions[0].states
    assert receipt.diagnostics.engine_status == "NO_ERROR"
    assert receipt.diagnostics.steps == direct_steps >= MINIMUM_TRAJECTORY_STEPS
    assert len(facade_states) == direct_profiles.sizes["time"]
    np.testing.assert_allclose(
        np.asarray([state.t for state in facade_states]),
        np.asarray(direct_profiles.coords["time"]),
        rtol=0.0,
        atol=PROFILE_ABSOLUTE_TOLERANCE,
    )

    profile_channels = (
        ("T_i", "T_i", PROFILE_ABSOLUTE_TOLERANCE),
        ("T_e", "T_e", PROFILE_ABSOLUTE_TOLERANCE),
        ("n_e", "n_e", DENSITY_ABSOLUTE_TOLERANCE),
        ("psi", "psi", PROFILE_ABSOLUTE_TOLERANCE),
    )
    for state_channel, output_channel, absolute_tolerance in profile_channels:
        facade_trajectory = np.stack(
            [
                np.asarray(
                    getattr(state.core_profiles, state_channel).cell_plus_boundaries()
                )
                for state in facade_states
            ]
        )
        np.testing.assert_allclose(
            facade_trajectory,
            np.asarray(direct_profiles[output_channel]),
            rtol=PROFILE_RELATIVE_TOLERANCE,
            atol=absolute_tolerance,
        )

    phi_b_dot = np.asarray(
        [state.geometry.Phi_b_dot for state in facade_states[1:]],
        dtype=np.float64,
    )
    peak_absolute_phi_b_dot = float(np.max(np.abs(phi_b_dot)))
    assert np.any(phi_b_dot != 0.0)
    print(
        "published-rampup-trajectory-metrics "
        f"facade_wall_seconds={facade_wall_seconds:.6f} "
        f"direct_wall_seconds={direct_wall_seconds:.6f} "
        f"facade_steps={receipt.diagnostics.steps} direct_steps={direct_steps} "
        f"peak_absolute_phi_b_dot={peak_absolute_phi_b_dot:.12e}"
    )
