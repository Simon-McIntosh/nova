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
PROFILE_RELATIVE_TOLERANCE = 1.0e-10
PROFILE_ABSOLUTE_TOLERANCE = 1.0e-12
DENSITY_ABSOLUTE_TOLERANCE = 1.0e8


def _published_config():
    from torax.examples.iterhybrid_rampup import CONFIG

    config = copy.deepcopy(CONFIG)
    config["numerics"]["t_final"] = REFERENCE_INTERVAL_SECONDS
    config["numerics"]["exact_t_final"] = True
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


def _published_request(config_data, config):
    initial_state = _initial_state(config)
    initial_current = 3.0e6
    final_current = initial_current + (10.5e6 - initial_current) * (
        REFERENCE_INTERVAL_SECONDS / 80.0
    )
    return ForwardTransportInput(
        geometry=TransportGeometry({"valid": True}),
        initial_state=initial_state,
        waveforms=TransportWaveforms(
            time=np.array([0.0, REFERENCE_INTERVAL_SECONDS]),
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
