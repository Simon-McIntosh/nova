"""Focused contracts for active-set streaming and receipt serialization."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import diiid_forward_gs_match
from nova.equilibrium.fixed_point import newton_krylov


CORROBORATION_SCRIPT = (
    Path(__file__).parents[1]
    / "docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py"
)


def _corroboration_module():
    spec = spec_from_file_location(
        "efit_topology_corroboration_telemetry", CORROBORATION_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _solve_with_streaming(enabled: bool):
    def mask_fn(state):
        return state >= 0.5

    def shadowed_map(_state, mask):
        return jnp.where(mask, 2.0, 1.0)

    def solve():
        return newton_krylov(
            lambda state: shadowed_map(state, mask_fn(state)),
            jnp.zeros(1),
            newton_steps=1,
            gmres_iterations=1,
            warmup=0,
            shadow_mask_fn=mask_fn,
            promoted_shadow_mask_fn=lambda state, _previous: mask_fn(state),
            shadowed_map_fn=shadowed_map,
            active_set_steps=4,
            stream_active_set=enabled,
        )

    result = jax.jit(solve)()
    result.state.block_until_ready()
    jax.effects_barrier()
    return result


def test_streaming_reports_each_trip_without_changing_numerics(capsys):
    quiet = _solve_with_streaming(False)
    assert capsys.readouterr().out == ""

    streamed = _solve_with_streaming(True)
    lines = capsys.readouterr().out.splitlines()

    np.testing.assert_array_equal(streamed.state, quiet.state)
    np.testing.assert_array_equal(streamed.residual, quiet.residual)
    np.testing.assert_array_equal(
        streamed.active_set_residuals, quiet.active_set_residuals
    )
    np.testing.assert_array_equal(
        streamed.active_set_mask_differences, quiet.active_set_mask_differences
    )
    assert len(lines) == int(streamed.active_set_iterations) == 2
    for trip_index, line in enumerate(lines):
        assert f"trip={trip_index}" in line
        assert "mask_difference=" in line
        assert "live_residual=" in line
        assert "inner_iterations=" in line


def test_streaming_omits_masked_trips_from_batched_solves(capsys):
    def lane(initial):
        def mask_fn(state):
            return state >= 0.5

        def shadowed_map(_state, mask):
            return jnp.where(mask, 2.0, 1.0)

        return newton_krylov(
            lambda state: shadowed_map(state, mask_fn(state)),
            initial,
            newton_steps=1,
            gmres_iterations=1,
            warmup=0,
            shadow_mask_fn=mask_fn,
            promoted_shadow_mask_fn=lambda state, _previous: mask_fn(state),
            shadowed_map_fn=shadowed_map,
            active_set_steps=4,
            stream_active_set=True,
        )

    result = jax.jit(jax.vmap(lane))(jnp.zeros((2, 1)))
    result.state.block_until_ready()
    jax.effects_barrier()
    lines = capsys.readouterr().out.splitlines()

    np.testing.assert_array_equal(result.state, np.full((2, 1), 2.0))
    assert len(lines) == int(np.sum(result.active_set_iterations)) == 4
    assert not any("trip=2" in line or "trip=3" in line for line in lines)


def test_receipt_adapters_slice_fixed_arrays_to_the_executed_trip_count():
    result = SimpleNamespace(
        active_set_iterations=np.asarray(2),
        active_set_residuals=np.asarray((0.5, np.nan, np.nan, np.nan)),
        active_set_mask_differences=np.asarray((1, 0, -1, -1)),
        active_set_cycle_damping_activations=np.asarray((0, 1, -1, -1)),
    )
    expected = {
        "active_set_iterations": 2,
        "active_set_residuals": [0.5, None],
        "active_set_mask_differences": [1, 0],
        "active_set_cycle_damping_activations": [0, 1],
    }

    corroboration = _corroboration_module()._active_set_receipt(result)
    forward_match = diiid_forward_gs_match._strict_json_value(
        diiid_forward_gs_match._active_set_receipt(result)
    )

    assert corroboration == expected
    assert forward_match == expected
    for receipt in (corroboration, forward_match):
        assert len(receipt["active_set_residuals"]) == receipt["active_set_iterations"]
        assert (
            len(receipt["active_set_mask_differences"])
            == receipt["active_set_iterations"]
        )
        assert (
            len(receipt["active_set_cycle_damping_activations"])
            == receipt["active_set_iterations"]
        )
        json.dumps(receipt, allow_nan=False)
