"""Cross-device pair-block sharding for the tiled polygon operator."""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def run_child():
    """Compare one-device and four-device results in a fresh JAX process."""
    from nova.biot.polygon import pad_batch
    from nova.biot.tiledassembly import TilePlan, tile_evaluator

    section = np.array([[5.9, -0.1], [6.1, -0.1], [6.1, 0.1], [5.9, 0.1]])
    edge, weight, norm = pad_batch([section, section])
    target_r = np.array([5.7, 6.3])
    target_z = np.array([0.2, -0.2])
    plan = TilePlan(2, 2, 1, 1, 2)
    one = np.stack(
        tile_evaluator(plan, batched=True, devices=1)(
            target_r, target_z, edge, weight, norm
        )
    )
    mapped = tile_evaluator(plan, batched=True, devices=4)
    four = np.stack(mapped(target_r, target_z, edge, weight, norm))
    np.testing.assert_allclose(four, one, rtol=2e-12, atol=1e-18)
    assert mapped.compile_count == 1


def test_pair_blocks_are_sharded_without_moving_the_operator():
    script = pathlib.Path(__file__).resolve()
    environment = os.environ.copy()
    environment.update(
        NOVA_MULTI_DEVICE_CHILD="1",
        JAX_PLATFORMS="cpu",
        NOVA_COMPILATION_CACHE="off",
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__" and os.environ.get("NOVA_MULTI_DEVICE_CHILD"):
    run_child()
