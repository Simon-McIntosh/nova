"""Load identical saved arrays before calling the unchanged trace capture."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import hashlib
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.bank_solve_trace import capture


def main():
    assert jax.config.jax_enable_x64
    build = capture.fixture.build
    common = Path(os.environ["NOVA_COMMON_SEEDS"])

    def common_batch(lanes):
        batch = build(lanes)
        with np.load(common) as saved:
            states = tuple(jnp.asarray(saved[f"state_0_{c}"][:lanes]) for c in range(3))
            seeds = jnp.asarray(saved["seeds"][:lanes])
            previous = jnp.asarray(saved["previous"][:lanes])
        capture.emit(
            "COMMON_SEEDS",
            path=str(common),
            variant=0,
            sha256=hashlib.sha256(common.read_bytes()).hexdigest(),
        )
        return replace(batch, states=states, seeds_batch=seeds, previous_batch=previous)

    capture.fixture.build = common_batch
    capture.main()


if __name__ == "__main__":
    main()
