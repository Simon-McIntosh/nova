"""Run the landed noise-floor instrument on the saved common seed population."""

import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from scripts.prototypes.bank_residual_noise_floor import measure


def main():
    build = measure.fixture.build

    def common(lanes, *args, **kwargs):
        batch = build(lanes, *args, **kwargs)
        with np.load(os.environ["NOVA_COMMON_SEEDS"]) as archive:
            return replace(
                batch,
                states=tuple(jnp.asarray(archive[f"state_0_{i}"]) for i in range(3)),
                seeds_batch=jnp.asarray(archive["seeds"]),
                previous_batch=jnp.asarray(archive["previous"]),
            )

    measure.fixture.build = common
    measure.main()


if __name__ == "__main__":
    main()
