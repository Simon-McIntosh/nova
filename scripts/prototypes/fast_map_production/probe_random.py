"""Scan a fixed-shape map along a reproducible perturbed moment direction."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import os

import numpy as np
import scipy.linalg

from scripts.prototypes.diverted_stall import measure
from scripts.prototypes.fast_map_production import kernel, program


def main():
    seed = int(os.environ["MOMENT_DIRECTION_SEED"])
    original_steps = np.logspace
    original_solve = scipy.linalg.lu_solve

    def fine_steps(start, stop, count, *args, **kwargs):
        if start == -6 and stop == 0 and count == 81:
            return np.linspace(7.49e-5, 8.93e-5, 401)
        return original_steps(start, stop, count, *args, **kwargs)

    def random_direction(factorization, rhs, *args, **kwargs):
        newton = original_solve(factorization, rhs, *args, **kwargs)
        direction = np.random.default_rng(seed).standard_normal(newton.shape)
        direction *= np.linalg.norm(newton) / np.linalg.norm(direction)
        print(
            f"RANDOM_DIRECTION seed={seed} size={direction.size} "
            f"newton_norm={np.linalg.norm(newton):.12g} "
            f"sample_norm={np.linalg.norm(direction):.12g}",
            flush=True,
        )
        return direction

    measure.np.logspace = fine_steps
    scipy.linalg.lu_solve = random_direction
    measure.fixed_shape_read = kernel.read
    measure.mapped_program = program
    measure.map_moments = program.map_moments
    measure.main()


if __name__ == "__main__":
    main()
