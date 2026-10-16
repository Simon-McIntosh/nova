"""Scan a stalled correction from zero to the first rejected line-search scales."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import numpy as np

from scripts.prototypes.diverted_stall import measure
from scripts.prototypes.fast_map_production import kernel, program


def main():
    original_steps = np.logspace

    def fine_steps(start, stop, count, *args, **kwargs):
        if start == -6 and stop == 0 and count == 81:
            return np.linspace(0.0, 0.03, 401)
        return original_steps(start, stop, count, *args, **kwargs)

    measure.np.logspace = fine_steps
    measure.fixed_shape_read = kernel.read
    measure.mapped_program = program
    measure.map_moments = program.map_moments
    measure.main()


if __name__ == "__main__":
    main()
