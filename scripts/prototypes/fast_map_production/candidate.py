"""Measure the continuous read with one moment representation per cell."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from scripts.prototypes.diverted_stall import measure
from scripts.prototypes.fast_map_production import kernel, program


def main():
    measure.fixed_shape_read = kernel.read
    measure.mapped_program = program
    measure.map_moments = program.map_moments
    measure.main()


if __name__ == "__main__":
    main()
