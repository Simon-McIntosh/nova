"""Rerun the unchanged fast read and moment map at a recorded state."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from scripts.prototypes.diverted_stall import measure
from scripts.prototypes.fast_read import program


def main():
    measure.mapped_program = program
    measure.map_moments = program.map_moments
    measure.main()


if __name__ == "__main__":
    main()
