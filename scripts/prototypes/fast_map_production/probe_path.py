"""Inspect path weights on both sides of a support exchange."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from scripts.prototypes.diverted_stall import measure
from scripts.prototypes.fast_map_production import kernel, program


def main():
    original = np.logspace

    def sampled_steps(start, stop, count, *args, **kwargs):
        if start == -6 and stop == 0 and count == 81:
            return np.linspace(7.8716e-5, 7.8752e-5, 101)
        return original(start, stop, count, *args, **kwargs)

    def traced_read(*args, **kwargs):
        reading = kernel.read(*args, **kwargs)
        jax.debug.print(
            "PATH_WEIGHT cell71={weight} geom71={geometry} member71={member} "
            "weights={weights} normal={normal} booking={booking} saddle={saddle}",
            weight=reading.path_weight[71],
            geometry=reading.geometric_membership[71],
            member=reading.membership[71],
            weights=reading.path_weight[np.asarray([4, 6, 9, 71])],
            normal=reading.normal_form_cells[np.asarray([4, 6, 9, 71])],
            booking=reading.booking_cells[np.asarray([4, 6, 9, 71])],
            saddle=reading.saddle_form.position,
            ordered=True,
        )
        return reading

    measure.np.logspace = sampled_steps
    measure.fixed_shape_read = traced_read
    measure.mapped_program = program
    measure.map_moments = program.map_moments
    measure.main()


if __name__ == "__main__":
    main()
