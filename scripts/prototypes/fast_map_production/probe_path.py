"""Inspect path weights on both sides of a support exchange."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import topology
from scripts.prototypes.diverted_stall import measure
from scripts.prototypes.fast_map_production import kernel, program


def main():
    original = np.logspace

    def sampled_steps(start, stop, count, *args, **kwargs):
        if start == -6 and stop == 0 and count == 81:
            return np.asarray((7.872248e-5, 7.872284e-5))
        return original(start, stop, count, *args, **kwargs)

    def traced_read(*args, **kwargs):
        reading = kernel.read(*args, **kwargs)
        geometry = args[2]
        direction = reading.axis - reading.saddle_form.position
        nearby = reading.saddle_form.position + direction * (
            0.5 * jnp.median(geometry.pitch) / jnp.linalg.norm(direction)
        )
        global_sector = topology._normal_form_sectors(
            reading.axis[None], reading.saddle_form
        )[0]
        local_sector = topology._normal_form_sectors(nearby[None], reading.saddle_form)[
            0
        ]
        jax.debug.print(
            "PATH_WEIGHT cell71={weight} geom71={geometry} member71={member} "
            "weights={weights} normal={normal} booking={booking} saddle={saddle} "
            "global_sector={global_sector} local_sector={local_sector} "
            "positive={positive} rays={rays}",
            weight=reading.path_weight[71],
            geometry=reading.geometric_membership[71],
            member=reading.membership[71],
            weights=reading.path_weight[np.asarray([4, 6, 9, 71])],
            normal=reading.normal_form_cells[np.asarray([4, 6, 9, 71])],
            booking=reading.booking_cells[np.asarray([4, 6, 9, 71])],
            saddle=reading.saddle_form.position,
            global_sector=global_sector,
            local_sector=local_sector,
            positive=reading.saddle_form.positive,
            rays=reading.saddle_form.direction,
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
