"""Expose the read's saddle ranking and support gates on the moving field."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import nova

from nova.equilibrium import topology
from scripts.prototypes.fast_map_production import kernel, tie_probe
from tests.equilibrium.test_topology_read import (
    _census_cache,
    _census_field,
    _census_tie_setup,
)


def main():
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} kernel={kernel.__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    state, prior, _ = _census_tie_setup(_census_cache())
    geometry = state["geometry"]
    sampled = tie_probe.make_sampler(state)
    sigma = float(state["convention"].sigma)
    for offset in (0.1732, 0.17325):
        field = _census_field(state, offset)
        roots, reading, booked, *_ = sampled(offset * float(state["scale"]), prior)
        axis = np.asarray(reading.axis)
        seed = jnp.argmin(jnp.sum((geometry.centre - reading.axis) ** 2, axis=1))
        ceiling = kernel._wall_path_ceiling(
            geometry,
            field.sampled_coefficient(geometry),
            None,
            reading.axis,
            None,
            -jnp.inf,
            seed,
            sigma,
            False,
        )
        saddles = []
        for index in np.flatnonzero(np.asarray(roots.valid))[1:]:
            position = np.asarray(roots.position[index])
            cell = int(
                np.argmax(
                    np.asarray(topology._inside_cells(geometry, roots.position[index]))
                )
            )
            level = float(roots.jet.value[index])
            passage = float(ceiling[cell])
            saddles.append(
                {
                    "index": int(index),
                    "position": position.tolist(),
                    "cell": cell,
                    "level": level,
                    "coarse_passage": passage,
                    "ranking_margin": passage - sigma * level,
                }
            )
        support = np.flatnonzero(np.asarray(reading.membership) > 0)
        row = {
            "offset": offset,
            "axis": axis.tolist(),
            "saddles": saddles,
            "selected_form": np.asarray(reading.saddle_form.position).tolist(),
            "boundary_class": int(reading.boundary_class),
            "boundary_flux": float(reading.boundary_flux),
            "wall_candidate": float(reading.wall_candidate),
            "saddle_candidate": float(reading.saddle_candidate),
            "booked": float(booked),
            "support": support.tolist(),
            "support_path_weights": np.asarray(reading.path_weight)[support].tolist(),
            "support_geometric_membership": np.asarray(reading.geometric_membership)[
                support
            ].tolist(),
            "support_membership": np.asarray(reading.membership)[support].tolist(),
        }
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
