"""Run the merged nova read on the captured escape comparison records."""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import nova  # noqa: E402
from nova.equilibrium import topology  # noqa: E402
from nova.equilibrium.topology import TopologyPolicy  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402
from scripts.prototypes.efit_field_swap.crosscheck import (  # noqa: E402
    domain_rings,
    fitted_geometry,
    lattice,
    read_trace_record,
)
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField  # noqa: E402
from scripts.prototypes.efit_field_swap.orientation import read_convention  # noqa: E402
from scripts.prototypes.efit_maximin_certificate.run import ROWS  # noqa: E402
from scripts.prototypes.fast_map_production.kernel import (  # noqa: E402
    _wall_path_ceiling,
    read as nova_read,
)
from scripts.prototypes.read_robustness.census import census  # noqa: E402


@dataclass(frozen=True)
class CensusField:
    """Expose the spline's sampled values to the shared rounding guard."""

    base: TraceSplineField
    jets: jax.Array
    sample_index: jax.Array

    def value(self, point):
        return self.base.value(point)

    def evaluate(self, point):
        return self.base.evaluate(point)

    def sampled_coefficient(self, geometry):
        return self.base.sampled_coefficient(geometry)

    def cell_value(self, index, point):
        return self.base.value(point)


jax.tree_util.register_dataclass(
    CensusField, data_fields=["base", "jets", "sample_index"], meta_fields=[]
)


def decision(record):
    """Use nova's own initial stationary roots with the merged fast read."""
    base = TraceSplineField.from_record(record)
    boundary, obstacles = domain_rings(record)
    convention = read_convention(polarity=record.polarity)
    policy = TopologyPolicy()
    for resolution in (16, 24, 32):
        cells, samples = lattice(boundary, resolution, obstacles)
        geometry, fitted = fitted_geometry(cells, samples, (boundary, *obstacles))
        if geometry is None:
            continue
        sampled = jax.vmap(base.value)(geometry.sample_points.reshape(-1, 2))
        sample_index = jnp.arange(len(sampled)).reshape(
            geometry.sample_points.shape[:2]
        )
        field = CensusField(base, sampled[:, None], sample_index)
        initial = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
        valid = np.asarray(initial.x_point_valid)
        previous = jnp.asarray(
            np.vstack(
                (np.asarray(initial.axis)[None], np.asarray(initial.x_points)[valid])
            )
        )
        roots = census(field, geometry, convention, policy, previous)
        if int(roots.saddles) == 0:
            continue
        reading = nova_read(field, roots.roots, geometry, convention, policy)
        if not bool(np.asarray(reading.valid)):
            continue
        wall_cell = int(
            np.argmax(np.asarray(topology._inside_cells(geometry, reading.boundary)))
        )
        seed = jnp.argmin(jnp.sum((geometry.centre - reading.axis) ** 2, axis=1))
        ceiling = _wall_path_ceiling(
            geometry,
            reading.field_coefficients,
            reading.saddle_form,
            reading.axis,
            reading.normal_form_cells,
            reading.saddle_candidate,
            seed,
            convention.sigma,
            True,
        )
        return {
            "resolution": resolution,
            "cells": len(cells),
            "capacity_fit": fitted,
            "axis": np.asarray(reading.axis).tolist(),
            "boundary": np.asarray(reading.boundary).tolist(),
            "boundary_flux": float(np.asarray(reading.boundary_flux)),
            "boundary_class": int(np.asarray(reading.boundary_class)),
            "wall_level": float(np.asarray(reading.wall_candidate)),
            "saddle_level": float(np.asarray(reading.saddle_candidate)),
            "selected_cells": int(np.count_nonzero(np.asarray(reading.selected_cells))),
            "contact_cell": wall_cell,
            "contact_cell_ceiling": float(np.asarray(ceiling)[wall_cell]),
            "contact_cell_selected": bool(
                np.asarray(reading.selected_cells)[wall_cell]
            ),
            "contact_cell_path_weight": float(
                np.asarray(reading.path_weight)[wall_cell]
            ),
            "root_valid": np.asarray(roots.roots.valid).tolist(),
            "root_candidates": int(roots.candidates),
            "root_saddles": int(roots.saddles),
            "initial_axis": np.asarray(initial.axis).tolist(),
        }
    raise RuntimeError("no valid merged read at resolutions 16, 24, 32")


def main(output: pathlib.Path, cases: tuple[str, ...] = ()):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    print(f"nova.__file__={nova.__file__}", flush=True)
    rows = []
    for spec in ROWS:
        if cases and spec["case"] not in cases:
            continue
        record = read_trace_record(spec["source"])
        result = decision(record)
        rows.append({"case": spec["case"], "source": spec["source"], **result})
        print(json.dumps(rows[-1]), flush=True)
        output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), tuple(sys.argv[2:]))
