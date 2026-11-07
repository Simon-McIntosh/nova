"""Print the axis-admission conjuncts for the frame 144 limited field.

The field is read through the operator's own qualification path with
``qualified_o_candidates`` wrapped, so the values printed are the ones the
qualification computed rather than a reimplementation of them. The field is
either the cached limited seed or one captured at a refusal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
for extra in (
    REPOSITORY,
    REPOSITORY / "scripts" / "prototypes" / "diiid_axis_identification",
    REPOSITORY / "scripts" / "prototypes" / "diiid_forward_convergence",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import measure_branches  # noqa: E402
from nova.equilibrium.topology import (  # noqa: E402
    _MATERIAL_CONNECTED_FRACTION,
    TopologyClass,
)
import benchmarks.diiid_forward_gs_match as bench  # noqa: E402

def _first(value) -> float | None:
    array = np.ravel(np.asarray(value))
    return float(array[0]) if array.size and np.isfinite(array[0]) else None


def _float(value) -> float | None:
    array = np.asarray(value)
    return float(array) if array.size == 1 and np.all(np.isfinite(array)) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capture",
        type=Path,
        default=None,
        help="npz written by capture_refusal.py; read its field instead of the cached seed",
    )
    arguments = parser.parse_args()

    expansion = float(bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION)
    prepared = measure_branches._prepare_frame(144, expansion)
    profile = prepared["profile"]
    topology = profile.operator.topology
    kind = type(topology)
    original = kind.qualified_o_candidates
    records: list[dict] = []

    def wrapper(self, vmap_o, vmap_x, data_w, polarity, psi_grid, inside_material, surface=None):
        import jax.numpy as jnp

        material_cell_count = int(np.asarray(jnp.sum(inside_material)))
        floor = int(
            max(np.floor(_MATERIAL_CONNECTED_FRACTION * material_cell_count), 1)
        )
        finite_o = np.asarray(jnp.isfinite(vmap_o[:, 0]))
        shapes = {
            "vmap_o": list(np.shape(vmap_o)),
            "vmap_x": list(np.shape(vmap_x)),
            "inside_material": list(np.shape(inside_material)),
        }
        entry = {
            "shapes": shapes,
            "material_cell_count": material_cell_count,
            "connection_floor_cells": floor,
            "candidates": [],
        }
        records.append(entry)

        coordinate = self.grid.coordinate
        finite_o = jnp.isfinite(vmap_o[:, 0])
        owner_index = jax.vmap(
            lambda position: jnp.argmin(
                jnp.sum((coordinate - position) ** 2, axis=1)
            )
        )(vmap_o[:, :2])
        seedable = (
            jnp.zeros(coordinate.shape[0], dtype=bool).at[owner_index].max(finite_o)
        )
        seed_material = inside_material | seedable

        def qualify(data_o):
            data_x = self.x_point_data(vmap_x, polarity, data_o[2])
            data_b = self.boundary(data_o, vmap_x, data_w, polarity)
            closed = self.psi_mask(polarity, psi_grid, data_b[2])
            component = self.axis_component(
                psi_grid,
                data_b[2],
                data_o[2],
                data_o[:2],
                closed,
                seed_material,
                jnp.equal(data_b[2], data_x[2]),
                data_x[:2],
                surface,
                polarity=polarity,
            )
            open_component = self.axis_component(
                psi_grid,
                data_b[2],
                data_o[2],
                data_o[:2],
                jnp.ones_like(closed),
                seed_material,
                jnp.equal(data_b[2], data_x[2]),
                data_x[:2],
                surface,
                polarity=polarity,
            )
            governed_size = jnp.sum(component & inside_material)
            governed_size_all_open = jnp.sum(open_component & inside_material)
            resolved = jnp.all(jnp.isfinite(data_b[:3]))
            admitted = (
                jnp.all(jnp.isfinite(data_o[:3])) & resolved & (governed_size >= floor)
            )
            return (
                admitted,
                governed_size,
                governed_size_all_open,
                resolved,
                jnp.isfinite(data_o[0]),
                data_o[2],
                data_b[0],
                data_b[1],
                data_b[2],
                component & inside_material,
            )

        measured = jax.vmap(qualify)(jnp.asarray(vmap_o))
        (admitted, governed, open_governed, resolved, finite,
         axis_flux, b0, b1, b2, masks) = measured
        candidates = entry["candidates"]
        for index in range(int(finite is not None and np.shape(finite)[0] or 0)):
            candidates.append(
                {
                    "index": index,
                    "r": _first(vmap_o[index, 0]),
                    "z": _first(vmap_o[index, 1]),
                    "flux": _first(axis_flux[index]),
                    "finite_o": bool(np.asarray(finite[index])),
                    "resolved": bool(np.asarray(resolved[index])),
                    "data_b": [
                        _first(b0[index]),
                        _first(b1[index]),
                        _first(b2[index]),
                    ],
                    "governed_size": int(np.asarray(governed[index])),
                    "governed_size_all_open": int(np.asarray(open_governed[index])),
                    "governed_inside_material": int(
                        np.asarray(jnp.sum(masks[index]))
                    ),
                    "admitted": bool(np.asarray(admitted[index])),
                    "connection_floor_cells": floor,
                }
            )
        entry["completed"] = True
        return original(
            self, vmap_o, vmap_x, data_w, polarity, psi_grid, inside_material, surface
        )

    if arguments.capture is None:
        psi = np.asarray(
            prepared["cold"].branches.flux[int(TopologyClass.LIMITED)], dtype=float
        )
        material = np.asarray(profile.operator.inside_material, dtype=bool)
        requested = int(TopologyClass.LIMITED)
        source = "cached limited seed"
    else:
        held = np.load(arguments.capture)
        psi = np.asarray(held["psi"], dtype=float)
        material = np.asarray(held["inside_material"], dtype=bool)
        requested = int(held["requested_class"])
        if requested < 0:
            requested = None
        source = f"captured field {arguments.capture.name}"
    polarity = int(getattr(profile.operator, "polarity", 1))
    status = "read_qualification returned"
    try:
        kind.qualified_o_candidates = wrapper
        with jax.disable_jit():
            topology.read_qualification(
                psi,
                polarity,
                material,
                requested_class=requested,
            )
    except Exception as error:
        status = repr(error)[:400]
    finally:
        kind.qualified_o_candidates = original

    payload = {"read_status": status, "field_source": source, "passes": records}
    arguments.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
