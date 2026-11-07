"""Print the axis-admission conjuncts the frame 144 limited seed was refused on.

The seed is read through the operator's own qualification path with
``qualified_o_candidates`` wrapped, so the values printed are the ones the
refusal was computed from rather than a reimplementation of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from nova.equilibrium.topology import TopologyClass  # noqa: E402
import benchmarks.diiid_forward_gs_match as bench  # noqa: E402

def _float(value) -> float | None:
    array = np.asarray(value)
    return float(array) if array.size == 1 and np.all(np.isfinite(array)) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
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
        floor = int(max(np.floor(0.01 * material_cell_count), 1))
        finite_o = np.asarray(jnp.isfinite(vmap_o[:, 0]))
        data_o = self.o_point_data(vmap_o, polarity)
        data_x = self.x_point_data(vmap_x, polarity, data_o[2])
        data_b = self.boundary(data_o, vmap_x, data_w, polarity)
        closed = self.psi_mask(polarity, psi_grid, data_b[2])
        seed_material = inside_material
        shapes = {
            "vmap_o": list(np.shape(vmap_o)),
            "data_o": [list(np.shape(part)) for part in data_o],
            "data_x": [list(np.shape(part)) for part in data_x],
            "data_b": [list(np.shape(part)) for part in data_b],
            "closed": list(np.shape(closed)),
        }
        entry = {
            "shapes": shapes,
            "material_cell_count": material_cell_count,
            "connection_floor_cells": floor,
            "candidates": [],
        }
        records.append(entry)
        candidates = entry["candidates"]
        for index in range(np.asarray(vmap_o).shape[0]):
            row = {
                "index": int(index),
                "r": _float(vmap_o[index, 0]),
                "z": _float(vmap_o[index, 1]),
                "flux": _float(data_o[2][index]),
                "finite_o": bool(finite_o[index]),
                "data_b": [_float(data_b[k][index]) for k in range(3)],
                "resolved": bool(np.all(np.isfinite(np.asarray(data_b[:3])[..., index] if np.asarray(data_b).ndim > 2 else np.asarray([data_b[k][index] for k in range(3)])))),
            }

            def component_size(mask) -> int:
                component = self.axis_component(
                    psi_grid,
                    data_b[2],
                    data_o[2],
                    data_o[:2],
                    mask,
                    seed_material,
                    jnp.equal(data_b[2], data_x[2]),
                    data_x[:2],
                    surface,
                    polarity=polarity,
                )
                return int(np.asarray(jnp.sum(component & inside_material)))

            try:
                row["governed_size"] = component_size(closed)
                row["governed_size_all_open"] = component_size(jnp.ones_like(closed))
                row["error"] = None
            except Exception as error:  # recorded, not swallowed silently
                row["governed_size"] = None
                row["governed_size_all_open"] = None
                row["error"] = repr(error)[:400]
            candidates.append(row)
        entry["completed"] = True
        return original(
            self, vmap_o, vmap_x, data_w, polarity, psi_grid, inside_material, surface
        )

    import jax

    kind.qualified_o_candidates = wrapper
    psi = np.asarray(prepared["cold"].branches.flux[0])
    material = np.asarray(profile.operator.inside_material, dtype=bool)
    polarity = int(getattr(profile.operator, "polarity", 1))
    status = "read_qualification returned"
    try:
        with jax.disable_jit():
            topology.read_qualification(
                psi,
                polarity,
                material,
                requested_class=int(TopologyClass.LIMITED),
            )
    except Exception as error:
        status = repr(error)[:400]
    finally:
        kind.qualified_o_candidates = original

    payload = {"read_status": status, "passes": records}
    arguments.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
