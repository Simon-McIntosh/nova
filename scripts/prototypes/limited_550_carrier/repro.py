"""Reproduce the limited 550-cell certificate carrier refusal."""

from __future__ import annotations

import sys
import traceback

from nova.jax.config import configure_dtypes

configure_dtypes()

import nova  # noqa: E402
from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium import topology  # noqa: E402
from nova.equilibrium.separatrix_clip import AtomicCellMesh  # noqa: E402
from nova.equilibrium.stencil_mesh import canonical_polygons  # noqa: E402

CASE = "weak-rotation-reactor-static"
REQUESTED = -500


def main() -> int:
    print(f"nova={nova.__file__}")
    print(f"case={CASE} requested_cells={REQUESTED}")
    carrier, source, exact = certificate._case(CASE, clip_mode="exact")
    machine = certificate._case_machine(CASE, carrier, exact, REQUESTED)
    print(f"realised_cells={len(machine.node)} cache_hit={machine.cache.get('hit')}")
    polygons = machine.cell_polygons
    canonical = canonical_polygons(polygons)
    for label, cells in (("raw", polygons), ("canonical", canonical)):
        try:
            mesh = AtomicCellMesh.from_cells(cells)
        except Exception:
            print(f"ATOMIC-REFUSED label={label}")
            traceback.print_exc()
        else:
            print(f"{label}: atomic nodes={len(mesh.node_coordinates)}")
    try:
        topology.TopologyGeometry.from_cells(
            polygons, machine.sampling_vertices, (machine.wall_node,)
        )
    except Exception:
        print("TOPOLOGY-GEOMETRY-REFUSED")
        traceback.print_exc()
        return 1
    print("TopologyGeometry.from_cells built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
