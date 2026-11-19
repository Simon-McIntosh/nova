"""The production point program's arms and the shapes each consumer asks for.

The point program is one jitted body, ``fixed_points._row_program`` for the row
window and the jets build's own ``jax.jit(jax.vmap(_source_jets))``. Either
lowers to one program per point-batch width and per source-cell count, so an arm
is named by the consumer it serves. The census arms resolve their coupling
through the certified machine store exactly as the caller does, so the program a
fill compiles here is the program that caller requests; the resolved cell and
node counts are recorded rather than assumed.
"""

# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass
import json
import os

from nova.jax.config import configure_dtypes

configure_dtypes()

from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402


@dataclass(frozen=True)
class Arm:
    """One measured shape of the production point program."""

    name: str
    batch: int
    requested_cells: int | None
    hexagon_cells: int | None
    node_target: int | None = None


ARMS = (
    Arm("row-132", 3, None, 132),
    Arm("census-250", 4, 250, None),
    Arm("census-500", 4, 500, None),
    Arm("census-2132", 4, 500, None, node_target=2132),
)
"""The row window at 132 cells, the 282-cell census build, the build the
resolver serves and the 2,132-node certified build the plan names.

The 500-cell request has more than one certified build in the store, and the
production resolver takes the first in key order, which holds 550 polygons. The
2,132-node build holds 551 and is a different program (the same body at another
cell count), so it is named by its node count rather than left to that order.
"""

ARM_BY_NAME = {arm.name: arm for arm in ARMS}


def unique_nodes(geometry) -> np.ndarray:
    """Return the geometry's node set exactly as both consumers deduplicate it."""
    samples = np.asarray(geometry.sample_points)
    nodes, _ = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    return nodes


def census_cache() -> Path:
    """The certified machine store the census build reads.

    A launch recipe that points the fixture store elsewhere names the store this
    row reads explicitly, so the build still resolves the certified machine
    instead of an empty directory.
    """
    override = os.environ.get("NOVA_CACHED_POINT_MACHINE_CACHE")
    if override:
        return Path(override)
    from tests.equilibrium.test_topology_read import _census_cache

    return _census_cache()


def _census_machine(carrier, cells: int, node_target: int | None):
    """Return the certified census build a request resolves to.

    Without a node target this is the production path's own first match. With
    one, the candidates are walked in the same key order the production resolver
    uses and the first build whose node set has that many nodes is returned, so a
    build the resolver does not serve can still be measured by name.
    """
    from nova.database.zarrstore import ZarrStore
    from nova.equilibrium import topology
    from scripts.analytic_oracle_fixtures import measure as fixture
    from scripts.prototypes.solve_options.profile import _machine

    directory = census_cache()
    if node_target is None:
        return _machine(directory, carrier, cells)
    filename = f"{fixture.CACHE_FILENAME}_{cells}"
    root = directory / f"{filename}.zarr"
    for group in sorted(root.iterdir()):
        header = group / "zarr.json"
        if not header.is_file():
            continue
        attrs = json.loads(header.read_text())["attributes"]
        identity = json.loads(attrs["semantic_identity"])
        if not (
            identity["analytic_case"] == carrier.name
            and identity["discretisation"]["requested_cells"] == -cells
            and identity.get("builder") in ("device", "host")
        ):
            continue
        key = attrs["cache_key"]
        store = ZarrStore(filename=filename, dirname=directory, group=key)
        store.load()
        machine = fixture._from_dataset(store.data, identity, key)
        geometry = topology.TopologyGeometry.from_cells(
            machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
        )
        if len(unique_nodes(geometry)) == node_target:
            machine.cache.update(semantic_key=key, hit=True, readonly=True)
            return machine, key, identity["builder"]
    raise FileNotFoundError(
        f"no certified build of {filename} holds {node_target} nodes"
    )


def resolve(name: str) -> dict:
    """Return the program, coupling and points one arm hands the compiler."""
    import jax

    from nova.equilibrium import fixed_points, topology
    from nova.equilibrium.fixed_points import _source_jets

    arm = ARM_BY_NAME[name]
    machine_key, machine_builder = "", ""
    if arm.hexagon_cells is not None:
        from tests.equilibrium import test_topology_read as source

        oracle, total, wall, _, _ = source._analytic_inputs("limited")
        geometry = source._realised_hex_geometry(wall, arm.hexagon_cells)
        field = source._kernel_backed_field("limited", oracle, total, geometry)
        coupling = field.coupling
        program = fixed_points._row_program
        shape = "row"
    else:
        from benchmarks import solovev_certificate as certificate

        carrier, _, _ = certificate._case("diverted-single-null", clip_mode="exact")
        machine, machine_key, machine_builder = _census_machine(
            carrier, arm.requested_cells, arm.node_target
        )
        geometry = topology.TopologyGeometry.from_cells(
            machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
        )
        coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
        # The jets build binds its own program object. The body is the same
        # program, so using the consumer's own object is what shows the entry a
        # fill writes is the entry that build reads.
        program = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))
        shape = "jets"
    return {
        "program": program,
        "coupling": coupling,
        "nodes": unique_nodes(geometry),
        "shape": shape,
        "source_cells": int(len(coupling.centre)),
    }


def batches(nodes: np.ndarray, batch: int):
    """Yield ``(first, last, points)`` exactly as the two consumers pad a batch."""
    for first in range(0, len(nodes), batch):
        last = min(first + batch, len(nodes))
        points = nodes[first:last]
        if len(points) < batch:
            padding = np.repeat(points[-1:], batch - len(points), axis=0)
            points = np.concatenate((points, padding))
        yield first, last, points


def desired_subset(nodes: np.ndarray, count: int | None) -> np.ndarray:
    """Return the node indices one arm compares, spread over the sorted grid."""
    if count is None or count >= len(nodes):
        return np.arange(len(nodes))
    return np.unique(np.linspace(0, len(nodes) - 1, count).round().astype(int))


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["list"]:
        print("\n".join(arm.name for arm in ARMS))
    elif arguments[:1] == ["resolve"]:
        for name in arguments[1:] or [arm.name for arm in ARMS]:
            resolved = resolve(name)
            print(
                "ARM=%s shape=%s source_cells=%d nodes=%d batch=%d"
                % (
                    name,
                    resolved["shape"],
                    resolved["source_cells"],
                    len(resolved["nodes"]),
                    ARM_BY_NAME[name].batch,
                ),
                flush=True,
            )
    else:
        raise SystemExit("usage: arms.py list | arms.py resolve [name ...]")